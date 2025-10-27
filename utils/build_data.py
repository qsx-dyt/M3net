import numpy as np
from typing import Tuple, Union, Optional
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class HSIDatasetBuilder:
    def __init__(self, data_name, data: np.ndarray, labels: np.ndarray, patch_size: int = 5, verbose: bool = True, pca_components: Optional[int] = 50):
        """
        初始化高光谱数据集构建器
        Args:
            data: 高光谱数据，形状为 (H, W, C)
            labels: 标签数据，形状为 (H, W)
            patch_size: patch大小，默认为5
        """
        self.data_name = data_name
        self.data = data.astype(np.float32)
        self.labels = labels
        self.patch_size = patch_size
        self.height, self.width, self.channels = data.shape
        self.num_classes = len(np.unique(labels)) - 1  # 不包括背景类
        self.verbose = verbose

        self.augmenter = HSIAugmenter()
        # 正则化
        for i in range(self.channels):
            input_max = np.max(data[:, :, i])
            input_min = np.min(data[:, :, i])
            self.data[:, :, i] = (data[:, :, i] - input_min) / (input_max - input_min)

        # 应用PCA降维
        if pca_components is not None:
            # 重塑数据为二维数组 (H*W, C)
            h, w, c = self.data.shape
            data_reshaped = self.data.reshape(-1, c)
            # 初始化并拟合PCA
            pca = PCA(n_components=pca_components)
            data_pca = pca.fit_transform(data_reshaped)
            # 重塑回三维数组 (H, W, new_C)
            self.data = data_pca.reshape(h, w, data_pca.shape[1])
            # 更新通道数
            self.channels = data_pca.shape[1]
            if verbose:
                print(f"应用PCA降维，保留 {self.channels} 个主成分")

        # 镜像填充数据
        self.padded_data = self.mirror_hsi(
            self.data,
            self.patch_size
        )

    def create_memmap(self, shape, dtype='float32', prefix='mmap'):
        import os
        temp_dir = 'utils/temp'
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f'{self.data_name}_{prefix}.dat')
        return np.memmap(temp_file, dtype=dtype, mode='w+', shape=shape)

    def batch_extract_patches(self, positions, batch_size=10000):
        total_samples = len(positions)
        patch_shape = (total_samples, self.patch_size * self.patch_size, self.channels)
        patches = self.create_memmap(patch_shape, prefix='patches')
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch_positions = positions[i:end_idx]
            batch_patches = self._extract_patches(batch_positions)
            patches[i:end_idx] = batch_patches
        return patches

    def build_pretrain_dataset(self, verbose=True, batch_size=10000) -> Tuple[np.ndarray, np.ndarray]:
        """构建无标签的预训练数据集"""
        sample_step = 1 if (self.height * self.width) < 1e5 else 2
        h_indices = np.arange(0, self.height, sample_step)
        w_indices = np.arange(0, self.width, sample_step)
        hw_grid = np.array(np.meshgrid(h_indices, w_indices)).T.reshape(-1, 2)
        # 提取所有位置的patches
        patches = self.batch_extract_patches(hw_grid, batch_size)
        dummy_labels = np.zeros(len(hw_grid), dtype=np.int64)
        if verbose:
            print(f"\n无标签预训练数据集包含 {len(hw_grid)} 个样本")
            print(f"数据形状: {patches.shape}")

        return patches, dummy_labels

    def build_label_dataset(self, verbose=True) -> Tuple[np.ndarray, np.ndarray]:
        """构建有标签的数据集"""
        positions = np.argwhere(self.labels > 0)
        patches = self.batch_extract_patches(positions)
        labels = self.labels[positions[:, 0], positions[:, 1]] - 1
        if verbose:
            self.print_class_distribution(positions, labels, "标签数据集")
        return patches, labels

    def build_finetune_dataset(
            self,
            size_per_class: Optional[int] = None,
            ratio_per_class: Optional[float] = None,
            augment: bool = True,
            aug_factor: int = 4  # 每个样本增强的倍数
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        构建微调数据集和验证集，支持数据增强
        Args:
            size_per_class: 每个类别的样本数量
            ratio_per_class: 每个类别样本的比例
            augment: 是否进行数据增强
            aug_factor: 每个样本增强的倍数
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (微调数据, 微调标签, 验证数据, 验证标签)
        """
        if size_per_class is None and ratio_per_class is None:
            raise ValueError("必须指定size_per_class或ratio_per_class其中之一")
        finetune_positions = []
        val_positions = []
        for i in range(1, self.num_classes + 1):
            pos = np.argwhere(self.labels == i)
            if ratio_per_class is not None:
                size = int(len(pos) * ratio_per_class)
            else:
                size = size_per_class if len(pos) > size_per_class else len(pos) // 2
            selected_idx = np.random.choice(len(pos), size, replace=False)
            mask = np.ones(len(pos), dtype=bool)
            mask[selected_idx] = False
            finetune_positions.append(pos[selected_idx])
            val_positions.append(pos[mask])
        # 构建基础数据集
        finetune_positions = np.vstack(finetune_positions)
        finetune_patches = self._extract_patches(finetune_positions)
        finetune_labels = self.labels[finetune_positions[:, 0], finetune_positions[:, 1]] - 1
        val_positions = np.vstack(val_positions)
        val_patches = self._extract_patches(val_positions)
        val_labels = self.labels[val_positions[:, 0], val_positions[:, 1]] - 1
        # 打印数据统计
        if self.verbose:
            self.print_class_distribution(finetune_positions, finetune_labels, "微调数据集")
            self.print_class_distribution(val_positions, val_labels, "验证数据集")

        # 数据增强
        if augment:
            aug_patches = []
            aug_labels = []
            for patch, label in zip(finetune_patches, finetune_labels):
                patch_spatial = patch.reshape(self.patch_size, self.patch_size, -1)
                aug_methods = [
                                  self.augmenter.random_flip(patch_spatial),
                                  self.augmenter.random_rotation(patch_spatial),
                                  self.augmenter.spectral_noise(patch_spatial),
                                  self.augmenter.spectral_shift(patch_spatial)
                              ][:aug_factor]
                aug_patches.extend(aug_methods)
                aug_labels.extend([label] * len(aug_methods))
            aug_patches = np.array([p.reshape(-1, self.channels) for p in aug_patches])
            aug_labels = np.array(aug_labels)
            finetune_patches = np.concatenate([finetune_patches, aug_patches], axis=0)
            finetune_labels = np.concatenate([finetune_labels, aug_labels], axis=0)
            # 随机打乱
            shuffle_idx = np.random.permutation(len(finetune_patches))
            finetune_patches = finetune_patches[shuffle_idx]
            finetune_labels = finetune_labels[shuffle_idx]

        return finetune_patches, finetune_labels, val_patches, val_labels

    def build_train_test_dataset(
            self,
            test_size: float = 0.3,
            random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        构建训练集和测试集
        Args:
            test_size: 测试集比例
            random_state: 随机种子
        """
        positions = np.argwhere(self.labels > 0)
        labels = self.labels[positions[:, 0], positions[:, 1]] - 1
        # 分层采样确保每个类别的比例一致
        train_pos, test_pos = train_test_split(
            positions,
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )
        # 构建训练集
        train_patches = self._extract_patches(train_pos)
        train_labels = self.labels[train_pos[:, 0], train_pos[:, 1]] - 1
        # 构建测试集
        test_patches = self._extract_patches(test_pos)
        test_labels = self.labels[test_pos[:, 0], test_pos[:, 1]] - 1
        self.print_class_distribution(train_pos, train_labels, "训练集")
        self.print_class_distribution(test_pos, test_labels, "测试集")

        return train_patches, train_labels, test_patches, test_labels

    def mirror_hsi(self, input_data: np.ndarray, patch: int = 5) -> np.ndarray:
        """
        对高光谱数据进行镜像填充
        Args:
            input_data: 输入数据，形状为(height, width, band)
            patch: patch大小，必须为奇数
        Returns:
            np.ndarray: 填充后的数据，形状为(height+2*padding, width+2*padding, band)
        """
        padding = patch // 2
        mirror_data = np.pad(
            input_data,
            ((padding, padding), (padding, padding), (0, 0)),
            mode='reflect'
        )
        if self.verbose:
            print("**************************************************")
            print(f"patch_size: {patch}")
            print(f"mirror_image shape: {mirror_data.shape}")
            print("**************************************************")

        return mirror_data

    def _extract_patches(self, positions: np.ndarray) -> np.ndarray:
        """从填充后的数据中提取patches"""
        patches = []
        for pos in positions:
            h, w = pos
            patch = self.padded_data[
                    h:h + self.patch_size,
                    w:w + self.patch_size,
                    :
                    ]
            patches.append(patch)
        patches = np.array(patches)

        return patches.reshape(len(patches), -1, self.channels)

    def print_class_distribution(self, positions: np.ndarray, labels: np.ndarray, dataset_name: str = ""):
        """
        打印数据集中每个类别的样本数量分布
        Args:
            positions: 样本位置数组
            labels: 标签数组
            dataset_name: 数据集名称
        """
        if not self.verbose:
            return
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_samples = len(positions)
        print("\n" + "=" * 50)
        print(f"{dataset_name}类别分布:")
        print("-" * 50)
        print(f"{'类别':^10}{'样本数':^12}{'占比':^10}")
        print("-" * 50)
        for label, count in zip(unique_labels, counts):
            percentage = count / total_samples * 100
            print(f"{label + 1:^10}{count:^12}{percentage:^10.2f}%")
        print("-" * 50)
        print(f"总样本数: {total_samples}")
        print("=" * 50 + "\n")


class HSIAugmenter:
    """高光谱图像增强器"""
    @staticmethod
    def random_flip(patch: np.ndarray) -> np.ndarray:
        """随机水平或垂直翻转"""
        if np.random.random() > 0.5:
            patch = np.flip(patch, axis=0)  # 水平翻转
        if np.random.random() > 0.5:
            patch = np.flip(patch, axis=1)  # 垂直翻转
        return patch
    @staticmethod
    def random_rotation(patch: np.ndarray) -> np.ndarray:
        """随机旋转(90°, 180°, 270°)"""
        k = np.random.randint(1, 4)  # 随机旋转次数
        return np.rot90(patch, k=k)
    @staticmethod
    def spectral_noise(patch: np.ndarray, sigma: float = 0.01) -> np.ndarray:
        """添加高斯光谱噪声"""
        noise = np.random.normal(0, sigma, patch.shape)
        return np.clip(patch + noise, 0, 1)  # 限制在[0,1]范围内
    @staticmethod
    def spectral_shift(patch: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
        """随机光谱偏移"""
        shift = np.random.uniform(-max_shift, max_shift)
        return np.clip(patch + shift, 0, 1)