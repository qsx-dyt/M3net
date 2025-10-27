import argparse
import os
import time

import numpy as np
import torch
from pyasn1_modules.rfc7906 import aa_classification
from scipy.io import loadmat
from torch.backends import cudnn
from torch.utils import data as Data
from thop import profile

from finetune import finetune
from models import build_model
from pretrain import pretrain
from utils.build_data import HSIDatasetBuilder
from utils.util import save_results_to_log, validate, calculate_metrics
from utils.visualise import plot_classification

parser = argparse.ArgumentParser("HSI_SSl")
parser.add_argument('--dataset', type=str, default='IndianPines', help='dataset name')
parser.add_argument('--patch_size', type=int, default=11, help='size of patch')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--pca_components', type=int, default=50, help='number of pca components')
parser.add_argument('--SpaMask_ratio', type=float, default=0.6, help='spatial mask ratio')
parser.add_argument('--samples', type=int, default=5, help='number of per sample')
parser.add_argument('--output_dir', type=str, default='./output', help='path where to save')
parser.add_argument('--model_path', type=str, default='./output/IndianPines_best.pth')
parser.add_argument('--pt_model', type=str, default=None, help='pretrained model name')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--flag', type=str, default='test', choices=['pretrain', 'finetune', 'test'], help='types of training')
parser.add_argument('--runs', type=int, default=3, help='number of runs')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# cudnn.benchmark = True
cudnn.deterministic = True

def generate_spatial_mask(x, mask_ratio=0.6):
    """
    x shape: (B, C, N) 其中 N = patch_size^2 是空间位置总数
    返回: (B, N) 的布尔掩码，True 表示需要遮蔽的位置
    """
    B, C, N = x.shape
    if mask_ratio == 0:
        return torch.zeros((B, C), dtype=torch.bool, device=x.device)
    # 生成随机排序索引
    rand_scores = torch.rand(B, N, device=x.device)
    k = int(N * mask_ratio)
    selected_indices = rand_scores.argsort(dim=1, descending=True)[:, :k]
    # 创建布尔掩码
    mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
    mask.scatter_(1, selected_indices, True)
    return mask

depth = 2
heads = 2
if args.dataset == 'IndianPines':
    hsi_data = loadmat('./data/IndianPines/Indian_pines_corrected.mat')['indian_pines_corrected']
    hsi_labels = loadmat('./data/IndianPines/Indian_pines_gt.mat')['indian_pines_gt']
elif args.dataset == 'PaviaU':
    hsi_data = loadmat('./data/PaviaU/PaviaU.mat')['paviaU']
    hsi_labels = loadmat('./data/PaviaU/PaviaU_gt.mat')['paviaU_gt']
elif args.dataset == 'HC':
    hsi_data = loadmat('./data/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat')['WHU_Hi_HanChuan']
    hsi_labels = loadmat('./data/WHU-Hi-HanChuan/WHU_Hi_HanChuan_gt.mat')['WHU_Hi_HanChuan_gt']
else:
    raise ValueError("Unknown dataset")
print(f"数据集: {args.dataset}, 大小: {hsi_data.shape}")

builder = HSIDatasetBuilder(
    data_name=args.dataset,
    data=hsi_data,
    labels=hsi_labels,
    patch_size=args.patch_size,
    pca_components=args.pca_components,
    verbose=True
)
num_classes = builder.num_classes
band = builder.channels
height = builder.height
width = builder.width

# pretrain
if args.flag == 'pretrain':
    pre_patches, pre_labels = builder.build_pretrain_dataset()
    pre_patches = torch.from_numpy(pre_patches.transpose(0, 2, 1)).float()
    pre_labels = torch.from_numpy(pre_labels).long()
    print("pre_patches shape: ", pre_patches.shape)

    # 生成掩码
    spatial_mask = generate_spatial_mask(pre_patches, mask_ratio=args.SpaMask_ratio)

    pre_datasets = Data.TensorDataset(
        pre_patches,
        spatial_mask,
    )
    # 创建数据加载器
    pre_loader = Data.DataLoader(
        pre_datasets,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    pretrain(args, pre_loader, 0, band, depth, heads)

# finetune
if args.flag == 'finetune':
    seeds = [0000, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]
    args.batch_size = 32
    all_metrics = []
    all_class_acc = {}
    for i in range(args.runs):
        np.random.seed(seeds[i])
        torch.manual_seed(seeds[i])
        torch.cuda.manual_seed(seeds[i])
        cudnn.deterministic = True

        ft_patches, ft_labels, val_patches, val_labels = builder.build_finetune_dataset(
            size_per_class=args.samples,
        )
        ft_patches = torch.from_numpy(ft_patches.transpose(0, 2, 1)).float()
        ft_labels = torch.from_numpy(ft_labels).long()
        val_patches = torch.from_numpy(val_patches.transpose(0, 2, 1)).float()
        val_labels = torch.from_numpy(val_labels).long()
        ft_datasets = Data.TensorDataset(ft_patches, ft_labels)
        val_datasets = Data.TensorDataset(val_patches, val_labels)
        # 创建数据加载器
        ft_loader = Data.DataLoader(
            ft_datasets,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True
        )
        val_loader = Data.DataLoader(
            val_datasets,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True
        )

        metrics, class_acc_str = finetune(args, ft_loader, num_classes, band, depth, heads, val_loader, seeds[i])
        # 记录结果
        class_acc = {
            int(cls_part.replace('C', '').strip()):
                float(acc_part.replace('%', '').strip()) / 100
            for item in class_acc_str.split('Class Acc: ')[-1].split(' | ')
            for cls_part, acc_part in [item.split(':')]
        }

        for cls, acc in class_acc.items():
            if cls not in all_class_acc:
                all_class_acc[cls] = []
            all_class_acc[cls].append(acc)
        all_metrics.append({
            'OA': metrics['OA'],
            'AA': metrics['AA'],
            'Kappa': metrics['Kappa']
        })
    # 计算统计量
    class_stats = []
    for cls in sorted(all_class_acc.keys()):
        acc_values = all_class_acc[cls]
        class_stats.append(f"Class {cls}: {np.mean(acc_values):.2%} ± {np.std(acc_values, ddof=1):.2%}")
    oa_values = [m['OA'] for m in all_metrics]
    aa_values = [m['AA'] for m in all_metrics]
    kappa_values = [m['Kappa'] for m in all_metrics]

    save_dir = 'results/cls_results/'
    os.makedirs(save_dir, exist_ok=True)
    result_str = (f"最终统计结果（{args.runs}次运行）:\n"
                  f"OA: {np.mean(oa_values):.2%} ± {np.std(oa_values, ddof=1):.2%}\n"
                  f"AA: {np.mean(aa_values):.2%} ± {np.std(aa_values, ddof=1):.2%}\n"
                  f"Kappa: {np.mean(kappa_values):.4f} ± {np.std(kappa_values, ddof=1):.4f}\n"
                  f"\n类别精度统计:\n" + '\n'.join(class_stats))
    with open(os.path.join(save_dir, f'{args.dataset}_result.txt'), 'a', encoding='utf-8') as f:
        f.write(result_str + '\n\n')

    print(result_str)

if args.flag == 'test':
    model = build_model(args, num_classes, band, depth, heads)
    checkpoint = torch.load(args.model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.cuda()
    model.eval()
    print(model)

    #********************************************查看单个模型结果********************************************#
    # 需要设置args.seed和samples与保存模型一致
    _, _, val_patches, val_labels = builder.build_finetune_dataset(
        size_per_class=args.samples,
    )
    val_patches = torch.from_numpy(val_patches.transpose(0, 2, 1)).float()
    val_labels = torch.from_numpy(val_labels).long()
    val_datasets = Data.TensorDataset(val_patches, val_labels)
    val_loader = Data.DataLoader(
        val_datasets,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )
    # flops, params = profile(model, inputs=(torch.randn(1, band, args.patch_size * args.patch_size).cuda(),))
    # print(f"模型参数数量: {params / 1e6:.2f}M, 模型计算量: {flops / 1e6:.2f}M")
    targets, predicts = validate(model, val_loader)
    metrics = calculate_metrics(targets, predicts)
    class_acc_str = " | ".join([f"C{i}:{acc:.2%}" for i, acc in enumerate(metrics['class_acc'])])
    print(f"验证集结果：\n"
          f"OA: {metrics['OA']:.2%} | "
          f"AA: {metrics['AA']:.2%} | "
          f"Kappa: {metrics['Kappa']:.4f}\n"
          f"Class Acc: {class_acc_str}")

