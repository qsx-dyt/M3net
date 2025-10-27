import numpy as np
import torch
import time
from sklearn.metrics import confusion_matrix, cohen_kappa_score


def validate(model, loader):
    all_targets, all_preds = [], []
    model.eval()
    with torch.no_grad():
        for batch_data in loader:
            batch_img, batch_target = batch_data
            # 数据迁移到设备
            batch_img = batch_img.cuda(non_blocking=True)
            batch_target = batch_target.cuda(non_blocking=True)
            # 前向传播
            batch_pred = model(batch_img)

            preds = torch.argmax(batch_pred.detach(), dim=1)
            targets = batch_target
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_targets), np.concatenate(all_preds)


def calculate_metrics(targets: np.ndarray, preds: np.ndarray) -> dict:
    """计算分类指标 (OA, AA, Kappa)"""
    # 转换为整数类型避免计算误差
    targets = targets.astype(int)
    preds = preds.astype(int)

    # 计算混淆矩阵
    cm = confusion_matrix(targets, preds)
    cls_num = cm.shape[0]
    # 总体准确率 OA
    oa = np.diag(cm).sum() / cm.sum()
    # 平均准确率 AA
    aa = np.nanmean([cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0 for i in range(cls_num)])
    # Kappa系数
    kappa = cohen_kappa_score(targets, preds)
    # 计算每类准确率
    class_acc = []
    for i in range(cls_num):
        if cm[i, :].sum() > 0:
            class_acc.append(cm[i, i] / cm[i, :].sum())
        else:
            class_acc.append(0.0)
    return {'OA': oa, 'AA': aa, 'Kappa': kappa, 'class_acc': class_acc, 'cm': cm}


def save_results_to_log(args, flops, params, metrics, class_acc_str, log_file):
    """保存验证结果到日志文件"""
    cm = metrics['cm']
    with open(log_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Model Config: Patch Size={args.patch_size}, Flops={flops / 1e6:.4f}M, Params={params / 1e6:.4f}M \n")
        # f.write(f"Epochs: {args.epochs}\n")
        # f.write(f"Batch Size: {args.batch_size}\n")
        # f.write(f"Pretrained Model: {args.pt_model if hasattr(args, 'load_pt') and args.load_pt else 'None'}\n")
        f.write("\nEvaluation Metrics:\n")
        f.write(f"OA: {metrics['OA']:.2%}\n")
        f.write(f"AA: {metrics['AA']:.2%}\n")
        f.write(f"Kappa: {metrics['Kappa']:.4f}\n")
        f.write("\nClass Accuracies:\n")
        f.write(class_acc_str.replace(" | ", "\n") + "\n")
        # 混淆矩阵记录
        f.write("\nConfusion Matrix (Row=Actual, Column=Predicted):\n")
        max_width = len(str(cm.max())) + 2  # 动态计算列宽
        header = " " * (max_width + 3) + " ".join([f"{i:^{max_width}}" for i in range(cm.shape[1])])
        f.write(header + "\n")

        for i, row in enumerate(cm):
            row_str = [f"{num:{max_width}d}" for num in row]
            f.write(f"C{i} | " + " ".join(row_str) + "\n")

        # 记录时间戳
        f.write(f"\nGenerated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
