import os
import time

import torch
from thop import profile

from timm.utils.metrics import AverageMeter
from torch.amp import GradScaler, autocast

from models import build_model
from utils.visualise import plot_training_curve
from utils.logger import create_logger


def train_one_epoch(model, pretrain_loader, optimizer, scaler, logger=None):
    # 初始化计量器和收集器
    loss_meter = AverageMeter()

    # 初始化进度条
    from utils.logger import ProgressBar
    length = len(pretrain_loader)
    progress_bar = ProgressBar(total=length, prefix='预训练', suffix='', length=40)

    # 记录训练开始
    if logger:
        logger.info(f"开始训练 - 批次数: {length}")
        logger.info(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")

    model.train()
    for batch_idx, (batch_data, spatial_mask) in enumerate(pretrain_loader):
        # 数据迁移到设备
        batch_data = batch_data.cuda(non_blocking=True)
        spatial_mask = spatial_mask.cuda(non_blocking=True)
        # 梯度清零
        optimizer.zero_grad(set_to_none=True)  # 内存优化
        # 使用混合精度训练
        with autocast(device_type='cuda'):
            loss = model(batch_data, spatial_mask)
        # 使用 scaler 处理梯度
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # 收集指标
        loss_meter.update(loss.item(), batch_data.size(0))

        # 更新进度条
        batch_info = f"批次: {batch_idx + 1}/{length}"
        suffix = f"损失: {loss.item():.4f}"
        progress_bar.update(batch_idx + 1, suffix=suffix, batch_info=batch_info)

    return loss_meter.avg


def pretrain(args, pretrain_loader, num_classes, band, depth, heads):
    # 确保日志目录存在
    os.makedirs('checkpoint/pretrain/logs', exist_ok=True)

    # 创建日志记录器
    log_dir = 'checkpoint/pretrain/logs'
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = f"{args.dataset}_{timestamp}.log"
    logger = create_logger(name=f"Pretrain_{args.dataset}", log_dir=log_dir, log_file=log_file)

    # 记录训练配置
    logger.info(f"{'=' * 20} 训练配置 {'=' * 20}")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info(f"掩码比例: 空间 {args.SpaMask_ratio}")
    logger.info(f"{'=' * 50}")

    # 构建模型
    model = build_model(args, num_classes, band, depth, heads, is_pretrain=True)
    model = model.cuda()
    print(model)
    # 参数打印
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Params: {n_params/1e6:.4f} M")

    # 优化器配置
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,  # 基础学习率
        weight_decay=0.005,  # 权重衰减
        betas=(0.9, 0.999)  # 动量参数
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,  # 每10轮降低一次学习率
        gamma=0.9  # 学习率衰减因子
    )

    # 加载检查点
    checkpoint_path = ('checkpoint/pretrain/' + args.dataset + '_' + str(args.patch_size) + '_' + str(args.SpaMask_ratio)
                       + '_pt.pth')
    times = 0
    train_losses = []
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        logger.info(f"加载检查点: {checkpoint_path}")
        print("Load checkpoint from {}".format(checkpoint_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        times = checkpoint['time']
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        logger.info(f"从轮次 {start_epoch} 继续训练")
    else:
        start_epoch = 0
        logger.info("未找到检查点，从头开始训练")

    # 训练模型
    print("------------------Start training------------------")
    logger.info("开始训练过程")
    scaler = GradScaler()
    tick = time.time()
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"{'=' * 15} 轮次 {epoch + 1}/{args.epochs} {'=' * 15}")
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"当前学习率: {current_lr:.6f}")
        # 训练一个轮次
        avg_loss = train_one_epoch(model, pretrain_loader, optimizer, scaler, logger)
        # 组合调度器更新
        scheduler.step()
        # 记录训练指标
        train_losses.append(avg_loss)

        # 记录轮次结果
        logger.info(f"轮次 {epoch + 1} 结果 - 损失: {avg_loss:.8f}")
        print("Epoch: {:03d} train_loss: {:.8f}".format(epoch + 1, avg_loss))

        ckp_dir = 'checkpoint/pretrain/'
        if not os.path.exists(ckp_dir):
            os.makedirs(ckp_dir)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            tock2 = time.time()
            # 保存检查点
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'time': tock2 - tick,
                        'train_losses': train_losses,
                        },
                       checkpoint_path)
            logger.info(f"保存检查点到: {checkpoint_path}")

    tock = time.time()
    training_time = tock - tick + times
    logger.info(f"训练完成，总用时: {training_time:.2f}秒")
    print("Training time: {:.2f}s".format(training_time))
    print("-----------------Training finished-----------------\n")

    # 绘制训练曲线
    logger.info("绘制训练曲线")
    plot_training_curve(train_losses)

    # 关闭日志记录器
    logger.info("任务完成，关闭日志记录器")
    return model




