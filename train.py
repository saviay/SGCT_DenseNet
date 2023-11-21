import cv2
import os
import math
import argparse
import torch
import torch.optim as optim
from adabound import adabound
# from albumentations import Cutout
from torchtoolbox.transform import Cutout

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from model import densenet121, load_state_dict
from my_dataset import MyDataSet
from utils import read_split_data, evaluate, show_loss_acc, train_one_epoch, save_training_results
import time


loss_function = torch.nn.CrossEntropyLoss()
result_data = {
    'epoch': [],
    'train_loss': [],
    'train_accuracy': [],
    'val_loss': [],
    'val_accuracy': []
}


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if not os.path.exists("./weights"):
        os.makedirs("./weights")
    if not os.path.exists("./results"):
        os.makedirs("./results")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),  # 修改为448
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.2),
            Cutout(scale=(0.02, 0.3)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),  # 修改为512
            transforms.CenterCrop(224),  # 修改为448
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    }

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(num_workers))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=num_workers,
                                               collate_fn=val_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=num_workers,
                                             collate_fn=val_dataset.collate_fn)

    model = densenet121(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            load_state_dict(model, args.weights)
        else:
            raise FileNotFoundError("Weights file not found: {}".format(args.weights))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "classifier" not in name:
                para.requires_grad_(False)

    # if args.freeze_layers:
    #     # 计算模型中总层数
    #     total_layers = len(list(model.parameters()))
    #
    #     # 计算你想要冻结的层的起始索引
    #     start_freeze_index = total_layers - 5
    #
    #     # 冻结最后五层
    #     for idx, (name, para) in enumerate(model.named_parameters()):
    #         if idx >= start_freeze_index:
    #             para.requires_grad_(True)

    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = adabound.AdaBound(pg, lr=args.lr, weight_decay=1E-4, final_lr=0.1)

    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    start_time = time.time()  # 定义start_time变量并赋予初始值

    processed_samples = 0
    time_interval = 1  # 每秒
    for epoch in range(args.epochs):
        mean_loss, train_acc = train_one_epoch(model=model,
                                               optimizer=optimizer,
                                               data_loader=train_loader,
                                               device=device,
                                               epoch=epoch,
                                               tb_writer=tb_writer)

        scheduler.step()

        val_loss, val_acc = evaluate(model=model, loss_function=loss_function,
                                     data_loader=val_loader, device=device)
        print(f"Epoch {epoch}: Augmented training set size: {len(train_loader.dataset)}")
        print("[epoch {}] train accuracy: {:.3f}, val accuracy: {:.3f}, train loss: {:.3f}, val loss: {:.3f}".format(
            epoch, train_acc, val_acc, mean_loss, val_loss))
        # print("Epoch {}: Val Group Accuracies: {}".format(epoch, val_group_accuracies))
        tb_writer.add_scalar("train_accuracy", train_acc, epoch)
        tb_writer.add_scalar("train_loss", mean_loss, epoch)
        tb_writer.add_scalar("val_accuracy", val_acc, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)

        train_losses.append(mean_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        # 记录相关数据
        result_data['epoch'].append(epoch)
        result_data['train_loss'].append(mean_loss)
        result_data['train_accuracy'].append(train_acc)
        result_data['val_loss'].append(val_loss)
        result_data['val_accuracy'].append(val_acc)

        # 计算每秒处理的样本数
        processed_samples += len(train_loader.dataset)
        if time.time() - start_time >= time_interval:
            samples_per_sec = processed_samples / (time.time() - start_time)
            print("Processed samples:", processed_samples)
            print("Samples per second:", samples_per_sec)
            start_time = time.time()
            processed_samples = 0

    # 将数据写入JSON文件
    save_training_results(result_data, args, batch_size)
    tb_writer.close()
    # 绘制训练曲线并保存图像
    show_loss_acc(train_losses, train_accuracies, val_losses, val_accuracies, batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=108)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--data-path', type=str, default="../data_set/NJrock/train")
    parser.add_argument('--weights', type=str, default='densenet121.pth', help='initial weights path')
    parser.add_argument('--freeze-layers', action='store_true', default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()

    train(opt)
