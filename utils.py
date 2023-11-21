import os
import sys
import json
import pickle
import random
import datetime
import torch
import matplotlib.font_manager as fm
from matplotlib import transforms
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('TkAgg')  # 或者使用 'QtAgg'
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import numpy as np
from torchvision import transforms as T
from typing import List
import seaborn as sns
from sklearn.metrics import confusion_matrix


def read_split_data(root: str, val_rate: float = 0.25):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def train_one_epoch(model, optimizer, data_loader, device, epoch, tb_writer,num_groups):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    total_samples = 0
    total_correct = 0
    total_group_samples = {group: 0 for group in range(num_groups)}
    correct_group_samples = {group: 0 for group in range(num_groups)}

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels, group_labels = data
        images = images.to(device)
        labels = labels.to(device)
        group_labels = group_labels.to(device)

        optimizer.zero_grad()
        pred = model(images)

        loss = loss_function(pred, labels)
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] mean loss {:.3f}".format(epoch, mean_loss.item())

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

        # 计算整体准确率
        _, predicted = torch.max(pred, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        # 计算大类别准确率
        for group in range(num_groups):
            group_mask = group_labels == group
            group_total_samples = group_mask.sum().item()
            group_total_correct = (predicted[group_mask] == labels[group_mask]).sum().item()

            total_group_samples[group] += group_total_samples
            correct_group_samples[group] += group_total_correct

    # 计算整体准确率和大类别准确率
    mean_acc = total_correct / total_samples
    group_accuracies = {group: correct_group_samples[group] / total_group_samples[group]
                        for group in range(num_groups)}

    # 记录训练整体准确率和损失到TensorBoard
    tb_writer.add_scalar("train_accuracy", mean_acc, epoch)
    tb_writer.add_scalar("train_loss", mean_loss.item(), epoch)

    return mean_loss.item(), mean_acc, group_accuracies


@torch.no_grad()
def evaluate(model, loss_function, data_loader, device, num_groups):
    model.eval()
    # 验证样本总个数
    total_num = len(data_loader.dataset)
    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)
    # 用于存储总的验证损失
    total_loss = torch.zeros(1).to(device)
    # 初始化大类别样本数和预测正确的样本数
    total_group_samples = {group: 0 for group in range(num_groups)}
    correct_group_samples = {group: 0 for group in range(num_groups)}

    data_loader = tqdm(data_loader, file=sys.stdout)

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            images, labels, group_labels = data
            images = images.to(device)
            labels = labels.to(device)
            group_labels = group_labels.to(device)

            pred = model(images)
            loss = loss_function(pred, labels)
            total_loss += loss

            _, predicted = torch.max(pred, 1)
            sum_num += torch.eq(predicted, labels).sum()

            # 计算大类别准确率
            for group in range(num_groups):
                group_mask = group_labels == group
                group_total_samples = group_mask.sum().item()
                group_total_correct = (predicted[group_mask] == labels[group_mask]).sum().item()

                total_group_samples[group] += group_total_samples
                correct_group_samples[group] += group_total_correct

    # 计算平均验证损失和整体准确率
    mean_loss = total_loss / len(data_loader)
    val_acc = sum_num.item() / total_num

    # 计算大类别准确率
    group_accuracies = {group: correct_group_samples[group] / total_group_samples[group]
                        for group in range(num_groups)}

    return mean_loss.item(), val_acc, group_accuracies


# 修改train_one_epoch函数，增加返回值，包括loss和accuracy
def train_one_epoch(model, optimizer, data_loader, device, epoch, tb_writer):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    total_samples = 0
    total_correct = 0

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        pred = model(images)

        loss = loss_function(pred, labels)
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] mean loss {:.3f}".format(epoch, mean_loss.item())

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

        # 计算准确率
        _, predicted = torch.max(pred, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    # 计算平均准确率
    mean_acc = total_correct / total_samples

    # 记录训练准确率和损失到TensorBoard
    tb_writer.add_scalar("train_accuracy", mean_acc, epoch)
    tb_writer.add_scalar("train_loss", mean_loss.item(), epoch)

    return mean_loss.item(), mean_acc

@torch.no_grad()
# 修改evaluate函数，使其返回验证损失和验证准确率

def evaluate(model, loss_function, data_loader, device):
    model.eval()
    # 验证样本总个数
    total_num = len(data_loader.dataset)
    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)
    # 用于存储总的验证损失
    total_loss = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader, file=sys.stdout)

    with torch.no_grad():
        for step, data in enumerate(data_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss = loss_function(pred, labels)
            total_loss += loss
            pred = torch.max(pred, dim=1)[1]
            sum_num += torch.eq(pred, labels).sum()
    # 计算平均验证损失和验证准确率
    mean_loss = total_loss / len(data_loader)
    val_acc = sum_num.item() / total_num

    return mean_loss.item(), val_acc



def show_loss_acc(train_losses, train_accuracies, val_losses, val_accuracies,batch_size):
    # 创建results文件夹
    os.makedirs("results", exist_ok=True)

    # 生成唯一的文件名
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # 使用当前时间戳作为文件名
    file_name = f'batch_{batch_size}_results_accuracy_{timestamp}.png'  # 修改文件名的格式，可以根据需要自定义

    # 保存训练和验证准确率的图像
    plt.figure(figsize=None)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(min(train_accuracies), min(val_accuracies)) - 0.1, 1])
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.savefig(f'results/{file_name}', dpi=100)  # 使用不同的文件名保存图像
    plt.show()

    # 生成新的文件名
    file_name = f'batch_{batch_size}_results_loss_{timestamp}.png'  # 修改文件名的格式，可以根据需要自定义

    # 保存训练和验证损失的图像
    plt.figure(figsize=None)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.savefig(f'results/{file_name}', dpi=100)  # 使用不同的文件名保存图像
    plt.show()

def save_training_results(result_data, args, batch_size):
    # 将结果数据和参数配置合并为一个字典
    results_with_args = {
        'results': result_data,
        'args': vars(args)
    }
    # 获取当前时间戳
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # 使用当前时间戳作为文件名
    # 构建文件名
    file_name = f'batch_{batch_size}_training_results_{timestamp}.json'
    # 将数据写入JSON文件
    result_file = os.path.join('./results', file_name)
    with open(result_file, 'w') as f:
        json.dump(results_with_args, f)


def evaluate_predict(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return accuracy, precision, recall, f1


def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100, font_size=12, value_font_size=24):
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    # Load Times New Roman font
    font_path = 'C:/Windows/Fonts/times.ttf'  # Path to Times New Roman font
    prop = fm.FontProperties(fname=font_path, size=font_size)

    plt.imshow(cm, cmap='Blues')
    plt.title(title, fontproperties=prop, fontsize=24)  # 使用Times New Roman字体设置标题
    plt.xlabel("Predict", fontproperties=prop, fontsize=24)  # 使用Times New Roman字体设置x轴标签并调整字体大小
    plt.ylabel("Truth", fontproperties=prop, fontsize=24)  # 使用Times New Roman字体设置y轴标签并调整字体大小
    plt.yticks(range(label_name.__len__()), label_name, rotation=90, fontproperties=prop, va='center')  # 使用Times New Roman字体设置y轴标签
    plt.xticks(range(label_name.__len__()), label_name, rotation=0, fontproperties=prop, fontsize=font_size)  # 使用Times New Roman字体设置x轴标签并调整字体大小

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)
            value = float(format('%.3f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color, fontproperties=prop, fontsize=value_font_size)  # 调整数字字体大小

    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)









