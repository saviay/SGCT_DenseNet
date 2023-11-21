from PIL import Image
import torch
from torch.utils.data import Dataset


# class MyDataSet(Dataset):
#     """自定义数据集"""
#     def __init__(self, images_path: list, images_class: list, images_group_class: list, transform=None):
#         # 初始化函数，用于创建 MyDataSet 类的实例
#         # images_path: 包含图像文件路径的列表
#         # images_class: 包含图像小类别标签的列表
#         # images_group_class: 包含图像大类别标签的列表
#         # transform: 图像预处理的操作
#         self.images_path = images_path  # 保存图像文件路径列表
#         self.images_class = images_class  # 保存图像小类别标签列表
#         self.images_group_class = images_group_class  # 保存图像大类别标签列表
#         self.transform = transform  # 图像预处理的操作
#
#     def __len__(self):
#         # 返回数据集中样本的数量
#         return len(self.images_path)
#
#     def __getitem__(self, item):
#         img = Image.open(self.images_path[item])
#         if img.mode != 'RGB':
#             raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
#         label = self.images_class[item]
#         group_label = self.images_group_class[item]  # 获取大类别索引
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         return img, label, group_label  # 返回图像、标签和大类别索引
#
#     @staticmethod
#     def collate_fn(batch):
#         # 用于自定义数据加载时对批次数据的处理函数
#         # 将 batch 中的图像、小类别标签和大类别标签分别提取出来
#         images, small_class_labels, big_class_labels = tuple(zip(*batch))
#         # 将图像堆叠成一个张量（Batch），dim=0 表示在批次维度上堆叠
#         images = torch.stack(images, dim=0)
#         # 将小类别标签和大类别标签转换为张量
#         small_class_labels = torch.as_tensor(small_class_labels)
#         big_class_labels = torch.as_tensor(big_class_labels)
#         # 返回处理后的批次图像、小类别标签和大类别标签
#         return images, small_class_labels, big_class_labels

class MyDataSet(Dataset):
    """自定义数据集"""
    def __init__(self, images_path: list, images_class: list, transform=None):
        # 初始化函数，用于创建 MyDataSet 类的实例
        # images_path: 包含图像文件路径的列表
        # images_class: 包含图像类别标签的列表
        # transform: 图像预处理的操作
        self.images_path = images_path  # 保存图像文件路径列表
        self.images_class = images_class  # 保存图像类别标签列表
        self.transform = transform  # 图像预处理的操作
    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.images_path)
    def __getitem__(self, item):
        # 通过索引获取一个样本，并进行数据加载和预处理
        # 加载图像
        img = Image.open(self.images_path[item])
        # 检查图像是否为彩色图像（RGB模式）
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        # 获取样本的标签
        label = self.images_class[item]
        # 进行图像预处理（如果预处理操作不为空）
        if self.transform is not None:
            img = self.transform(img)
        # 返回预处理后的图像和标签
        return img, label
    @staticmethod
    def collate_fn(batch):
        # 用于自定义数据加载时对批次数据的处理函数
        # 将 batch 中的图像和标签分别提取出来
        images, labels = tuple(zip(*batch))
        # 将图像堆叠成一个张量（Batch），dim=0 表示在批次维度上堆叠
        images = torch.stack(images, dim=0)
        # 将标签转换为张量
        labels = torch.as_tensor(labels)
        # 返回处理后的批次图像和标签
        return images, labels
