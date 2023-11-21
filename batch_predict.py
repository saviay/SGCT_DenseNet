import os
import json
import torch
from PIL import Image
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from utils import evaluate_predict, draw_confusion_matrix
from sklearn.metrics import confusion_matrix
from model import densenet121
import numpy as np
import seaborn as sns


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # num_groups = 3
    # Define the folder path for the test dataset
    folder_path = "../data_set/NJrock/test"

    # Read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "File '{}' does not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Create model
    model = densenet121(num_classes=108).to(device)

    test_dataset = torchvision.datasets.ImageFolder(
        root=folder_path,
        transform=data_transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=1
    )

    for i in range(105, 106):
        # Generate the model_weight_path for the current iteration
        model_weight_path = f"./weights/model-{i}.pth"
        if not os.path.exists(model_weight_path):
            print(f"Skipping {model_weight_path} as it does not exist.")
            continue

        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()


        accuracy, precision, recall, f1= evaluate_predict(model, test_loader, device)

        print(f"Model: {model_weight_path}")
        print("Accuracy: {:.4f}".format(accuracy))
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1 Score: {:.4f}".format(f1))






    # 获取真实标签和预测标签的数值表示
    true_labels = []
    predicted_labels = []
    misclassified_images = []

    # 在循环中检查每个样本的预测结果是否正确，如果不正确则记录
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            # 检查每个样本的预测结果是否正确，如果不正确则将图片文件名和错误分类类别添加到列表中
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    image_path = test_dataset.samples[i][0]
                    misclassified_images.append((image_path, class_indict[str(predicted[i].item())]))

    # 打印错误图片文件名和错误分类类别列表
    print("Misclassified Images and their Predicted Classes:")
    for image_path, predicted_class in misclassified_images:
        print(f"Image: {image_path}, Predicted Class: {predicted_class}")

    # draw_confusion_matrix(label_true=np.array(true_labels),
    #                       label_pred=np.array(predicted_labels),
    #                       label_name=["Igneous", "Metamorphic", "Sedimentary"],
    #                       title="Confusion Matrix",
    #                       pdf_save_path="Confusion_Matrix.jpg",
    #                       dpi=600,
    #                       font_size=18)


if __name__ == '__main__':
    main()

