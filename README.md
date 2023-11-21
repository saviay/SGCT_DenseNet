# SGCT_DenseNet

## Project Overview:
SGCT_DenseNet is a deep learning project based on the DenseNet architecture, designed for tasks related to image processing. The dataset for this project can be found here: https://doi.org/10.11922/sciencedb.j00001.00097.

## Project Structure:
- model.py: Contains the network model.
- train.py: Script for training the model.
- se_module.py: Custom module implementation.
- my_dataset.py: Custom dataset loading script.
- predict.py: Script for single image prediction.
- batch_predict.py: Script for batch image prediction.
- densenet121.pth: Pre-trained weights for DenseNet-121
- densenet169.pth: Pre-trained weights for DenseNet-169
- densenet201.pth: Pre-trained weights for DenseNet-201

## Installation:
Before running the scripts, make sure to install the required dependencies. 

## Usage:
Train the Model:
Use the train.py script to train the model. Ensure that the dataset is prepared and configure the parameters in the script accordingly.
    
    ```python train.py```

## Single Image Prediction:
Use the predict.py script to make predictions on a single image.

    ```python predict.py --image_path=path/to/your/image.jpg```

## Batch Image Prediction:
Use the batch_predict.py script to predict on a whole folder of images.

   ``` python batch_predict.py --input_folder=path/to/your/images --output_folder=path/to/save/predictions```

## Contribution:
If you are interested in contributing code, suggesting improvements, or reporting issues, please refer to the contribution guidelines.

## License:
This project is licensed under the MIT License.
