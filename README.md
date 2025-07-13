# Cotton Leaf Disease Detection (SAR-CLD-2024)

This project uses the **SAR-CLD-2024** dataset to build a deep learning model for detecting various cotton leaf diseases using a Convolutional Neural Network (CNN). The dataset contains high-quality, augmented images of cotton leaves categorized into multiple disease classes.

We utilize a **pretrained ResNet-18** model, fine-tuned to classify the disease types. The training pipeline includes custom dataset loading using Pandas DataFrame splits (train/validation/test), image preprocessing, early stopping, and best model checkpointing. The model is trained and evaluated using PyTorch on GPU-enabled environments like Kaggle.

This project aims to help in early detection of cotton diseases to support sustainable agriculture and reduce crop loss using deep learning.

## Dataset

- Dataset: [SAR-CLD-2024: A Comprehensive Dataset for Cotton Leaf Disease Detection](https://www.kaggle.com/datasets/saikatmukherjee7011/sarcld2024-cotton-leaf-disease)
- Format: PNG/JPEG images with disease labels in folder names
- Split: Train (70%), Validation (15%), Test (15%)

## Tools Used

- Python
- PyTorch
- torchvision
- PIL
- tqdm
- pandas

## Notebook

> All training code is included in the [`pre-trained-cnn.ipynb`](./pre-trained-cnn.ipynb) file.
