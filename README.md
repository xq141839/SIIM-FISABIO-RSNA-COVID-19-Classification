# A Simple Training Strategy for SIIM-FISABIO-RSNA-COVID-19-Detection-Classification Challenge 
## Requirements
* Pytorch >= 1.5.0
* Albumentations == 1.0.1
* [efficientnet_pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) 
## Methods
1. Model: EfficientNet-B7
2. GroupKFold (5 folds and based on the image id)
3. Augmentations (HorizontalFlip + ShiftScaleRotate + RandomBrightnessContrast + CoarseDropout)
4. Batch Size = 32
5. Image Size = 512 x 512
6. Loss: Cross Entropy
7. Optimizer: Adam
## Results
### We get 0.417 score (private datasets) for a single fold and 0.430 for 5 ensemble folds (mean).
