# Detecting COVID-19 with Chest X-Ray using PyTorch

A deep learning image classification model built with PyTorch and ResNet-18 to distinguish between Normal, Viral Pneumonia, and COVID-19 chest X-ray scans — achieving over 98% validation accuracy with 100% recall on COVID-19 cases.

> **Disclaimer:** This model and dataset are strictly for educational purposes. They cannot and must not be used to diagnose COVID-19 or any pulmonary condition in a clinical context.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Scientific Rationale](#2-scientific-rationale)
3. [Dataset](#3-dataset)
4. [Approach & Architecture](#4-approach--architecture)
5. [Training Configuration](#5-training-configuration)
6. [Results](#6-results)
7. [Project Structure](#7-project-structure)
8. [Installation & Usage](#8-installation--usage)
9. [Step-by-Step Walkthrough](#9-step-by-step-walkthrough)
10. [Limitations](#10-limitations)
11. [References](#11-references)

---

## 1. Project Overview

This project implements a transfer learning pipeline on top of a pre-trained ResNet-18 convolutional neural network to classify chest X-ray images into three categories:

- **Normal** — healthy patient
- **Viral Pneumonia** — non-COVID pulmonary infection
- **COVID-19** — SARS-CoV-2 related pulmonary involvement

The model is fine-tuned on a publicly available radiography dataset sourced from Kaggle, and converges in under one full training epoch to above 98% validation accuracy — a direct consequence of the representational power of ImageNet pre-training applied to grayscale medical imaging.

---

## 2. Scientific Rationale

### Why Transfer Learning on Medical Images

Training a CNN from scratch on a dataset of ~3,000 images would result in severe overfitting. Transfer learning addresses this by initializing the network with weights learned from ImageNet (1.2M images, 1,000 classes), which encode generic low-level and mid-level visual features (edges, textures, shapes) that transfer effectively to medical imaging domains.

The strategy applied here:

```
ImageNet pre-trained ResNet-18
        |
        v
Freeze feature extractor (optional)
        |
        v
Replace final FC layer: 512 -> 3 classes
        |
        v
Fine-tune all parameters end-to-end
```

### Why ResNet-18

ResNet-18 offers a favorable trade-off for this task:

| Property | Value |
|---|---|
| Parameters | 11.2M |
| Depth | 18 layers |
| Residual connections | Yes (avoids vanishing gradient) |
| ImageNet Top-1 accuracy | 69.8% |
| Inference speed | Fast |
| Suitability for small datasets | High |

Deeper variants (ResNet-50, ResNet-101) would introduce unnecessary complexity for a 3-class problem on ~3,000 images.

### Why Recall Matters More Than Accuracy for COVID-19

On imbalanced medical datasets, overall accuracy is misleading. For COVID-19 detection specifically:

| Error type | Consequence |
|---|---|
| False Negative (missed COVID-19) | Untreated patient, potential transmission |
| False Positive (healthy classified as COVID-19) | Unnecessary isolation, follow-up testing |

The cost of a false negative is substantially higher. **COVID-19 Recall (sensitivity) is the primary clinical metric**, and the model achieves 100% recall on the COVID-19 class.

---

## 3. Dataset

- **Source:** [Kaggle — COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **Total images:** 2,924 grayscale chest X-ray scans
- **Format:** PNG, grayscale, variable resolution → resized to 224×224

### Class Distribution

| Class | Count | Share |
|---|---|---|
| Normal | 1,341 | 45.9% |
| Viral Pneumonia | 1,345 | 46.0% |
| COVID-19 | 219 | 7.5% |

The COVID-19 class is significantly under-represented (~8× fewer samples than the other two classes). This imbalance is addressed through validation set stratification and is reflected in the evaluation strategy.

### Train / Validation Split

| Split | Normal | Viral Pneumonia | COVID-19 | Total |
|---|---|---|---|---|
| Validation | 30 | 30 | 30 | 90 |
| Training | 1,311 | 1,315 | 189 | 2,815 |

A fixed validation set of 30 images per class (90 total) was held out before training to ensure balanced evaluation across all three classes.

---

## 4. Approach & Architecture

### Custom Dataset Class

```python
import os
from PIL import Image
from torch.utils.data import Dataset

class ChestXRayDataset(Dataset):
    def __init__(self, image_dirs, transform):
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = ['COVID-19', 'Normal', 'Viral Pneumonia']

        for label, directory in enumerate(image_dirs):
            for fname in os.listdir(directory):
                if fname.endswith('.png'):
                    self.images.append(os.path.join(directory, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        return self.transform(image), self.labels[idx]
```

### Image Transformations

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

Normalization values are the ImageNet channel statistics — required for correct behavior of pre-trained weights.

### Model Definition

```python
import torch
import torch.nn as nn
from torchvision import models

resnet18 = models.resnet18(pretrained=True)

# Replace final classification layer
resnet18.fc = nn.Linear(resnet18.fc.in_features, 3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet18 = resnet18.to(device)
```

Only the final fully connected layer is replaced. All upstream parameters remain trainable (full fine-tuning, not frozen feature extraction).

---

## 5. Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 3e-5 |
| Loss function | Cross-Entropy |
| Batch size | 6 |
| Input size | 224 × 224 |
| Epochs to convergence | < 1 |
| Device | CUDA (GPU) / CPU |

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.parameters(), lr=3e-5)
```

The low learning rate (3e-5) is deliberate for fine-tuning: it prevents catastrophic forgetting of pre-trained representations while allowing gradual adaptation to the medical imaging domain.

### Training Loop

```python
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), correct / total
```

---

## 6. Results

| Metric | Value |
|---|---|
| Overall validation accuracy | > 98% |
| COVID-19 class recall | 100% |
| Epochs to convergence | < 1 |

The model achieves perfect recall on the COVID-19 class — meaning zero missed COVID-19 cases in the validation set. This is the clinically relevant outcome given the class imbalance and the asymmetric cost of false negatives in epidemic screening contexts.

### Evaluation

```python
def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total

val_accuracy = evaluate(resnet18, val_loader, device)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
```

---

## 7. Project Structure

```
Detecting-COVID-19-with-Chest-X-Ray-using-PyTorch/
├── data/
│   ├── train/
│   │   ├── COVID-19/
│   │   ├── Normal/
│   │   └── Viral Pneumonia/
│   └── val/
│       ├── COVID-19/          # 30 images per class
│       ├── Normal/
│       └── Viral Pneumonia/
├── notebook/
│   └── COVID19_XRay_Classification.ipynb   # Main project notebook
├── models/
│   └── resnet18_covid_finetuned.pth        # Saved model weights
├── requirements.txt
└── README.md
```

> The dataset is not included in this repository. Download it from [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) and place the class folders under `data/train/` and `data/val/`.

---

## 8. Installation & Usage

### Requirements

```bash
pip install -r requirements.txt
```

```
torch==2.0.1
torchvision==0.15.2
Pillow==10.0.0
numpy==1.25.2
matplotlib==3.7.2
scikit-learn==1.3.0
jupyter==1.0.0
```

### Run the notebook

```bash
jupyter notebook notebook/COVID19_XRay_Classification.ipynb
```

### Inference on a new image

```python
import torch
from torchvision import transforms, models
from PIL import Image

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 3)
model.load_state_dict(torch.load('models/resnet18_covid_finetuned.pth'))
model.eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('path/to/xray.png').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# Predict
class_names = ['COVID-19', 'Normal', 'Viral Pneumonia']
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)

print(f"Predicted class: {class_names[predicted.item()]}")
```

---

## 9. Step-by-Step Walkthrough

The notebook is organized into the following sequential steps:

| Step | Description |
|---|---|
| 1 | Introduction — project context and objectives |
| 2 | Importing Libraries — PyTorch, torchvision, NumPy, Matplotlib |
| 3 | Creating Custom Dataset — `ChestXRayDataset` class |
| 4 | Image Transformations — resize, normalize, augment |
| 5 | Prepare DataLoader — batching and shuffling |
| 6 | Data Visualization — sample images per class |
| 7 | Creating the Model — ResNet-18 + custom FC layer |
| 8 | Training the Model — Adam, Cross-Entropy, fine-tuning loop |
| 9 | Final Results — accuracy metrics, confusion matrix |

**Prerequisites:** Python programming experience, theoretical understanding of CNNs and gradient descent.

---

## 10. Limitations

- **Not for clinical use.** This model cannot and must not be used to diagnose COVID-19 in real patients.
- **Small dataset.** 219 COVID-19 images is insufficient for robust clinical generalization.
- **Class imbalance.** Despite stratified validation, the training set remains imbalanced.
- **Single modality.** The model uses X-ray only; clinical diagnosis integrates PCR, CT, and clinical symptoms.
- **No external validation.** The model has not been tested on data from different scanners, hospitals, or patient populations.
- **Grayscale converted to RGB.** ImageNet normalization assumes 3-channel input; grayscale images are replicated across channels, which is a pragmatic approximation.

---

## 11. References

[1] He, K., Zhang, X., Ren, S. and Sun, J., 2016. Deep Residual Learning for Image Recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp.770–778.

[2] Tawsifur Rahman et al., 2021. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection Using Chest X-ray Images. *Computers in Biology and Medicine*, 132, p.104319.

[3] COVID-19 Radiography Database — [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

[4] PyTorch Documentation — [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

[5] Coursera Guided Project — Detecting COVID-19 with Chest X-Ray using PyTorch

---

*This project was completed as part of a Coursera guided project curriculum. For questions, open an issue in the repository.*
