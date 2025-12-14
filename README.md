# Wafer Fault Detection using Deep Learning



<img width="696" height="398" alt="image" src="https://github.com/user-attachments/assets/09254186-071a-4566-85c9-30211e47fbe1" />

# Overview

This project focuses on automated **semiconductor wafer defect classification** using deep learning. A custom Convolutional Neural Network (CNN) and a **ResNet18 transfer learning model** were developed and evaluated to identify wafer faults from image data. The best-performing model was deployed as an interactive **Streamlit web application**, demonstrating a complete end-to-end machine learning workflow.

# Key Features

* Wafer defect classification using deep learning
* Custom-built CNN architecture from scratch
* Transfer learning with pre-trained ResNet18
* Image preprocessing and data augmentation in PyTorch
* Model performance comparison and evaluation
* Real-time inference through a Streamlit web app

# Tech Stack

* **Programming Language:** Python
* **Deep Learning:** PyTorch, Torchvision
* **Model Architectures:** Custom CNN, ResNet18 (Transfer Learning)
* **Web Framework:** Streamlit
* **Tools:** NumPy, Matplotlib

## Dataset

* Image-based semiconductor wafer defect dataset

## Model Performance

| Model                        | Validation Accuracy | Test Accuracy |
| ---------------------------- | ------------------- | ------------- |
| Custom CNN                   | 95.39%              | 95.43%        |
| ResNet18 (Transfer Learning) | **96.66%**          | **96.71%**    |

## Installation

```bash
git clone https://github.com/Jayanth-0407/wafer-fault-detection.git
cd wafer-fault-detection.git
pip install -r requirements.txt
```

## Usage

### Run Streamlit App

```bash
streamlit run app.py
```

Upload a wafer image through the web interface to get real-time defect classification results.

## Project Structure

```
├── data/                 # Dataset and preprocessing scripts              
├── notebooks/            # Experiments and analysis
├── app.py                # Streamlit application
├── wafer_fault.pth       # stores the weights used in the training for further using in streamlit
├── requirements.txt
└── README.md
```

## Results

* ResNet18 outperformed the custom CNN in both validation and test accuracy
* Demonstrated strong generalization due to transfer learning and augmentation
* Successfully deployed as a real-time inference application

## Author

**Jayanth**
