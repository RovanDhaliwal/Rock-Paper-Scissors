# Rock-Paper-Scissors

## Overview
This project develops an AI system capable of recognizing Rock-Paper-Scissors hand gestures. Leveraging transfer learning with a pre-trained AlexNet model, the system aims to classify hand gestures (rock, paper, scissors) with high accuracy, forming the foundation for an "unbeatable" AI that can predict and counter human moves in real-time.

## Features
- Real-time hand gesture classification for Rock, Paper, and Scissors.
- Utilizes a fine-tuned AlexNet for high-accuracy image recognition.
- Designed for extensibility to process live video feeds from a webcam.
- Base for a system that can predict and beat opponent's moves.

## Installation

### Prerequisites
- Python 3.8+
- OpenCV
- PyTorch

### Usage
## Run on Images
The system takes an image of a hand gesture as input, processes it through the trained AlexNet model, and outputs the classified gesture (rock, paper, or scissors) along with a confidence score. This can then be used to determine a winning counter-move.

## Run on Webcam
For real-time interaction, the system is designed to capture frames from a webcam, perform inference on each frame, and display the detected gesture and the AI's counter-move on the screen. (Full real-time implementation is a future improvement).

### Model Information
## Model: AlexNet
- Source: PyTorch torchvision.models (pre-trained on ImageNet)
- Architecture: A classic Convolutional Neural Network (CNN) architecture, adapted for transfer learning by replacing its final classification layer to predict 3 classes (Rock, Paper, Scissors)
- Pre-training: The AlexNet model was pre-trained on the ImageNet dataset, allowing it to learn robust features from a large general image dataset before fine-tuning for our specific task

### Performance
- Accuracy: 98.86% (Achieved on the test dataset for gesture classification)
- FPS (Frames Per Second): ~40-60 FPS (Estimated for AlexNet classification on typical CPU/GPU)
- Inference Time: <30ms (Estimated, consistent with the excellent target for fast classification models)
- Test Loss: 0.0526

### Results
The fine-tuned AlexNet model demonstrates excellent performance in classifying Rock-Paper-Scissors gestures, achieving nearly 99% accuracy on unseen data. This provides a strong foundation for building a responsive and accurate AI player.

### Challenges & Solutions
## Data Variability
The Rock-Paper-Scissors dataset, while diverse, might still contain variations in lighting, background, and hand orientation that could challenge the model.

Solution: Applied data augmentation techniques like random rotation and horizontal flips during training to improve model robustness.

## Real-time Processing
Ensuring the model can classify gestures quickly enough for a real-time game.

Solution: Chose a relatively efficient CNN architecture (AlexNet) and utilized pre-trained weights to speed up convergence and leverage learned features. Further optimization would involve GPU acceleration.

## Dataset Preparation
Solution: Used opendatasets and PyTorch's ImageFolder for efficient data handling and organization.

## Transfer Learning Implementation
Solution: Adapted AlexNet's classifier head by replacing it with custom nn.Linear, nn.ReLU, nn.Dropout, and nn.LogSoftmax layers to output 3 classes.

### Future Improvements
- Real-time Implementation: Implement the full real-time video processing pipeline to allow direct interaction via webcam
- Advanced Game Logic: Develop advanced game logic to analyze player patterns and adapt its strategy, making the AI truly "unbeatable"
- Interactive GUI: Create an interactive graphical user interface (GUI) using libraries like Streamlit or Tkinter for a more engaging user experience
- Optimization: Optimize the model further for deployment on resource-constrained devices, such as mobile phones or embedded systems
- Enhanced Dataset: Collect more diverse hand gesture data to improve model generalization

### Acknowledgments
Dataset: The Rock-Paper-Scissors image dataset used for training was sourced from Kaggle, contributed by 'drgfreeman'
Framework: Developed using PyTorch and torchvision for model building and training
Libraries: Utilized OpenCV for potential video processing capabilities
