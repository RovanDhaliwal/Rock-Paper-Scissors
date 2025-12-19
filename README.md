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

### Setup
```bash
pip install torch torchvision opencv-python numpy pandas matplotlib
# Optional: for web UI, if implemented
pip install streamlit
