# Brain Tumor Classification using CNN

## Overview

This project implements a Convolutional Neural Network (CNN) to classify brain tumors from MRI images. The model is trained on a dataset of MRI scans and can predict whether a given scan indicates the presence of a brain tumor.

## Dataset

The dataset consists of MRI scans labeled with tumor or non-tumor categories. It is divided into training, validation, and testing sets.

## Features

- Image augmentation for improved generalization
- CNN architecture optimized for image classification
- Visualization of class distribution and training metrics
- Evaluation using a confusion matrix and test predictions

## Installation

To run this project, install the required dependencies:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## Running the Project

1. Prepare the dataset and ensure images are structured in training and validation folders.
2. Run the Jupyter Notebook to train the CNN model.
3. Evaluate the model using test data and visualize results.

## Model Architecture

The CNN model includes:

- Convolutional layers with ReLU activation
- Max pooling layers for downsampling
- Fully connected layers for classification
- Softmax activation for final predictions

## Results

- Training and validation accuracy over multiple epochs
- Confusion matrix to assess classification performance
- Example predictions on test data

## Future Improvements

- Experiment with different CNN architectures
- Implement transfer learning for improved accuracy
- Deploy the model as a web application

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- CNN Architectures
