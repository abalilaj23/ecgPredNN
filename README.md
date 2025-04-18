# Project 1: Predicting Heart Disease: Binary Classification Neural Network (heartDiseasePredNN)

This project applies a neural network to classify the presence of heart disease using structured tabular data. The workflow includes data preprocessing, model building with TensorFlow/Keras, training, evaluation, and performance visualization.

## Key Steps

- **Import and prepare data**: Load the heart disease dataset and perform preprocessing (one-hot encoding, normalization).
- **Train-test split**: Stratified split to preserve class distribution.
- **Model architecture**: A simple feedforward neural network with one hidden layer (ReLU) and an output sigmoid activation for binary classification.
- **Training**: The model is trained over 300 epochs with binary cross-entropy loss and the Adam optimizer.
- **Evaluation**: Assessed on accuracy and loss using both training/validation plots and test set performance.
- **Results**: The model achieved ~77% accuracy on the test set. Validation performance suggests room for optimization.

## Notes

- Preprocessing (encoding, normalization) was done outside the model. To deploy this pipeline, the same preprocessing steps must be applied to new data.
- For improved portability and robustness, consider integrating Keras preprocessing layers into the model pipeline.


# Project 2: Classifying Abnormal ECGs: Neural Network Model (ecgScansNNClass)

This project focuses on building a classification neural network to detect abnormal electrocardiogram (ECG) signals using the ECG5000 dataset. Each ECG signal is represented by 140 time-series data points, with a binary target variable:

- **1** = Normal ECG
- **0** = Abnormal ECG

## Project Workflow

### Packages & Data

The project leverages key Python libraries including:
- **TensorFlow**
- **Keras**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**

The dataset is loaded from TensorFlow's public storage and consists of 4,998 samples and 141 columns (140 signal points + 1 target).

### Data Preprocessing

- The dataset contains only numerical values.
- Data is split into training and testing sets using **stratified sampling** to preserve class balance.
- **Normalization** is applied using the training set’s mean and standard deviation to standardize the features.

### Model Architecture

A simple feedforward neural network is implemented:

- **Input layer** with 140 features
- **1 hidden layer** with 16 ReLU-activated neurons
- **Output layer** with 1 sigmoid-activated neuron for binary classification

The model is compiled with:
- **Adam optimizer**
- **Binary cross-entropy loss**

###  Training

- The model is trained for 50 epochs with a batch size of 32 and a 20% validation split.
- The training reaches over **99% accuracy** on the training and validation sets within a few epochs, indicating effective learning.
