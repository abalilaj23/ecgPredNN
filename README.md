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
