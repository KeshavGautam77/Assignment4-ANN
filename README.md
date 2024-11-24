# Assignment4-ANN

# Project Title: Breast Cancer Analysis Using ANN
# Description:
Breast cancer is the most prevalent cancer among women worldwide, making up approximately 25% of all cancer cases. In 2015, over 2.1 million new cases were reported. It occurs when cells in the breast grow uncontrollably, often forming tumors visible on X-rays or felt as lumps. Diagnosing breast cancer presents the challenge of distinguishing between malignant (cancerous) and benign (non-cancerous) tumors. This project aims to address this challenge by using machine learning techniques, particularly an Artificial Neural Network (ANN), to analyze and classify breast cancer tumors based on the Breast Cancer Wisconsin (Diagnostic) Dataset.

# Dataset Information:
The dataset contains measurements from digitized images of fine needle aspirate (FNA) samples from breast masses. These samples represent features extracted from the nuclei of cells, which provide crucial information to classify the tumor as benign or malignant.

Source: The dataset is available through multiple channels:
UW CS FTP server
UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic)
Kaggle: Breast Cancer Wisconsin Data (Kaggle)
Attribute Information:
ID number: Unique identifier for each record.
Diagnosis: Class label (M = Malignant, B = Benign).
Features (30 attributes derived from the characteristics of the tumor):
Radius: Mean distance from the center to points on the perimeter.
Texture: Standard deviation of grayscale values.
Perimeter: Perimeter of the nucleus.
Area: Area of the nucleus.
Smoothness: Local variation in the radius lengths.
Compactness: Perimeter² / area - 1.0.
Concavity: Severity of concave portions of the contour.
Concave Points: Number of concave portions in the contour.
Symmetry: Symmetry of the nucleus.
Fractal Dimension: "Coastline approximation" - 1.
Each feature is calculated in three ways:

Mean: Average value for each feature.
Standard Error: Standard deviation of the feature.
Worst: Largest value of the feature (mean of the three largest values).
This results in 30 features used for classification.

Class Distribution:
Benign: 357 instances
Malignant: 212 instances
There are no missing values in the dataset.
# Steps Followed:
# 1. Exploratory Data Analysis (EDA):
The dataset was analyzed to understand its structure and identify patterns in the features.
Visualizations were created to explore the distribution of features and their correlation with the diagnosis (malignant vs benign).
Outliers and feature distributions were examined, as well as the relationships between the target variable and features.
# 2. Data Cleaning:
The target variable (Diagnosis) was encoded as follows: Benign = 0, Malignant = 1.
Any irrelevant or duplicate data points were removed.
Missing data: None were found in the dataset.
# 3. Preprocessing and Feature Selection:
Standardization: The dataset was standardized using the StandardScaler to ensure that all features are on the same scale. This is important since the ANN model (MLPClassifier) is sensitive to the scale of the input data.
Feature Selection: The most important features were selected using techniques such as ANOVA F-test and Mutual Information. This helps reduce the complexity of the model and focuses on the most predictive features. The selected features mainly focused on tumor radius, perimeter, area, and concavity.
# 4. Model Building:
A Multi-layer Perceptron (MLP) classifier was used, which is a type of Artificial Neural Network (ANN).
The model architecture included two hidden layers, each with 50 neurons.
The model was trained on the training data, and its performance was evaluated using accuracy, precision, recall, and F1-score.
# 5. Model Tuning:
Grid Search Cross-validation was applied to tune the hyperparameters of the MLP model. Important hyperparameters such as activation function, regularization (alpha), batch size, hidden layer sizes, learning rate, and solver were optimized.
The best hyperparameters found were:
Activation: 'relu'
Alpha: 0.0001
Batch Size: 64
Hidden Layer Sizes: (100,)
Learning Rate: 'constant'
Solver: 'adam'
These optimized parameters were used to train the final model, resulting in improved performance.

#6. Model Evaluation:
The final model achieved an accuracy of 97% on the test set.
Precision for benign cases: 0.97
Precision for malignant cases: 0.98
Recall for benign cases: 0.99
F1-Score: High for both benign and malignant classes.
The confusion matrix for model performance on the test set:

lua
Copy code
[[70  1]
 [ 2 41]]
This matrix shows that the model correctly identified most of the malignant and benign cases.

# 7. Deployment:
The trained model (mlp_model_imp.pkl) and the scaler (scaler.pkl) were saved using pickle for later use.
These files can be used to deploy the model in a Streamlit app, allowing users to input feature values (such as radius, area, texture) and predict whether a tumor is benign or malignant.
Project Structure:
Dataset:

This folder contains the Breast Cancer Wisconsin (Diagnostic) dataset used for training and evaluation.
Dataset Link
Notebook:

Includes Jupyter notebooks for exploratory data analysis, feature engineering, model development, and evaluation.
Notebook Link

# Pickle-Files:
Contains saved model files (mlp_model_imp.pkl) and the scaler file (scaler.pkl).


#streamlit.py:
The main file for the Streamlit app that loads the model and scaler and takes input from the user for tumor classification.
streamlit.py

#readme.md:
This file explains the dataset, methodology, results, and how to run the project.
README Link

#requirements.txt:

#Lists the Python libraries needed to run the project.
requirements.txt Link

# Conclusion:
This project demonstrates the application of machine learning, specifically Artificial Neural Networks (ANNs), in the classification of breast cancer tumors. By performing thorough data preprocessing, feature selection, and model tuning, we were able to develop a robust model that achieves high accuracy in distinguishing between benign and malignant tumors. The model is ready for deployment in real-world applications and can assist healthcare professionals in diagnosing breast cancer with high reliability.
