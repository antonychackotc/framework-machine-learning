# framework-machine-learning

st.subheader("1) Upload File")
st.write("In summary, this tab is designed for file upload, reading different file formats into pandas DataFrames, and enabling users to select and view their datasets for further exploration")

st.subheader("2) Dataset Analysis")
st.write("Tab 2 provides a comprehensive data analysis summary:")
st.write("Displays basic information about the dataset (shape, column types).")
st.write("Identifies issues like missing values, outliers, and columns requiring encoding or scaling.")
st.write("It helps users understand the dataset better and prepare it for further preprocessing or modeling.")

st.subheader("3) Supervised Learning")


1. Target Column Selection:
The user starts by selecting a target column (the dependent variable) from the dataset using a dropdown. This is the variable the model will predict.
If a target column is selected:
Task Type Inference:
The code checks if the target column is continuous (e.g., numeric values, such as prices or temperatures) or discrete (e.g., categorical values, such as classes or categories).
If the target column is continuous with more than 20 unique values, the task is inferred as Regression. Otherwise, it's treated as Classification.
If the target column contains categorical data (non-numeric), the task is inferred as Classification.
Target Data Type: The type of the target column (Continuous or Discrete) is displayed for clarity.
2. Class Imbalance Check (for Classification):
If the task is Classification, the class distribution of the target column is checked.
Balanced vs. Unbalanced:
The distribution of the classes is calculated and visualized as a bar chart. If one class dominates over others (i.e., one class has more than 70% of the data), the dataset is considered Unbalanced.
If the dataset is unbalanced, techniques like oversampling (e.g., SMOTE) or undersampling are discussed, along with their pros and cons:
Oversampling: Adds synthetic data for the minority class to balance the classes, but risks overfitting.
Undersampling: Reduces the majority class samples to balance the classes, but may lose valuable information.
3. Manual Task Type Selection:
The user can manually override the inferred task type (either Classification or Regression) and target type (either Continuous or Discrete) by selecting from radio buttons.
Based on the manual selection, the recommended algorithms are displayed:
Regression Algorithms: If the task is regression, algorithms like Linear Regression, Polynomial Regression, Ridge Regression, Random Forest Regressor, XGBoost, etc., are suggested.
Classification Algorithms: If the task is classification, algorithms like Logistic Regression, Decision Trees, Random Forest Classifier, XGBoost, Naive Bayes, etc., are suggested.
4. Feature Selection:
After selecting the target column, the user selects the feature columns (independent variables) from a multiselect dropdown. These are the columns used to predict the target.
The user can select any combination of columns except for the target column.
5. Data Preparation:
Once the feature columns are selected, the dataset is split into a training set and a testing set using train_test_split(). By default, 30% of the data is allocated for testing, and the remaining 70% is used for training.
Target Encoding: If the task is classification and the target column is binary, the target column values are mapped to 0 and 1. This is necessary for binary classification algorithms.
6. Algorithm Selection:
The user selects the machine learning algorithm to use for training the model from a dropdown. The available options are dependent on the chosen task type (either Regression or Classification).
Based on the selected algorithm, the corresponding model is initialized.
7. Model Training and Evaluation:
Training: The selected model is trained on the training data (X_train and y_train).
Prediction: After training, predictions are made on the test data (X_test), and the time taken for training is displayed.
Evaluation Metrics:
For Regression tasks:
The Mean Squared Error (MSE) is computed to evaluate the model’s performance.
For Classification tasks:
Various evaluation metrics such as Accuracy, Precision, Recall, and F1 Score are calculated using the corresponding functions from sklearn.metrics.
Confusion Matrix: A confusion matrix is displayed as a heatmap, showing the distribution of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
Classification Report: A detailed classification report is generated, showing precision, recall, and F1 scores for each class.
The Confusion Matrix is visualized as a heatmap using seaborn, with labels added for clarity (e.g., "Predicted 0" and "Predicted 1").
8. Model Performance Details:
The model's performance is described in terms of the classification metrics (Accuracy, Precision, Recall, and F1 score), and the Confusion Matrix is annotated with the counts of TP, TN, FP, and FN.
The user can also view additional information about the meaning of these values (e.g., True Positive, False Positive) for clarity.
9. Dataset Download:
The user is given the option to download the updated dataset containing the predictions and the corresponding true labels.
A new column is added to the test dataset showing the predicted values, and a "Result" column is added to indicate whether the prediction was a True Positive, True Negative, False Positive, or False Negative.
The updated dataset is available for download as a CSV file using a st.download_button.
10. Conclusion:
The user has the flexibility to select a model, train it, evaluate its performance, and even download the predictions alongside the original data for further analysis or reporting.
Summary of Key Components:
Target Selection: Choose the variable to predict.
Task Type Inference: Determine if the problem is a regression or classification task.
Class Imbalance Check: Evaluate if oversampling or undersampling is needed.
Algorithm Suggestions: Receive recommendations based on the task type.
Feature Selection: Choose independent variables to predict the target.
Model Training and Evaluation: Train and evaluate models, with performance metrics and visualizations.
Confusion Matrix: For classification tasks, visualize and interpret the performance in terms of TP, TN, FP, FN.
Model Performance: Display various evaluation metrics (e.g., accuracy, precision, recall).
Download Predictions: Option to download the updated dataset with predictions.

Summary of Pros and Cons
Linear Regression: Simple, interpretable, but assumes linearity.
Polynomial Regression: Models non-linear relationships, but risk of overfitting.
Ridge Regression: Handles multicollinearity and overfitting, but needs careful tuning.
Lasso Regression: Performs feature selection, but can over-penalize features.
Decision Tree Regressor/Classifier: Easy to interpret, but prone to overfitting.
Random Forest Regressor/Classifier: Reduces overfitting, but computationally expensive.
XGBoost: High performance, but requires hyperparameter tuning.
KNN Regressor/Classifier: Simple, but computationally expensive with large datasets.
Logistic Regression: Simple, but assumes linearity and performs poorly with non-linear data.
SVM: Effective for high-dimensional data, but computationally expensive.
Naive Bayes: Simple, works well with small datasets, but assumes feature independence.

st.header("1. Regression Algorithms")

st.subheader("Continous")

1)Linear Regression
Function: Predicts a continuous target variable based on the linear relationship with one or more independent variables.

2)Polynomial Regression
Function: Extends linear regression by adding polynomial terms to the model to capture non-linear relationships.

3)Ridge Regression
Function: Linear regression with L2 regularization to penalize large coefficients, preventing overfitting.

4)Lasso Regression
Function: A form of linear regression that uses L1 regularization to shrink some coefficients to zero, resulting in a sparse model.

st.subheader("Discret")

1) Decision Tree Regressor
Function: Uses a tree structure to partition the data into regions and make predictions by averaging the target variable within each region.

2) Random Forest Regressor
Function: An ensemble method that combines multiple decision trees to improve predictive accuracy and reduce overfitting.

3) XGBoost (Extreme Gradient Boosting)
Function: An optimized gradient boosting algorithm that sequentially builds decision trees to correct the errors made by previous trees

4) KNN Regressor (K-Nearest Neighbors)
Function: Predicts the target value by averaging the values of the k-nearest neighbors.

###########################################################################################

st.header("2. Classification Algorithms")

st.subheader("Binary Classification")

1) Logistic Regression
Function: A linear model used for binary classification by predicting probabilities using the logistic function.

2) SVM (Support Vector Machine) Classifier
Function: Finds the hyperplane that best separates different classes in a higher-dimensional space.

3) Decision Tree Classifier
Function: Uses a tree structure to classify data by splitting the data at each node based on feature values.

4) Random Forest Classifier
Function: An ensemble of decision trees where each tree is trained on a random subset of the data and features.

5) XGBoost Classifier
Function: A gradient boosting algorithm that builds decision trees sequentially, correcting previous errors.

6) Naive Bayes Classifier
Function: Based on Bayes’ Theorem, it predicts the probability of a class label given the input features, assuming independence between features.

7) KNN Classifier (K-Nearest Neighbors)
Function: Classifies a sample based on the majority class of its k-nearest neighbors.

#############################################################################################

Key Characteristics of Unsupervised Learning:
No labeled data: Unsupervised learning algorithms work with datasets that do not have predefined labels or outcomes.
Discover hidden patterns: The goal is to find patterns, groupings, or structures in the data, like clusters or associations between features.
Common tasks: The most common tasks in unsupervised learning include clustering, dimensionality reduction, and anomaly detection.
Common Unsupervised Learning Algorithms:
Clustering Algorithms: These algorithms group similar data points together based on their characteristics or features. Some popular clustering algorithms include:

KMeans Clustering: Divides data into a set number of clusters based on proximity to the centroid of each cluster.
DBSCAN (Density-Based Spatial Clustering of Applications with Noise): Groups data based on density, making it great for discovering clusters of arbitrary shapes and handling noise.
Agglomerative Clustering: A hierarchical clustering method that builds a tree of clusters, allowing flexible group shapes and scalability.
Gaussian Mixture Models (GMM): Assumes that data is generated from a mixture of several Gaussian distributions and tries to identify the different clusters.
Dimensionality Reduction: Unsupervised algorithms can also help reduce the number of features in a dataset while preserving its underlying structure. Some popular algorithms include:

Principal Component Analysis (PCA): A method that transforms features into a smaller set of uncorrelated features, which explain the variance in the data.
t-SNE (t-Distributed Stochastic Neighbor Embedding): A technique used for visualizing high-dimensional data in a lower-dimensional space, typically 2D or 3D.
Anomaly Detection: In unsupervised learning, anomaly detection algorithms are used to identify rare or unusual observations in data, which might signify fraud, errors, or outliers.

Why Use Unsupervised Learning?
Exploring Unknown Structures: Unsupervised learning is useful when we don’t know what to look for in the data. It helps in discovering patterns that were not predefined or labeled.
Data Exploration: It aids in understanding the structure and characteristics of the dataset before proceeding with further analysis.
Preprocessing for Supervised Learning: Some unsupervised learning algorithms, like clustering and dimensionality reduction, can be used to preprocess data for supervised tasks.
Example Use Cases of Unsupervised Learning:
Customer Segmentation: By clustering customers based on their buying patterns, businesses can target specific groups with personalized marketing strategies.
Anomaly Detection in Network Security: Identifying unusual patterns in network traffic that could signal security breaches or fraud.
Dimensionality Reduction for Data Visualization: Reducing the dimensions of large datasets to make them easier to visualize and interpret.

KMeans: Simple, fast, but assumes spherical clusters and requires specifying the number of clusters.
DBSCAN: Can find clusters of arbitrary shape and handle noise, but sensitive to parameters.
Agglomerative Clustering: Hierarchical and flexible, but computationally expensive for large datasets.
GMM: Soft clustering and flexible cluster shapes, but assumes Gaussian distribution and is sensitive to initialization.
PCA: Effective for dimensionality reduction and speeding up algorithms, but linear and hard to interpret.
t-SNE: Excellent for visualization and preserving local structure, but slow and hard to interpret.

#################################################################################################################


