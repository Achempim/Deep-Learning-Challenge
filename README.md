# Deep-Learning-Challenge

# Report on the Neural Network Model for Alphabet Soup

## Overview of the Analysis
The purpose of this analysis is to build a deep learning model to predict the success of charitable donations based on various features from Alphabet Soup’s dataset. The goal is to help Alphabet Soup make informed decisions by classifying which charity funding applications are more likely to be successful.

# Results
## Data Preprocessing
- Target Variable(s)
  - - The target variable for this model is IS_SUCCESSFUL, which indicates whether a donation application was successful or not.

# Feature Variable(s)
The features used in this model are all remaining columns after dropping IS_SUCCESSFUL, EIN, and NAME columns. These include categorical variables such as APPLICATION_TYPE and CLASSIFICATION, along with others relevant to the application.

# Variables Removed:
- EIN and NAME columns were removed from the input data since they are unique identifiers and do not provide predictive power for the target variable. These variables are neither features nor targets for our analysis.

# Compiling, Training, and Evaluating the Model
 - Neurons, Layers, and Activation Functions:
  - - The neural network model has three layers:
   - - - First Hidden Layer: 80 neurons, using ReLU activation.
   - - - Second Hidden Layer: 30 neurons, using ReLU activation.
   - - - Output Layer: 1 neuron, using a Sigmoid activation function for binary classification.
 - An optimized model was also implemented with slightly different configurations:
   - - - First Hidden Layer: 100 neurons, using ReLU activation.
   - - - Second Hidden Layer: 50 neurons, using ReLU activation.
   - - - Output Layer: 1 neuron with Sigmoid activation.

- Model Performance:
   - - The model was trained on scaled data with a validation split of 20% to monitor its performance on unseen data. Despite optimization attempts, the model's accuracy fluctuated but may not have fully reached the desired level of performance. This is a common outcome in classification models, especially with imbalanced or categorical-heavy datasets.

- Steps to Increase Model Performance:
   - - Data Scaling: StandardScaler was used to normalize the features for optimal model performance.
   - - Handling Rare Categories: Low-frequency categories in categorical features were grouped as 'Other' to reduce noise and complexity.
   - - Hyperparameter Tuning: An increased number of neurons and epochs were tested to improve learning capacity and convergence. Additionally, different batch sizes and validation splits were used to ensure a balanced training approach.

# Summary
The deep learning model developed achieved moderate accuracy, with training and validation accuracy visualized in the accuracy and loss graphs. The model has room for improvement; further optimizations such as tuning dropout rates or implementing regularization could enhance performance. Additionally, a different model, such as a Random Forest Classifier or XGBoost, could be explored, as they often perform well on tabular data with categorical features. These models may yield higher interpretability and performance on structured datasets like this one.