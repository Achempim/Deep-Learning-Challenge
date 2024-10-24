# Deep-Learning-Challenge

# Report on the Performance of the Deep Learning Model for Alphabet Soup

## Overview of the Analysis
The purpose of this analysis was to build a binary classification model that predicts whether a non-profit organization, funded by Alphabet Soup, will be successful or not, based on various features from historical data. By accurately identifying successful organizations, the goal is to optimize resource allocation and funding strategies, ensuring that Alphabet Soup supports initiatives with a high likelihood of positive outcomes.

## Results

### Data Preprocessing
- Target Variable:

  - IS_SUCCESSFUL (1 = successful, 0 = not successful)

- Feature Variables:

  - All columns from the dataset except EIN, NAME, and IS_SUCCESSFUL.
  - These columns include:
    - APPLICATION_TYPE
    - AFFILIATION
    - CLASSIFICATION
    - USE_CASE
    - ORGANIZATION
    - STATUS
    - INCOME_AMT
    - SPECIAL_CONSIDERATIONS
    - ASK_AMT

- Removed Variables:

  - EIN and NAME: These columns are identifiers, not relevant for training the model.
  - Reason: These identifiers are unique to each record and provide no predictive value.

### Compiling, Training, and Evaluating the Model
- Neurons, Layers, and Activation Functions:

  - First Hidden Layer:
    - 80 neurons, ReLU activation
    - Chosen to provide enough learning capacity to capture complex patterns.

- Second Hidden Layer:
    - 30 neurons, ReLU activation
    - Helps the model learn additional patterns and relationships between features.

- Output Layer:
    - 1 neuron, Sigmoid activation (used for binary classification).
    - Sigmoid outputs a probability between 0 and 1 to predict the target class.

- Were Target Performance Metrics Achieved?

  - Target Accuracy: 75% or higher
  - Achieved Accuracy:
    - Initial Test Accuracy: ~85%
- The model successfully met the target performance, demonstrating high predictive accuracy on the test data.

- Steps Taken to Improve Model Performance:

  - Experiment 1: Increased the number of neurons in the first hidden layer (from 80 to 100).
  - Experiment 2: Added an additional hidden layer to capture more complex relationships.
  - Experiment 3: Adjusted activation functions from ReLU to tanh to explore alternative transformations.
  - Experiment 4: Increased the number of epochs during training to 150 for better convergence.
  - Experiment 5: Used StandardScaler to normalize the input features and reduce model bias.

### Summary

The deep learning model performed well, achieving an accuracy of approximately 85% on the test data, surpassing the target of 75%. The model effectively captured relationships between features and produced reliable predictions. While the model achieved strong performance, it may still benefit from further tuning, particularly to reduce overfitting or enhance generalization.

### Recommendation for Improvement:

- Alternative Models to Explore:
  - Random Forest Classifier: This model might better handle categorical variables and offer interpretability in feature importance.
  - XGBoost or LightGBM: These gradient boosting algorithms excel in classification tasks, particularly with tabular data.
  - Logistic Regression: As a simpler model, this could serve as a baseline for comparison to ensure that the neural network isn't over-complicating the problem.

### Conclusion:

Based on the model's strong performance, it is recommended for deployment to help Alphabet Soup identify successful organizations. However, further exploration of ensemble models like Random Forest or XGBoost might yield additional benefits, such as improved interpretability or slight accuracy gains. This would help ensure the robustness and reliability of the funding decisions.