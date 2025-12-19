# Aman-TASK
Title:
Implementation of Logistic Regression from Scratch and Performance Evaluation

1. Aim

The aim of this experiment is to implement a Logistic Regression model from scratch using Python without using any inbuilt machine learning libraries. The model is trained using gradient descent and its performance is evaluated using accuracy, precision, recall, and F1-score.

2.Introduction to Logistic Regression:-

Logistic Regression is a supervised machine learning algorithm used for binary classification problems. It is mainly used when the output has only two possible classes such as pass/fail, yes/no, or 0/1.

Unlike linear regression, logistic regression uses a sigmoid function to convert the output into a probability value between 0 and 1. Based on this probability, the final class is decided using a threshold value, usually 0.5.

Logistic regression is widely used in applications such as medical diagnosis, spam email detection, and student performance prediction.

3. Algorithm Steps

The step-by-step procedure for implementing logistic regression is as follows:

1,Collect and prepare the dataset.

2, weights and bias with zero values.

3,Multiply input features with their corresponding weights.

4,Add the bias term to the result.

5,Apply the sigmoid function to obtain probability values.

6,Calculate the error between predicted output and actual output.

7,Update weights and bias using gradient descent.

8,Repeat the process for a fixed number of iterations.

9,Use the trained model to make final predictions.

10,Evaluate the model using performance metrics.

CODE :-

import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Dataset (Study Hours vs Pass/Fail)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])

# Initialize parameters
w = 0.0
b = 0.0
learning_rate = 0.1
epochs = 1000

# Training the model
for _ in range(epochs):
    z = w * X.flatten() + b
    y_pred = sigmoid(z)

    dw = np.mean((y_pred - y) * X.flatten())
    db = np.mean(y_pred - y)

    w -= learning_rate * dw
    b -= learning_rate * db

# Prediction function
def predict(X):
    z = w * X.flatten() + b
    y_pred = sigmoid(z)
    return [1 if i >= 0.5 else 0 for i in y_pred]

y_pred = predict(X)

EXPLANATION:-

W-Weight tells how important an input is
In model language:
-Attendance weight = 0.2
-Study hours weight = 0.8
-Study hours affect result more than attendance

Multiply input with weight-
- attendance × weight
  
  Bias-b:-
  - Bias is a starting point or base value.
  - example:

Teacher already thinks:
“Most students pass”
That thinking is bias.

Sigmoid Function:-
Any number → between 0 and 1
Before sigmoid:

Output can be big number (like 10, -5, 3)

After sigmoid:

Output becomes probability

Example:
-0.9 → 90% chance PASS
-0.2 → 20% chance PASS

- sigmoid(z) = 1 / (1 + e^-z)
  
-   Complete working example:-

  Inputs:
-Attendance = 70%
-Weight = 0.06
-Bias = -3

Step 1: Multiply
70 × 0.06 = 4.2

Step 2: Add bias
4.2 - 3 = 1.2

Step 3: Apply sigmoid
sigmoid(1.2) ≈ 0.77

Step 4: Final decision
0.77 ≥ 0.5 → PASS (Class 1)


Error simply means:

-How wrong is our prediction
-Error = Predicted value − Actual value
-Error tells us how much to change weights and bias

If error is:

-Big → change weights more
-Small → change weights slightly

Updating weights and bias (Gradient Descent):-

-Big word, simple meaning:
-Gradient Descent = Learn from mistakes step by step

Step 1: Find error
Step 2: Update weight
New weight = Old weight − (learning rate × error × input)
Step 3: Update bias
New bias = Old bias − (learning rate × error)

IT WORKS LIKE THIS:-

-If prediction is too small, weights increase
-If prediction is too large, weights decrease

So next prediction becomes closer to correct answer.


Evaluation Metrics:-

-To evaluate the performance of the model, the following metrics are used:
-Accuracy: Measures overall correctness of the model.
-Precision: Measures how many predicted positive values are actually positive.
-Recall: Measures how many actual positive values are correctly predicted.
-F1-Score: Harmonic mean of precision and recall.

y_true = y
y_pred = predict(X)

TP = sum((y_true == 1) & (np.array(y_pred) == 1))
TN = sum((y_true == 0) & (np.array(y_pred) == 0))
FP = sum((y_true == 0) & (np.array(y_pred) == 1))
FN = sum((y_true == 1) & (np.array(y_pred) == 0))

accuracy = (TP + TN) / len(y_true)
precision = TP / (TP + FP + 1e-10)
recall = TP / (TP + FN + 1e-10)
f1 = 2 * precision * recall / (precision + recall + 1e-10)



6. Result

The logistic regression model was successfully implemented from scratch. The trained model was able to classify the given dataset correctly. The obtained accuracy, precision, recall, and F1-score indicate that the model performs well for the given binary classification problem.


7. Conclusion

In this experiment, logistic regression was implemented without using any built-in machine learning libraries. The model was trained using gradient descent and evaluated using standard performance metrics. This experiment helped in understanding the internal working of logistic regression, including weight updates and probability-based classification. The results show that logistic regression is an effective algorithm for simple binary classification tasks.
