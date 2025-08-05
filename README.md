# Spam-Detection

Spam-Detection is a Python package for detecting and filtering spam messages using Machine Learning models. The
package integrates with Django or any other project that uses python and offers different types of classifiers: Naive
Bayes, Random Forest, and Support Vector Machine (SVM). Since version 2.1.0, two new classifiers have been added:
Logistic Regression and XGBClassifier.


## Installation

You can install the spam detection package via pip:

```sh
pip install spam-detection
```

Make sure you have the following dependencies installed:

- scikit-learn
- nltk
- pandas
- numpy
- joblib
- xgboost

Additionally, you'll need to download the NLTK data and to do so, use the python interpreter to run the following
commands:

```python
import nltk

nltk.download('wordnet')
nltk.download('stopwords')
```

## Usage

### Training the Models

Before using the classifiers, you must train the models. Training data is loaded from a CSV file. You can find the
training data in the `data` directory in the GitHub's page of the project. The CSV file must have 3 columns: `label`,
`text` and `label_num`. The `text` column contains the content of the message to analyze and the `label` column
contains the labels `ham` or `spam` and `label_num` contains the number `0` (not spam) or `1`(spam).

The more data you have, the better the models will perform.

To train the models, run the following command:

```sh
python3 spam_detection/trainer.py
```



### Tests

The test results are shown below:

#### _Model: NAIVE_BAYES_

##### Confusion Matrix:

|                  | Predicted: Ham       | Predicted: Spam      |
|------------------|----------------------|----------------------|
| **Actual: Ham**  | 1935 (True Negative) | 170 (False Positive) |
| **Actual: Spam** | 221 (False Negative) | 633 (True Positive)  |

- True Negative (TN): 1935 messages were correctly identified as ham (non-spam).
- False Positive (FP): 170 ham messages were incorrectly identified as spam.
- False Negative (FN): 221 spam messages were incorrectly identified as ham.
- True Positive (TP): 633 messages were correctly identified as spam.

##### Performance Metrics:

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Ham          | 0.90      | 0.92   | 0.91     | 2105    |
| Spam         | 0.79      | 0.74   | 0.76     | 854     |
| **Accuracy** |           |        | **0.87** | 2959    |
| Macro Avg    | 0.84      | 0.83   | 0.84     | 2959    |
| Weighted Avg | 0.87      | 0.87   | 0.87     | 2959    |

##### Accuracy: 0.8678607637715444

<br>

#### _Model: RANDOM_FOREST_

##### Confusion Matrix:

|                  | Predicted: Ham       | Predicted: Spam     |
|------------------|----------------------|---------------------|
| **Actual: Ham**  | 2067 (True Negative) | 38 (False Positive) |
| **Actual: Spam** | 36 (False Negative)  | 818 (True Positive) |

- True Negative (TN): 2067 messages were correctly identified as ham (non-spam).
- False Positive (FP): 38 ham messages were incorrectly identified as spam.
- False Negative (FN): 36 spam messages were incorrectly identified as ham.
- True Positive (TP): 818 messages were correctly identified as spam.

##### Performance Metrics:

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Ham          | 0.98      | 0.98   | 0.98     | 2105    |
| Spam         | 0.96      | 0.96   | 0.96     | 854     |
| **Accuracy** |           |        | **0.97** | 2959    |
| Macro Avg    | 0.97      | 0.97   | 0.97     | 2959    |
| Weighted Avg | 0.98      | 0.97   | 0.98     | 2959    |

##### Accuracy: 0.9749915511997297

<br>

#### _Model: SVM_

##### Confusion Matrix:

|                  | Predicted: Ham       | Predicted: Spam     |
|------------------|----------------------|---------------------|
| **Actual: Ham**  | 2080 (True Negative) | 25 (False Positive) |
| **Actual: Spam** | 41 (False Negative)  | 813 (True Positive) |

- True Negative (TN): 2080 messages were correctly identified as ham (non-spam).
- False Positive (FP): 25 ham messages were incorrectly identified as spam.
- False Negative (FN): 41 spam messages were incorrectly identified as ham.
- True Positive (TP): 813 messages were correctly identified as spam.

##### Performance Metrics:

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Ham          | 0.98      | 0.99   | 0.98     | 2105    |
| Spam         | 0.97      | 0.95   | 0.96     | 854     |
| **Accuracy** |           |        | **0.98** | 2959    |
| Macro Avg    | 0.98      | 0.97   | 0.97     | 2959    |
| Weighted Avg | 0.98      | 0.98   | 0.98     | 2959    |

##### Accuracy: 0.9773572152754308

<br>

#### _Model: LOGISTIC_REGRESSION_

##### Confusion Matrix:

|                  | Predicted: Ham       | Predicted: Spam     |
|------------------|----------------------|---------------------|
| **Actual: Ham**  | 2065 (True Negative) | 48 (False Positive) |
| **Actual: Spam** | 46 (False Negative)  | 989 (True Positive) |

- True Negative (TN): 2065 messages were correctly identified as ham (non-spam).
- False Positive (FP): 48 ham messages were incorrectly identified as spam.
- False Negative (FN): 46 spam messages were incorrectly identified as ham.
- True Positive (TP): 989 messages were correctly identified as spam.

##### Performance Metrics:

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Ham          | 0.98      | 0.98   | 0.98     | 2113    |
| Spam         | 0.95      | 0.96   | 0.95     | 1035    |
| **Accuracy** |           |        | **0.97** | 3148    |
| Macro Avg    | 0.97      | 0.97   | 0.97     | 3148    |
| Weighted Avg | 0.97      | 0.97   | 0.97     | 3148    |

##### Accuracy: 0.9707680491551459

<br>

#### _Model: XGB_

##### Confusion Matrix:

|                  | Predicted: Ham       | Predicted: Spam      |
|------------------|----------------------|----------------------|
| **Actual: Ham**  | 2050 (True Negative) | 63 (False Positive)  |
| **Actual: Spam** | 28 (False Negative)  | 1007 (True Positive) |

- True Negative (TN): 2050 messages were correctly identified as ham (non-spam).
- False Positive (FP): 63 ham messages were incorrectly identified as spam.
- False Negative (FN): 28 spam messages were incorrectly identified as ham.
- True Positive (TP): 1007 messages were correctly identified as spam.

##### Performance Metrics:

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Ham          | 0.99      | 0.97   | 0.98     | 2113    |
| Spam         | 0.94      | 0.97   | 0.96     | 1035    |
| **Accuracy** |           |        | **0.97** | 3148    |
| Macro Avg    | 0.96      | 0.97   | 0.97     | 3148    |
| Weighted Avg | 0.97      | 0.97   | 0.97     | 3148    |

##### Accuracy: 0.9710927573062261

The models that performed the best are the SVM and Logistic Regression, with the SVM model achieving slightly higher
accuracy than Logistic Regression.
Given that no single model achieved perfect accuracy, I have decided to implement a voting classifier.
This classifier will combine the predictions of the five models (Naive Bayes, Random Forest, SVM,
Logistic Regression, and XGB) using a majority vote system to make the final prediction.
This approach aims to leverage the strengths of each model to improve overall prediction accuracy.

##### Weighted Voting System

To enhance the decision-making process, I've refined our approach to a weighted voting system. This new system assigns
different weights to each model's vote based on their respective accuracies. The weights are proportional to the
accuracy of each model relative to the sum of the accuracies of all models. The models with higher accuracy have a
greater influence on the final decision.

The models and their respective proportional weights are as follows:

- Naive Bayes: Weight = 0.1822
- Random Forest: Weight = 0.2047
- SVM (Support Vector Machine): Weight = 0.2052
- Logistic Regression: Weight = 0.2039
- XGBoost (XGB): Weight = 0.2039

These weights were calculated based on the accuracy of each model as a proportion of the total accuracy of all models.
The final decision whether a message is spam or not is determined by the weighted spam score. Each model casts a vote
(spam or not spam), and this vote is multiplied by the model's weight. The weighted spam scores from all models are then
summed up. If this total weighted spam score exceeds 50% of the total possible weight, the message is classified as
spam. Otherwise, it's classified as not spam (ham).

This approach ensures that the more accurate models have a larger say in the final decision, thereby increasing the
reliability of spam detection. It combines the strengths of each model, compensating for individual weaknesses and
provides a more nuanced classification.

##### System Output

The system provides a detailed output for each message, showing the vote (spam or ham) from each model, along with its
weight. It also displays the total weighted spam score and the final classification decision (Spam or Not Spam). This
transparency in the voting process allows for easier understanding and debugging of the model's performance on different
messages.

If you have trained the models on new data, you can test them by running the following command:

```sh
python tests/test.py
```



### Making Predictions

To use the spam detector in your Django project:

1. Import the `VotingSpamDetector` from the `prediction` module.
2. Create an instance of the detector.
3. Use the `is_spam` method to check if a message is spam.

```python
from spam_detection.prediction.predict import VotingSpamDetector

# Create the spam detector
spam_detector = VotingSpamDetector()

# Check if a message is spam
message = "Enter the message here"
is_spam = spam_detector.is_spam(message)
print(f"Is spam: {is_spam}")
```

## Project Structure

- `classifiers/`: Contains the different classifiers (Naive Bayes, Random Forest, SVM, XGB & Logistic Regression).
- `data/`: Contains the sample dataset for training the classifiers.
- `loading_and_processing/`: Contains utility functions for loading and preprocessing data.
- `models/`: Contains the trained models and their vectorizers.
- `prediction/`: Contains the main spam detector class.
- `tests/`: Contains scripts for testing
- `tuning/`: Contains scripts for tuning the classifiers.
- `training/`: Contains scripts for training the classifiers.
