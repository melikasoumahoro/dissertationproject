from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
dataset = load_dataset("conll2003")

ner_tags = dataset["train"].features["ner_tags"].feature
class_names = ner_tags.names

# Convert numerical tags in the training set to string labels
y_train = [dataset["train"].features["ner_tags"].feature.int2str(tag) 
           for sentence in dataset["train"]["ner_tags"] 
           for tag in sentence]

# Convert numerical tags in the test set to string labels
y_test = [dataset["test"].features["ner_tags"].feature.int2str(tag) 
          for sentence in dataset["test"]["ner_tags"] 
          for tag in sentence]


# Flatten all tokens from sentences
tokens_train = [token for sentence in dataset["train"]["tokens"] for token in sentence]
tokens_test = [token for sentence in dataset["test"]["tokens"] for token in sentence]

# Extract features
vectorizer = CountVectorizer()
X_train_features = vectorizer.fit_transform(tokens_train)
X_test_features = vectorizer.transform(tokens_test)

# Train the logistic regression model
model = LogisticRegression(multi_class='multinomial', max_iter=1000)
model.fit(X_train_features, y_train)

# Predictions
y_pred = model.predict(X_test_features)

target_names = class_names
print(classification_report(y_test, y_pred, labels=target_names, target_names=target_names, zero_division=0))


# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=target_names)

# Plotting the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Logistic Regression Confusion Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()
