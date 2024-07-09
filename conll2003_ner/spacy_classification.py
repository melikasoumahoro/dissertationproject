import spacy
import numpy as np

from datasets import load_dataset
from spacy_conv import SpacyToConllConverter

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

import evaluate

# Loading dataset
dataset = load_dataset("conll2003")
train_data = dataset["train"]
valid_data = dataset["validation"]
test_data = dataset["test"]

ner_tags = dataset["train"].features["ner_tags"].feature
class_names = ner_tags.names

# Initialise lists to hold sentence tokens and their corresponding NER tags
# Store tuples - (sentence, ner_tags)
true_classes = []
predicted_classes = []

converter = SpacyToConllConverter()

# Make predictions using the spacy pipeline in SpacyToConllConverter() 
# and convert them to the standard format
for i in range(len(test_data)):
    words = test_data[i]["tokens"]
    ner_tags = test_data[i]["ner_tags"]
    spacy_tags = converter.convert_tokens_to_conll_int(words)

    true_classes.append((words, ner_tags))
    predicted_classes.append((words, spacy_tags))


# Flatten the true and predicted labels for evaluation
true_classes_flat = [label for sentence in true_classes for label in sentence[1]]
predicted_classes_flat = [label for sentence in predicted_classes for label in sentence[1]]

print(classification_report(true_classes_flat, predicted_classes_flat, target_names=class_names, zero_division=0))

# Compute the confusion matrix
cm = confusion_matrix(true_classes_flat, predicted_classes_flat)

# Plot confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('SpaCy Confusion Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()


predicted_classes_str = []
true_classes_str = []

for sentence in predicted_classes:
    tags = sentence[1]
    tags_str = [dataset["test"].features["ner_tags"].feature.int2str(tag) for tag in tags]
    predicted_classes_str.append(tags_str)

for sentence in true_classes:
    tags = sentence[1]
    tags_str = [dataset["test"].features["ner_tags"].feature.int2str(tag) for tag in tags]
    true_classes_str.append(tags_str)


# Full named entity performance
metric = evaluate.load("seqeval")
results = metric.compute(predictions=predicted_classes_str, references=true_classes_str, scheme="IOB2", mode="strict")
print("Full entity evaluation results:", results)