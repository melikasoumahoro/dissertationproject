from datasets import load_dataset
from itertools import chain
import scipy.stats

import sklearn
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

import evaluate

dataset = load_dataset("conll2003")

train_data = dataset["train"]
valid_data = dataset["validation"]
test_data = dataset["test"]

ner_tags = dataset["train"].features["ner_tags"].feature
class_names = ner_tags.names


def extract_sentences(data):
    """
    Extracts and structures sentences from a dataset.
    
    This function processes a dataset where each item represents a sentence. Each sentence is a dictionary
    containing tokens and their associated numeric NER tags. It converts these numeric tags into their
    string representations and pairs each token with its corresponding tag. The output is a list of sentences,
    where each sentence is represented as a list of (token, tag) tuples.
    
    Parameters:
    data (List[Dict[str, Any]]): A list of dictionaries, each dictionary is a sentence and contains 
        its 'tokens' and 'ner_tags'
    
    Returns:
    List[List[Tuple[str, str]]]: A nested list where each inner list contains tuples of tokens paired with their
        corresponding NER tag in string format.
    """
    sentences = []
    for sentence in data:
        words = [token for token in sentence['tokens']]
        ner_tags = [data.features['ner_tags'].feature.int2str(tag) for tag in sentence['ner_tags']]
        sentences.append(list(zip(words, ner_tags)))
    return sentences

# Extract sentences for training and testing
train_sentences = extract_sentences(dataset['train'])
test_sentences = extract_sentences(dataset['test'])


# The following functions are retrieved from:
# https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html

def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True  # Beginning of Sentence

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True  # End of Sentence

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

X_train = [sent2features(s) for s in train_sentences]
y_train = [sent2labels(s) for s in train_sentences]

X_test = [sent2features(s) for s in test_sentences]
y_test = [sent2labels(s) for s in test_sentences]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)
labels = list(crf.classes_)

y_pred = crf.predict(X_test)

# Flatten the true and predicted labels for evaluation
y_test_flat = [label for sublist in y_test for label in sublist]
y_pred_flat = [label for sublist in y_pred for label in sublist]

target_names = list(class_names)  
print(classification_report(y_test_flat, y_pred_flat, labels=target_names, target_names=target_names, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test_flat, y_pred_flat, labels=target_names)

plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('CRF Confusion Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Full named entity performance
metric = evaluate.load("seqeval")
results = metric.compute(predictions=y_pred, references=y_test, scheme="IOB2", mode="strict")
print("Full entity evaluation results:", results)