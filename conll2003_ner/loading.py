import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from datasets import load_dataset
import transformers
from collections import Counter

dataset = load_dataset("conll2003")

train_data = dataset["train"]
valid_data = dataset["validation"]
test_data = dataset["test"]

first_sentence = train_data[0]["tokens"]
first_labels = train_data[0]["ner_tags"]

ner_first = [train_data.features['ner_tags'].feature.int2str(tag) for tag in train_data[0]["ner_tags"]]
#print(ner_first)

ner_labels= {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 
                  'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}

pos_labels = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 
                  'CC': 10, 'CD': 11, 'DT': 12,'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 
                  'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,'NNS': 24, 'NN|SYM': 25, 
                  'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,'SYM': 34,
                  'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
                  'WP': 44, 'WP$': 45, 'WRB': 46}


def tag_conversion(diction:dict, class_int):
    key_list = list(diction.keys())
    value_list = list(diction.values())
    position = value_list.index(class_int) 
    return key_list[position]

#print(tag_conversion(ner_labels,8))
#print(first_sentence)
#print(first_labels)
#print(ner_labels.keys())

"""keys
tokens: This column contains the words in a sentence
pos_tags: Contains part-of-speech tags for each word
ner_tags: Contains Named Entity Recognition (NER) labels for each word.
"""

"""# Take a look at the first few examples in the training data
for i in range(1):
    # The "tokens" column contains the words
    words = train_data[i]["tokens"]
                                            
    # The "pos" column contains the part-of-speech tags
    pos_tags = train_data[i]["pos_tags"]
    
    # The "ner_tags" column contains the NER labels
    ner_tags = train_data[i]["ner_tags"]
    
  # Combine the words, POS tags, and NER labels for each example
    for word, pos, ner in zip(words, pos_tags, ner_tags):
        print(f"{word}: {tag_conversion(ner_labels,ner)}")

    print()  # Add a newline between examples
    """

def extract_sentences(data):
    sentences = []
    tags = []
    for sentence in data:
        words = [token for token in sentence['tokens']]
        ner_tags = [data.features['ner_tags'].feature.int2str(tag) for tag in sentence['ner_tags']]
        sentences.append(list(words))
        tags.append(list(ner_tags))
    return sentences, tags

# Extract sentences and tags for training
train_sentences, train_labels = extract_sentences(dataset['train'])
test_sentences, test_labels = extract_sentences(dataset['test'])
#print(train_labels)

label_list = dataset["train"].features["ner_tags"].feature.names
#print(label_list)

#print(dataset['train'][slice(0, 5, None)])


def calculate_entity_proportions(data):
    # Counting NER tags in the dataset
    tag_counts = Counter([tag for entry in data for tag in entry["ner_tags"]])
    
    # Total number of tags
    total_tags = sum(tag_counts.values())
    
    # Calculate proportions
    tag_proportions = {tag: count for tag, count in tag_counts.items()}
    
    return tag_proportions

# Calculating proportions for each part of the dataset
train_proportions = calculate_entity_proportions(train_data)
validation_proportions = calculate_entity_proportions(valid_data)
test_proportions = calculate_entity_proportions(test_data)

# Output the proportions
print("Train Data Entity Proportions:")
for tag, proportion in train_proportions.items():
    print(f"{tag}: {proportion}")

print("\nValidation Data Entity Proportions:")
for tag, proportion in validation_proportions.items():
    print(f"{tag}: {proportion}")

print("\nTest Data Entity Proportions:")
for tag, proportion in test_proportions.items():
    print(f"{tag}: {proportion}")
