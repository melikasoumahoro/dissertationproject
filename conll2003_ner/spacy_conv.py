import spacy
import re
from spacy.tokens import Token
from spacy.tokenizer import Tokenizer

# Regular expressions for the custom tokenizer
prefix_re = re.compile(r'''^[\[\("']''')
suffix_re = re.compile(r'''[\]\)"']$''')
infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')

def custom_token_match(text):
    """
    Checks if a given text contains common English contractions.

    Uses a regular expression to search for common English contractions
    within the input text.

    Parameters:
    - text (str): The given text.

    Returns:
    bool: True if at least one contraction is found in the text, False otherwise.
    """

    # Regex to match contractions like "n't", "'re", "'s", etc.
    contraction_regex = re.compile(r"\b\w+'t\b|\b\w+'re\b|\b\w+'s\b|\b\w+'d\b|\b\w+'ll\b|\b\w+'ve\b")
    if contraction_regex.search(text):  # Use search instead of match to find the pattern anywhere in the text
        return True
    return False

# Adapted from
# https://stackoverflow.com/questions/51012476/spacy-custom-tokenizer-to-include-only-hyphen-words-as-tokens-using-infix-regex

def custom_tokenizer(nlp):
    """
    Initialises a new Tokenizer with custom_token_match as the token matching function.

    Parameters:
    - nlp (spacy.lang): The spaCy language model.

    Returns:
    spacy.tokenizer.Tokenizer: A custom tokenizer instance for the specified spaCy language model.
    """
    return Tokenizer(nlp.vocab,
                     token_match=custom_token_match)


class SpacyToConllConverter:
    """
    A converter that transforms NER tags from spaCy format to CoNLL format.

    Initialises with a specified spaCy model and optionally a set of NER labels.
    Applies the custom tokenizer to the spaCy pipeline

    Attributes:
    - nlp (spacy.lang): The spaCy language model loaded with the specified model name.
    - ner_labels (dict): A dictionary mapping CoNLL format entity labels to their corresponding numeric codes.

    Parameters:
    - model_name (str): The name of the spaCy language model to be loaded. Defaults to "en_core_web_lg".
    - ner_labels (dict, optional): An optional dictionary for custom NER labels mapping. If not provided,
      a default mapping for a set of common entity types (e.g., PER, ORG, LOC, MISC) in B- (Beginning) and I- (Inside)
      formats is used.

    Usage example:
    converter = SpacyToConllConverter(model_name="en_core_web_sm")
    conll_format_data = converter.convert("Your text here")
    """

    def __init__(self, model_name="en_core_web_lg", ner_labels=None):
        self.nlp = spacy.load(model_name)
        self.nlp.tokenizer = custom_tokenizer(self.nlp)
        
        self.ner_labels = ner_labels or {
            'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4,
            'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8
        }


    def spacy_ent_to_conll_int(self, token: Token) -> int:
        """
        Converts a spaCy token's entity type to an integer code based on a predefined mapping
        to the CoNLL-2003 NER tags.

        First determines whether a token is part of a named entity and if so,
        whether it is the beginning ('B-') or inside ('I-') of an entity. 
        Then constructs the CoNLL tag for recognized entity types ('ORG', 'LOC', 'PERSON', 'MISC'). 
        If the token's entity type does not match any of the specified types or the token is not part 
        of a named entity, it is tagged as 'O'.

        The CoNLL tag is then mapped to an integer code

        Parameters:
        - token (Token): A spaCy Token object.

        Returns:
        - int: An integer code corresponding to the CoNLL tag of the token's entity type
        """
        if token.ent_type_:
            prefix = "B-" if token.ent_iob_ == 'B' else "I-"
            if token.ent_type_ == 'PERSON':
                conll_tag = f"{prefix}PER"
            elif (token.ent_type_ == 'LOC' or token.ent_type_ == 'GPE' or token.ent_type_ == 'FAC'):
                conll_tag = f"{prefix}LOC"
            elif token.ent_type_ in {'ORG'}:
                conll_tag = f"{prefix}{token.ent_type_}"
            else:
                conll_tag = f"{prefix}MISC" # Default for unrecognized entity types
        else:
            conll_tag = "O"

        return self.ner_labels.get(conll_tag, self.ner_labels['O'])


    def convert_tokens_to_conll_int(self, tokens):
        """
        Converts a sentence into a list of their integer NER tags

        First join tokens to construct a single string and then processes this text with the spaCy NLP pipeline.
        Each token's entity type is then mapped to the integer code

        Parameters:
        - tokens (List[str]): A list of strings, where each string is a token (word or punctuation mark)

        Returns:
        - List[int]: A list of integer NER tags of the tokens

        """
        text = " ".join(tokens)
        doc = self.nlp(text)

        #print("Original tokens:", tokens)
        spacy_tokens = [token.text for token in doc]
        #print("SpaCy tokens:   ", spacy_tokens)
        return [self.spacy_ent_to_conll_int(token) for token in doc]

 
converter = SpacyToConllConverter()
#converter.convert_tokens_to_conll_int([ "\"", "Donald", "Trump", "do", "n't", "support", "any", "such", "recommendation", "because", "we", "do", "n't", "see", "any", "grounds", "for", "it", ",", "\"", "the", "Commission", "'s", "chief", "spokesman", "Nikolaus", "van", "der", "Pas", "told", "a", "news", "briefing", "." ])

