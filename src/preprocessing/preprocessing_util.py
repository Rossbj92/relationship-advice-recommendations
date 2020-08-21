import pandas as pd
import numpy as np
import re
import spacy
from sqlalchemy import create_engine
nlp = spacy.load('en_core_web_sm')

CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

def load_raw_data(file):
    """Loads csv file."""
    return pd.read_csv(file)

class TextCleaner:
    """Contains several preprocessing utility methods.

    This class's purpose is for text cleaning rather than
    any NLP-based methods. For example, these include expanding
    contractions and formatting age/gender conventions.

    Attributes:
        text (list): A list of strings.
    """
    def __init__(self, text):
        self.text = text

    def touchup(self, text = None):
        """Replaces new-line characters, lowercases, and strips whitespace.

        Args:
            text (list): A list of strings.

        Returns:
            The same list of strings free of uppercase letters, new-line
            characters, and whitespace.
        """
        if text == None:
            text = self.text
        docs = []
        for doc in text:
            joined_comments = ((''.join([word for word in doc]))
                                  .replace('\\n', ' ')
                                  .lower()
                                  .strip()
                              )
            docs.append(joined_comments)
        return docs

    def expand_contractions(self, text = None, contraction_mapping=CONTRACTION_MAP):
        """Expands string contractions.

        Adapted from https://github.com/dipanjanS/practical-machine
        -learning-with-python/tree/master/bonus%20content/nlp%20proven
        %20approach.

        Args:
            text (list): List of strings.
            contraction_mapping (dict): Dictionary with keys containing
              contractions and values containing replacement words.

        Returns:
            The same list of strings with any contractions expanded.
        """
        if text == None:
            text = self.text
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                          flags=re.IGNORECASE|re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match)\
                                    if contraction_mapping.get(match)\
                                    else contraction_mapping.get(match.lower())
            expanded_contraction = first_char+expanded_contraction[1:]
            return expanded_contraction
        docs = []
        for doc in text:
            expanded_text = contractions_pattern.sub(expand_match, doc)
            docs.append(re.sub("'", "", expanded_text))
        return docs

    def standardize_age_gender(self, text = None):
        """Formats r/relationshipadvice age/gender identifier convention.

        A convention in r/relationshipsadvice is referring to oneself and
        one's partner with the format "(agegender)", e.g., 28m. Processing
        these causes issues, with the age and gender portion usually split
        and nonsensical. This function catches all strings containing 2
        numeric characters and 1 letter, splits them on the second numeric
        character, and combines them with an underscore.

        Args:
            text (list): A list of strings

        Returns:
            The same list of strings with formatted age/gender characters.
        """
        if text == None:
            text = self.text
        docs = []
        for doc in text:
            matches = re.findall(r"\b[0-9]{2}[A-Za-z]\b", doc)
            for idx,match in enumerate(matches):
                doc = doc.replace(matches[idx], match[:2] + '_' + match[-1])
            docs.append(doc)
        return docs

    def final_pass(self, text = None):
        """Removes special and new-line characters.

        Args:
            text (list): A list of strings.

        Returns:
            The same list of strings free of special characters.
        """
        if text == None:
            text = self.text
        docs = []
        for doc in text:
            final = re.sub(r'[\"\'\[\]\(\)\:?\\+]', ' ', doc).replace('\n', ' ')
            docs.append(final)
        return docs

    def main_clean(self, text = None):
        """Main cleaning function.

        This runs all other methods in the TextCleaner class. For
        more information, see doc strings for `.touchup()`,
        `.standardize_age_gender()`, `.expand_contractions()`, and
        `.final_pass()`.

        Args:
            text (list): A list of strings.

        Returns:
            The same list of strings cleaned with the above methods.
        """
        if text == None:
            text = self.text
        first_pass = self.touchup(text)
        age_gender_formatting = self.standardize_age_gender(first_pass)
        expanding_contractions = self.expand_contractions(age_gender_formatting)
        final_pass = self.final_pass(expanding_contractions)
        return final_pass

class NlpPipe(TextCleaner):
    """Processing pipeline with tokenization and lemmatization methods.

    Attributes:
        text (list): A list of strings.
        tokenizer (obj): NLP tokenizer with Spacy as default.
    """
    def __init__(self, text = None, tokenizer = spacy.load('en_core_web_sm')):
        """Inits class with text and tokenizer"""
        self.text = text
        self.tokenizer = tokenizer

    def clean_text(self,
                   text = None
                  ):
        """Main text cleaning function.

        For clarification on the methods of cleaning, refer to the
        TextCleaner class docs.

        Args:
            text (list): A list of strings.

        Returns:
            The tokenized list of strings.
        """
        if text == None:
            text = self.text
        return super().main_clean(text)

    def tokenize(self,
                 text = None,
                 clean = True,
                 remove_punct = True,
                 remove_stop = True
                ):

        """Tokenizes corpus documents.

        Args:
            text (list): A list of strings.
            clean (bool): If true, calls the `.clean_text()` method. Default is true.
            remove_punct (bool): If true, removes punctuation. Default is true.
            remove_stop (bool): If true, removes stopwords. Default is true.

        Returns:
            List of documents.
        """
        if text == None:
            text = self.text
        if clean == True:
            text = self.clean_text(text)
        docs = []
        for doc in text:
            tokens = self.tokenizer(doc)
            if remove_punct:
                tokens = [tok for tok in tokens if tok.is_punct == False and str(tok) not in '                 ']
            if remove_stop:
                tokens = [tok for tok in tokens if tok.is_stop == False and str(tok) not in '                 ']
            docs.append(tokens)
        return docs

    def lemmatize(self,
                  text=None,
                  tokenize=True
                 ):
        """Lemmatizes list of documents.

        Args:
            text (list): A list of strings.
            tokenize (bool): If true, calls `.tokenize()` method with defaults set.
              Default is true.

        Returns:
            Corpus of lemmatized documents.
        """
        if text == None:
            text = self.tokenize(self.text)
        else:
            text = self.tokenize(text)
        docs = []
        for doc in text:
            docs.append([tok.lemma_ for tok in doc])
        return docs

def concat_processed(corpus,
                     df,
                     cols
                    ):
    """Concatenates corpus to subset of original dataframe.

    Args:
        corpus (list): List of documents.
        df (dataframe): Pandas dataframe object.
        cols (list): List of columns in df argument.

    Returns:
        Dataframe with the corpus as a new series.
    """
    corpus_series = pd.Series(corpus, name = 'processed')
    main_df = df[cols]

    concatted = pd.concat([main_df, corpus_series], axis = 1)

    return concatted

def data_save_df(data,
                 user,
                 pw,
                 ip,
                 port
                ):
    """Saves dataframe to a postgresql database.

    Args:
        data (dataframe): Pandas dataframe.
        user, pw, ip, port (str): Database credentials, ip, and port.

    Returns:
        None.
    """
    engine = create_engine(f'postgresql+psycopg2://{user}:{pw}@{ip}:{port}/')
    engine.connect()
    pd.to_sql(data, engine)
    return


