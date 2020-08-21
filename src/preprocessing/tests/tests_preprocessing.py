import sys
sys.path.append("..")
import preprocessing_util as clean
import pytest

class TestTextCleaner:
    @pytest.mark.parametrize("test_input,expected", [(["wEiRd"], ["weird"]), #lower-case
                                                     (["i've got alot\\nto say"], ["i've got alot to say"]), #remove double line breaks
                                                     ([" some whitespace  "], ["some whitespace"])]) #remove whitespace
    def test_touchup(self, test_input, expected):
        assert clean.TextCleaner(test_input).touchup() == expected

    @pytest.mark.parametrize("test_input, expected", [(["don't"], ['do not']),
                                                      (["haven't"], ['have not']),
                                                      (["could've"], ['could have'])])
    def test_contractions(self, test_input, expected):
        assert clean.TextCleaner(test_input).expand_contractions() == expected


    def test_standardize_age_gender(self):
        assert clean.TextCleaner(['26m']).standardize_age_gender() == ['26_m']

    def test_main_clean(self):
        df = clean.pd.read_csv('test_docs.csv')
        assert clean.TextCleaner(df['raw'].tolist()).main_clean() == df['clean'].tolist()

class TestNlpPipe:

    @pytest.fixture
    def tokenizer(self):
        return clean.spacy.load('en_core_web_sm')

    @pytest.fixture
    def data(self):
        return clean.pd.read_csv('test_docs.csv')

    @pytest.mark.parametrize("test_input,expected", [([".?!,;-"], [""]), #lower-case
                                                 (["the has have is my yours"], [""]), #remove double line breaks
                                                 ([".?!,;- the has have is my yours "], ["some whitespace"])]) #remove whitespace
    def test_clean(self, test_input, expected, data, tokenizer):
        assert clean.NlpPipe(data['raw'].tolist(), tokenizer = tokenizer).clean_text() == data['clean'].tolist()

    @pytest.mark.parametrize("test_tokens", [([". ? ! , ; - the has have is my yours"])])
    def test_tokenizer(self, test_tokens, tokenizer):
        assert clean.NlpPipe(text = test_tokens, tokenizer = tokenizer).tokenize(clean = False) == [[]]
        assert str(clean.NlpPipe(text = test_tokens, tokenizer = tokenizer).tokenize(clean = False, remove_punct = False)) == '[[., ?, !, ,, ;, -]]'
        assert str(clean.NlpPipe(text = test_tokens, tokenizer = tokenizer).tokenize(clean = False, remove_stop = False)) == str('[[the, has, have, is, my, yours]]')
        assert str(clean.NlpPipe(text = test_tokens, tokenizer = tokenizer).tokenize(clean = False, remove_punct = False, remove_stop = False)) == '[. ? ! , ; - the has have is my yours]'

    def test_lemmatize(self, tokenizer):
        assert clean.NlpPipe(text = ['ducks, going'], tokenizer = tokenizer).lemmatize() == [['duck', 'go']]

    def test_full_pipe(self, tokenizer, data):
        assert len(clean.NlpPipe(text = data['raw'].tolist(), tokenizer = tokenizer).lemmatize()) == data.shape[0]
