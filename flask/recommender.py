from sqlalchemy import create_engine
import load_models as load
import en_core_web_sm
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import preprocessing as prep
from sklearn.metrics import pairwise_distances




lda, bert, encoder, pretrained_vecs = load.load_models()

def clean_input(text):
    """Preprocesses text for transformation.

    This function formats text to be transformed for tf-idf.
    Text is lower-cased, line break characters are removed,
    and any whitespace is removed. After tokenization, text
    is lemmatized with no stop words or punctuation.

    Args:
        text (str): User entered form text.

    Returns:
        A list of lemmatized words.
    """
    lemmas = prep.NlpPipe([text]).lemmatize()
    return lemmas

def lda_vec(lda, text):
    """

    """
    bow_input = lda.id2word.doc2bow(text[0])
    topics_probs = lda.get_document_topics(bow_input)

    lda_preds = np.zeros((1, 20)) #matrix of topics + probabiliies
    for topic, prob in topics_probs:
        lda_preds[0, topic] = prob

    return lda_preds

def bert_vec(bert, text):
    """

    """
    encoded_preds = np.array(bert.encode(' '.join(word for word in text[0])))
    return encoded_preds

def autoencode_vecs(lda_preds, encoded_preds, encoder):
    """

    """
    bert_lda_combo = np.c_[lda_preds * 15, encoded_preds]
    final_encoding = encoder.predict(bert_lda_combo)
    return final_encoding

def recommend(text, lda=lda, bert=bert, encoder=encoder, pretrained_vecs=pretrained_vecs):
    try:
        engine = create_engine('postgresql+psycopg2://postgres:bryanross1@34.94.44.13:5432/')
        engine.connect()
        processed_text = clean_input(text)
        lda_input = lda_vec(lda, processed_text)
        bert_input = bert_vec(bert, processed_text)
        encoded_vecs = autoencode_vecs(lda_input, bert_input, encoder)

        dists = np.zeros(len(pretrained_vecs))
        for i in range(len(pretrained_vecs)):
            dists[i] = pairwise_distances(encoded_vecs.reshape(1,-1), pretrained_vecs[i].reshape(1,-1), metric = 'cosine')

        top_five = dists.argsort()[:5]
        indexes_to_query = tuple(top_five.tolist())

        query = f'SELECT index,title,url FROM data WHERE index IN {indexes_to_query}'

        df = pd.read_sql(query, engine)

        recommended_posts = []
        for idx,post in df.iterrows():
            info = {
                'title': post['title'],
                'url': post['url']
            }
            recommended_posts.append(info)

        return recommended_posts
    except TypeError:
        return

