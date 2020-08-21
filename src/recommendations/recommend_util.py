import preprocessing.preprocessing_util as prep
import pandas as pd
import numpy as np
import gensim.models

import keras
from keras.layers import Input, Dense
from keras.models import Model
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

import pickle
import warnings
warnings.filterwarnings('ignore')

def autoencoder_load(model, weights):
    """Loads pretrained autoencoder.

    Params:
    model, weights (str): path to files.

    Returns:
    Loaded Keras model.
    """
    file = open(model, 'r')
    loaded_model = file.read()
    file.close()

    encoder = keras.models.model_from_json(loaded_model)
    encoder.load_weights(weights)

    return encoder

def load():
    """Loads all models/vectors from modeling notebook."""
    df = pd.read_csv('data/interim/processed_sample.csv')

    lda_bert_model = autoencoder_load('models/lda_bert/lda_bert_autoencoder.json', 'models/lda_bert/lda_bert_model.h5')
    with open('models/lda_bert/lda_bert_vectors.pkl', 'rb') as f:
        bert_lda_vectors = pickle.load(f)

    lda_d2v_model = autoencoder_load('models/lda_d2v/lda_d2v_autoencoder.json', 'models/lda_d2v/lda_d2v_model.h5')
    with open('models/lda_d2v/lda_d2v_vectors.pkl', 'rb') as f:
        lda_d2v_vectors = pickle.load(f)

    bert = SentenceTransformer('bert-base-nli-max-tokens')
    with open('models/bert/bert_docvecs.pkl', 'rb') as f:
        bert_vectors = pickle.load(f)

    lda = gensim.models.LdaModel.load('models/lda/lda_model')
    with open('models/lda/lda_vecs.pkl', 'rb') as f:
        lda_vectors = pickle.load(f)

    d2v = gensim.models.Doc2Vec.load('models/d2v/d2v_model')
    doc_vectors = np.array([d2v.docvecs[f'doc_{num}'] for num in range(df.shape[0])])

    return df, \
           lda_bert_model, bert_lda_vectors, \
           lda_d2v_model, lda_d2v_vectors, \
           bert, bert_vectors, \
           lda, lda_vectors, \
           d2v, doc_vectors

class Recommender:
    """This class makes post recommendations based on user input.

    While the final model only uses the lda-bert model, all are in
    here for any future experimentation. Recommendations are able
    to be made using 5 procedures: LDA alone, Doc2Vec alone, BERT
    alone, LDA-Doc2Vec, and LDA-BERT.

    Attributes:
        text (str): Text to receive recommendations on.
        df (dataframe): Pandas dataframe containing any post
          information one would like (e.g., titles).
    """
    """Inits class with text and df"""
    def __init__(self, text, df):
        self.text = text
        self.df = df
        self.predicted_vec = None
        self.rec_idxes = None


    def process_text(self, processor):
        """Preprocesses text input.

        Args:
            Processor (obj): Preprocessing pipeline.

        Returns:
            Self. The processed_text attribute stores the
            processed text.
        """
        self.processed_text = processor[0]
        return self

    def predicted_topics(self,
                         lda,
                         text=None,
                         num_topics = 20
                        ):
        """Retrieves topic probabilities for text.

        Args:
            Lda (obj): Gensim LDA model.
            Text (str): String to get probabilities for.
            Num_topics (int): Number of topics in model.

        Returns:
            Array of topic probabilites.
        """
        if not text and self.processed_text:
            text = self.processed_text
        elif not text:
            text = self.text
        bow_input = lda.id2word.doc2bow(text)
        topics_probs = lda.get_document_topics(bow_input)
        topic_pred_vector = np.zeros((1, num_topics)) #vector of topics + probabilities
        for topic, prob in topics_probs:
            topic_pred_vector[0, topic] = prob
        return topic_pred_vector

    def compute_dists(self,
                      predicted_vector,
                      trained_vectors,
                      num_recs = 5,
                      dist_metric = 'cosine'
                     ):
        """Computes pairwise distances between vectors.

        Args:
            Predicted_vector (array): Vector to compare against all others.
            Trained_vectors (list): List of arrays to iterate through for comparisons.
            Num_recs (int): Number of recommendations to return, which are displayed
              as indices in descending order of similarity -- i.e., "5" will return the
              5 most similar vectors' indices.
            Dist_metric (str): Distance metric to use. Default is cosine.

        Returns:
            Array indices the same length as num_recs.
        """
        dists = np.zeros(len(trained_vectors))
        for i in range(len(trained_vectors)): #I hate this for-loop so much...
            dists[i] = pairwise_distances(predicted_vector, trained_vectors[i].reshape(1,-1), metric = dist_metric)
        recommendation_idxes = dists.argsort()[:num_recs]
        return recommendation_idxes

    def print_info(self, idxes, df=None):
        """Prints recommendation information.

        Args:
            Idxes (array): Array containing indices to index in dataframe.
            Df (dataframe): Pandas dataframe containing information for posts.

        Returns:
            Post title and URL for each index represented in idxes.
        """
        if not df:
            df = self.df
        for idx in idxes:
            print(f'Post: {df.loc[idx, "title"]}')
            print(f'URL: {df.loc[idx, "url"]}')
            print('\n')

    def lda_preds(self, lda, lda_vectors):
        """Retrieves recommendations based on LDA model.

        Args:
            Lda (obj): Gensim LDA model.
            Lda_vectors (list): List of arrays representing already
              trained document topic probabilities.

        Returns:
            Post URL and title for top 5 recommended posts.
        """
        if self.processed_text:
            text = self.processed_text
        else:
            text = self.text
        predicted_topics = self.predicted_topics(lda, text)
        recommendation_indices = self.compute_dists(predicted_topics, lda_vectors)
        return self.print_info(recommendation_indices)

    def d2v_preds(self, d2v, d2v_vectors):
        """Retrieves recommendations based on Doc2Vec model.

        Args:
            D2v (obj): Gensim Doc2Vec model.
            D2v_vectors (list): List of arrays representing already
              trained document vectors.

        Returns:
            Post URL and title for top 5 recommended posts.
        """
        if self.processed_text:
            text = self.processed_text
        else:
            text = self.text
        predicted_vec = d2v.infer_vector(text).reshape(1,-1)
        recommendation_indices = self.compute_dists(predicted_vec, d2v_vectors)
        return self.print_info(recommendation_indices)

    def bert_preds(self, bert, bert_vectors):
        """Retrieves recommendations based on BERT embeddings.

        Args:
            Bert (obj): Sentence-transformer object.
            Bert_vectors (list): List of arrays representing already
              trained BERT embeddings.

        Returns:
            Post URL and title for top 5 recommended posts.
        """
        if self.processed_text:
            text = self.processed_text
        else:
            text = self.text
        predicted_vec = np.array(bert.encode(' '.join(word for word in text)))
        recommendation_indices = self.compute_dists(predicted_vec, bert_vectors)
        return self.print_info(recommendation_indices)

    def lda_d2v_preds(self,
                      lda_model,
                      d2v_model,
                      encoder,
                      vectors
                     ):
        """Retrieves recommendations based on LDA-D2V embeddings.

        Args:
            Lda_model (obj): Gensim LDA model.
            D2v_model (obj): Gensim Doc2Vec model.
            Encoder (obj): Previously trained Keras autoencoder model.
              Vector input dimensions are equal to the LDA+D2V vector sizes.
            Vectors (list): List of arrays representing already
              encoded vectors.

        Returns:
            Post URL and title for top 5 recommended posts.
        """
        if self.processed_text:
            text = self.processed_text
        else:
            text = self.text
        predicted_topics = self.predicted_topics(lda_model, text).reshape(1,-1)
        predicted_d2v = d2v_model.infer_vector(text).reshape(1,-1)
        d2v_lda_combo = np.concatenate((predicted_topics * 15, predicted_d2v), axis = 1)
        encoded_vec = encoder.predict(d2v_lda_combo)
        recommendation_indices = self.compute_dists(encoded_vec, vectors)
        return self.print_info(recommendation_indices)

    def lda_bert_preds(self,
                       lda_model,
                       bert,
                       encoder,
                       vectors,
                       num_recs = 5,
                       save_vec = False,
                       save_idxes = False,
                       print_recs = True
                      ):
        """Retrieves recommendations based on LDA-BERT embeddings.

        Args:
            Lda_model (obj): Gensim LDA model.
            Bert (obj): Sentence-transformer object.
            Encoder (obj): Previously trained Keras autoencoder model.
              Vector input dimensions are equal to the LDA+D2V vector sizes.
            Vectors (list): List of arrays representing already
              encoded vectors.
            Num_recs (int): Number of recommendations to return.
            Save_vec, save_idxes (bool): Whether to store the text's predicted
              vector and recommendation indices, and primarily done so here
              for visualization purposes. Defaults are False.
            Print_recs (bool): Whether to print recommendation information.
              Default is true.

        Returns:
            Either self with predicted vector and recommendation indices or
            recommendation information.
        """
        if self.processed_text:
            text = self.processed_text
        else:
            text = self.text
        predicted_topics = self.predicted_topics(lda_model, text).reshape(1,-1)
        bert_vec = np.array(bert.encode(' '.join(word for word in text))).reshape(1, -1)
        lda_bert_combo = np.concatenate((predicted_topics * 15, bert_vec), axis = 1)
        encoded_vec = encoder.predict(lda_bert_combo)
        if save_vec:
            self.predicted_vec = encoded_vec
        if num_recs == 'all':
            recommendation_indices = self.compute_dists(encoded_vec, vectors, len(vectors))
        else:
            recommendation_indices = self.compute_dists(encoded_vec, vectors, num_recs)
        if save_idxes:
            self.rec_idxes = recommendation_indices
        if print_recs:
            return self.print_info(recommendation_indices)
        return self
