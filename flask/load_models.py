from keras import models
import gensim.models
import pickle
import boto3
from io import BytesIO
from sentence_transformers import SentenceTransformer


def load_models():
    """Loads models from s3 bucket."""
    s3 = boto3.resource('s3',
                  aws_access_key_id='',
                  aws_secret_access_key= '')
    with BytesIO() as lda_model:
        s3.Bucket('bert-sentence-model').download_fileobj('lda_model.pkl', lda_model)
        lda_model.seek(0)
        lda = pickle.load(lda_model)

    with BytesIO() as bert:
        s3.Bucket('bert-sentence-model').download_fileobj('sentence_trans.pkl', bert)
        bert.seek(0)
        bert = pickle.load(bert)

    with BytesIO() as vectors:
        s3.Bucket('bert-sentence-model').download_fileobj('encoded_bert_lda.pkl', vectors)
        vectors.seek(0)
        pretrained_vecs = pickle.load(vectors)


    session = boto3.session.Session(region_name='us-east-2')
    s3client = session.client('s3',
                              aws_access_key_id='',
                              aws_secret_access_key= '')
    s3client.download_file('bert-sentence-model',
                         'autoencoder.json',
                         'autoencoder.json')
    model = open('autoencoder.json', 'r')
    loaded = model.read()
    encoder = models.model_from_json(loaded)
    s3client.download_file('bert-sentence-model',
                           'model.h5', 'model.h5')
    encoder.load_weights('model.h5')



    return lda, bert, encoder, pretrained_vecs
