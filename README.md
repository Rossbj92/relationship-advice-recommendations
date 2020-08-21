My project objective is to identify general relationship problems that people have and recommend resources based on an individual's relationship issue(s).
If time allows, I'd like to also generate a model to give advice to users.

# Reddit Relationship Recommendations

Using approximately 200,000 posts from r/relationshipadvice, I built a content-based recommendation system.

Flask app code is also included using the final model to make recommendations for user-inputted text.

## Replication Instructions

1. Clone repo
2. Install package requirements in ```requirements.txt```.
3. Process data with [preprocessing](notebooks/preprocessing.ipynb) notebook.
4. Extract vectors with the [feature engineering](notebooks/feature-engineering.ipynb) notebook.
5. Experiment with recommendation methods in the [recommendations](notebooks/recommendations.ipynb) notebook.

*(Optional) Run [webscraper](notebooks/scraping.ipynb) notebook*

A test sample of data is provided to run the notebooks. Should you want to gather new data, simply refer to the ```scraping``` notebook.

## Directory Descriptions

```Data```
- ```raw``` - csv test sample of raw data
- ```interim``` - cleaned data used in the feature engineering notebook.

```Notebooks```
- [scraping](notebooks/scraping.ipynb) - used for querying the Reddit Pushshift API
- [preprocessing](notebooks/preprocessing.ipynb) - text preprocessing
- [feature engineering](notebooks/feature-engineering.ipynb) - various text vectorization methods
- [recommendations](notebooks/recommendations.ipynb) - tested recommendation methods

```Models```
- Necessary models fit on sample data to use the ```recommendations``` notebook.

```Reports```
- ```presentation``` - pdf and ppt format of final project presentation
- ```figures``` - images from original output used as references in notebooks

```src```
- Contains code for functions used in the notebooks. Each directory corresponds to one notebook, as well as a separate ```visualizations``` directory

```flask```
- Contains templates, static files, and python scripts used to build the flask app.

## Conclusions

5 methods for recommendations were tested:
1. LDA
2. Doc2Vec
3. BERT sentence embeddings
4. LDA-Doc2Vec embeddings
5. LDA-BERT embeddings

Of those, the LDA-BERT embeddings provided the most sensible recommendations. Future work will include optimizing the pipeline in processing new predictions, as well as making the recommendation system more robust to shorter length searches.

## Methods

- LDA
- BERT
- Doc2Vec
- Autoencoding
- Flask
- Content-based recommendations
- Google cloud services (SQL, app engine)
- AWS S3
