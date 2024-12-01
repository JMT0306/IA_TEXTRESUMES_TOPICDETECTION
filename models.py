from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary


def create_bow(preprocessed_docs):
    """
    Cria uma matriz Bag-of-Words a partir dos documentos pré-processados.
    """
    vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
    bow_matrix = vectorizer.fit_transform(preprocessed_docs)
    return bow_matrix, vectorizer


def apply_kmeans(bow_matrix, num_clusters):
    """
    Aplica o algoritmo k-means aos vetores BoW.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(bow_matrix)
    return kmeans


def apply_lda(preprocessed_docs, num_topics):
    """
    Aplica o modelo LDA aos documentos pré-processados.
    """
    dictionary = Dictionary(preprocessed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]

    # Ajustar parâmetros para maior precisão
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=42)

    return lda_model, corpus, dictionary
