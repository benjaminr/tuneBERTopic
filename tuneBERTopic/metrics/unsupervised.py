import mlflow
from bertopic import BERTopic
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from sklearn.metrics import silhouette_score


def silhouette(topic_model: BERTopic, docs):
    topics, _ = topic_model.fit_transform(docs)
    topics_ = topic_model.get_topics()
    topics_ = {k: [word for word, _ in v] for k, v in topics_.items()}
    topic_words = [topics_[t] for t in topics_ if t != -1]
    umap_model = topic_model.umap_model
    if hasattr(umap_model, 'embedding_'):
        umap_embeddings = umap_model.embedding_
    else:
        original_embeddings = topic_model.transform(docs)[1]
        umap_embeddings = umap_model.fit_transform(original_embeddings)

    topics = np.array(topics)

    if topic_words:
        silhouette = silhouette_score(umap_embeddings, topics)
        mlflow.log_param("silhouette", silhouette)
        mlflow.log_param("topics", topic_words)
        return 1-silhouette
    return -1

def coherence(topic_model: BERTopic, documents):
    topics = topic_model.get_topics()
    topics = {k: [word for word, _ in v] for k, v in topics.items()}
    # Create a dictionary and corpus needed for the coherence model
    texts = [doc.split() for doc in documents]
    dictionary = Dictionary(texts)
    # Filter out the -1 topic which represents outliers
    topic_words = [topics[t] for t in topics if t != -1]
    # Remove empty topics
    topic_words = [
        topic for topic in topic_words if not all(v == "" for v in topic)
    ]
    topic_words = [topic for topic in topic_words if all(word in dictionary.token2id for word in topic)]
    if topic_words:
        # log to mlflow
        mlflow.log_param("num_topics", len(topic_words))
        mlflow.log_param("num_documents", len(documents))
        mlflow.log_param("num_unique_words", len(dictionary))
        mlflow.log_param("topics", topic_words)
        # return c_v coherence score
        coherence_model = CoherenceModel(
            topics=topic_words, texts=texts, dictionary=dictionary, coherence="c_v"
        )
        coherence_score = coherence_model.get_coherence()
        mlflow.log_metric(f"coherence_score", coherence_score)
        return coherence_score
    return -1