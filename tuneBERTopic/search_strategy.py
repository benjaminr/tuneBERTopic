import mlflow
import os
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary


class SearchStrategy:
    """The parent class for the search strategies.

    Defines the methods that the search strategies should implement.
    Defines the evaluate_model method that calculates the coherence score.

    Four coherence metrics are calculated and logged, c_v is used to optimise.
    """

    def __init__(self, param_grid):
        self.param_grid = param_grid
        mlflow.set_tracking_uri(
            uri=os.getenv("MLFLOW_TRACKING_URL", "http://127.0.0.1:8080")
        )

    def search(self, documents, model_class):
        raise NotImplementedError("Search method not implemented")

    def evaluate_model(self, topic_model: BERTopic, documents):
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
