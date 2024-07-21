import mlflow
import os
from bertopic import BERTopic
from tunebertopic.metrics import evaluation_metrics
from tunebertopic.llm.models import llms



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

    def evaluate_model(self, topic_model: BERTopic, documents, metric="coherence", llm="openai"):
        evaluator = evaluation_metrics.get(metric)
        if evaluator:
            if metric in ["bleu", "rouge"]:
                llm = llms.get(llm)()
                return evaluator(topic_model, documents, llm_model=llm)
            else:
                return evaluator(topic_model, documents)
        else:
            raise ValueError(f"Unsupported evaluation metric: {metric}")
        
