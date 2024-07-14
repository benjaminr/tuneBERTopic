import mlflow
from hyperopt import fmin, tpe, hp, Trials
from tuneBERTopic.search_strategy import SearchStrategy
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer


class BayesianOptimizationSearch(SearchStrategy):
    """Uses Bayesian Optimization to search for the best hyperparameters.

    This class inherits from the SearchStrategy class and implements the search
    method. It uses the hyperopt library to perform a Bayesian Optimization search.
    The search space is defined by the parameter grid passed to the class. The
    search method uses an objective function to evaluate the model with the given
    hyperparameters. The best hyperparameters and score are returned.

    The current evaluation metric is the inverse of the coherence score. It is
    the inverse because the Bayesian Optimization search is a minimization problem.

    Args:
        SearchStrategy (class): The parent class for the search strategies.
    """

    def search(self, documents, model_class):
        """Search for the best hyperparameters using Bayesian Optimization.

        Args:
            documents (List): The documents to be modeled.
            model_class (class): The model class to be used for the search.
        """

        def objective(params):
            """The objective function to be minimized.

            Args:
                params (dict): The hyperparameters to be evaluated.

            Returns:
                float: The inverse of the coherence score.
            """
            with mlflow.start_run():
                param_dict = {
                    key: value for key, value in zip(self.param_grid.keys(), params)
                }
                mlflow.log_params(param_dict, "params")
                embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                vectorizer = CountVectorizer(
                    min_df=param_dict['vectorizer__min_df'],
                    max_df=param_dict['vectorizer__max_df'],
                    ngram_range=param_dict["vectorizer__ngram_range"],
                    stop_words="english",
                )
                umap_model = UMAP(
                    n_neighbors=param_dict["umap__n_neighbors"],
                    n_components=param_dict["umap__n_components"],
                    metric=param_dict["umap__metric"],
                    random_state=42,
                )
                hdbscan_model = HDBSCAN(
                    min_cluster_size=param_dict["hdbscan__min_cluster_size"],
                    min_samples=param_dict["hdbscan__min_samples"],
                )
                model = model_class(
                    embedding_model=embedding_model,
                    vectorizer_model=vectorizer,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                )
                model.fit(documents)
                score = self.evaluate_model(model, documents)
                return 1 - score

        search_space = [hp.choice(key, value) for key, value in self.param_grid.items()]
        trials = Trials()
        best = fmin(
            objective, search_space, algo=tpe.suggest, max_evals=100, trials=trials
        )

        best_params = {
            key: self.param_grid[key][list(best.values())[idx]]
            for idx, key in enumerate(self.param_grid.keys())
        }
        best_score = 1 - trials.best_trial["result"]["loss"]
        return best_params, best_score
