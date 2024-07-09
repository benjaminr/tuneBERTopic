
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic
from tuneBERTopic.bayesian_search import BayesianOptimizationSearch


if __name__ == '__main__':
    param_grid = {
        'vectorizer__min_df': [1, 2, 5],
        'vectorizer__max_df': [0.9, 0.95, 1.0],
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'umap__n_neighbors': [5, 10, 15],
        'umap__n_components': [5, 10, 15],
        'umap__metric': ['euclidean', 'cosine'],
        'hdbscan__min_cluster_size': [5, 10, 15],
        'hdbscan__min_samples': [1, 5, 10],
    }

    # Fetch sample dataset
    categories = ['sci.space']
    newsgroups_subset = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
    documents = newsgroups_subset.data[:1000]

    # Select and run the search strategy
    strategy = 'bayesian'
    search = BayesianOptimizationSearch(param_grid)
    best_params, best_score = search.search(documents, BERTopic)

    # Print the best parameters and score
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)
