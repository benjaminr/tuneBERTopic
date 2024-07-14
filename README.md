# Tune BERTopic

This library provides optimisation search strategies for evaluating the best hyperparameters for BERTopic models.

It currently implements a Bayesian approach, minimising the inverse of the topic coherence.

There are plans to include additional search strategies and evaluation metrics in future versions.

## Installation

```
poetry install
```

## Run

```
# setup the mlflow tracking server
mlflow server --host 127.0.0.1 --port 8080

# Run against a document file (newline delimetered)
poetry run python main.py parameters.yaml --data-path /path/to/documents_file

# Example run against the 'sci.space' category with the example parameters file
poetry run python main.py parameters.yaml --categories sci.space

# Example run against the 'sci.space' category with the example parameters file and a max number of samples
poetry run python main.py parameters.yaml --categories sci.space --max-num=samples 100
```

