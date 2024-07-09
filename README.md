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

# start the runs
poetry run python main.py
```

