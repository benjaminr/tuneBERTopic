import argparse
import logging
from bertopic import BERTopic
from tuneBERTopic.bayesian_search import BayesianOptimizationSearch
from tuneBERTopic.data import load_data, load_parameter_file


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("parameter_file", type=str)
    argparser.add_argument("--data_path", type=str, default=None)
    argparser.add_argument("--categories", type=str, nargs="*", default=None)
    argparser.add_argument("--max_num_samples", type=int, default=1000)
    argparser.add_argument("--strategy", type=str, default="bayesian")
    argparser.add_argument("--log-level", type=str, default="INFO")
    args = argparser.parse_args()

    logging.basicConfig(level=args.log_level)

    params = load_parameter_file(args.parameter_file)
    logging.info(f"Loaded parameters: {params}")

    if args.data_path:
        logging.info(f"Loading data from {args.data_path}")
        documents = load_data(data_path=args.data_path)
    else:
        logging.info(
            f"Loading sample data, with categories: {args.categories} and max_num_samples: {args.max_num_samples}"
        )
        documents = load_data(
            use_sample=True,
            max_num_samples=args.max_num_samples,
            categories=args.categories,
        )

    if args.strategy == "bayesian":
        logging.info("Using Bayesian Optimization search strategy")
        search = BayesianOptimizationSearch(params)

    logging.info("Starting search...")
    best_params, best_score = search.search(documents, BERTopic)

    # Print the best parameters and score
    logging.info(f"Best Parameters: {best_params}")
    logging.info(f"Best Score: {best_score}")
