import argparse
import logging
from bertopic import BERTopic
from tunebertopic.tuning.bayesian_search import BayesianOptimizationSearch
from tunebertopic.data import load_data, load_parameter_file


logger = logging.getLogger("tunebertopic")
console = logging.StreamHandler()
logger.addHandler(console)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("parameter_file", type=str)
    argparser.add_argument("--data-path", type=str, default=None)
    argparser.add_argument("--categories", type=str, nargs="*", default=None)
    argparser.add_argument("--max-num-samples", type=int, default=1000)
    argparser.add_argument("--strategy", type=str, default="bayesian")
    argparser.add_argument("--metric", type=str, default="coherence")
    argparser.add_argument("--llm", type=str, default="openai")
    argparser.add_argument("--log-level", type=str, default="INFO")
    args = argparser.parse_args()

    logger.setLevel(args.log_level)

    params = load_parameter_file(args.parameter_file)
    logger.info(f"Loaded parameters: {params}")

    if args.data_path:
        logger.info(f"Loading data from {args.data_path}")
        documents = load_data(data_path=args.data_path)
    else:
        logger.info(
            f"Loading sample data, with categories: {args.categories} and max_num_samples: {args.max_num_samples}"
        )
        documents = load_data(
            use_sample=True,
            max_num_samples=args.max_num_samples,
            categories=args.categories,
        )
    documents = [doc for doc in documents if len(doc.strip()) > 0]

    if args.strategy == "bayesian":
        logger.info("Using Bayesian Optimization search strategy")
        search = BayesianOptimizationSearch(params)

    logger.info("Starting search...")
    best_params, best_score = search.search(documents, BERTopic, args.metric, args.llm)

    # Print the best parameters and score
    logger.info(f"Best Parameters: {best_params}")
    logger.info(f"Best Score: {best_score}")
