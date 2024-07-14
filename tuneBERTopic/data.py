from sklearn.datasets import fetch_20newsgroups
import yaml


def load_parameter_file(parameter_file):
    return yaml.load(open(parameter_file, "r"), Loader=yaml.FullLoader)

def load_data(
    data_path=None, use_sample=False, max_num_samples=1000, categories=None
):
    """Gets the data from the provided source.

    Either loads the sample dataset or reads the data from a file.

    Args:
        data_path (Pathlike, optional): The path of the data file. Defaults to None.
        use_sample (bool, optional): Use sample data instead. Defaults to False.
        max_num_samples (int, optional): The max number of samples. Defaults to 1000.
        categories (_type_, optional): The categories of newsgroups sample dataset. Defaults to None.

    Raises:
        ValueError: If no data file is provided and not using the sample dataset.

    Returns:
        List[str]: The documents to be modeled.
    """
    if use_sample:
        newsgroups_subset = fetch_20newsgroups(
            subset="all", categories=categories, remove=("headers", "footers", "quotes")
        )
        documents = newsgroups_subset.data[:max_num_samples]
        return documents
    elif data_path is not None:
        return open(data_path, "r").readlines()
    else:
        raise ValueError("Data file must be provided unless using the sample dataset.")
