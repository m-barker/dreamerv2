import pandas as pd  # type: ignore


def csv_to_json(csv_path: str, json_path: str) -> None:
    """Converts a csv file to a json file,
    where the json file contains a list where each element in
    the list is a dictionary with the keys being the column names
    of the csv file.

    Args:
        csv_path (str): path to input csv file
        json_path (str): output path to json file
    """
    df = pd.read_csv(csv_path)
    df.to_json(json_path, orient="records")
