import pandas as pd


def load_person_preferences(file_path):
    matrix = pd.read_csv(file_path).to_numpy()
    titles = pd.read_csv(file_path, nrows=0).columns.tolist()
    return titles, matrix
