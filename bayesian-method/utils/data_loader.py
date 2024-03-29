import numpy as np
import pandas as pd
import xlrd


def load_person_preferences(file_path):
    matrix = pd.read_csv(file_path).to_numpy()
    titles = pd.read_csv(file_path, nrows=0).columns.tolist()
    return titles, matrix


def load_news_dataset(file_path):
    matrix = pd.read_excel(file_path).to_numpy()
    matrix = matrix[:, 0:4]
    # titles = pd.read_excel(file_path, nrows=0).columns.tolist()
    # titles = titles[0:4]
    titles = ['fecha', 'titular', 'fuente', 'categoria']
    return titles, matrix


def transform_news_dataset(matrix, news_categories, output_file=None):
    rows = matrix.shape[0]
    columns = matrix.shape[1]
    processed_matrix = matrix
    count = 0
    for row in range(0, rows):
        found = False
        for category in news_categories:
            if matrix[row][columns - 1] == category:
                found = True
        if not found:
            processed_matrix = np.delete(processed_matrix, row - count, axis=0)
            count += 1

    if output_file is not None:
        np.savetxt(output_file, np.asarray(processed_matrix, dtype=object), delimiter="\t", fmt='%s')
    return processed_matrix


def load_transformed_news_dataset(file_path):
    matrix = pd.read_csv(file_path, delimiter="\t").to_numpy()
    titles = ['fecha', 'titular', 'fuente', 'categoria']
    return titles, matrix


def load_binary_dataset(file_path):
    matrix = pd.read_csv(file_path).to_numpy()
    titles = pd.read_csv(file_path, nrows=0).columns.tolist()
    return titles, matrix


def discretize_binary_dataset(titles, matrix):
    # GRE ∈ {GRE ≥500, GRE < 500} y GPA ∈ {GPA ≥ 3, GPA < 3}.
    rows = matrix.shape[0]
    columns = matrix.shape[1]
    discretized_matrix = np.zeros((rows, columns))
    for row in range(0, rows):
        for column in range(0, columns):
            if titles[column] == "gre":
                discretized_matrix[row][column] = 0 if matrix[row][column] < 500 else 1
            elif titles[column] == "gpa":
                discretized_matrix[row][column] = 0 if matrix[row][column] < 3 else 1
            else:
                discretized_matrix[row][column] = matrix[row][column]
    return discretized_matrix

