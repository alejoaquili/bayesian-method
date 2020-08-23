import numpy as np


def get_value_of_cell(inferences, expected_values, actual_class, current_classification):
    count = 0
    for i in range(0, len(expected_values)):
        if expected_values[i] == actual_class:
            if inferences[i] == current_classification:
                count += 1
    return count


def calculate_confusion_matrix(inferences, expected_values, classes_dict):
    indexes_of_classes = np.empty(len(classes_dict), dtype=object)
    confusion_matrix = np.empty((len(classes_dict) + 1, len(classes_dict) + 1), dtype=object)
    classes = []
    for current_class in classes_dict:
        classes.append(current_class)
    for i in range(0, len(classes)):
        indexes_of_classes[i] = classes[i]
        confusion_matrix[i][0] = classes[i]
        confusion_matrix[0][i] = classes[i]
    for row in range(0, len(classes)):
        for col in range(0, len(classes)):
            confusion_matrix[row + 1][col + 1] = get_value_of_cell(inferences, expected_values, indexes_of_classes[row],
                                                           indexes_of_classes[col])
    return confusion_matrix
