import numpy as np


def equal_comparison(expected_value, current_value):
    return expected_value == current_value


def bayesian_naive_classifier(training_matrix, matrix_to_classify, columns_true_values, comparison_function=equal_comparison):
    rows = training_matrix.shape[0]
    columns = training_matrix.shape[1] - 1
    sorted_matrix = training_matrix[training_matrix[:, columns].argsort()]
    current_class = sorted_matrix[0, columns]
    start_row = 0
    probabilities_for_classes = {}
    probabilities_of_classes = {}
    class_quantity = get_class_quantity(sorted_matrix)
    for i in range(1, rows):
        if sorted_matrix[i][columns] != current_class:
            data_for_class = sorted_matrix[start_row:i, 0:columns]
            probabilities_for_classes[current_class] = calculate_probabilities_for_class(data_for_class, class_quantity,
                                                                                         columns_true_values,
                                                                                         comparison_function)
            probabilities_of_classes[current_class] = (i - start_row) / rows
            start_row = i
            current_class = sorted_matrix[i][columns]
    data_for_class = sorted_matrix[start_row:rows, 0:columns]
    probabilities_for_classes[current_class] = calculate_probabilities_for_class(data_for_class, class_quantity,
                                                                                 columns_true_values,
                                                                                 comparison_function)
    probabilities_of_classes[current_class] = (rows - start_row) / rows
    return classify(matrix_to_classify, probabilities_for_classes, probabilities_of_classes, columns_true_values,
                    comparison_function)


def calculate_probabilities_for_class(data_for_class, class_quantity, columns_true_values, comparison_function):
    columns = data_for_class.shape[1]
    probabilities = np.zeros(columns)
    for i in range(0, columns):
        column_true_value = columns_true_values[i]
        probabilities[i] = calculate_probability_for_attribute(data_for_class, i, class_quantity, column_true_value,
                                                               comparison_function)
    return probabilities


def calculate_probability_for_attribute(data_for_class, column, class_quantity, true_value, comparison_function):
    rows = data_for_class.shape[0]
    count = 0
    for i in range(0, rows):
        current_value = data_for_class[i][column]
        if apply_comparison_function(true_value, current_value, comparison_function):
            count += 1
    return (count + 1) / (rows + class_quantity)


def get_class_quantity(sorted_matrix):
    rows = sorted_matrix.shape[0]
    columns = sorted_matrix.shape[1]
    class_quantity = 1
    current_class = sorted_matrix[0][columns - 1]
    for i in range(1, rows):
        if sorted_matrix[i][columns - 1] != current_class:
            current_class = sorted_matrix[i][columns - 1]
            class_quantity += 1
    return class_quantity


def classify(matrix_to_classify, probability_for_classes, probability_of_classes, columns_true_values,
             comparison_function):
    rows = len(matrix_to_classify)
    class_predictions = {}
    class_inferences = []
    for i in range(0, rows):
        for current_class in probability_of_classes.keys():
            class_predictions[current_class] = get_class_prediction(matrix_to_classify[i], current_class,
                                                                    probability_for_classes, probability_of_classes,
                                                                    columns_true_values, comparison_function)
        max_probability = 0
        class_inference = None
        for current_class in class_predictions.keys():
            if class_predictions[current_class] >= max_probability:
                class_inference = current_class
                max_probability = class_predictions[current_class]
        class_inferences.append(class_inference)
    return class_inferences


def get_class_prediction(input_data, current_class, probability_for_classes, probability_of_class, columns_true_values,
                         comparison_function):
    cols = len(input_data)
    current_probability = 1
    for i in range(0, cols):
        if apply_comparison_function(input_data[i], columns_true_values[i], comparison_function):
            variable_probability = probability_for_classes[current_class][i]
            current_probability *= variable_probability
        else:
            variable_probability = 1 - probability_for_classes[current_class][i]
            current_probability *= variable_probability
    return current_probability * probability_of_class[current_class]


def apply_comparison_function(expected_value, current_value, function):
    return function(expected_value, current_value)

