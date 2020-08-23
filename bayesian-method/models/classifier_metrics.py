import numpy as np


class ClassifierMetrics:
    def __init__(self):
        self.classes = None
        self.confusion_matrix = None
        self.true_positives_rate = None
        self.false_positives_rate = None

    def calculate_confusion_matrix(self, inferences, expected_values, classes):
        indexes_of_classes = np.empty(len(classes), dtype=object)
        confusion_matrix = np.empty((len(classes) + 1, len(classes) + 1), dtype=object)
        confusion_matrix[0][0] = ""
        for i in range(0, len(classes)):
            indexes_of_classes[i] = classes[i]
            confusion_matrix[i + 1][0] = classes[i]
            confusion_matrix[0][i + 1] = classes[i]
        for row in range(0, len(classes)):
            for col in range(0, len(classes)):
                confusion_matrix[row + 1][col + 1] = self.get_value_of_cell(inferences, expected_values,
                                                                            indexes_of_classes[row],
                                                                            indexes_of_classes[col])
        return confusion_matrix

    @staticmethod
    def get_value_of_cell(inferences, expected_values, actual_class, current_classification):
        count = 0
        for i in range(0, len(expected_values)):
            if expected_values[i] == actual_class:
                if inferences[i] == current_classification:
                    count += 1
        return count

    def calculate_true_positives_rate(self):
        pass

    def calculate_false_positives_rate(self):
        pass

    def calculate_accuracy(self):
        pass

    def calculate_all_metrics(self, inferences, expected_values, classes_dict):
        classes = []
        for current_class in classes_dict:
            classes.append(current_class)
        self.classes = classes
        self.confusion_matrix = self.calculate_confusion_matrix(inferences, expected_values, classes)



