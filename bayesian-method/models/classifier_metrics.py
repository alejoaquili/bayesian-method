import numpy as np


class ClassifierMetrics:
    def __init__(self):
        self.classes = None
        self.confusion_matrix = None
        self.true_positives_rate = {}
        self.false_positives_rate = {}
        self.recall = {}
        self.accuracy = {}
        self.precision = {}
        self.f1_score = {}
        self.matthews_correlation_coefficient = {}
        self.estimated_classifier_error = None
        self.estimated_classifier_error_relative = None

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

    @staticmethod
    def calculate_true_positives_rate(current_class, confusion_matrix):
        # true positives = TP / TP + FN
        for row in range(0, len(confusion_matrix)):
            if confusion_matrix[0][row] == current_class:
                true_positives = confusion_matrix[row][row]
                false_negatives = 0
                for column in range(1, len(confusion_matrix[0])):
                    if column != row:
                        false_negatives += confusion_matrix[row][column]
        return true_positives / (true_positives + false_negatives)

    @staticmethod
    def calculate_false_positives_rate(current_class, confusion_matrix):
        # false positives = FP / FP + TN
        class_row = None
        for row in range(0, len(confusion_matrix)):
            if confusion_matrix[0][row] == current_class:
                class_row = row
        false_positives = np.sum(confusion_matrix[1:, class_row]) - confusion_matrix[class_row][class_row]
        true_negatives = np.sum(confusion_matrix[1:, 1:]) - np.sum(confusion_matrix[class_row, 1:]) - \
                         np.sum(confusion_matrix[1:, class_row]) + confusion_matrix[class_row][class_row]
        return false_positives / (false_positives + true_negatives)

    def calculate_recall(self, current_class, confusion_matrix):
        return self.calculate_true_positives_rate(current_class, confusion_matrix)

    @staticmethod
    def calculate_accuracy(current_class, confusion_matrix):
        # accuracy = (TP + TN) / (TP + TN + FP + FN)
        total = np.sum(confusion_matrix[1:, 1:])
        for row in range(0, len(confusion_matrix)):
            if confusion_matrix[0][row] == current_class:
                class_row = row
        true_positives = confusion_matrix[class_row][class_row]
        true_negatives = total - np.sum(confusion_matrix[class_row, 1:]) - np.sum(confusion_matrix[1:, class_row]) + \
                         confusion_matrix[class_row][class_row]
        return (true_positives + true_negatives) / total

    @staticmethod
    def calculate_precision(current_class, confusion_matrix):
        # precision = TP / (TP + FP)
        for row in range(0, len(confusion_matrix)):
            if confusion_matrix[0][row] == current_class:
                class_row = row
        true_positives = confusion_matrix[class_row][class_row]
        false_positives = np.sum(confusion_matrix[1:, class_row]) - confusion_matrix[class_row][class_row]
        return true_positives / (true_positives + false_positives)

    def calculate_f_score(self, current_class, confusion_matrix, beta):
        # F1 score = (1 + beta^2) * precision * recall / (beta^2 * precision + recall)
        precision = self.calculate_precision(current_class, confusion_matrix)
        recall = self.calculate_recall(current_class, confusion_matrix)
        return (1 + beta**2) * precision * recall / (beta ** 2 * precision + recall)

    def calculate_f1_score(self, current_class, confusion_matrix):
        # F1 score = 2 * precision * recall / (precision + recall)
        return self.calculate_f_score(current_class, confusion_matrix, 1)

    @staticmethod
    def calculate_classifier_error(confusion_matrix):
        error = np.sum(confusion_matrix[1:, 1:])
        for i in range(1, len(confusion_matrix)):
            error -= confusion_matrix[i][i]
        print("Error calculated")
        return error

    @staticmethod
    def calculate_classifier_error_relaitve(confusion_matrix):
        total = np.sum(confusion_matrix[1:, 1:])
        error = total
        for i in range(1, len(confusion_matrix)):
            error -= confusion_matrix[i][i]
        return error / total

    def calculate_matthews_correlation_coefficient(self, current_class, confusion_matrix):
        # MCC = (TP * TN - FP * FN) / ( (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))^(1/2)
        class_row = ClassifierMetrics.get_class_row(current_class, confusion_matrix)
        total = np.sum(confusion_matrix[1:, 1:])
        true_positives = confusion_matrix[class_row][class_row]
        false_positives = np.sum(confusion_matrix[1:, class_row]) - confusion_matrix[class_row][class_row]
        true_negatives = total - np.sum(confusion_matrix[class_row, 1:]) - np.sum(confusion_matrix[1:, class_row]) + \
                         confusion_matrix[class_row][class_row]
        false_negatives = 0
        for column in range(1, len(confusion_matrix[0])):
            if column != class_row:
                false_negatives += confusion_matrix[class_row][column]
        numerator = true_positives * true_negatives - false_positives * false_negatives
        denominator = (true_positives + false_positives) * (true_positives + false_negatives) * (true_negatives +
                                                              false_positives) * (true_negatives + false_negatives)
        return numerator / np.sqrt(denominator)

    @staticmethod
    def get_class_row(current_class, confusion_matrix):
        for row in range(0, len(confusion_matrix)):
            if confusion_matrix[0][row] == current_class:
                class_row = row
        return class_row

    def calculate_all_metrics(self, inferences, expected_values, classes_dict):
        classes = []
        for current_class in classes_dict:
            classes.append(current_class)
        self.classes = classes
        self.confusion_matrix = self.calculate_confusion_matrix(inferences, expected_values, classes)
        self.estimated_classifier_error = self.calculate_classifier_error(self.confusion_matrix)
        self.estimated_classifier_error_relative = self.calculate_classifier_error_relaitve(self.confusion_matrix)
        for current_class in classes:
            self.true_positives_rate[current_class] = self.calculate_true_positives_rate(current_class,
                                                                                         self.confusion_matrix)
            self.false_positives_rate[current_class] = self.calculate_false_positives_rate(current_class,
                                                                                           self.confusion_matrix)
            self.recall[current_class] = self.true_positives_rate[current_class]
            self.accuracy[current_class] = self.calculate_accuracy(current_class, self.confusion_matrix)
            self.precision[current_class] = self.calculate_precision(current_class, self.confusion_matrix)
            self.f1_score[current_class] = self.calculate_f1_score(current_class, self.confusion_matrix)
            self.matthews_correlation_coefficient[current_class] = \
                self.calculate_matthews_correlation_coefficient(current_class, self.confusion_matrix)


