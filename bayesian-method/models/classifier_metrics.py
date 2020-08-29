import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import trapz


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
        self.roc_point = {}
        self.general_roc_points = []

    def calculate_confusion_matrix(self, inferences, expected_values, classes, heatmap_path=None):
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
        self.plot_confusion_matrix_heatmap(confusion_matrix, heatmap_path)
        return confusion_matrix

    def calculate_general_roc_curve(self, expected_values, classes, probability_for_classes, out_path=None):
        thresholds_total = 10
        for current_class in classes:
            class_roc_points = []
            for i in range(1, thresholds_total + 1):
                threshold = i * 0.1
                false_positive_rate, true_positive_rate = self.calculate_general_roc_point_for_class(expected_values,
                                                            current_class, classes, probability_for_classes, threshold)
                class_roc_points.append([false_positive_rate, true_positive_rate])
            self.plot_threshold_roc_curve(class_roc_points, current_class, out_path)

    def calculate_general_roc_point_for_class(self, expected_values, current_class, classes, probability_for_classes, threshold):
        # true positives = TP / TP + FN
        # false positives = FP / FP + TN
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        for i in range(0, len(expected_values)):
            total = self.get_total_probability_for_row(i, classes, probability_for_classes)
            normalized_probability = probability_for_classes[current_class][0][i] / total
            if normalized_probability >= threshold:
                if expected_values[i] == current_class:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                if expected_values[i] == current_class:
                    false_negatives += 1
                else:
                    true_negatives += 1
        false_positives_rate = false_positives / (true_negatives + false_positives)
        true_positives_rate = true_positives / (true_positives + false_negatives)
        return false_positives_rate, true_positives_rate

    def get_total_probability_for_row(self, row_index, classes, probability_for_classes):
        total = 0
        for current_class in classes:
            total += probability_for_classes[current_class][0][row_index]
        return total

    @staticmethod
    def calculate_general_roc_point(expected_values, classes, probability_for_classes, threshold):
        true_positives = 0
        false_positives = 0
        for i in range(0, len(expected_values)):
            false_positives += 1
            for current_class in classes:
                if current_class == expected_values[i]:
                    if probability_for_classes[current_class][0][i] >= threshold:
                        true_positives += 1
                        false_positives -= 1
        total = true_positives + false_positives
        return false_positives / total, true_positives / total


    @staticmethod
    def plot_confusion_matrix_heatmap(confusion_matrix, heatmap_path):
        fig, ax = plt.subplots()
        sns.heatmap(np.asarray(confusion_matrix[1:, 1:], dtype=int), annot=True, fmt='d', cmap='Blues', square=True)
        ax.set_xticks(np.arange(len(confusion_matrix[0, 0:])))
        ax.set_yticks(np.arange(len(confusion_matrix[0, 0:])))
        ax.set_xticklabels(confusion_matrix[0, 0:])
        ax.set_yticklabels(confusion_matrix[0, 0:])

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
                 rotation_mode="anchor")

        ax.set_title("Confusion Matrix")
        ax.set_ylabel('Ground Truth label')
        ax.set_xlabel('Predicted label')
        fig.tight_layout()
        if heatmap_path is not None:
            plt.savefig(heatmap_path + "/confusion_matrix_heatmap.png", format="png")
        else:
            plt.show()
        plt.close('all')

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
        return error

    @staticmethod
    def calculate_classifier_error_relative(confusion_matrix):
        total = np.sum(confusion_matrix[1:, 1:])
        error = total
        for i in range(1, len(confusion_matrix)):
            error -= confusion_matrix[i][i]
        return error / total

    @staticmethod
    def calculate_matthews_correlation_coefficient(current_class, confusion_matrix):
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

    @staticmethod
    def plot_roc_curves(roc_point, current_class, curve_path=None):
        auc = trapz([0, roc_point[1], 1], [0, roc_point[0], 1])
        legend = "ROC Curve (AUC = {:.4f})".format(auc)
        plt.subplots()
        plt.plot([(0, 0), (1, 1)], color='black', linewidth=1, linestyle='--')
        plt.plot([0, roc_point[0], 1], [0, roc_point[1], 1], color='blue', linewidth=1)
        plt.scatter(roc_point[0], roc_point[1], marker='o', s=30, facecolor='blue', edgecolor='blue', label=legend)
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.title("{} - ROC Curve".format(current_class))
        plt.xlabel("False positive rate (FP Rate)")
        plt.ylabel("True positive rate (TP Rate)")
        plt.grid()
        plt.legend(loc="lower right")
        if curve_path is not None:
            plt.savefig(curve_path + "/{} - ROC Curve.png".format(current_class), format="png")
        else:
            plt.show()
        plt.close('all')

    @staticmethod
    def plot_threshold_roc_curve(roc_points, current_class, curve_path=None):
        values = np.asarray(roc_points)
        x_values = np.zeros(1)
        x_values = np.hstack((x_values, values[:, 0]))
        x_values = np.hstack((x_values, np.ones(1)))
        y_values = np.zeros(1)
        y_values = np.hstack((y_values, values[:, 1]))
        y_values = np.hstack((y_values, np.ones(1)))
        # x_values = np.zeros(10)
        # y_values = np.zeros(10)
        # x_values[9] = 1
        # y_values[9] = 1
        # for i in range(0, 8):
        #     x_values[i + 1] = 1 - 0.1 * i
        #     y_values[i + 1] = 0.1 * i
        auc = trapz(y_values, x_values)
        legend = "{class_name} ROC Curve (AUC = {auc:.4f})".format(class_name=current_class, auc=auc)
        fig, ax = plt.subplots()
        # labels = []
        # for i in range(0, 10):
        #     labels.append("{:.1f}".format((i + 1) / 10))
        # for i in range(0, len(labels)):
        #     ax.annotate(labels[i], (x_values[i], y_values[i]))
        plt.plot([(0, 0), (1, 1)], color='black', linewidth=1, linestyle='--')
        plt.plot(x_values, y_values, color='blue', linewidth=1, label=legend)
        plt.scatter(x_values, y_values, marker='o', s=30, facecolor='blue', edgecolor='blue')
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.title("General ROC Curve")
        plt.xlabel("False positive rate (FP Rate)")
        plt.ylabel("True positive rate (TP Rate)")
        plt.grid()
        plt.legend(loc="lower right")
        if curve_path is not None:
            plt.savefig(curve_path + "/" + current_class + "-threshold-ROC-Curve.png", format="png")
        else:
            plt.show()
        plt.close('all')

    def calculate_all_metrics(self, inferences, expected_values, classes_dict, probabilities_for_classes,
                              output_folder=None):
        classes = []
        for current_class in classes_dict:
            classes.append(current_class)
        self.classes = classes
        self.confusion_matrix = self.calculate_confusion_matrix(inferences, expected_values, classes, output_folder)
        self.estimated_classifier_error = self.calculate_classifier_error(self.confusion_matrix)
        self.estimated_classifier_error_relative = self.calculate_classifier_error_relative(self.confusion_matrix)
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
            self.roc_point[current_class] = (self.false_positives_rate[current_class],
                                             self.true_positives_rate[current_class])
            self.plot_roc_curves(self.roc_point[current_class], current_class, output_folder)
            self.calculate_general_roc_curve(expected_values, classes, probabilities_for_classes, output_folder)


