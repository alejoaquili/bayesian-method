import numpy as np

from models.bayesian_network import BayesianNetwork
from utils.data_loader import load_news_dataset, load_transformed_news_dataset, load_binary_dataset, discretize_binary_dataset
from bayesian_naive_classifier import bayesian_classifier
from models.news_classifier import NewsClassifier
from sklearn.model_selection import train_test_split


########################################################################################################################
# Ej 2

#titles, training_matrix = load_person_preferences("../data/PreferenciasBritanicos.csv")
#print(bayesian_classifier(training_matrix, [[1, 0, 1, 1, 0], [0, 0, 1, 1, 1]], [1, 1, 1, 1, 1]))

# titles, news_data = load_news_dataset("../data/Noticias_argentinas.xlsx")
# transform_news_dataset(news_data, ["Internacional", "Deportes", "Ciencia y Tecnologia", "Economia"],
#                       "../data/transformed_news.tsv")


########################################################################################################################
# Ej 3

# titles, news_data = load_transformed_news_dataset("../data/transformed_news.tsv")
# train_matrix = news_data[:, :3]
# categories = news_data[:, 3]
# training_news, test_news, training_target, test_target = train_test_split(train_matrix, categories, train_size=0.8,
#                                                                           stratify=categories)
# news_classifier = NewsClassifier()
# column = np.transpose
# extra_column = training_target.reshape(len(training_target), 1)
# training_news = np.append(training_news, extra_column, axis=1)
# news_classifier.train(training_news, ignore_words=True)
# # result = news_classifier.classify(test_news, test_target)
# # print(result)
# result, metrics = news_classifier.classify(test_news, test_target, generate_metrics=True, output_path="../output",
#                                            use_sources=False, ignore_words=False)
# print(result)
# print(metrics.confusion_matrix)
# print(metrics.estimated_classifier_error)
# print(metrics.estimated_classifier_error_relative)
# print(metrics.true_positives_rate)
# print(metrics.false_positives_rate)
# print(metrics.recall)
# print(metrics.accuracy)
# print(metrics.precision)
# print(metrics.f1_score)
# print(metrics.matthews_correlation_coefficient)
# print(metrics.roc_point)


########################################################################################################################
# Ej 4

titles, matrix = load_binary_dataset("../data/binary.csv")
matrix = discretize_binary_dataset(titles, matrix)
print(titles)
bayesian_network = BayesianNetwork()
bayesian_network.generate_structure()
