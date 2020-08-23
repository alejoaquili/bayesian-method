import numpy as np
from utils.data_loader import load_news_dataset, load_transformed_news_dataset
from bayesian_naive_classifier import bayesian_classifier
from models.news_classifier import NewsClassifier
from sklearn.model_selection import train_test_split
#titles, training_matrix = load_person_preferences("../data/PreferenciasBritanicos.csv")
#print(bayesian_classifier(training_matrix, [[1, 0, 1, 1, 0], [0, 0, 1, 1, 1]], [1, 1, 1, 1, 1]))

# titles, news_data = load_news_dataset("../data/Noticias_argentinas.xlsx")
# transform_news_dataset(news_data, ["Internacional", "Deportes", "Ciencia y Tecnologia", "Economia"],
#                       "../data/transformed_news.tsv")


titles, news_data = load_transformed_news_dataset("../data/transformed_news.tsv")
train_matrix = news_data[:, :3]
categories = news_data[:, 3]
training_news, test_news, training_target, test_target = train_test_split(train_matrix, categories, train_size=0.8,
                                                                          stratify=categories)
news_classifier = NewsClassifier()
column = np.transpose
extra_column = training_target.reshape(len(training_target), 1)
training_news = np.append(training_news, extra_column, axis=1)
news_classifier.train(training_news)
result = news_classifier.classify(test_news, test_target)
print(result)

# print(news_classifier.classify(np.asarray([["10/10/10","Los cambios en las notificaciones de WhatsApp que le pueden generar un dolor de cabeza a los usuarios", "TycSports"]]), ["Deportes"]))
