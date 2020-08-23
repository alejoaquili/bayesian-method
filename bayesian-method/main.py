import numpy as np
from models import token_counter
from utils.data_loader import load_person_preferences, load_transformed_news_dataset, transform_news_dataset
from bayesian_naive_classifier import bayesian_classifier

#titles, training_matrix = load_person_preferences("../data/PreferenciasBritanicos.csv")
#print(bayesian_classifier(training_matrix, [[1, 0, 1, 1, 0], [0, 0, 1, 1, 1]], [1, 1, 1, 1, 1]))

# titles, news_data = load_news_dataset("../data/Noticias_argentinas.xlsx")
# transform_news_dataset(np.asarray([["Internacional"], ["Nacional"], ["Internacional"], ["Nacional"], ["Deportes"], ["Ciencia y Tecnología"], ["Nacional"], ["Nacional"], ["hola"], ["Economía"]]), ["Internacional", "Deportes", "Ciencia y Tecnología", "Economía"])
# transform_news_dataset(news_data, ["Internacional", "Deportes", "Ciencia y Tecnologia", "Economia"],
          #             "../data/transformed_news.csv")

# tokenizer = token_counter.TokenCounter()
# tokenizer.tokenize(test)
# print(tokenizer.word_frequencies)
# print(tokenizer.total_words)

titles, news_data = load_tranformed_news_dataset("../data/transformed_news.csv")
