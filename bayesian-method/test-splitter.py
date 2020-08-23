from utils.data_loader import load_news_dataset, load_transformed_news_dataset
from sklearn.model_selection import train_test_split


titles, news_data = load_transformed_news_dataset("../data/transformed_news.tsv")
train_matrix = news_data[0:11, 1:3]
categories = news_data[0:11, 3]
xTrain, xTest, yTrain, yTest = train_test_split(train_matrix, categories, train_size=0.8, stratify=categories)
print(xTrain)
