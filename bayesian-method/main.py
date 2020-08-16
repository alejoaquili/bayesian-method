from utils.data_loader import load_person_preferences
from bayesian_naive_classifier import bayesian_classifier

titles, training_matrix = load_person_preferences("../data/PreferenciasBritanicos.csv")
print(bayesian_classifier(training_matrix, [[1, 0, 1, 1, 0]]))

