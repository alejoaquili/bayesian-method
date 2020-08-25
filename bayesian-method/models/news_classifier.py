from models.token_counter import TokenCounter
from models.classifier_metrics import ClassifierMetrics


class NewsClassifier:

    def __init__(self):
        self.trained = False
        self.class_quantity = None
        self.ignored_words = {"a", "ante", "bajo", "cabe", "con", "contra", "de", "desde", "en", "entre", "hacia",
                              "hasta", "para", "por", "según", "sin", "so", "sobre", "tras", "la", "las", "el", "ellos",
                              "ellas", "tu", "una", "unas", "uno", "unos", "nosotros", "nosotras", "yo", "vos",
                              "vosotros", "vosotras", "que", "como", "para", "donde", "cuando", "cual", "porque", "por",
                              "si", "no", "e", "y", "o", "u", "mediante", "también", "quienes", "quien", "se", "del",
                              "su", "es", "al", "lo", "mas", "le", "sus", "pero", "me", "esta", "esto", "este", "ser",
                              "son", "fue", "habia", "hubo", "era", "muy", "mi", "ya", "puede", "nos", "nose", "ni",
                              "ese", "eso", "esa", "tan"}
        self.tokenizer_of_classes = {}
        self.probabilities_of_classes = {}
        self.probabilities_for_classes = {}

    def train(self, training_matrix):
        rows = training_matrix.shape[0]
        columns = training_matrix.shape[1] - 1
        sorted_matrix = training_matrix[training_matrix[:, columns].argsort()]
        current_class = sorted_matrix[0, columns]
        start_row = 0
        self.get_class_quantity(sorted_matrix)

        for i in range(1, rows):
            if sorted_matrix[i][columns] != current_class:
                data_for_class = sorted_matrix[start_row:i, 1:columns]
                self.process_news_titles(current_class, data_for_class)
                self.probabilities_of_classes[current_class] = (i - start_row) / rows
                start_row = i
                current_class = sorted_matrix[i][columns]
        data_for_class = sorted_matrix[start_row:rows, 1:columns]
        self.process_news_titles(current_class, data_for_class)
        self.probabilities_of_classes[current_class] = (rows - start_row) / rows
        # TODO: process news sources too
        self.trained = True

    def classify(self, news_matrix, expected_classes=None, generate_metrics=False, use_sources=False):
        if not self.trained:
            raise Exception('Cannot classify without training')

        if expected_classes is None and generate_metrics:
            raise ValueError('Cannot generate metrics of a unsupervised classification')

        rows = len(news_matrix)
        # columns = len(news_matrix[0])
        inferences = []
        for current_class in self.probabilities_of_classes.keys():
            probabilities_for_news = [self.calculate_probability_of_titles(current_class, news_matrix[:, 1])]
            if use_sources:
                probabilities_for_news.append(self.calculate_probability_of_source(current_class, news_matrix[:, 1]))
            self.probabilities_for_classes[current_class] = probabilities_for_news
        for row in range(0, rows):
            max_probability = 0
            max_probability_class = None
            for current_class in self.probabilities_of_classes.keys():
                current_probability = self.probabilities_for_classes[current_class][0][row] * \
                                      self.probabilities_of_classes[current_class]
                                      # self.probabilities_for_classes[current_class][1][row] * \
                if current_probability > max_probability:
                    max_probability = current_probability
                    max_probability_class = current_class
            inferences.append(max_probability_class)
        if generate_metrics:
            metrics = ClassifierMetrics()
            metrics.calculate_all_metrics(inferences, expected_classes, self.probabilities_of_classes.keys())
            return inferences, metrics
        return inferences

    def get_class_quantity(self, sorted_matrix):
        rows = sorted_matrix.shape[0]
        columns = sorted_matrix.shape[1]
        class_quantity = 1
        current_class = sorted_matrix[0][columns - 1]
        for i in range(1, rows):
            if sorted_matrix[i][columns - 1] != current_class:
                current_class = sorted_matrix[i][columns - 1]
                class_quantity += 1
        self.class_quantity = class_quantity
        return class_quantity

    def process_news_titles(self, current_class, titles):
        tokenizer = TokenCounter()
        tokenizer.tokenize(titles)
        self.tokenizer_of_classes[current_class] = tokenizer

    def calculate_probability_of_titles(self, current_class, titles):
        current_class_words_frequencies = self.tokenizer_of_classes[current_class].word_frequencies
        current_class_words_quantity = self.tokenizer_of_classes[current_class].total_words
        probability_of_titles = []
        for title in titles:
            title_words = TokenCounter.tokenize_string(title) # TODO: add words to ignore
            probability_of_title = 1
            for word in title_words: # TODO: add words to ignore
                word_frequency = 0
                if word in current_class_words_frequencies:
                    word_frequency = current_class_words_frequencies[word]
                probability_of_word = (word_frequency + 1) / (current_class_words_quantity + self.class_quantity)
                probability_of_title *= probability_of_word
            probability_of_titles.append(probability_of_title)
        return probability_of_titles

    def calculate_probability_of_titles(self, current_class, titles):
        current_class_words_frequencies = self.tokenizer_of_classes[current_class].word_frequencies
        current_class_words_quantity = self.tokenizer_of_classes[current_class].total_words
        probability_of_titles = []
        for title in titles:
            title_words = TokenCounter.tokenize_string(title)  # TODO: add words to ignore
            probability_of_title = 1
            for word in title_words:  # TODO: add words to ignore
                word_frequency = 0
                if word in current_class_words_frequencies:
                    word_frequency = current_class_words_frequencies[word]
                probability_of_word = (word_frequency + 1) / (current_class_words_quantity + self.class_quantity)
                probability_of_title *= probability_of_word
            probability_of_titles.append(probability_of_title)
        return probability_of_titles

