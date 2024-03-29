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
        self.sources_for_classes = {}

    def train(self, training_matrix, ignore_words=False):
        rows = training_matrix.shape[0]
        columns = training_matrix.shape[1] - 1
        sorted_matrix = training_matrix[training_matrix[:, columns].argsort()]
        current_class = sorted_matrix[0, columns]
        start_row = 0
        self.get_class_quantity(sorted_matrix)

        for i in range(1, rows):
            if sorted_matrix[i][columns] != current_class:
                data_for_class = sorted_matrix[start_row:i, 1:columns]
                self.process_news_titles(current_class, data_for_class, ignore_words)
                self.process_sources(current_class, data_for_class[:, 1])
                self.probabilities_of_classes[current_class] = (i - start_row) / rows
                start_row = i
                current_class = sorted_matrix[i][columns]
        data_for_class = sorted_matrix[start_row:rows, 1:columns]
        self.process_news_titles(current_class, data_for_class, ignore_words)
        self.process_sources(current_class, data_for_class[:, 1])
        self.probabilities_of_classes[current_class] = (rows - start_row) / rows
        self.trained = True

    def classify(self, news_matrix, expected_classes=None, generate_metrics=False, output_path=None, use_sources=False,
                 ignore_words=False):
        if not self.trained:
            raise Exception('Cannot classify without training')

        if expected_classes is None and generate_metrics:
            raise ValueError('Cannot generate metrics of a unsupervised classification')

        rows = len(news_matrix)
        # columns = len(news_matrix[0])
        inferences = []
        for current_class in self.probabilities_of_classes.keys():
            probabilities_for_news = [self.calculate_probability_of_titles(current_class, news_matrix[:, 1],
                                                                           ignore_words)]
            if use_sources:
                probabilities_for_news.append(self.calculate_probability_of_source(current_class, news_matrix[:, 2]))
            self.probabilities_for_classes[current_class] = probabilities_for_news
        for row in range(0, rows):
            max_probability = 0
            max_probability_class = None
            for current_class in self.probabilities_of_classes.keys():
                current_probability = self.probabilities_for_classes[current_class][0][row] * \
                                      self.probabilities_of_classes[current_class]
                if  use_sources:
                    current_probability  *= self.probabilities_for_classes[current_class][1][row]
                if current_probability > max_probability:
                    max_probability = current_probability
                    max_probability_class = current_class
            inferences.append(max_probability_class)
        if generate_metrics:
            metrics = ClassifierMetrics()
            metrics.calculate_all_metrics(inferences, expected_classes, self.probabilities_of_classes.keys(),
                                          self.probabilities_for_classes, output_folder=output_path)
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

    def process_news_titles(self, current_class, titles, ignore_words):
        tokenizer = TokenCounter()
        if ignore_words:
            tokenizer.tokenize(titles, self.ignored_words)
        else:
            tokenizer.tokenize(titles)
        self.tokenizer_of_classes[current_class] = tokenizer

    def process_sources(self, current_class, sources):
        sources_map = {}
        for source in sources:
            lower_source = source.lower()
            if lower_source not in sources_map:
                current_count = 0
            else:
                current_count = sources_map[lower_source]
            sources_map[lower_source] = current_count + 1
        self.sources_for_classes[current_class] = sources_map

    def calculate_probability_of_titles(self, current_class, titles, ignore_words):
        current_class_words_frequencies = self.tokenizer_of_classes[current_class].word_frequencies
        current_class_words_quantity = self.tokenizer_of_classes[current_class].total_words
        probability_of_titles = []
        for title in titles:
            if ignore_words:
                title_words = TokenCounter.tokenize_string(title, self.ignored_words)
            else:
                title_words = TokenCounter.tokenize_string(title)
            probability_of_title = 1
            for word in title_words:
                word_frequency = 0
                if word in current_class_words_frequencies:
                    word_frequency = current_class_words_frequencies[word]
                probability_of_word = (word_frequency + 1) / (current_class_words_quantity + self.class_quantity)
                probability_of_title *= probability_of_word
            probability_of_titles.append(probability_of_title)
        return probability_of_titles

    def calculate_probability_of_source(self, current_class, sources):
        sources_map = self.sources_for_classes[current_class]
        sources_quantity = len(sources_map)
        probability_of_sources = []
        for source in sources:
            lower_source = source.lower()
            source_frequency = 0
            if lower_source in sources_map:
                source_frequency = sources_map[lower_source]
            probability_of_source = (source_frequency + 1) / (sources_quantity + self.class_quantity)
            probability_of_sources.append(probability_of_source)
        return probability_of_sources

