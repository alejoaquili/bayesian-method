import re


class TokenCounter:

    def __init__(self):
        self.word_frequencies = {}
        self.total_words = 0

    def tokenize(self, texts, words_to_ignore=None):
        for text in texts:
            for word in text[0].split():
                current_word = str.lower(re.sub('[^a-zA-Z ]+', '', word))
                if words_to_ignore is None or current_word not in words_to_ignore:
                    if current_word not in self.word_frequencies:
                        current_count = 0
                    else:
                        current_count = self.word_frequencies[current_word]
                    self.word_frequencies[current_word] = current_count + 1
                    self.total_words += 1

    @staticmethod
    def tokenize(text, words_to_ignore=None):
        words = []
        for word in text.split():
            current_word = str.lower(re.sub('[^a-zA-Z ]+', '', word))
            if words_to_ignore is None or current_word not in words_to_ignore:
                words.append(current_word)
        return words
