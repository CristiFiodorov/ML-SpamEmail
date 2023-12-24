import json
from collections import defaultdict

from utils import process_data


class NaiveBayes:
    def __init__(self):
        self.class_prob = None
        self.conditional_prob = None

    def get_prob_from_json(self, path):
        with open(path, 'r') as f:
            j = json.load(f)
        self.conditional_prob = j['conditional_prob']
        self.class_prob = j['class_prob']

    def train(self, data, save_name=None):
        total_len = len(data['spam']) + len(data['non_spam'])
        class_freq = {key: len(value) / total_len for key, value in data.items()}
        self.class_prob = {label: freq / len(class_freq) for label, freq in class_freq.items()}

        token_freq = defaultdict(lambda: defaultdict(int))

        for cls, tokens_count in data.items():
            for token_count in tokens_count.values():
                for token, count in token_count.items():
                    token_freq[cls][token] += count

        not_in_spam = set(token_freq['non_spam'].keys()) - set(token_freq['spam'].keys())

        not_in_non_spam = set(token_freq['spam'].keys()) - set(token_freq['non_spam'].keys())

        for token in not_in_spam:
            token_freq['spam'][token] = 1

        for token in not_in_non_spam:
            token_freq['non_spam'][token] = 1

        self.conditional_prob = defaultdict(dict)
        for label, words in token_freq.items():
            total_words = sum(words.values())
            self.conditional_prob[label] = {word: (count / total_words) for word, count in words.items()}

        if save_name is None:
            return

        with open(save_name, 'w') as f:
            json.dump({
                'conditional_prob': self.conditional_prob,
                'class_prob': self.class_prob
            }, f, indent=4)

    def predict(self, text):
        if self.conditional_prob is None or self.class_prob is None:
            return None

        processed_data = set(process_data(text))
        processed_data = list(filter(lambda token: token in self.conditional_prob['non_spam'], processed_data))
        processed_data = sorted(processed_data,
                                key=lambda token: self.conditional_prob['non_spam'][token]
                                                  + self.conditional_prob['spam'][token],
                                reverse=True)[:50]

        class_scores = defaultdict(float)

        for cls in self.class_prob.keys():
            class_scores[cls] = self.class_prob[cls]
            for token in processed_data:
                class_scores[cls] *= self.conditional_prob[cls][token]

        print(class_scores)
        return max(class_scores, key=class_scores.get)