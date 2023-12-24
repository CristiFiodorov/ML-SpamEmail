import os
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))


def process_data(data):
    # lower case
    data = data.lower()

    # keep only alphanumeric characters
    data = re.sub(r'[^a-z\s]', '', data)

    # tokenize data
    tokens = word_tokenize(data)

    # delete stop_words + words with len less than 2
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    porter_stemmer = PorterStemmer()

    # reducing words to its basic form
    return [porter_stemmer.stem(word) for word in filtered_tokens]


def get_email_training_data(folder_path, test_sub_folder='part10'):
    normal = defaultdict(lambda: defaultdict(int))
    spam = defaultdict(lambda: defaultdict(int))

    for sub_folder in os.listdir(folder_path):
        if sub_folder == test_sub_folder:
            continue

        full_sub_folder_path = os.path.join(folder_path, sub_folder)

        for file in os.listdir(full_sub_folder_path):
            full_file_path = os.path.join(full_sub_folder_path, file)
            with open(full_file_path) as f:
                data = f.read()
            processed_tokens = process_data(data)

            if file.startswith('spm'):
                for token in processed_tokens:
                    spam[file][token] += 1
                continue
            for token in processed_tokens:
                normal[file][token] += 1

    return {'spam': spam, 'non_spam': normal}