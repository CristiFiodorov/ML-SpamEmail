import os

from naive_bayes import NaiveBayes
from utils import get_email_training_data


def main():
    path = fr'D:\work\bd\samples\fii\3S1\ML\lingspam_public'

    for folder in os.listdir(path):
        if folder == 'readme.txt':
            continue

        folder_full_path = os.path.join(path, folder)
        os.makedirs('probs', exist_ok=True)
        json_path = f'probs/{folder}.json'

        naive_bayes = NaiveBayes()
        if os.path.isfile(json_path):
            naive_bayes.get_prob_from_json(json_path)
        else:
            training_data = get_email_training_data(folder_full_path)
            naive_bayes.train(training_data, json_path)

        part10_folder_path = os.path.join(folder_full_path, 'part10')

        errors = 0
        for file in os.listdir(part10_folder_path):
            with open(os.path.join(part10_folder_path, file)) as f:
                data = f.read()

            bayes_class = naive_bayes.predict(data)

            if bayes_class == 'spam' and not file.startswith('spm'):
                errors += 1
            if bayes_class == 'non_spam' and file.startswith('spm'):
                errors += 1

            print(file, bayes_class)

        print(errors)


if __name__ == '__main__':
    main()
