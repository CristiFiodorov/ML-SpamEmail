import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt

from naive_bayes import NaiveBayes
from utils import get_email_training_data


def run_bayes():
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


def cvloo(base_folder, validation_sub_folder):
    naive_bayes = NaiveBayes()

    training_data = get_email_training_data(base_folder, validation_sub_folder)
    naive_bayes.train(training_data)

    total_errors = 0
    abs_validation_sub_folder_path = os.path.join(base_folder, validation_sub_folder)
    for file in os.listdir(abs_validation_sub_folder_path):
        with open(os.path.join(abs_validation_sub_folder_path, file)) as f:
            data = f.read()

        bayes_class = naive_bayes.predict(data)

        if bayes_class == 'spam' and not file.startswith('spm'):
            total_errors += 1
        if bayes_class == 'non_spam' and file.startswith('spm'):
            total_errors += 1

        print(file, bayes_class)

    return total_errors


def run_bayes_cvloo():
    path = fr'D:\work\bd\samples\fii\3S1\ML\lingspam_public'

    total_errors = 0
    errors_dict = defaultdict(int)
    for folder in os.listdir(path):
        if folder == 'readme.txt':
            continue

        folder_full_path = os.path.join(path, folder)

        for sub_folder in os.listdir(folder_full_path):
            errors = cvloo(folder_full_path, sub_folder)
            total_errors += errors
            errors_dict[f'{folder}/{sub_folder}'] = errors
            print(f"Folder: {folder}/{sub_folder}: {errors}")

    print(total_errors)
    print(errors_dict)
    return errors_dict


def make_plot_for_cvloo():
    error_dict = run_bayes_cvloo()
    with open('cvloo_errors.json', 'w') as cvloo_errors:
        json.dump(error_dict, cvloo_errors, indent=4)

    plt.figure(figsize=(15, 6))
    plt.bar(list(error_dict.keys()), list(error_dict.values()), color='skyblue')

    plt.title("CVLOO")
    plt.xlabel("Folders")
    plt.ylabel("Errors")
    plt.xticks(rotation=90)
    
    plt.show()


def main():
    # run_bayes()
    make_plot_for_cvloo()


if __name__ == '__main__':
    main()
