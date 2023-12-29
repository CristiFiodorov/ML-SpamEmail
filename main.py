import json
import os
from collections import defaultdict
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from naive_bayes import NaiveBayes
from utils import get_email_training_data, get_email_training_data_without_file


def run_bayes():
    path = fr'D:\work\bd\samples\fii\3S1\ML\lingspam_public'

    error_dict = defaultdict(lambda: (0, 0))

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
        total = 0
        for file in os.listdir(part10_folder_path):
            with open(os.path.join(part10_folder_path, file)) as f:
                data = f.read()

            bayes_class = naive_bayes.predict(data)

            if bayes_class == 'spam' and not file.startswith('spm'):
                errors += 1
            if bayes_class == 'non_spam' and file.startswith('spm'):
                errors += 1

            total += 1
            print(file, bayes_class)

        error_dict[folder] = total, errors
        print(errors)

    return error_dict


def make_plot_for_alg(run_alg, title):
    error_dict = run_alg()
    folders = list(error_dict.keys())
    values = list(error_dict.values())

    accuracies = [(total - errors) / total for total, errors in values]

    plt.barh(folders, accuracies, color='skyblue')

    plt.title(f'{title} Errors')
    plt.xlabel('Accuracy')
    plt.ylabel('Folders')

    for index, value in enumerate(accuracies):
        plt.text(value, index, f'{value:.2f}')

    plt.show()


def cvloo(base_folder, validation_file):
    naive_bayes = NaiveBayes()

    training_data = get_email_training_data_without_file(base_folder, validation_file)
    naive_bayes.train(training_data)

    abs_validation_path = os.path.join(base_folder, validation_file)

    with open(abs_validation_path) as f:
        data = f.read()

    bayes_class = naive_bayes.predict(data)

    if bayes_class == 'spam' and not os.path.basename(validation_file).startswith('spm'):
        return 1
    if bayes_class == 'non_spam' and os.path.basename(validation_file).startswith('spm'):
        return 1

    return 0


def process_file(argv):
    base_folder = os.path.dirname(argv[0])
    validation_file = os.path.join(os.path.basename(argv[0]), argv[1])
    errors = cvloo(base_folder, validation_file)
    return os.path.join(os.path.basename(base_folder), validation_file), errors


def run_bayes_cvloo():
    path = r'D:\work\bd\samples\fii\3S1\ML\lingspam_public'
    total_errors = 0
    errors_dict = defaultdict(int)

    file_paths = []
    for folder in os.listdir(path):
        if folder == 'readme.txt':
            continue
        folder_full_path = os.path.join(path, folder)
        for sub_folder in os.listdir(folder_full_path):
            sub_folder_full_path = os.path.join(folder_full_path, sub_folder)
            for file in os.listdir(sub_folder_full_path):
                file_paths.append((sub_folder_full_path, file))

    with ProcessPoolExecutor(max_workers=10) as executor:
        future_to_file = {executor.submit(process_file, file_path): file_path for file_path in file_paths}
        for future in concurrent.futures.as_completed(future_to_file):
            file_path, errors = future.result()
            total_errors += errors
            errors_dict[f'{file_path}'] = errors
            print(f"File: {file_path}: {errors}")

    print(total_errors)
    print(errors_dict)
    return errors_dict


def make_pie_chart_for_cvloo():
    if not os.path.isfile('cvloo_errors.json'):
        error_dict = run_bayes_cvloo()
        with open('cvloo_errors.json', 'w') as cvloo_errors_json:
            json.dump(error_dict, cvloo_errors_json, indent=4)
    else:
        with open('cvloo_errors.json')as cvloo_errors_json:
            error_dict = json.load(cvloo_errors_json)

    cvloo_errors = error_dict.values()

    num_errors = sum(cvloo_errors)
    num_no_errors = len(cvloo_errors) - num_errors

    labels = 'Errors', 'No Errors'
    sizes = [num_errors, num_no_errors]
    colors = ['lightcoral', 'lightskyblue']
    explode = (0.1, 0)

    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)

    plt.axis('equal')
    plt.title('CVLOO Error Distribution')
    plt.show()

    error_folders_dict = defaultdict(list)
    for key, value in error_dict.items():
        folder = key.split("\\", 1)
        error_folders_dict[folder[0]].append(value)

    num_pies = len(error_folders_dict)
    fig, axes = plt.subplots(1, num_pies, figsize=(num_pies * 5, 6))

    for i, (category, values) in enumerate(error_folders_dict.items()):
        num_errors = sum(values)
        num_no_errors = len(values) - num_errors
        labels = 'Errors', 'No Errors'
        sizes = [num_errors, num_no_errors]
        axes[i].pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=140)
        axes[i].set_title(category)

    plt.show()


def run_lib_alg(alg, arg_dict):
    path = r'D:\work\bd\samples\fii\3S1\ML\lingspam_public'

    error_dict = defaultdict(lambda x: (0, 0))

    for folder in os.listdir(path):
        if folder == 'readme.txt':
            continue

        folder_full_path = os.path.join(path, folder)
        training_data = get_email_training_data(folder_full_path)

        corpus = []
        y = []
        for category in training_data:
            for file in training_data[category]:
                text_representation = ' '.join(
                    [f'{word} ' * count for word, count in training_data[category][file].items()])
                corpus.append(text_representation)
                y.append(1 if category == 'spam' else 0)

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)

        knn = alg(**arg_dict)
        knn.fit(X, y)

        part10_folder_path = os.path.join(folder_full_path, 'part10')

        errors = 0
        total = 0
        for file in os.listdir(part10_folder_path):
            with open(os.path.join(part10_folder_path, file)) as f:
                data = f.read()

            new_text_transformed = vectorizer.transform([data])
            predicted_label = knn.predict(new_text_transformed)

            if predicted_label[0] == 1 and not file.startswith('spm'):
                errors += 1
            if predicted_label[0] != 1 and file.startswith('spm'):
                errors += 1

            total += 1
            print("Spam" if predicted_label[0] == 1 else "Not Spam")

        error_dict[folder] = total, errors
        print(errors)

    return error_dict


def run_knn():
    return run_lib_alg(KNeighborsClassifier, {'n_neighbors': 20})


def run_adaboost():
    return run_lib_alg(AdaBoostClassifier, {"n_estimators": 200, "learning_rate": 1})


def run_lib_naive_bayes():
    return run_lib_alg(MultinomialNB, {})


def run_id3():
    return run_lib_alg(DecisionTreeClassifier, {"criterion": 'entropy'})


def main():
    # run_bayes()
    # make_pie_chart_for_cvloo()

    make_plot_for_alg(run_bayes, "Bayes Naive")
    make_plot_for_alg(run_lib_naive_bayes, "Lib Bayes Naive")
    make_plot_for_alg(run_knn, "K-NN")
    make_plot_for_alg(run_adaboost, "Ada-Boost")
    make_plot_for_alg(run_id3, "ID3")


if __name__ == '__main__':
    main()
