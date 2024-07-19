import argparse
import numpy as np
import pandas as pd
from hmmlearn import hmm
import os
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="Classify devices based on power consumption using HMM.")
    parser.add_argument('--train', required=True, help='Path to the training data CSV file')
    parser.add_argument('--test', required=True, help='Path to the test data folder')
    parser.add_argument('--output', required=True, help='Path to the output results file')
    return parser.parse_args()


def classify(power_series, models, device_names, n_times=7):
    votes = []
    for _ in range(n_times):
        log_likelihoods = [model.score(power_series) for model in models]
        best_model_index = np.argmax(log_likelihoods)
        classified_device = device_names[best_model_index]
        votes.append(classified_device)
    most_common_device = Counter(votes).most_common(1)[0][0]
    return most_common_device


def main():
    args = parse_args()

    data = pd.read_csv(args.train)
    data.drop('time', axis=1, inplace=True)
    power = [data[device].values.reshape(-1, 1) for device in data.columns]
    light2 = power[0]
    light5 = power[1]
    light4 = power[2]
    refrig = power[3]
    microv = power[4]

    models = []

    model_light2 = hmm.GaussianHMM(n_components=8, covariance_type="diag", n_iter=200, random_state=22)
    model_light2.fit(light2)
    models.append(model_light2)

    model_light5 = hmm.GaussianHMM(n_components=8, covariance_type="diag", n_iter=200, random_state=42)
    model_light5.fit(light5)
    models.append(model_light5)

    model_light4 = hmm.GaussianHMM(n_components=8, covariance_type="diag", n_iter=200, random_state=42)
    model_light4.fit(light4)
    models.append(model_light4)

    model_refrig = hmm.GaussianHMM(n_components=8, covariance_type="diag", n_iter=200, random_state=42)
    model_refrig.fit(refrig)
    models.append(model_refrig)

    model_microv = hmm.GaussianHMM(n_components=6, covariance_type="diag", n_iter=200, random_state=42)
    model_microv.fit(microv)
    models.append(model_microv)

    test_files = os.listdir(args.test)
    test_data = {}
    for file in test_files:
        file_path = os.path.join(args.test, file)
        test_data[file] = pd.read_csv(file_path, usecols=['dev'])

    device_names = data.columns.tolist()
    results = {}

    for filename, power_data in test_data.items():
        power_series = power_data['dev'].values.reshape(-1, 1)
        classified_device = classify(power_series, models, device_names, n_times=7)
        results[filename] = classified_device

    with open(args.output, 'w') as f:
        f.write(f"file, device_classified \n")
        for file, device in results.items():
            f.write(f"{file}, {device}\n")

    print("Wyniki klasyfikacji zapisane do pliku:", args.output)


if __name__ == "__main__":
    main()
