# k-nearest neighbors on the Iris Flowers Dataset
from random import seed
from random import randrange
from csv import reader
import numpy as np
import argparse

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    if len(actual) == 0:
        return 0.0
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, n_folds, num_neighbors, algorithm):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = k_nearest_neighbors(train_set, test_set, num_neighbors, algorithm)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Calculate the vectorized Hamming distance
def hamming_distance(x, y):
    xarr = np.array(x[:len(x) - 1])
    yarr = np.array(y[:len(y) - 1])
    dists = len((xarr != yarr).nonzero()[0])
    # print("Distance: ", dists)
    return dists


# Calculate the vectorized Euclidean distance
def euclidean_distance(x, y):
    xarr = np.array(x[:len(x) - 1])
    yarr = np.array(y[:len(y) - 1])
    dists = np.sqrt((np.square(xarr[:, np.newaxis] - yarr).sum(axis=None)))  # Euclidean Distance
    # print("Distance: ", dists)
    return dists


# Calculate the vectorized Manhattan distance
def manhattan_distance(x, y):
    xarr = np.array(x[:len(x) - 1])
    yarr = np.array(y[:len(y) - 1])
    dists = np.abs(xarr[:, np.newaxis] - yarr).sum(axis=None)  # Manhattan Distance
    # print(dists)
    return dists


# Calculate the vectorized Minkowski distance
def minkowski_distance(x, y):
    xarr = np.array(x[:len(x) - 1])
    yarr = np.array(y[:len(y) - 1])
    c = 5
    dists = (np.abs(xarr[:, np.newaxis] - yarr) ** c).sum(axis=None) ** (1 / c)  # Minkowski Distance
    # print("Distance: ", dists)
    return dists


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors, dist_algorithm):
    distances = list()
    for train_row in train:
        dist = dist_algorithm(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors, dist_algorithm):
    neighbors = get_neighbors(train, test_row, num_neighbors, dist_algorithm)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors, dist_algorithm):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors, dist_algorithm)
        predictions.append(output)
    return (predictions)

def print_result(score, num_neighbors, algorithm):
    print('\n\t Distance calculation method: %s' % algorithm)
    print('\t Number of Neighbors: %s' % num_neighbors)
    print('\t Scores: %s' % score)
    print('\t Mean Accuracy: %.3f%%\n' % (sum(score) / float(len(score))))

def process_dataset(filename, dataset_name):
    dataset = load_csv(filename)

    # convert class column to integers
    for i in range(len(dataset[0])):
        str_column_to_int(dataset, i)
    
    print('--------------------------------------------------------')
    print('  %s Dataset' % dataset_name)
    print('--------------------------------------------------------')
    

    # evaluate using Hamming distance
    scores = evaluate_algorithm(dataset, n_folds, num_neighbors, hamming_distance)
    print_result(scores, num_neighbors, "Hamming")

    # evaluate using Euclidean distance
    scores = evaluate_algorithm(dataset, n_folds, num_neighbors, euclidean_distance)
    print_result(scores, num_neighbors, "Euclidean")

    # evaluate using Manhattan distance
    scores = evaluate_algorithm(dataset, n_folds, num_neighbors, manhattan_distance)
    print_result(scores, num_neighbors, "Manhattan")

    # evaluate using Minkowski distance
    scores = evaluate_algorithm(dataset, n_folds, num_neighbors, minkowski_distance)
    print_result(scores, num_neighbors, "Minkowski")
    

# Test the kNN on the Iris Flowers dataset
seed(1)

n_folds = 10
num_neighbors = 15


parser = argparse.ArgumentParser()
parser.add_argument("-f", help="dataset file to evaluate algorithm")
args = parser.parse_args()

filepath = args.f

if filepath == "car.csv":
    # evaluate using car dataset
    process_dataset(filepath, "Car")
elif filepath == "breast-cancer.csv":
    # evaluate using breast cancer dataset
    process_dataset(filepath, "Breast Cancer")
elif filepath == "hayes-roth.csv":
    # evaluate using breast cancer dataset
    process_dataset(filepath, "Hayes Roth")
else:
    process_dataset("car.csv", "Car")
    process_dataset("breast-cancer.csv", "Breast Cancer")
    process_dataset("hayes-roth.csv", "Hayes Roth")
