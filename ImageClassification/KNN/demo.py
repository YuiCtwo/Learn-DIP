# 本代码修改自 CS 131 hw6, 使用 KNN 做人脸识别

from k_nearest_neighbor import compute_distances
from k_nearest_neighbor import predict_labels
from k_nearest_neighbor import split_folds
from utils_knn import load_dataset
import numpy as np
import matplotlib.pyplot as plt

X_train, y_train, classes_train = load_dataset('faces', train=True, as_gray=True)
X_test, y_test, classes_test = load_dataset('faces', train=False, as_gray=True)

assert classes_train == classes_test
X_train_flat = np.reshape(X_train, (X_train.shape[0], -1))
X_test_flat = np.reshape(X_test, (X_test.shape[0], -1))
# =================
# Step 1: compute the distances between all features from X_train and from X_test
dists = compute_distances(X_test_flat, X_train_flat)
assert dists.shape == (160, 800)
print("dists shape:", dists.shape)
# We use k = 1 (which corresponds to only taking the nearest neighbor to decide)
y_test_pred = predict_labels(dists, y_train, k=1)
# Compute and print the fraction of correctly predicted examples
num_test = y_test.shape[0]
num_correct = np.sum(y_test_pred == y_test, axis=None)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
# ===================
# Step 2: split the data into 5 folds to perform cross-validation.
num_folds = 5
X_trains, y_trains, X_vals, y_vals = split_folds(X_train_flat, y_train, num_folds)
assert X_trains.shape == (5, 640, 4096)
assert y_trains.shape == (5, 640)
assert X_vals.shape == (5, 160, 4096)
assert y_vals.shape == (5, 160)
# ======================
# Step 3: Measure the mean accuracy for each value of `k`
# List of k to choose from
k_choices = list(range(5, 101, 5))
# Dictionnary mapping k values to accuracies
# For each k value, we will have `num_folds` accuracies to compute
# k_to_accuracies[1] will be for instance [0.22, 0.23, 0.19, 0.25, 0.20] for 5 folds
k_to_accuracies = {}
for k in k_choices:
    print("Running for k=%d" % k)
    accuracies = []
    for i in range(num_folds):
        # Make predictions
        fold_dists = compute_distances(X_vals[i], X_trains[i])
        y_pred = predict_labels(fold_dists, y_trains[i], k)

        # Compute and print the fraction of correctly predicted examples
        num_correct = np.sum(y_pred == y_vals[i])
        accuracy = float(num_correct) / len(y_vals[i])
        accuracies.append(accuracy)

    k_to_accuracies[k] = accuracies

# plot the raw observations
plt.figure(figsize=(12, 8))
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k, v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k, v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

# Based on the cross-validation results above, choose the best value for k,
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 26% accuracy on the test data.


# choose a best k using the figure above
# best_k = 1
# y_test_pred = predict_labels(dists, y_train, k=best_k)
#
# # Compute and display the accuracy
# num_correct = np.sum(y_test_pred == y_test)
# accuracy = float(num_correct) / num_test
# print('For k = %d, got %d / %d correct => accuracy: %f' % (best_k, num_correct, num_test, accuracy))
