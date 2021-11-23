import numpy as np


def compute_distances(X1, X2):
    """Compute the L2 distance between each point in X1 and each point in X2.
    It's possible to vectorize the computation entirely (i.e. not use any loop).

    Args:
        X1: numpy array of shape (M, D) normalized along axis=1
        X2: numpy array of shape (N, D) normalized along axis=1

    Returns:
        dists: numpy array of shape (M, N) containing the L2 distances.
    """
    M = X1.shape[0]
    N = X2.shape[0]
    assert X1.shape[1] == X2.shape[1]

    # YOUR CODE HERE
    # Compute the L2 distance between all X1 features and X2 features.
    # Don't use any for loop, and store the result in dists.
    #
    # You should implement this function using only basic array operations;
    # in particular you should not use functions from scipy.
    #
    # HINT: Try to formulate the l2 distance using matrix multiplication
    a2 = np.diag(np.dot(X1, X1.T))  # (M, D) * (D, M) -> (M, M) 只使用对角元素
    b2 = np.diag(np.dot(X2, X2.T))  # (N, D) * (D, N) -> (N, N)
    a2 = np.reshape(np.repeat(a2, N), (M, N))
    b2 = np.reshape(np.repeat(b2, M), (N, M))
    ab2 = 2 * np.dot(X1, X2.T)  # (M, D) * (D, N) -> (M, N)
    dists = np.sqrt(-ab2 + a2 + b2.T)

    assert dists.shape == (M, N), "dists should have shape (M, N), got %s" % dists.shape

    return dists


def predict_labels(dists, y_train, k=1):
    """Given a matrix of distances `dists` between test points and training points,
    predict a label for each test point based on the `k` nearest neighbors.

    Args:
        dists: A numpy array of shape (num_test, num_train) where dists[i, j] gives
               the distance betwen the ith test point and the jth training point.

    Returns:
        y_pred: A numpy array of shape (num_test,) containing predicted labels for the
                test data, where y[i] is the predicted label for the test point X[i].
    """
    num_test, num_train = dists.shape
    y_pred = np.zeros(num_test, dtype=np.int)

    # Use the distance matrix to find the k nearest neighbors of the ith
    # testing point, and use y_train to find the labels of these
    # neighbors.

    # Once you have found the labels of the k nearest neighbors, you
    # need to find the most common label in the list closest_y of labels.
    # Store this label in y_pred[i]. Break ties by choosing the smaller
    # label.

    # Hint: Look up the functions numpy.argsort and numpy.bincount

    sorted_idx = np.argsort(dists)  # index of the descending value
    for one in range(num_test):
        neighbor_idx = sorted_idx[one, :k]
        k_neighbor = y_train[neighbor_idx]
        # statistics the frequency of number 0 to max(L)
        max_label = np.argmax(np.bincount(k_neighbor))
        y_pred[one] = max_label

    return y_pred


def split_folds(train_data, labels, num_folds):
    """Split up the training data into `num_folds` folds.

    The goal of the functions is to return training sets (features and labels) along with
    corresponding validation sets. In each fold, the validation set will represent (1/num_folds)
    of the data while the training set represent (num_folds-1)/num_folds.
    If num_folds=5, this corresponds to a 80% / 20% split.

    For instance, if X_train = [0, 1, 2, 3, 4, 5], and we want three folds, the output will be:
        X_trains = [[2, 3, 4, 5],
                    [0, 1, 4, 5],
                    [0, 1, 2, 3]]
        X_vals = [[0, 1],
                  [2, 3],
                  [4, 5]]

    Return the folds in this order to match the staff solution!
        
    hint: you may find np.hstack and np.vstack helpful for this part

    """
    assert train_data.shape[0] == labels.shape[0]

    validation_size = train_data.shape[0] // num_folds
    training_size = train_data.shape[0] - validation_size

    X_trains = np.zeros((num_folds, training_size, train_data.shape[1]))
    y_trains = np.zeros((num_folds, training_size), dtype=np.int)
    X_vals = np.zeros((num_folds, validation_size, train_data.shape[1]))
    y_vals = np.zeros((num_folds, validation_size), dtype=np.int)

    split_X_data = np.array_split(train_data, num_folds)
    split_y_data = np.array_split(labels, num_folds)
    for i in range(num_folds):
        selector = list(range(num_folds))
        selector.pop(i)  # len = num_folds - 1
        X_tmp = split_X_data[selector[0]]
        y_tmp = split_y_data[selector[0]]
        for j in range(1, num_folds-1):
            X_tmp = np.vstack((X_tmp, split_X_data[selector[j]]))
            y_tmp = np.hstack((y_tmp, split_y_data[selector[j]]))

        X_trains[i, :, :] = X_tmp
        X_vals[i, :, :] = split_X_data[i]
        y_trains[i, :] = y_tmp
        y_vals[i, :] = split_y_data[i]

    return X_trains, y_trains, X_vals, y_vals
