import numpy as np
import utils
from math import log

class Node:
    """
    Helper data structure for a decision tree node.
    Students can use this to build their tree recursively.
    DO NOT MODIFY
    """
    def __init__(self, is_leaf=False, prediction=None, feature_index=None, split_value=None, left=None, right=None):
        self.is_leaf = is_leaf              # True if this is a leaf node
        if not self.is_leaf and prediction:
            raise Exception("Cannot assign a prediction value to a non-leaf node")
        self.prediction = prediction        # The majority class label (if is_leaf=True)
        self.feature_index = feature_index  # The index of the feature to split on
        self.split_value = split_value      # The value to split at (for continuous/binary splits)
        self.left = left                    # Child node for condition: X[feature_index] <= split_value
        self.right = right                  # Child node for condition: X[feature_index] > split_value

def entropy(y):
    """
    Calculates the Shannon entropy of a set of labels.
    
    Args:
        y (numpy.ndarray): 1D array of class labels.
        
    Returns:
        float: The entropy value.
    """
    # TODO: student code here
    entropy = 0.0
    labels_dict = {}
    for label in y:
        if label in labels_dict:
            labels_dict[label] += 1
        else:
            labels_dict.update({label: 1})
    for val in labels_dict:
        prob = labels_dict[val] / len(y)
        entropy -= prob * log(prob, 2)
    return entropy

def info_gain(X, y, feature_index, split_value):
    """
    Calculates the Information Gain of a potential split.
    
    Args:
        X (numpy.ndarray): 2D array of features.
        y (numpy.ndarray): 1D array of class labels.
        feature_index (int): The index of the feature to evaluate.
        split_value (float): The value to threshold the split on (<= vs >).
        
    Returns:
        float: The information gain.
    """
    # TODO: student code here
    # X[data_point][feature_no]

    #Entropy of parent
    H_parent = entropy(y)

    #Split nodes
    y_greater = []
    y_lesser = []
    for i in range(len(y)):
        if X[i][feature_index] <= split_value:
            y_lesser.append(y[i])
        else:
            y_greater.append(y[i])

    #Entropy of Children
    H_greater = entropy(y_greater)
    H_lesser = entropy(y_lesser)

    #Final Calculation
    info_gain = H_parent - (len(y_lesser) / len(y) * H_lesser + len(y_greater) / len(y) * H_greater)

    return info_gain

def build_tree(X, y, max_depth=None, current_depth=0, max_features=None):
    """
    Recursively builds a decision tree.
    
    Args:
        X (numpy.ndarray): 2D array of features.
        y (numpy.ndarray): 1D array of class labels.
        max_depth (int, optional): Maximum depth of the tree to prevent overfitting.
        current_depth (int): Current depth in the recursion.
        max_features (int, optional): Number of features to subsample at each split (for bagging).
        
    Returns:
        Node: The root node of the built decision tree.

    Hint: use `thresholds = np.percentile(unique_values, np.linspace(10, 90, 9))` to implement the 
    optimization for continuous variables.
    """
    # TODO: student code here
    root = Node()

    #Check if node should stop here
    if len(X) <= 0:
        root.is_leaf = True
    elif (max_depth != None):
        if current_depth >= max_depth:
            root.is_leaf = True
        
    unique_labels = list(set(y))
    if (len(unique_labels) <= 1): #if it's a pure node
        root.is_leaf = True
    
    
    if root.is_leaf:
        labels_dict = dict.fromkeys(unique_labels, 0)
        for label in y:
            labels_dict[label] += 1
        if not labels_dict:
            root.prediction = 0
        else:
            root.prediction = max(labels_dict, key=labels_dict.get)
        return root

    #try splits
    #optional bagging feature subsampling
    feature_set = []
    if max_features == None:
        feature_set = range(len(X[0]))
    elif max_features >= len(X[0]):
        feature_set = range(len(X[0]))
    else:
        feature_set = sample_features(len(X[0]), max_features)
    # for each feature,
    best_feature = 0
    best_feature_gain = 0
    best_split = 0
    for i in feature_set:
        #find unique values
        unique_values = list(set(X[:, i]))
        potential_splits = []
        if len(unique_values) <= 10:
            potential_splits = unique_values
        else:
            potential_splits = np.percentile(unique_values, np.linspace(10, 90, 9))
        #try potential splits
        best_split_gain = 0
        best_split_ind = 0
        for j in range(len(potential_splits)):
            cur_gain = info_gain(X, y, i, potential_splits[j])
            if cur_gain > best_split_gain:
                best_split_gain = cur_gain
                best_split_ind = j
        if best_split_gain > best_feature_gain:
            best_feature_gain = best_split_gain
            best_feature = i
            best_split = potential_splits[best_split_ind]

    #assign splits
    root.split_value = best_split
    root.feature_index = best_feature

    #split X and y into two parts each, eliminating the feature that we split on
    X_left = np.empty((0, len(X[0])))
    X_right = np.empty((0, len(X[0])))
    y_left = []
    y_right = []
    for i in range(len(y)):
        if X[i][best_feature] <= best_split:
            X_left = np.append(X_left, [X[i]], axis=0)
            y_left.append(y[i])
        else: 
            X_right = np.append(X_right, [X[i]], axis=0)
            y_right.append(y[i])
    #In the case that the node results in a zero node child
    if len(y_left) == 0 or len(y_right) == 0:
        root.is_leaf = True
        labels_dict = dict.fromkeys(unique_labels, 0)
        for label in y:
            labels_dict[label] += 1
        if not labels_dict:
            root.prediction = 0
        else:
            root.prediction = max(labels_dict, key=labels_dict.get)
        return root
    #zero out feature we split on
    #Create left and right children, passing these splits
    root.left = build_tree(X_left, y_left, max_depth, current_depth + 1, max_features)
    root.right = build_tree(X_right, y_right, max_depth, current_depth + 1, max_features)
    return root

def bootstrap_sample(X, y):
    """
    Creates a bootstrap sample of the dataset by sampling with replacement.
    
    Args:
        X (numpy.ndarray): 2D array of features.
        y (numpy.ndarray): 1D array of class labels.
        
    Returns:
        tuple: (X_sample, y_sample) - The bootstrapped dataset.
    """
    # TODO: student code here

    # should have the same number of rows as the original
    sample_indices = np.random.choice(len(y), len(y), replace=True)
    X_samp = np.empty((0, len(X[0])))
    y_samp = []
    for index in sample_indices:
        X_samp = np.append(X_samp, [X[index]], axis=0)
        y_samp.append(y[index])

    return (X_samp, np.array(y_samp))

def sample_features(total_features, max_features):
    """
    Randomly samples a subset of features to consider for a split.
    
    In a random forest, this should be done without replacement at each node.
    
    Args:
        total_features (int): The total number of features available in the dataset.
        max_features (int): The number of features to randomly select.
        
    Returns:
        numpy.ndarray: A 1D array of randomly selected feature indices.
    """
    # TODO: student code here
    sample_indices = np.random.choice(total_features, max_features, replace=False)
    return sample_indices

def build_random_forest(X, y, n_trees=10, max_depth=None, max_features=None):
    """
    Builds a random forest ensemble.
    
    Args:
        X (numpy.ndarray): 2D array of features.
        y (numpy.ndarray): 1D array of class labels.
        n_trees (int): Number of trees in the forest.
        max_depth (int, optional): Maximum depth of each individual tree.
        max_features (int, optional): Number of features to evaluate at each split.
        
    Returns:
        list: A list of Node objects representing the root of each tree in the forest.
    """
    # TODO: student code here
    trees = []

    for i in range(n_trees):
        X_samp, y_samp = bootstrap_sample(X, y)
        root = build_tree(X_samp, y_samp, max_depth, 0, max_features)
        trees.append(root)

    return trees