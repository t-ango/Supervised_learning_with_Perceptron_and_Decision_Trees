"""
This module implements supervised learning experiments using Perceptron and Decision Tree models 
on the Palmer Penguins dataset. The experiments demonstrate binary classification and multi-run 
evaluation of model performance.

Features:
- **Data Preprocessing**: Reads and preprocesses the Palmer Penguins dataset, including z-score 
  normalization and binary label conversion.
- **Perceptron Model**: Implements a perceptron for binary classification with support for 
  training, prediction, and decision boundary visualization.
- **Decision Tree Model**: Constructs decision trees for classification, supports visualization, 
  and evaluates performance through multi-run experiments.
- **Experiments**:
  1. Binary classification using Perceptron (Gentoo vs others, Chinstrap vs others).
  2. Binary classification using Decision Tree (Gentoo vs others, Chinstrap vs others).
  3. Multi-run evaluation using Decision Tree with all features.

This module is designed for educational and demonstration purposes, highlighting supervised 
learning concepts and model evaluation techniques.
"""

import csv
import random as rnd
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats  # Used for "mode" - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html
from decision_tree_nodes import DecisionTreeBranchNode, DecisionTreeLeafNode
from matplotlib import lines
from numpy.typing import NDArray


#########################################
#   Data input / prediction evaluation
#########################################

def read_data() -> tuple[NDArray, NDArray]:
    """Read data with defined data types and encoding from CSV file, remove rows with missing data,
    and normalize using z-score.

    Returns
    -------
    X : NDArray
        Numpy array, shape (n_samples, 4), containing the normalized values 
        of the four numeric columns (bill length, bill depth, flipper length, body mass).
    y : NDArray
        Numpy array, shape (n_samples,), with integer labels representing species.
    """
    # Full path to the file
    file_path = "/Users/mango/UNI OFFLINE/AI/Oppgaver/g2_supervised_learning-main/palmer_penguins.csv"
    
    try:
        data = np.genfromtxt(
            file_path,
            delimiter=',',
            dtype=[('species', 'U20'), ('island', 'U20'), ('bill_length', 'f8'),
                   ('bill_depth', 'f8'), ('flipper_length', 'f8'), ('body_mass', 'f8'), 
                   ('sex', 'U20')],
            encoding='utf-8',
            skip_header=1,
            missing_values="NA",
            filling_values=np.nan)

        # Extract the numeric columns and species column
        species = np.array(data['species'])
        numeric_data = np.column_stack((
            data['bill_length'], data['bill_depth'], 
            data['flipper_length'], data['body_mass'] ))

        mask = ~np.isnan(numeric_data).any(axis=1)
        numeric_data = numeric_data[mask]
        species = species[mask]

        # Map species names to integer labels
        species_mapping = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
        y = np.vectorize(species_mapping.get)(species)

        X = (numeric_data - np.mean(numeric_data, axis=0)) / np.std(numeric_data, axis=0)

        return X, y

    except Exception as e:
        print(f"Error reading the data file: {e}")
        return None, None


def convert_y_to_binary(y: NDArray, y_value_true: int) -> NDArray:
    """Convert integer valued y to binary (0 or 1) valued vector by 
    creating a binary array: 1 where y equals y_value_true, otherwise 0.

    Parameters
    ----------
    y : NDArray
        Integer valued NumPy vector, shape (n_samples,)
    y_value_true : int
        Value of y which will be converted to 1 in output.
        All other values are converted to 0.

    Returns
    -------
    y_binary : NDArray
        Binary vector, shape (n_samples,)
        1 for values in y that are equal to y_value_true, 0 otherwise.
    """

    y_binary = np.where(y == y_value_true, 1, 0)
    return y_binary


def train_test_split(
    X: NDArray, y: NDArray, train_frac: float
) -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
    """Shuffle and split dataset into training and testing datasets.

    Parameters
    ----------
    X : NDArray
        Dataset, shape (n_samples, n_features)
    y : NDArray
        Values to be predicted, shape (n_samples,)
    train_frac : float
        Fraction of data to be used for training

    Returns
    -------
    (X_train, y_train) : tuple[NDArray, NDArray]
        Training dataset
    (X_test, y_test) : tuple[NDArray, NDArray]
        Test dataset
    """
    
    # Shuffle the indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Calculate the split index
    split_idx = int(len(X) * train_frac)
    
    # Split the data into training and testing sets
    X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
    y_train, y_test = y_shuffled[:split_idx], y_shuffled[split_idx:]
    
    return (X_train, y_train), (X_test, y_test)


def accuracy(y_pred: NDArray, y_true: NDArray) -> float:
    """Calculate accuracy of model based on predicted and true values.

    Parameters
    ----------
    y_pred : NDArray
        Numpy array with predicted values, shape (n_samples,)
    y_true : NDArray
        Numpy array with true values, shape (n_samples,)

    Returns
    -------
    accuracy : float
        Fraction of cases where the predicted values
        are equal to the true values. Number in range [0,1]
    """
    
    correct_predictions = np.sum(y_pred == y_true)
    accuracy = correct_predictions / len(y_true)
    
    return accuracy


##############################
#   Gini impurity functions
##############################


def gini_impurity(y: NDArray) -> float:
    """Calculate Gini impurity of a vector:
    1. Calculate the probabilities of each class
    2. Apply Gini impurity formula.

    Parameters
    ----------
    y : NDArray, integers
        1D NumPy array with class labels.

    Returns
    -------
    impurity : float
        Gini impurity, scalar in range [0,1).
    """

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    
    impurity = 1 - np.sum(probabilities ** 2)
    
    return impurity


def gini_impurity_reduction(y: NDArray, left_mask: NDArray) -> float:
    """Calculate the reduction in mean impurity from a binary split:
    1.Calculate original Gini impurity
    2.Split y into left and right subsets and calculate Gini impurity for each
    3.Calculate weighted mean impurity
    4.Calculate impurity reduction

    Parameters
    ----------
    y : NDArray
        1D numpy array with class labels.
    left_mask : NDArray
        1D numpy boolean array, True for "left" elements, False for "right".

    Returns
    -------
    impurity_reduction : float
        Reduction in mean Gini impurity, scalar in range [0, 0.5].
    """
    original_impurity = gini_impurity(y)
    
    y_left = y[left_mask]
    y_right = y[~left_mask]
    
    left_impurity = gini_impurity(y_left)
    right_impurity = gini_impurity(y_right)
    
    left_weight = len(y_left) / len(y)
    right_weight = len(y_right) / len(y)
    weighted_mean_impurity = left_weight * left_impurity + right_weight * right_impurity
    
    impurity_reduction = original_impurity - weighted_mean_impurity
    
    return impurity_reduction


def best_split_feature_value(X: NDArray, y: NDArray) -> tuple[float, int, float]:
    """Find feature and value "split" that yields highest impurity reduction.

    Parameters
    ----------
    X : NDArray
        NumPy feature matrix, shape (n_samples, n_features)
    y : NDArray
        NumPy class label vector, shape (n_samples,)

    Returns
    -------
    impurity_reduction : float
        Reduction in Gini impurity for best split.
        Zero if no split that reduces impurity exists.
    feature_index : int
        Index of X column with best feature for split.
    feature_value : float
        Value of feature in X yielding best split of y.
        Dataset is split using X[:,feature_index] <= feature_value.
    """
    best_impurity_reduction = 0.0
    best_feature_index = None
    best_feature_value = None

    # Loop over each feature
    n_features = X.shape[1]
    for feature_index in range(n_features):
        unique_values = np.unique(X[:, feature_index])
        
        # Evaluate each unique value as a potential split
        for value in unique_values:
            left_mask = X[:, feature_index] <= value
            impurity_reduction = gini_impurity_reduction(y, left_mask)
            
            # Check if this is the best split so far
            if impurity_reduction > best_impurity_reduction:
                best_impurity_reduction = impurity_reduction
                best_feature_index = feature_index
                best_feature_value = value

    return best_impurity_reduction, best_feature_index, best_feature_value


###################
#   Perceptron
###################


class Perceptron:
    """Perceptron model for classifying two classes.

    Attributes
    ----------
    weights : NDArray
        Array, shape (n_features,), with perceptron weights.
    bias : float
        Perceptron bias value.
    converged : bool | None
        Boolean indicating if Perceptron has converged during training.
        Set to None if Perceptron has not yet been trained.
    """

    def __init__(self):
        """Initialize perceptron with no weights or bias."""
        self.weights = None
        self.bias = 0.0
        self.converged = None

    def predict_single(self, x: NDArray) -> int:
        """Predict / calculate perceptron output for single observation / row x.

        Parameters
        ----------
        x : NDArray
            1D numpy array with shape (n_features,)

        Returns
        -------
        int
            Predicted class label (0 or 1)
        """
        # Compute the perceptron output
        linear_output = np.dot(self.weights, x) + self.bias
        return 1 if linear_output >= 0 else 0

    def predict(self, X: NDArray) -> NDArray:
        """Predict / calculate perceptron output for data matrix X.

        Parameters
        ----------
        X : NDArray
            2D numpy array, shape (n_samples, n_features)

        Returns
        -------
        NDArray
            Predicted class labels (0 or 1), shape (n_samples,)
        """
        # Vectorized prediction for all samples in X
        linear_outputs = np.dot(X, self.weights) + self.bias
        return (linear_outputs >= 0).astype(int)

    def train(self, X: NDArray, y: NDArray, learning_rate: float, max_epochs: int):
        """Fit perceptron to training data X with binary labels y.
        Comments in running code for step-by-step explanation.

        Parameters
        ----------
        X : NDArray
            2D numpy array with training samples, shape (n_samples, n_features)
        y : NDArray
            1D numpy array with binary labels, shape (n_samples,)
        learning_rate : float
            Learning rate for perceptron weight updates
        max_epochs : int
            Maximum number of epochs for training

        Notes
        -----
        If the perceptron converges before reaching `max_epochs`, training stops early.
        """
        n_samples, n_features = X.shape
        # Initialize weights and bias if not set
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.converged = False

        for epoch in range(max_epochs):
            errors = 0
            for i in range(n_samples):
                # Predict the class label
                prediction = self.predict_single(X[i])
                # Calculate error 
                error = y[i] - prediction
                if error != 0:
                    # Update weights and bias
                    self.weights += learning_rate * error * X[i]
                    self.bias += learning_rate * error
                    errors += 1
            # Check for convergence (no errors in this epoch)
            if errors == 0:
                self.converged = True
                break
        else:
            # If we exit the loop without breaking, it has not converged
            self.converged = False

    def decision_boundary_slope_intercept(self) -> tuple[float, float]:
        """Calculate slope and intercept for decision boundary line (2-feature data only).

        Returns
        -------
        tuple[float, float]
            Slope and intercept of decision boundary line.
            Returns (None, None) if data has more or less than 2 features.
        """
        if self.weights is None or len(self.weights) != 2:
            return None, None
        
        slope = -self.weights[0] / self.weights[1]
        intercept = -self.bias / self.weights[1]
        
        return slope, intercept


####################
#   Decision tree
####################


class DecisionTree:
    """Decision tree model for classification

    Attributes
    ----------
    _root: DecisionTreeBranchNode | None
        Root node in decision tree
    """

    def __init__(self):
        """Initialize decision tree"""
        self._root = None

    def __str__(self) -> str:
        """Return string representation of decision tree (based on binarytree.Node.__str__())"""
        if self._root is not None:
            return str(self._root)
        else:
            return "<Empty decision tree>"

    def train(self, X: NDArray, y: NDArray):
        """Train decision tree based on labelled dataset

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        y: NDArray, integers
            NumPy class label vector, shape (n_samples,)

        Notes
        -----
        Creates the decision tree by calling _build_tree() and setting
        the root node to the "top" DecisionTreeBranchNode.

        """
        self._root = self._build_tree(X, y)

    def _build_tree(self, X: NDArray, y: NDArray):
        """Recursively build decision tree

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        y: NDArray
            NumPy class label vector, shape (n_samples,)

        Notes
        -----
        - Determines the best possible binary split of the dataset. If no impurity
        reduction can be achieved, a leaf node is created, and its value is set to
        the most common class in y. If a split can achieve impurity reduction,
        a decision (branch) node is created, with left and right subtrees created by
        recursively calling _build_tree on the left and right subsets.

        """
        # Find best binary split of dataset
        impurity_reduction, feature_index, feature_value = best_split_feature_value(
            X, y
        )

        # If impurity can't be reduced further, create and return leaf node
        if impurity_reduction == 0:
            leaf_value = scipy.stats.mode(y, keepdims=False)[0]
            return DecisionTreeLeafNode(leaf_value)

        # If impurity _can_ be reduced, split dataset, build left and right
        # branches, and return branch node.
        else:
            left_mask = X[:, feature_index] <= feature_value
            left = self._build_tree(X[left_mask], y[left_mask])
            right = self._build_tree(X[~left_mask], y[~left_mask])
            return DecisionTreeBranchNode(feature_index, feature_value, left, right)

    def predict(self, X: NDArray):
        """Predict class (y vector) for feature matrix X

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)

        Returns
        -------
        y: NDArray, integers
            NumPy class label vector (predicted), shape (n_samples,)
        """
        if self._root is not None:
            return self._predict(X, self._root)
        else:
            raise ValueError("Decision tree root is None (not set)")

    def _predict(
        self, X: NDArray, node: Union["DecisionTreeBranchNode", "DecisionTreeLeafNode"]
    ) -> NDArray:
        """Predict class (y vector) for feature matrix X

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        node: "DecisionTreeBranchNode" or "DecisionTreeLeafNode"
            Node used to process the data. If the node is a leaf node,
            the data is classified with the value of the leaf node.
            If the node is a branch node, the data is split into left
            and right subsets, and classified by recursively calling
            _predict() on the left and right subsets.

        Returns
        -------
        y: NDArray
            NumPy class label vector (predicted), shape (n_samples,)

        Notes
        -----
        The prediction follows the following logic:

            if the node is a leaf node
                return y vector with all values equal to leaf node value
            else (the node is a branch node)
                split the dataset into left and right parts using node question
                predict classes for left and right datasets (using left and right branches)
                "stitch" predictions for left and right datasets into single y vector
                return y vector (length matching number of rows in X)
        """
        # If the current node is a leaf node, return the value (class label) for all samples
        if isinstance(node, DecisionTreeLeafNode):
            return np.full(X.shape[0], node.value)

        # If the current node is a branch node, recursively split the data and predict
        if isinstance(node, DecisionTreeBranchNode):
            # Split the data based on the current node's condition (feature index and value)
            left_mask = X[:, node.feature_index] <= node.feature_value
            right_mask = ~left_mask

            # Recursively predict for left and right branches
            left_predictions = self._predict(X[left_mask], node.left)
            right_predictions = self._predict(X[right_mask], node.right)

            # Combine the left and right predictions into a single array
            predictions = np.zeros(X.shape[0], dtype=int)
            predictions[left_mask] = left_predictions
            predictions[right_mask] = right_predictions

            return predictions

    
############
#   MAIN
############
    
import matplotlib.pyplot as plt
import numpy as np

def perceptron_experiment_1():
    """
    Conduct a binary classification experiment using a perceptron model on the Palmer Penguins dataset.

    This function performs the following steps:
    1. Reads and preprocesses the Palmer Penguins dataset.
    2. Splits the data into training and testing sets (80% training, 20% testing).
    3. Converts class labels to binary: Gentoo (1) and others (0).
    4. Selects two features (`bill_depth_mm` and `flipper_length_mm`) for training.
    5. Trains a perceptron model on the training data.
    6. Evaluates the perceptron on the test data and calculates accuracy.
    7. Visualizes the training data and plots the decision boundary of the perceptron.

    Notes
    -----
    - This experiment demonstrates the use of a perceptron for linear classification.
    - The decision boundary visualization provides insights into the model's classification behavior.

    Outputs
    -------
    - Prints the training and test set sizes.
    - Prints the accuracy of the perceptron model on the test set.
    - Displays a scatter plot of the training data with the perceptron's decision boundary.

    Returns
    -------
    None
    """
    X, y = read_data()

    train_frac = 0.8
    (X_train, y_train), (X_test, y_test) = train_test_split(X, y, train_frac)
    print("Training set size:", X_train.shape)
    print("Test set size:", X_test.shape)

    y_train_binary = convert_y_to_binary(y_train, y_value_true=2)  # 2 represents Gentoo
    y_test_binary = convert_y_to_binary(y_test, y_value_true=2)

    X_train_relevant = X_train[:, [1, 2]]  # bill_depth_mm (index 1), flipper_length_mm (index 2)
    X_test_relevant = X_test[:, [1, 2]]   

    perceptron = Perceptron()
    perceptron.train(X_train_relevant, y_train_binary, learning_rate=0.1, max_epochs=100)

    y_pred = perceptron.predict(X_test_relevant)

    model_accuracy = accuracy(y_pred, y_test_binary)
    print(f"Model Accuracy: {model_accuracy * 100:.2f}%")

  
    plt.figure(figsize=(8, 6))

    # Plot points where y_train_binary == 0 (Adelie and Chinstrap) with one color
    plt.scatter(X_train_relevant[y_train_binary == 0, 0], X_train_relevant[y_train_binary == 0, 1], color='blue', label='Adelie/Chinstrap', marker='o')

    # Plot points where y_train_binary == 1 (Gentoo) with another color
    plt.scatter(X_train_relevant[y_train_binary == 1, 0], X_train_relevant[y_train_binary == 1, 1], color='red', label='Gentoo', marker='x')

    slope, intercept = perceptron.decision_boundary_slope_intercept()

    # Create an array of x values (for plotting the line)
    x_vals = np.linspace(min(X_train_relevant[:, 0]), max(X_train_relevant[:, 0]), 100)

    # Calculate the corresponding y values for the decision boundary line
    y_vals = slope * x_vals + intercept

    plt.plot(x_vals, y_vals, color='green', label='Decision Boundary')

    plt.xlabel('Bill Depth (mm)')
    plt.ylabel('Flipper Length (mm)')
    plt.title('Perceptron Decision Boundary')
    plt.legend()

    plt.grid(True)
    plt.show()


def perceptron_experiment_2():
    """
    Conducts the second perceptron experiment on the Palmer Penguins dataset to classify Chinstrap penguins
    (class 1) from all other species (class 0). 

    This function performs the following steps:
    1. Reads the Palmer Penguins data.
    2. Splits the data into training (80%) and testing (20%) sets.
    3. Converts the target labels to binary (Chinstrap vs. others).
    4. Extracts relevant features for training (bill length and bill depth).
    5. Trains a Perceptron model on the training data.
    6. Checks for convergence and reports if the model did not converge.
    7. Makes predictions on the test data.
    8. Calculates and prints the model's accuracy.
    9. Plots the training data with the decision boundary.

    Differences from Experiment 1:
    - Focuses on classifying Chinstrap penguins (1) vs. non-Chinstrap (0).
    - Uses different features (bill length and bill depth).
    - Includes a convergence check to handle cases where the perceptron may fail to converge.

    Returns
    -------
    None
    """

    X, y = read_data()

    train_frac = 0.8
    (X_train, y_train), (X_test, y_test) = train_test_split(X, y, train_frac)
    print("Training set size:", X_train.shape)
    print("Test set size:", X_test.shape)

    y_train_binary = convert_y_to_binary(y_train, y_value_true=1)  # 1 represents Chinstrap
    y_test_binary = convert_y_to_binary(y_test, y_value_true=1)

    X_train_relevant = X_train[:, [0, 1]]  # bill_length_mm (index 0), bill_depth_mm (index 1)
    X_test_relevant = X_test[:, [0, 1]]   

    perceptron = Perceptron()
    perceptron.train(X_train_relevant, y_train_binary, learning_rate=0.1, max_epochs=100)

    if perceptron.converged is None or not perceptron.converged:
        print("Perceptron did not converge. This may be because the data is not linearly separable.")

    y_pred = perceptron.predict(X_test_relevant)

    model_accuracy = accuracy(y_pred, y_test_binary)
    print(f"Model Accuracy: {model_accuracy * 100:.2f}%")


    plt.figure(figsize=(8, 6))

    # Plot points where y_train_binary == 0 (Not Chinstrap) with one color
    plt.scatter(X_train_relevant[y_train_binary == 0, 0], X_train_relevant[y_train_binary == 0, 1], color='blue', label='Not Chinstrap', marker='o')

    # Plot points where y_train_binary == 1 (Chinstrap) with another color
    plt.scatter(X_train_relevant[y_train_binary == 1, 0], X_train_relevant[y_train_binary == 1, 1], color='red', label='Chinstrap', marker='x')

    slope, intercept = perceptron.decision_boundary_slope_intercept()
    x_vals = np.linspace(min(X_train_relevant[:, 0]), max(X_train_relevant[:, 0]), 100)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color='green', label='Decision Boundary')

    plt.xlabel('Bill Length (mm)')
    plt.ylabel('Bill Depth (mm)')
    plt.title('Perceptron Decision Boundary (Chinstrap vs Other)')
    plt.legend()

    plt.grid(True)
    plt.show()


def decision_tree_experiment_1():
    """
    Performs an experiment using a decision tree classifier to classify Gentoo penguins
    from all other species in the Palmer Penguins dataset. The experiment includes data 
    preparation, model training, accuracy evaluation, decision tree visualization, and 
    decision boundary plotting.

    Steps:
    1. Loads the Palmer Penguins dataset.
    2. Splits the data into training (80%) and testing (20%) sets.
    3. Converts target labels to binary (Gentoo -> 1, others -> 0).
    4. Selects relevant features (bill depth and flipper length) for training.
    5. Trains a decision tree classifier on the training data.
    6. Predicts labels for the test set and calculates the accuracy.
    7. Prints the decision tree structure (root, branches, and leaves).
    8. Plots the decision boundary and visualizes the decision regions along with training data.

    Outputs:
    - Model accuracy (printed).
    - Visual representation of the decision tree structure.
    - Contour plot showing decision boundaries with training data points.

    Notes:
    - This experiment focuses on classifying Gentoo penguins (class 1) versus all other species (class 0).
    - The model's decision boundary is visualized in a 2D plot based on the selected features.
    """
    
    X, y = read_data()

    train_frac = 0.8
    (X_train, y_train), (X_test, y_test) = train_test_split(X, y, train_frac)

    y_train_binary = convert_y_to_binary(y_train, y_value_true=2)  # 2 represents Gentoo
    y_test_binary = convert_y_to_binary(y_test, y_value_true=2)

    X_train_relevant = X_train[:, [1, 2]]  # bill_depth_mm and flipper_length_mm
    X_test_relevant = X_test[:, [1, 2]]

    decision_tree = DecisionTree()
    decision_tree.train(X_train_relevant, y_train_binary)

    y_pred = decision_tree.predict(X_test_relevant)

    model_accuracy = accuracy(y_pred, y_test_binary)
    print(f"Model Accuracy: {model_accuracy * 100:.2f}%")

    print("Visualizing Decision Tree experiment 1:")
    print(decision_tree)

    # Plot the decision boundary
    # Generate a grid of points to classify
    x_min, x_max = X_train_relevant[:, 0].min() - 1, X_train_relevant[:, 0].max() + 1
    y_min, y_max = X_train_relevant[:, 1].min() - 1, X_train_relevant[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Use the decision tree to predict the class for each point in the grid
    Z = decision_tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    #Plot the data
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

    plt.scatter(X_train_relevant[y_train_binary == 0, 0], X_train_relevant[y_train_binary == 0, 1], color='blue', label='Non-Gentoo', marker='o')
    plt.scatter(X_train_relevant[y_train_binary == 1, 0], X_train_relevant[y_train_binary == 1, 1], color='red', label='Gentoo', marker='x')

    plt.xlabel('Bill Depth (mm)')
    plt.ylabel('Flipper Length (mm)')
    plt.title('Decision Tree Decision Boundary')
    plt.legend()

    plt.grid(True)
    plt.show()


def decision_tree_experiment_2():
    """
    Conducts an experiment using a Decision Tree classifier to classify Chinstrap penguins 
    from the Palmer Penguins dataset based on selected features.

    The experiment follows these steps:
    1. Loads the Palmer Penguins dataset and splits it into training and testing sets (80% train, 20% test).
    2. Converts the labels to binary, with Chinstrap penguins labeled as 1 and all other species as 0.
    3. Extracts the relevant features (bill length and bill depth) for training the model.
    4. Trains a Decision Tree model on the training data.
    5. Evaluates the model by calculating and printing its accuracy on the test set.
    6. Visualizes the structure of the trained decision tree.
    7. Plots the decision boundary of the classifier over the feature space, showing how the model separates Chinstrap and non-Chinstrap species.

    Returns
    -------
    None
    """
    
    X, y = read_data()

    train_frac = 0.8
    (X_train, y_train), (X_test, y_test) = train_test_split(X, y, train_frac)

    y_train_binary = convert_y_to_binary(y_train, y_value_true=1)  # 1 represents Chinstrap
    y_test_binary = convert_y_to_binary(y_test, y_value_true=1)

    X_train_relevant = X_train[:, [0, 1]]  # bill_length_mm and bill_depth_mm
    X_test_relevant = X_test[:, [0, 1]]

    decision_tree = DecisionTree()
    decision_tree.train(X_train_relevant, y_train_binary)

    y_pred = decision_tree.predict(X_test_relevant)

    model_accuracy = accuracy(y_pred, y_test_binary)
    print(f"Decision Tree Model Accuracy: {model_accuracy * 100:.2f}%")


    print("Visualizing Decision Tree experiment 2:")
    print(decision_tree)

    # Step 10: Plot the decision boundary
    # Generate a grid of points to classify
    x_min, x_max = X_train_relevant[:, 0].min() - 1, X_train_relevant[:, 0].max() + 1
    y_min, y_max = X_train_relevant[:, 1].min() - 1, X_train_relevant[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # Use the decision tree to predict the class for each point in the grid
    Z = decision_tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create a plot showing the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

    # Plot the training data
    plt.scatter(X_train_relevant[y_train_binary == 0, 0], X_train_relevant[y_train_binary == 0, 1], color='blue', label='Non-Chinstrap', marker='o')
    plt.scatter(X_train_relevant[y_train_binary == 1, 0], X_train_relevant[y_train_binary == 1, 1], color='red', label='Chinstrap', marker='x')

    plt.xlabel('Bill Length (mm)')
    plt.ylabel('Bill Depth (mm)')
    plt.title('Decision Tree Decision Boundary - Chinstrap vs Other Species')
    plt.legend()

    plt.grid(True)
    plt.show()


def decision_tree_experiment_3(num_runs=3):
    """
    Perform a decision tree classification experiment with multiple runs to evaluate model stability and performance.
    This experiment involves the following steps for each run:
    1. The Palmer Penguins dataset is loaded and shuffled.
    2. The data is split into training (80%) and testing (20%) sets.
    3. A Decision Tree classifier is trained using all four features (bill length, bill depth, flipper length, and body mass).
    4. Predictions are made on the test set, and the accuracy is calculated.
    5. The accuracy for each run is printed and stored.
    6. After the specified number of runs, statistics (mean, standard deviation, min, max) are computed and displayed.
    7. The decision tree structure from the final run is visualized.

    Args:
    num_runs (int): The number of runs to perform. Default is 3.

    Returns:
    None
    """
    accuracies = []
    decision_tree = None 
        
    for run in range(num_runs):
        X, y = read_data()

        # Shuffle the data
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        train_frac = 0.8
        (X_train, y_train), (X_test, y_test) = train_test_split(X, y, train_frac)
        
        #train the tree using all 4 features
        decision_tree = DecisionTree()
        decision_tree.train(X_train, y_train)

        y_pred = decision_tree.predict(X_test)
        accuracy_val = accuracy(y_pred, y_test)
        accuracies.append(accuracy_val)
        print(f"Run {run + 1}: Accuracy = {accuracy_val * 100:.2f}%")

    accuracies = np.array(accuracies)
    print("\nStatistics after", num_runs, "runs:")
    print(f"Fraction: {train_frac}")
    print(f"Mean Accuracy: {accuracies.mean() * 100:.2f}%")
    print(f"Standard Deviation: {accuracies.std() * 100:.2f}%")
    print(f"Min Accuracy: {accuracies.min() * 100:.2f}%")
    print(f"Max Accuracy: {accuracies.max() * 100:.2f}%")

    print("\nVisualizing the Decision Tree structure from the last run:")
    print(decision_tree)


if __name__ == "__main__":
    
    # Demonstrate your code / solutions here.
    # Experiments are implemented as separate functions that are called here; 
    # Functions are using the same plotter window, simultaneous running not advisable. Uncomment the desired function. 

    perceptron_experiment_1()
    #perceptron_experiment_2()

    #decision_tree_experiment_1()
    #decision_tree_experiment_2()
    #decision_tree_experiment_3(10)












