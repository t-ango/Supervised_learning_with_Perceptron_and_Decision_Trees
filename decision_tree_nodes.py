"""Branch and leaf node classes for building decision trees

This module contains two classes which inherit from binarytree.Node;
DecisionTreeBranchNode and DecisionTreeLeafNode. The classes are used
for implementing the DecisionTree class.

The main benefit of inheriting from binarytree.Node in this context is
that it provides a useful method for visualizing the tree -
the __str__() method (invoked by calling str(root_node)).

There is no need to change the code below, unless you want to add
or change existing functionality.

"""

from __future__ import annotations

from typing import Union

import binarytree


class DecisionTreeBranchNode(binarytree.Node):
    def __init__(
        self,
        feature_index: int,
        feature_value: float,
        left=Union["DecisionTreeBranchNode", "DecisionTreeLeafNode"],
        right=Union["DecisionTreeBranchNode", "DecisionTreeLeafNode"],
    ):
        """Initialize decision node

        Parameters
        ----------
        feature_index: int
            Index of X column used in question
        feature_value: float
            Value of feature used in question
        left: DecisionTreeBranchNode or DecisionTreeLeafNode
            Node, root of left subtree
        right: DecisionTreeBranchNode or DecisionTreeLeafNode
            Node, root of right subtree

        Notes
        -----
        - DecisionTreeBranchNode is a subclass of binarytree.Node. This
        has the advantage of inheriting useful methods for general binary
        trees, e.g. visualization through the __str__ method.
        - Each decision node corresponds to a question of the form
        "is feature x <= value y". The features and values are stored as
        attributes "feature_index" and "feature_value".
        - A string representation of the question is saved in the node's
        "value" attribute.
        """
        question_string = f"f{feature_index} <= {feature_value:.3g}"  # "General" format - fixed point/scientific
        super().__init__(value=question_string, left=left, right=right)  # type: ignore
        self.feature_index = feature_index
        self.feature_value = feature_value


class DecisionTreeLeafNode(binarytree.Node):
    def __init__(self, y_value: Union[int, str]):
        """Initialize leaf node

        Parameters
        ----------
        y_value: int or string
            class in dataset (e.g. integer or string) represented by leaf

        Notes
        -----
        The attribute "value" is set to the string representation of the value,
        to be used for visualization. The numeric value is stored in the attribute
        "y_value".
        """
        super().__init__(str(y_value))
        self.y_value = y_value
