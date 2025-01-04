# Supervised_learning_with_Perceptron_and_Decision_Trees

This repository showcases supervised learning experiments using **Perceptron** and **Decision Tree** models applied to the [Palmer Penguins Dataset](https://allisonhorst.github.io/palmerpenguins/). The project highlights binary classification, decision boundary visualization, and multi-run evaluation for model performance.

---

## Features

- **Data Preprocessing**:
  - Reads and preprocesses the dataset, including handling missing values and z-score normalization.
  - Maps penguin species to integer labels for classification.

- **Perceptron Model**:
  - Binary classification with adjustable learning rate and epochs.
  - Decision boundary visualization for selected features.

- **Decision Tree Model**:
  - Classification based on binary splits with Gini impurity reduction.
  - Supports multi-run evaluation with performance statistics.

- **Experiments**:
  1. **Perceptron Binary Classification**:
     - Classifies Gentoo vs others and Chinstrap vs others.
     - Visualizes decision boundaries.
  2. **Decision Tree Binary Classification**:
     - Similar to Perceptron experiments but uses decision trees.
  3. **Multi-Run Decision Tree Evaluation**:
     - Assesses model stability and performance with all features.

---

## Project Structure

. 
├── decision_tree_nodes.py # Contains tree node classes for the Decision Tree model 
├── Supervised_learning1.py # Main program with Perceptron and Decision Tree experiments 
├── palmer_penguins.csv # Dataset (Palmer Penguins) 
├── README.md # Documentation for the repository


---

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries:
  - NumPy
  - SciPy
  - Matplotlib

Install dependencies using:
```bash
pip install numpy scipy matplotlib
Usage

Clone the repository:
git clone https://github.com/your_username/supervised-learning.git
cd supervised-learning

Run experiments:
Open Supervised_learning1.py and uncomment the desired experiment function.

Execute the script:
python Supervised_learning1.py

Available experiments:
perceptron_experiment_1(): Classify Gentoo vs others with Perceptron.
perceptron_experiment_2(): Classify Chinstrap vs others with Perceptron.
decision_tree_experiment_1(): Classify Gentoo vs others with Decision Tree.
decision_tree_experiment_2(): Classify Chinstrap vs others with Decision Tree.
decision_tree_experiment_3(num_runs): Multi-run Decision Tree evaluation.

Example Outputs

Perceptron Classification
Accuracy: 92.5%

Decision Tree Classification
Accuracy: 94.8%


Future Enhancements

Add support for more datasets.
Extend Decision Tree to support multi-class classification.
Implement additional supervised learning models (e.g., SVM, Random Forest).

License

This project is licensed under the MIT License.
