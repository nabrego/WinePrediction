import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
import graphviz
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

import sys
import os
sys.path.append(os.path.abspath('../../../src'))

from scripts.data_processing import load_and_preprocess_data
data_path = '../../../data/raw/winequalityN.csv'

X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

top_Features = ['alcohol', 'volatile acidity', 'free sulfur dioxide','citric acid']
X_drop = X_train.drop(columns = top_Features)
X_test = X_test.drop(columns = top_Features)

# grab the path total leaf impurities and then grab the alphas
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clf_list = []


for ccp_alpha in ccp_alphas:
    Decision = DecisionTreeClassifier(max_depth=10,min_samples_leaf=1,min_samples_split=2 ,ccp_alpha = ccp_alpha, random_state=42)
    Decision.fit(X_train,y_train)
    clf_list.append(Decision)
    


train_scores = [clf.score(X_train, y_train) for clf in clf_list]
test_scores = [clf.score(X_test, y_test) for clf in clf_list]

plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, marker='o', label='Training Accuracy', drawstyle="steps-post")
plt.plot(ccp_alphas, test_scores, marker='o', label='Testing Accuracy', drawstyle="steps-post")
plt.xlabel("Effective Alpha (ccp_alpha)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Effective Alpha for Training and Testing Sets")
plt.legend()
plt.grid(True)
plt.show()

optimal_index = test_scores.index(max(test_scores))
optimal_alpha = ccp_alphas[optimal_index]
print(f"Optimal ccp_alpha: {optimal_alpha}")


pruned_tree = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    ccp_alpha=optimal_alpha,
    random_state=42
)
pruned_tree.fit(X_train, y_train)

y_pred_pruned = pruned_tree.predict(X_test)

# Accuracy
accuracy_pruned = accuracy_score(y_test, y_pred_pruned)
print("Pruned Tree Accuracy:", accuracy_pruned)

# Classification Report
print("\nPruned Tree Classification Report:\n", classification_report(y_test, y_pred_pruned))


data_graph = tree.export_graphviz(
    pruned_tree, 
    out_file=None, 
    feature_names=X_drop.columns,  
    class_names=[str(i) for i in sorted(y.unique())],  
    filled=True, 
    rounded=True,  
    special_characters=True,
    precision=2
)

# Generate the Graphviz visualization
graph = graphviz.Source(data_graph)
graph