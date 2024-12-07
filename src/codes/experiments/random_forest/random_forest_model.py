import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
from joblib import dump

import sys
import os
sys.path.append(os.path.abspath('../../'))

from scripts.data_processing import load_and_preprocess_data

data_path = '../../../data/raw/winequalityN.csv'
X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

clf = RandomForestClassifier(random_state= 42)

params = {
    'max_depth': range(1 , 30, 5),
    'n_estimators': range(1 , 25, 5),
    'min_samples_split': range(2 , 10 , 2)
}

# We use GridSearchCV to find the best hyperparameters for our model
grid_search = GridSearchCV(clf , param_grid= params , cv= 5 , n_jobs= -1, verbose= 1, scoring= 'accuracy')
grid_search.fit(X_train, y_train)
best = [grid_search.best_params_, grid_search.best_score_]

grid_search.best_params_
y_pred = grid_search.predict(X_test)
print(y_pred)

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

print(classification_report(y_test, grid_search.predict(X_test)))

dump(grid_search, 'random_forest_model.joblib')