import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
from joblib import dump

df = pd.read_csv('../../../data/raw/winequalityN.csv')
df = df[df['type'] == 'white']
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

# Mapping the quality to low, medium and high
quaity_mapping = { 3 : "Low",4 : "Low",5: "Medium",6 : "Medium",7: "Medium",8 : "High",9 : "High"}
df["quality"] =  df["quality"].map(quaity_mapping)

X = df.drop(['type', 'density','quality'], axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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