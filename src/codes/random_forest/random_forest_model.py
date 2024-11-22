import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report

clf = RandomForestClassifier(random_state= 42)

params = {
    'max_depth': range(10 , 60 , 10),
    'n_estimators': range(25 , 100 , 25),
    'min_samples_split': range(2 , 10 , 2)
}

grid_search = GridSearchCV(clf , param_grid= params , cv= 5 , n_jobs= -1, verbose= 1, scoring= 'accuracy')

