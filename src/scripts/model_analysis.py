import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Add this import


def get_model_metrics(model, X_test, y_test):
    """Calculate performance metrics for a model."""
    y_pred = model.predict(X_test)
    return [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, average='weighted'),
        recall_score(y_test, y_pred, average='weighted'),
        f1_score(y_test, y_pred, average='weighted')
    ]

def analyze_feature_impact(model_class, X_train, X_test, y_train, y_test, features_to_analyze, model_params=None):
    """Analyze impact of removing different features."""
    metrics_scores = {}
    
    # Apply scaling if the model is SVM
    if model_class == SVC:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    else:
        X_train_df = X_train
        X_test_df = X_test
    
    # Baseline - train model with all features
    baseline_model = model_class(**(model_params or {}))
    baseline_model.fit(X_train_df, y_train)
    metrics_scores["None"] = get_model_metrics(baseline_model, X_test_df, y_test)
    
    # For each feature, train a new model without that feature
    for feature in features_to_analyze:
        # Create new training and test sets without the feature
        X_train_mod = X_train_df.drop(columns=[feature])
        X_test_mod = X_test_df.drop(columns=[feature])
        
        # Train a new model on the modified dataset
        model = model_class(**(model_params or {}))
        model.fit(X_train_mod, y_train)
        
        # Get metrics for this model
        metrics_scores[feature] = get_model_metrics(model, X_test_mod, y_test)
    
    return metrics_scores

def plot_feature_impact(metrics_scores, title):
    """Create bar plot comparing metrics across different feature removals."""
    features = list(metrics_scores.keys())
    metrics = np.array(list(metrics_scores.values()))
    
    # Metrics names for x-axis
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    # Set up the bar plot
    x = np.arange(len(metric_names))
    width = 0.2  # Width of bars
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#a8dadc',   # Light blue
              '#e9c46a',   # Muted yellow
              '#2a9d8f',   # Teal
              '#f4a261']   # Soft orange
    
    # Calculate positions for bars
    positions = [x + (i - (len(features)-1)/2) * width for i in range(len(features))]
    
    # Plot bars for each feature
    for i, (feature, scores) in enumerate(metrics_scores.items()):
        bars = ax.bar(positions[i], 
                     scores, 
                     width, 
                     label=feature if feature != "None" else "No Feature Removed",
                     color=colors[i])
        
        # Add value labels on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + 0.001,
                   f'{height:.3f}',
                   ha='center', 
                   va='bottom',
                   fontsize=9)
    
    # Customize the plot
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Scores', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis ticks
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    
    # Set y-axis limits to focus on the relevant range
    all_values = metrics.flatten()
    min_val = np.min(all_values) - 0.01
    max_val = np.max(all_values) + 0.01
    ax.set_ylim(min_val, max_val)
    
    # Set y-axis ticks at appropriate intervals
    tick_range = np.arange(np.floor(min_val*100)/100, 
                          np.ceil(max_val*100)/100, 
                          0.01)
    ax.set_yticks(tick_range)
    
    # Add grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Customize legend
    ax.legend(title='Features Removed',
             bbox_to_anchor=(1.05, 1),
             loc='upper left',
             fontsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig


def compare_label_schemes(model_class, data_path, model_params=None):
    """Compare model performance between original and consolidated labels."""
    # Load data with original labels
    df = pd.read_csv(data_path)
    df = df[df['type'] == 'white']
    df.fillna(df.select_dtypes(include='number').mean(), inplace=True)
    
    X = df.drop(['type', 'quality', 'density'], axis=1)
    y_orig = df['quality']
    
    # Create consolidated labels
    quality_mapping = {3: "Low", 4: "Low", 5: "Medium", 
                      6: "Medium", 7: "Medium", 8: "High", 9: "High"}
    y_cons = y_orig.map(quality_mapping)
    
    results = {}
    for label_type, y in [("Original", y_orig), ("Consolidated", y_cons)]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = model_class(**(model_params or {}))
        model.fit(X_train, y_train)
        results[label_type] = get_model_metrics(model, X_test, y_test)
    
    return results