import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path, features_to_drop=None, test_size=0.2, random_state=42):
    """
    Preprocess the dataset with configurable feature dropping.

    Args:
        file_path (str): Path to the data file
        features_to_drop (list): List of feature names to drop (excluding 'type' and 'quality' which are always handled separately)
        test_size (float): Fraction of data to use for the test set
        random_state (int): Random seed for reproducibility
    Returns:
        X_train, X_test, y_train, y_test: Train/test split of features and target
    """
    df = pd.read_csv(file_path)
    df = df[df['type'] == 'white']
    df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

    # Mapping the quality to low, medium and high
    quality_mapping = {3: "Low", 4: "Low", 5: "Medium", 
                      6: "Medium", 7: "Medium", 8: "High", 9: "High"}
    df["quality"] = df["quality"].map(quality_mapping)

    # Always drop 'type' and 'quality', then drop additional features if specified
    features_to_drop = features_to_drop or []
    all_features_to_drop = ['type', 'quality'] + features_to_drop
    
    X = df.drop(all_features_to_drop, axis=1)
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                       random_state=random_state)
    
    return X_train, X_test, y_train, y_test