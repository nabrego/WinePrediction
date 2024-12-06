import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path, test_size=0.2, random_state=42):
    """
    Preprocess the dataset: read from a file, split into features and target, scale features, and split into train/test sets.

    Args:
        data (pd.DataFrame): Input dataset as a Pandas DataFrame.
        target_column (str): Name of the target column.
        test_size (float): Fraction of data to use for the test set.
        random_state (int): Random seed for reproducibility.
    Returns:
        X_train, X_test, y_train, y_test: Train/test split of features and target.
    """
    df = pd.read_csv(file_path)
    df = df[df['type'] == 'white']
    df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

    # Mapping the quality to low, medium and high
    quaity_mapping = { 3 : "Low",4 : "Low",5: "Medium",6 : "Medium",7: "Medium",8 : "High",9 : "High"}
    df["quality"] =  df["quality"].map(quaity_mapping)

    X = df.drop(['type', 'density','quality'], axis=1)
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test