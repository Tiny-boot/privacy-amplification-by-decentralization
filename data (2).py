from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler
import numpy as np

def load():
    # Load dataset
    X, y = fetch_openml('Cardiovascular-Disease-dataset', version=1, return_X_y=True, as_frame=False)
    
    # Binarize the labels for binary classification (e.g., classify digit 0 vs. digit 1)
    y = y.astype(int)
    y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1 for binary classification

    # Standardize the features
    standardizer = StandardScaler()
    X = standardizer.fit_transform(X)

    # Normalize the data
    normalizer = Normalizer()
    X = normalizer.transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test