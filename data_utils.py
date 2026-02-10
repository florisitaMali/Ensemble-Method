import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# function to load the dataset and prepare it for modeling
def load_and_prepare_data(csv_path, target_column):
    
    # read the csv file into a pandas dataframe
    df = pd.read_csv(csv_path)

    # separate features (X) from the target variable (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # encode the target labels into numeric values (for example 0 and 1)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # convert any categorical feature columns into numeric form
    for col in X.columns:
        if X[col].dtype == object:
            # apply label encoding to categorical columns
            X[col] = LabelEncoder().fit_transform(X[col])

    # standardize features so they have mean=0 and std=1
    # this helps many machine learning models perform better
    X = StandardScaler().fit_transform(X)

    # split the dataset into training and testing sets
    # test_size=0.25 means 25% for testing, 75% for training
    # random_state ensures reproducibility
    # stratify=y keeps class distribution balanced in both splits
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y), le
