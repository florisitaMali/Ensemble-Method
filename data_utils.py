import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_prepare_data(csv_path, target_column):
    df = pd.read_csv(csv_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    le = LabelEncoder()
    y = le.fit_transform(y)

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = LabelEncoder().fit_transform(X[col])

    X = StandardScaler().fit_transform(X)

    return train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    ), le
