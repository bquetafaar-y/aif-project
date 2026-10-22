from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from numpy.typing import ArrayLike
from joblib import dump
import pandas as pd


def train_classification_model(X_train: ArrayLike,y_train: ArrayLike) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model

def save_classification_model(model: RandomForestClassifier,filename: str = "random_forest_model.joblib"):
    dump(model, filename)

def evaluate_classification_model(model: RandomForestClassifier,X_test: ArrayLike,y_test: ArrayLike):
    y_pred = model.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    print(f"Accuracy score: {acc_score:.4f}")

def save_initial_datasets(X: ArrayLike, y: ArrayLike):
    dump(X, "X.joblib")
    dump(y, "y.joblib")


if __name__ == "__main__":
    # Prepare the dataset
    df = pd.read_csv("dataset.csv")

    X = df.drop(columns=["applicant_id", "chose_lnu"])
    y = df["chose_lnu"]

    X = pd.get_dummies(X)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_classification_model(X_train, y_train)

    # Evaluate the model
    evaluate_classification_model(model, X_test, y_test)

    # Save the model
    save_classification_model(model)

    # Save datasets for the Streamlit app
    save_initial_datasets(X, y)
