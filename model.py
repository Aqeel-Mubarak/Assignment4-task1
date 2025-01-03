# model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_model():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, "iris_model.pkl")
    print("Model trained and saved as iris_model.pkl")

if __name__ == "__main__":
    train_model()
