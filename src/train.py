import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import load_data, clean_data, split_data, scale_data

def train_model():
    df = load_data("data/diabetes.csv")
    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, scaler = scale_data(X_train, X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    pickle.dump(model, open("models/model.pkl", "wb"))
    pickle.dump(scaler, open("models/scaler.pkl", "wb"))

    print("\nModel Trained & Saved Successfully")

if __name__ == "__main__":
    train_model()