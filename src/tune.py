from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_data, clean_data, split_data, scale_data

def tune_model():
    df = load_data("data/diabetes.csv")
    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test, scaler = scale_data(X_train, X_test)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='recall'
    )

    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    print("Best Score:", grid.best_score_)

if __name__ == "__main__":
    tune_model()