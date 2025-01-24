from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_model(X_train, y_train):
    """Train a Random Forest Classifier."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
