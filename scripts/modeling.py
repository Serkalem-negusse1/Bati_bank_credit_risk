import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Function to save the model as a pickle file
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

# Function to visualize feature importance (for Random Forest, Gradient Boosting)
def plot_feature_importance(model, feature_names):
    if isinstance(feature_names, np.ndarray):  # Handle NumPy array
        feature_names = [f"Feature {i}" for i in range(len(feature_names))]
    elif hasattr(feature_names, "columns"):  # Handle DataFrame
        feature_names = feature_names.columns.tolist()
        
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.show()

# Function to visualize the confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Function to plot the ROC Curve
def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

# Function to perform GridSearchCV for hyperparameter tuning
def perform_grid_search(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_}")
    return grid_search.best_estimator_

# Function to train and evaluate different models with hyperparameter tuning
def train_and_evaluate_models_with_tuning(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True, False]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.5],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    }

    # Compute class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    for name, model in models.items():
        print(f"Training and tuning {name}...")

        # If using RandomForest or GradientBoosting, pass class weights
        if isinstance(model, RandomForestClassifier):
            model = RandomForestClassifier(class_weight=class_weight_dict)
        elif isinstance(model, GradientBoostingClassifier):
            model = GradientBoostingClassifier()

        # Perform GridSearchCV
        best_model = perform_grid_search(model, param_grids[name], X_train, y_train)

        # Get predictions and probabilities
        y_prob = best_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC
        y_pred = (y_prob >= 0.5).astype(int)  # Convert probabilities to binary labels

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=1)
        recall = recall_score(y_test, y_pred, zero_division=1)
        f1 = f1_score(y_test, y_pred, zero_division=1)
        roc_auc = roc_auc_score(y_test, y_prob)

        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"{name} Precision: {precision:.4f}")
        print(f"{name} Recall: {recall:.4f}")
        print(f"{name} F1 Score: {f1:.4f}")
        print(f"{name} ROC-AUC: {roc_auc:.4f}")

        # Visualizations
        plot_confusion_matrix(y_test, y_pred)
        plot_roc_curve(y_test, y_prob)

        # Feature importance visualization for applicable models
        if hasattr(best_model, "feature_importances_"):
            plot_feature_importance(
                best_model, 
                X_train if isinstance(X_train, pd.DataFrame) else np.arange(X_train.shape[1])
            )
        
        # Save the trained model as a pickle file
        save_model(best_model, f"{name.replace(' ', '_').lower()}_model.pkl")
        print("\n")
