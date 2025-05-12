# metrics.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,    
    roc_curve,
    auc,
    accuracy_score,
    f1_score,
    roc_auc_score
)
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
import graphviz

import html
from IPython.display import display

from sklearn.model_selection import GridSearchCV
import time

import re


# --------------------------------------------
# Model metrics visualisations
# --------------------------------------------

# Plot confusion matrix
def plot_confusion(y_true, y_pred, model_name="Model"):
    """
    Plot confusion matrix
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix ({model_name})")
    plt.grid(False)
    plt.show()

# Plot ROC curve and AUC
def plot_roc_curve(y_true, y_proba, model_name="Model"):
    """
    Plot ROC curve and AUC
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name of the model
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({model_name})')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Plot feature importance
def plot_feature_importance(model, X_train, top_n=15, model_name="Model"):
    """
    Plot feature importance
    Args:
        model: Trained model
        X_train: Training features
        top_n: Number of top features to plot
        model_name: Name of the model
    """
    if hasattr(model, "coef_"):
        importances = pd.Series(model.coef_[0], index=X_train.columns)
    elif hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
    else:
        raise ValueError("Model does not have feature importances or coefficients.")

    top_features = importances.abs().sort_values(ascending=False).head(top_n)
    top_features.plot(kind='barh', figsize=(8, 6))
    plt.title(f"Top {top_n} Important Features ({model_name})")
    plt.xlabel("Importance")
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.show()

# Plot predictions vs true values
def plot_predictions_vs_truth(y_true, y_pred, model_name="Model"):
    """
    Plot predictions vs true values
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
    """
    df = pd.DataFrame({
        "True": y_true.reset_index(drop=True),
        "Predicted": y_pred
    })
    df["Match"] = df["True"] == df["Predicted"]

    plt.figure(figsize=(10, 3))
    plt.scatter(df.index, df["True"], label="True", c="blue", alpha=0.5)
    plt.scatter(df.index, df["Predicted"], label="Predicted", c="orange", marker="x", alpha=0.6)
    plt.title(f"Predictions vs True Values ({model_name})")
    plt.xlabel("Sample Index")
    plt.yticks([0, 1])
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot prediction errors (False Positives and False Negatives)
def plot_prediction_errors(y_true, y_pred, model_name="Model"):
    """
    Plot prediction errors (False Positives and False Negatives)
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
    """
    df = pd.DataFrame({
        "True": y_true.reset_index(drop=True),
        "Pred": y_pred
    })
    df["Error"] = df["True"] - df["Pred"]

    plt.figure(figsize=(10, 3))
    plt.plot(df["Error"].values, marker='o', linestyle='', color='red', alpha=0.5)
    plt.yticks([-1, 0, 1], labels=["FP", "Correct", "FN"])
    plt.title(f"Classification Errors (True - Pred) — {model_name}")
    plt.xlabel("Sample Index")
    plt.grid(True)
    plt.show()

def plot_error_summary(y_true, y_pred, model_name="Model"):
    """
    Visualizes counts of TP, TN, FP, FN for binary classification.
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
    """
    df = pd.DataFrame({"True": y_true, "Pred": y_pred})
    df["Error Type"] = df.apply(
        lambda row: "TP" if row["True"] == 1 and row["Pred"] == 1 else
                    "TN" if row["True"] == 0 and row["Pred"] == 0 else
                    "FP" if row["True"] == 0 and row["Pred"] == 1 else
                    "FN", axis=1
    )

    error_counts = df["Error Type"].value_counts().reindex(["TP", "TN", "FP", "FN"], fill_value=0)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=error_counts.index, y=error_counts.values, palette="pastel")
    plt.title(f"Error Summary ({model_name})")
    plt.xlabel("Error Type")
    plt.ylabel("Count")
    plt.grid(True, axis="y")
    plt.show()

# --------------------------------------------
# Model tree visualisations
# --------------------------------------------

def plot_decision_tree(model, feature_names, class_names=None, max_depth=3, figsize=(20, 10), model_name="Decision Tree"):
    """
    Plot decision tree
    Args:
        model: Trained model
        feature_names: List of feature names
        class_names: List of class names
        max_depth: Maximum depth of the tree
        figsize: Size of the plot
        model_name: Name of the model
    """
    plt.figure(figsize=figsize)
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=8
    )
    plt.title(f"{model_name} Visualization")
    plt.show()

# --------------------------------------------
# Generate rename map
# --------------------------------------------
def generate_rename_map(ohe, ordinal_cols, ordinal_categories):
    """
    Generate rename map from OneHotEncoder and OrdinalEncoder category labels.
    Args:
        ohe: Trained OneHotEncoder instance
        ordinal_cols: List of ordinal column names
        ordinal_categories: List of lists with category order per ordinal column
    """
    rename_map = {}

    # Binary OneHot-encoded features
    for feat in ohe.get_feature_names_out():
        prefix, val = feat.split("_", 1)
        safe_val = html.escape(val)
        rename_map[f"{feat} &le; 0.5"] = f"{prefix} ≠ {safe_val}"
        rename_map[f"{feat} &gt; 0.5"]  = f"{prefix} = {safe_val}"

    # Ordinal features: comparisons
    for col, cats in zip(ordinal_cols, ordinal_categories):
        for i, cat in enumerate(cats):
            safe_cat = html.escape(cat)
            rename_map[f"{col} = {i}"] = f"{col} = {safe_cat}"

            if i < len(cats) - 1:
                rename_map[f"{col} &le; {i + 0.5}"] = f"{col} ≤ {safe_cat}"
                rename_map[f"{col} &gt; {i + 0.5}"] = f"{col} > {safe_cat}"

    return rename_map

def export_tree_to_pdf(model, feature_names, class_names, output_path, rename_map=None, max_depth=None, show_graph=True):
    """
    Export decision tree to PDF using Graphviz.
    Args:
        model: Trained decision tree
        feature_names: List of feature names
        class_names: List of class names
        output_path: Path to save PDF
        rename_map: Dictionary of feature name mappings
        max_depth: Maximum depth of the tree
        show_graph: Whether to display the graph inline
    """
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=max_depth
    )

    # Then do label renaming (e.g. "Title_Mr" → "Title = Mr")
    if rename_map:
        for old, new in rename_map.items():
            dot_data = dot_data.replace(old, new)

    # List of discrete integer features to display nicely
    discrete_int_features = ["Age", "Pclass", "SibSp", "Parch", "FamilySize"]

    def prettify_discrete_thresholds(dot_data, discrete_features):
        """
        Replace 'Feature &le; X.5' on 'Feature ≤ X' for descreet features.
        """
        for feature in discrete_features:
            # For example: FamilySize &le; 3.5 → FamilySize ≤ 3
            pattern = fr"({feature}) &le; (\d+)\.5"
            dot_data = re.sub(pattern, lambda m: f"{m.group(1)} ≤ {int(m.group(2))}", dot_data)

        return dot_data    
    dot_data = prettify_discrete_thresholds(dot_data, discrete_int_features)

    graph = graphviz.Source(dot_data)
    graph.render(output_path, format="pdf", cleanup=True)

    if show_graph:
        display(graph)

    print(f"✅ Tree saved to: ./{output_path}.pdf")

# --------------------------------------------
# Log model results
# --------------------------------------------

# Log model results to global results_df
def log_model_result(results_df, model_name, params, metrics, training_time, notes=""):
    """
    Log model results to global results_df
    Args:
        results_df: Global results DataFrame
        model_name: Name of the model
        params: Model parameters
        metrics: Model metrics
        training_time: Training time
        notes: Additional notes
    """
    row = {
        "model_name": model_name,
        "params": params,
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "training_time": training_time,
        "notes": notes
    }
    results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    print(f"✅ Logged result for {model_name}")

    return results_df

# --------------------------------------------
# CV & Hyperparameters tuning
# --------------------------------------------
def evaluate_model_cv(
    model, param_grid, X_train, y_train, X_test, y_test,
    model_name, results_df, notes="", scoring="f1", cv=5, verbose=0
):
    """
    Evaluate model using cross-validation and hyperparameter tuning.
    Args:
        model: Trained model
        param_grid: Parameter grid for grid search
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        results_df: Global results DataFrame
        notes: Additional notes
        scoring: Scoring metric
        cv: Number of cross-validation folds
        verbose: Verbosity level
    """
    start = time.time()

    grid = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=verbose)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    train_time = time.time() - start

    # Test set predictions
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Metrics
    test_metrics = {
        "accuracy_test": accuracy_score(y_test, y_pred),
        "f1_test": f1_score(y_test, y_pred),
        "roc_auc_test": roc_auc_score(y_test, y_proba)
    }

    # GridSearchCV cross-val scores
    cv_metrics = {
        "accuracy_cv": grid.cv_results_["mean_test_score"] if scoring == "accuracy" else None,
        "f1_cv": grid.best_score_ if scoring == "f1" else None,
        "roc_auc_cv": grid.best_score_ if scoring == "roc_auc" else None,
    }

    # Log results
    results_df = pd.concat([
        results_df,
        pd.DataFrame([{
            "model_name": model_name,
            "params": grid.best_params_,
            **cv_metrics,
            **test_metrics,
            "training_time": train_time,
            "notes": notes
        }])
    ], ignore_index=True)

    return results_df, best_model, grid