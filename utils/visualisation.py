# visualisation.py — universal EDA plotting utilities

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance
import shap

from IPython.display import HTML
from IPython.display import display

import io

from math import ceil

# Set default style
sns.set_theme("paper")

# --------------------------------------------
# Dataframe visualisation
# --------------------------------------------
def display_df(df, rows = 10, show_index = False):
    """
    Display a DataFrame in HTML format.
    Args:
        df: DataFrame to display
        rows: Number of rows to display
        show_index: Whether to show row indices
    """
    if rows==0:
        display(HTML(df.to_html(index=show_index)))
    else:
        display(HTML(df.head(rows).to_html(index=show_index)))

def summarize_df(df: pd.DataFrame, name: str = "Dataset", rows: int = 10, mode: str = "basic"):
    """
    Summarize a DataFrame.
    Args:
        df: DataFrame to summarize
        name: Name of the dataset
        rows: Number of rows to display
        mode: Mode of summarization
    """
    print(f"\n{'='*10} {name} Summary {'='*10}\n")

    # Shape
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    # Info
    print("🧾 DataFrame Info:")
    df.info()
    # Describe
    print("\n📊 Statistical Description:")
    display_df(df.describe(include='all').round(2),0,True)
    # Nulls
    print("\n❓ Missing Values:")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    if nulls.empty:
        print("✅ No missing values.")
    else:
        display_df(nulls.to_frame("Missing values"),0,True)
    # Duplicates
    num_duplicates = df.duplicated().sum()
    print(f"\n🔁 Duplicated rows: {num_duplicates}")

    if mode == "extended":
        print("\n🔍 Extended Summary:")
        # Unique values
        print("\n🔢 Unique Values per Column:")
        display_df(df.nunique().sort_values().to_frame("Unique values"),0,True)
        # Data types
        print("\n🧪 Data Types:")
        display_df(df.dtypes.to_frame("Type"),0,True)

    print("\n" + "="*40)

# --------------------------------------------
# Data visualisation
# --------------------------------------------
def plot_target_distribution(df, target):
    """
    Plots the distribution of the binary target variable (e.g. Survived).
    Args:
        df: DataFrame containing the target variable
        target: Name of the target variable
    """
    counts = df[target].value_counts()
    percentages = df[target].value_counts(normalize=True).round(3) * 100

    plt.figure(figsize=(6, 5))
    ax = sns.countplot(data=df, x=target, palette='pastel')

    for i, p in enumerate(ax.patches):
        value = counts[i]
        percent = percentages[i]
        ax.annotate(f'{value} ({percent:.1f}%)',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, weight='bold')

    ax.set_title(f"Distribution of Target: {target}")
    ax.set_ylabel("Count")
    ax.set_xlabel(target)
    plt.tight_layout()
    plt.show()

def plot_categorical_vs_target(
    df,
    features,
    target,
    mode="count",  # "count", "percentage", or "rate"
    annotate=True,
    cols=3
):
    """
    Plots the distribution of a categorical feature vs the target variable.
    Args:
        df: DataFrame containing the categorical feature and target variable
        features: List of categorical features to plot
        target: Name of the target variable
        mode: Mode of visualization ("count", "percentage", "rate")
        annotate: Whether to annotate the bars
        cols: Number of columns in the plot
    """

    rows = (len(features) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(features):
        ax = axes[i]

        if mode == "count":
            sns.countplot(data=df, x=col, hue=target, palette="pastel", ax=ax)
            if annotate:
                for p in ax.patches:
                    height = int(p.get_height())
                    if height > 0:
                        ax.annotate(f"{height}", (p.get_x() + p.get_width() / 2, height),
                                    ha='center', va='bottom')

        elif mode == "percentage":
            pct_df = df.groupby([col, target]).size().reset_index(name="count")
            total = pct_df.groupby(col)["count"].transform("sum")
            pct_df["percentage"] = pct_df["count"] / total * 100
            sns.barplot(data=pct_df, x=col, y="percentage", hue=target, palette="pastel", ax=ax)
            if annotate:
                for p in ax.patches:
                    height = p.get_height()
                    if height > 0:
                        ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2, height),
                                    ha='center', va='bottom')
            ax.set_ylabel("Percentage (%)")

        elif mode == "rate":
            rate_df = df.groupby(col)[target].mean().reset_index()
            sns.barplot(data=rate_df, x=col, y=target, palette="pastel", ax=ax)
            if annotate:
                for p in ax.patches:
                    height = p.get_height()
                    if height > 0:
                        ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2, height),
                                    ha='center', va='bottom')
            ax.set_ylabel(f"Mean {target} rate")

        ax.set_title(f"{col} vs {target} ({mode})")
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    # Remove empty subplots if there are fewer features than subplots
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_numerical_distributions(df, features, cols=3):
    """
    Plots histograms and KDEs for numerical features in a grid layout.
    Args:
        df: DataFrame containing the numerical features
        features: List of numerical features to plot
        cols: Number of columns in the plot
    """
    rows = (len(features) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.histplot(df[col].dropna(), kde=True, bins=30, color="skyblue", ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def plot_boxplots_by_target(df, features, target, cols=3):
    """
    Plots boxplots for each numerical feature grouped by the target variable in a grid layout.
    Args:
        df: DataFrame containing the numerical features and target variable
        features: List of numerical features to plot
        target: Name of the target variable
        cols: Number of columns in the plot
    """
    rows = (len(features) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.boxplot(data=df, x=target, y=col, palette="Set3", ax=axes[i])
        axes[i].set_title(f"{col} by {target}")
    
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, features, target):
    """
    Plots a heatmap of correlations between numerical features and the target.
    Args:
        df: DataFrame containing the numerical features and target variable
        features: List of numerical features to plot
        target: Name of the target variable
    """
    cols = features + [target]
    corr = df[cols].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# --------------------------------------------
# Models interpreting
# --------------------------------------------

# Permutation Importance
def plot_permutation_importance(model, X_val, y_val, feature_names, model_name):
    """
    Plots the permutation importance of features in a model.
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target
        feature_names: List of feature names
        model_name: Name of the model
    """
    if feature_names is None:
        feature_names = X_val.columns
    result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance Mean": result.importances_mean,
        "Importance Std": result.importances_std
    }).sort_values("Importance Mean", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x="Importance Mean", y="Feature", palette="viridis")
    plt.title(f"Permutation Importance: {model_name}")
    plt.xlabel("Mean Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return importance_df

# Partial Dependence
def manual_partial_dependence(model, X_df, feature_name, target_class_idx=1):
    """
    Manually computes Partial Dependence values for a given feature.
    Returns numeric numpy arrays for grid points and PDP values.
    Args:
        model: Trained model
        X_df: DataFrame containing the features
        feature_name: Name of the feature to compute PDP for
        target_class_idx: Index of the target class (default is 1 for binary classification)
    Returns:
        grid_points: Numeric numpy array of grid points
    """
    if not isinstance(X_df, pd.DataFrame):
        raise ValueError("X_df must be a pandas DataFrame for manual_partial_dependence.")
    if feature_name not in X_df.columns:
        raise ValueError(f"Feature '{feature_name}' not found in X_df columns.")

    X_np_full = X_df.values
    original_column_names = X_df.columns.tolist()
    
    try:
        feature_idx = X_df.columns.get_loc(feature_name)
    except KeyError:
        print(f"Error: Could not find feature index for '{feature_name}'.")
        # Return empty float arrays on error
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Get unique values and attempt conversion to float
    try:
        grid_points_raw = np.unique(X_np_full[:, feature_idx])
        # Attempt conversion early; if it fails, it's likely non-numeric
        grid_points = np.array(grid_points_raw, dtype=np.float64)
    except ValueError:
         print(f"Warning: Could not convert grid points for '{feature_name}' to numeric. Skipping feature.")
         return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    pdp_values_list = []

    for value in grid_points: # Iterate through numeric grid points
        X_modified_np = np.copy(X_np_full)
        X_modified_np[:, feature_idx] = value
        
        X_modified_df = pd.DataFrame(X_modified_np, columns=original_column_names)
        
        try:
            probabilities = model.predict_proba(X_modified_df)
            
            if probabilities.shape[1] <= target_class_idx:
                print(f"  ERROR: target_class_idx {target_class_idx} is out of bounds "
                      f"for probabilities shape {probabilities.shape} (feature: '{feature_name}').")
                pdp_values_list.append(np.nan)
                continue
            
            prob_target_class = probabilities[:, target_class_idx]
            mean_prob_target_class = np.mean(prob_target_class)
            pdp_values_list.append(mean_prob_target_class)
        except Exception as e_manual_pred:
            print(f"  ERROR during predict_proba or averaging for feature '{feature_name}', value {value}: {e_manual_pred}")
            pdp_values_list.append(np.nan) # Add NaN if prediction failed

    # Convert final list to float64 array
    pdp_values = np.array(pdp_values_list, dtype=np.float64)
        
    return grid_points, pdp_values

def plot_partial_dependence_wrapper(
    model,
    X, # Should be a pandas DataFrame
    features=None,
    model_name="Model",
    top_n=6,
    importance_df=None,
    n_cols=3,
    figsize_per_plot=(5, 4)
):
    """
    Plot partial dependence for selected features using manual calculation.
    Simplified version with less debugging output.
    Args:
        model: Trained model
        X: DataFrame containing the features
        features: List of features to plot
        model_name: Name of the model
        top_n: Number of features to plot
        importance_df: DataFrame containing feature importance
        n_cols: Number of columns in the plot
        figsize_per_plot: Size of each plot
    """
    if not isinstance(X, pd.DataFrame):
        print("⚠️ plot_partial_dependence_wrapper expects X to be a pandas DataFrame.")
        return

    # --- Determine and Validate Features ---
    selected_features = []
    if features is not None:
        if isinstance(features, str): selected_features = [features]
        else: selected_features = list(features)
    elif importance_df is not None and "Feature" in importance_df.columns:
        selected_features = importance_df["Feature"].head(top_n).tolist()
    else: # importance_df is None and features is None
        selected_features = X.columns[:top_n].tolist()

    valid_features = [f for f in selected_features if f in X.columns]

    if not valid_features:
        print("⚠️ No valid features found in X to plot from the selection:", selected_features)
        return
    # --- End Feature Selection ---

    print(f"Plotting PDP for features: {valid_features}")

    n_rows = ceil(len(valid_features) / n_cols)
    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # --- Determine Target Class ---
    target_class_idx = 1 # Default for binary classification [0, 1], for class '1'
    target_class_label = 'Class 1' # Default label
    if hasattr(model, 'classes_'):
        if len(model.classes_) > target_class_idx:
            target_class_label = f"Class {model.classes_[target_class_idx]}"
        print(f"Model classes: {model.classes_}. Plotting PDP for: {target_class_label} (index {target_class_idx})")
    else:
        print(f"⚠️ Model has no 'classes_'. Assuming index {target_class_idx} is desired target.")
    # --- End Target Class ---

    for i, feature_name in enumerate(valid_features):
        ax = axes[i]
        try:
            # Call the manual function
            grid_points, pdp_values = manual_partial_dependence(
                model, X, feature_name, target_class_idx=target_class_idx
            )

            # Basic check if data is valid for plotting
            if grid_points.size > 0 and pdp_values.size > 0 and grid_points.size == pdp_values.size and not np.isnan(pdp_values).all():
                ax.plot(grid_points, pdp_values, marker='o', linestyle='-')
                ax.set_xlabel(feature_name)
                ax.set_ylabel(f'Mean Prob. ({target_class_label})')
                ax.set_title(f'PDP: {feature_name}')
                # Set x-ticks for categorical-like features
                if len(grid_points) <= 10:
                    ax.set_xticks(grid_points)
                ax.grid(True)
            else:
                 # If data is invalid (empty, NaN, etc.), display message
                 error_msg = f"Could not generate valid PDP\ndata for {feature_name}"
                 if grid_points.size != pdp_values.size: error_msg += "\n(Size mismatch)"
                 elif grid_points.size == 0: error_msg += "\n(No grid points)"
                 elif np.isnan(pdp_values).all(): error_msg += "\n(All NaN results)"

                 ax.text(0.5, 0.5, error_msg,
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, color='red', fontsize=9)
                 ax.set_title(f'PDP: {feature_name} (Error)')
                 print(f"⚠️ Skipping plot for {feature_name} due to invalid/empty data.")

        except Exception as plot_err:
             # Catch unexpected errors during PDP calculation or plotting for this feature
             print(f"ERROR plotting PDP for {feature_name}: {plot_err}")
             ax.text(0.5, 0.5, f"Error generating PDP\nfor {feature_name}",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='red', fontsize=9)
             ax.set_title(f'PDP: {feature_name} (Plot Error)')

    # --- Final Figure Adjustments ---
    # Hide any unused subplots
    for j in range(len(valid_features), len(axes)):
        fig.delaxes(axes[j])

    try:
        fig.suptitle(f"Manual Partial Dependence Plots: {model_name}", fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()
    except Exception as tight_layout_err:
         # Handle potential errors during layout adjustment
         print(f"\n⚠️ ERROR during plt.tight_layout: {tight_layout_err}")
         print("    Plot might not be perfectly arranged.")
         plt.show() # Still show the plot if possible
    # --- End Final Figure Adjustments ---

# SHAP Summary
def plot_shap_summary(model, X_sample):
    """
    Plots a SHAP summary plot for a given model and sample data.
    Args:
        model: Trained model
        X_sample: DataFrame containing the sample data
    """
    try:
        explainer = _choose_shap_explainer(model, X_sample)
        shap_values = explainer(X_sample)
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=True, show_values_in_legend=True, max_display=10)       

    except Exception as e:
        print(f"⚠️ SHAP summary plot failed: {e}")

# SHAP Dependence
def plot_shap_dependence(model, X_sample, X_original, feature):
    """
    Plots a SHAP dependence plot for a given model and sample data.
    Args:
        model: Trained model
        X_sample: DataFrame containing the sample data
        X_original: DataFrame containing the original data
        feature: Name of the feature to plot
    """
    try:
        explainer = _choose_shap_explainer(model, X_sample)
        shap_values = explainer(X_sample)


        if hasattr(shap_values, 'values'):
            values = shap_values.values
            if isinstance(values, np.ndarray):
                if values.ndim == 3:
                    values = values[:, :, 1]  # класс 1
                elif values.ndim == 2:
                    pass  # ok
                else:
                    raise ValueError(f"Unsupported SHAP value shape: {values.shape}")
            else:
                raise TypeError(f"Unexpected SHAP value type: {type(values)}")
        else:
            raise ValueError("Missing .values in shap_values")

        shap.dependence_plot(feature, values, X_original, show=True)

    except Exception as e:
        print(f"⚠️ SHAP dependence plot failed: {e}")


# Helper: Determine SHAP explainer
def _choose_shap_explainer(model, X_sample):
    """
    Determines the appropriate SHAP explainer based on the model type.
    Args:
        model: Trained model
        X_sample: DataFrame containing the sample data
    """
    model_type = type(model).__name__.lower()
    if any(tree_kw in model_type for tree_kw in ["tree", "forest", "boost"]):
        return shap.Explainer(model, X_sample)
    else:
        return shap.KernelExplainer(model.predict_proba, shap.sample(X_sample, 100))

# SHAP Force   
# Helper:map features and values from original data
def _map_feature_values(index, df_original, feature_order):
    """
    Maps raw feature values from the original dataset to model-encoded inputs
    for a given row (passenger).
    Args:
        index: Index of the row in the original dataset
        df_original: DataFrame containing the original data
        feature_order: List of feature names
    """
    row = df_original.loc[index]
    mapped_values = []

    for feat in feature_order:
        if "_" in feat:
            prefix, category = feat.split("_", 1)
            raw_val = row.get(prefix)
            val = 1.0 if str(raw_val) == category else 0.0
        else:
            val = row.get(feat, np.nan)
        
        mapped_values.append(val)

    return mapped_values

def plot_shap_force_for_instance(model, X_scaled, index, feature_values=None, feature_order=None):
    """
    Display a SHAP force plot for a single prediction.
    Args:
        model: Trained model
        X_scaled: DataFrame containing the scaled features
        index: Index of the instance to explain
        feature_values: List of feature values for the instance
        feature_order: List of feature names
    """
    # Initialize SHAP JavaScript visualization
    shap.initjs()
    
    # Create SHAP explainer and compute SHAP values
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)

    if feature_values is None:
        feature_values = X_scaled.iloc[index]   
        
    feature_values = np.array(feature_values).reshape(-1)

    # Get raw model output (logit) and convert to probability
    logit = shap_values[index].values.sum() + explainer.expected_value
    prob = 1 / (1 + np.exp(-logit))

    print(f"🔢 Model logit (f(x)): {logit:.4f}")
    print(f"✅ Predicted probability: {prob:.2%}")

    # Return the force plot for the selected instance
    shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values[index].values,
        features=feature_values,
        feature_names=list(feature_order),
        matplotlib=True
    )  

def explain_passenger(index, model, X_scaled, X_original=None, df_original=None):
    """
    Explains a passenger's prediction using SHAP force plot.
    Args:
        index: Index of the instance to explain
        model: Trained model
        X_scaled: DataFrame containing the scaled features
        X_original: DataFrame containing the original data
        df_original: DataFrame containing the original data
    """
    true_index = X_scaled.index[index]

    # Show passenger info from full dataset (e.g., name)
    if df_original is not None:
        true_index = X_scaled.index[index]
        display_df(df_original.loc[[true_index]])
    else:
        # Show original features (if available)
        if X_original is not None:
            display_df(X_original.iloc[index].to_frame(), show_index=True)

    feature_values = None
    if df_original is not None:
        feature_order = X_scaled.columns
        feature_values = _map_feature_values(true_index, df_original, feature_order)

    # SHAP explanation
    plot_shap_force_for_instance(model, X_scaled, index = index, feature_values=feature_values, feature_order=feature_order)