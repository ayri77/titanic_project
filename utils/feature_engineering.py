# feature_engineering.py

import pandas as pd
import numpy as np

from collections import defaultdict

# --------------------------------------------
# Missing values
# --------------------------------------------

def fill_missing_values(df):
    """
    Fill missing values in the DataFrame
    Embarked: fill with most common value
    Age: fill with median per Title group
    Fare: fill with median per Pclass
    Args:
        df (pd.DataFrame): DataFrame containing missing values
    Returns:
        pd.DataFrame: DataFrame with missing values filled
    """
    df = df.copy()

    # Embarked: fill with most common value
    most_common_embarked = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(most_common_embarked)

    # Age: fill with median per Title group
    df["Age"] = df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))

    # Fare: fill with median per Pclass
    df["Fare"] = df["Fare"].fillna(df.groupby(["Pclass"])["Fare"].transform("median"))

    return df

# --------------------------------------------
# Feature extraction
# --------------------------------------------

# Extract standardized title from passenger name
def extract_title(df):
    """
    Extract standardized title from passenger name    
    Args:
        df (pd.DataFrame): DataFrame containing passenger names
    Returns:
        pd.DataFrame: DataFrame with standardized titles
    """
    def _extract_title_from_name(name):
        name = name.lower()
        if "mr." in name:
            return "Mr"
        elif "mrs." in name or "mme." in name:
            return "Mrs"
        elif "miss." in name or "ms." in name or "mlle." in name or "lady." in name:
            return "Miss"
        elif "master." in name:
            return "Master"
        elif "dr." in name:
            return "Dr"
        elif "rev." in name:
            return "Rev"
        elif "sir." in name or "don." in name or "jonkheer." in name:
            return "Sir"
        elif "countess." in name:
            return "Countess"
        elif "col." in name or "major." in name or "capt." in name:
            return "Officer"
        else:
            return "Other"
    
    df["Title"] = df["Name"].apply(_extract_title_from_name)

    return df

# Create features related to family
def create_family_features(df):
    """
    Create features related to family
    Args:
        df (pd.DataFrame): DataFrame containing passenger data
    Returns:
        pd.DataFrame: DataFrame with family features
    """
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = df["FamilySize"].apply(lambda x: int(x == 1))

    return df

# Extract deck letter from cabin number
def simplify_deck(df):
    """
    Extract deck letter from cabin number
    Args:
        df (pd.DataFrame): DataFrame containing cabin numbers
    Returns:
        pd.DataFrame: DataFrame with deck letter
    """
    if "Cabin" in df.columns:
        df["Deck"] = df["Cabin"].apply(lambda x: x[0] if pd.notna(x) else "Unknown")
        # drop old column with missing data
        df.drop("Cabin", axis=1, inplace=True)
    
    return df

# Bin age into categorical age groups
def create_age_group(df):
    """
    Bin age into categorical age groups
    Bins: 0-1, 1-3, 3-12, 12-18, 18-35, 35-60, 60-100
    Labels: baby, kids<3, kids<12, teenager, young, adult, senior
    Args:
        df (pd.DataFrame): DataFrame containing passenger data
    Returns:
        pd.DataFrame: DataFrame with age groups
    """
    bins = [0, 1, 3, 12, 18, 35, 60, 100]
    labels = ["baby", "kids<3", "kids<12", "teenager", "young", "adult", "senior"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

    return df

# Round fare and create custom fare bands
def create_fare_band(df):
    """
    Round fare and create custom fare bands
    Bins: 0-7.5, 7.5-15, 15-25, 25-50, 50-9999
    Original fare is rounded to 2 decimal places
    Args:
        df (pd.DataFrame): DataFrame containing passenger data
    Returns:
        pd.DataFrame: DataFrame with fare bands
    """
    def _round_fare_custom(value):
        if pd.isna(value):
            return value
        if value < 1:
            return round(value, 2)
        elif value < 10:
            return round(value / 0.05) * 0.05
        elif value < 20:
            return round(value / 0.5) * 0.5
        elif value < 50:
            return round(value / 5) * 5
        else:
            return round(value / 10) * 10
        
    df["Fare"] = np.round(df["Fare"], 2)
    df["FareRounded"] = df["Fare"].apply(_round_fare_custom)
    fare_bins = [0, 7.5, 15, 25, 50.0, 9999.0]
    df["FareBand"] = pd.cut(df["FareRounded"], bins=fare_bins)
    df["FareBand"] = df["FareBand"].apply(lambda x: x.left).astype(float)
    # After creating 'FareRounded'
    df.drop(columns=['Fare'], inplace=True)
    df.rename(columns={'FareRounded': 'Fare'}, inplace=True)

    df["FareBand"] = df["FareBand"].fillna(0) # when Fare = 0

    return df

# Extract prefix from ticket (e.g., 'PC 17599' -> 'PC')
def create_ticket_prefix(df):
    """
    Extract prefix from ticket (e.g., 'PC 17599' -> 'PC')
    Args:
        df (pd.DataFrame): DataFrame containing passenger data
    Returns:
        pd.DataFrame: DataFrame with ticket prefixes
    """
    df["TicketPrefix"] = df["Ticket"].apply(lambda x: x.split()[0] if not x.isnumeric() else "")

    return df

# --------------------------------------------
# Feature normalization
# --------------------------------------------

# General function for grouping rare categories under "Other"
def normalize_rare_categories(df, col, top_n=None, min_count=None, threshold=None, other_label="Other", inplace=False, verbose=True):
    """
    Collapse rare categories into 'Other' based on frequency threshold, count, or top-N.
    Args:
        df (pd.DataFrame): DataFrame containing categorical data
        col (str): Column name to normalize
        top_n (int): Keep top N most frequent categories
        min_count (int): Keep categories with at least min_count rows
        threshold (float): Keep categories with frequency threshold
        other_label (str): Label for collapsed categories
        inplace (bool): If True, modify original DataFrame; else return copy
        verbose (bool): If True, print which values were replaced
    Returns:
        pd.DataFrame: DataFrame with rare categories collapsed
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    
    if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_categorical_dtype(df[col]):
        raise TypeError(f"Column '{col}' must be categorical (object or category).")

    data = df if inplace else df.copy()
    value_counts = data[col].value_counts()

    if top_n is not None:
        keep = value_counts.nlargest(top_n).index
    elif min_count is not None:
        keep = value_counts[value_counts >= min_count].index
    elif threshold is not None:
        min_count_thresh = threshold * len(df)
        keep = value_counts[value_counts >= min_count_thresh].index
    else:
        if verbose:
            print("⚠️ No threshold provided. No changes made.")
        return data

    replaced = data[col].apply(lambda x: x if x in keep else other_label)
    if verbose:
        removed = set(data[col]) - set(keep)
        print(f"🔁 {col}: Replaced {len(removed)} rare categories with '{other_label}': {sorted(removed)}")
    data[col] = replaced
    return data

# Specific wrappers for Deck and TicketPrefix normalization
def normalize_deck(df):
    """
    Normalize deck categories
    Args:
        df (pd.DataFrame): DataFrame containing deck data
    Returns:
        pd.DataFrame: DataFrame with normalized deck categories
    """
    return normalize_rare_categories(df, "Deck", min_count=15)

def normalize_ticket_prefix(df):
    """
    Normalize ticket prefix categories
    Args:
        df (pd.DataFrame): DataFrame containing ticket prefix data
    Returns:
        pd.DataFrame: DataFrame with normalized ticket prefix categories
    """
    return normalize_rare_categories(df, "TicketPrefix", top_n=5)

def calculate_rare_categories(df, categorical_features, threshold=0.05):
    """
    Calculate rare categories in categorical features
    Args:
        df (pd.DataFrame): DataFrame containing categorical data
        categorical_features (list): List of categorical feature names
        threshold (float): Threshold for rare categories (default: 0.05)
    Returns:
        pd.DataFrame: DataFrame with rare categories
    """
    # Dictionary to store rare categories for each feature
    rare_categories = defaultdict(list)
    summary = []

    # Threshold: less than 5% of total rows
    threshold = threshold * len(df)

    # Analyze each categorical feature for rare values
    for col in categorical_features:
        value_counts = df[col].value_counts(dropna=False)
        rare = value_counts[value_counts < threshold]
        if not rare.empty:
            rare_categories[col] = list(rare.index)
            summary.append((col, len(rare), list(rare.items())))

    # Prepare a summary DataFrame for display
    rare_summary_df = pd.DataFrame([
        {
            "Feature": col,
            "Rare Count": rare_count,
            "Rare Values": ", ".join([f"{k} ({v})" for k, v in values])
        }
        for col, rare_count, values in summary
    ])

    return rare_summary_df