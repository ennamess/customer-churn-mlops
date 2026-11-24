
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV_PATH = Path(__file__).resolve().parent.parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

RANDOM_STATE = 42  
TEST_SIZE = 0.15   
VAL_SIZE = 0.15    
TARGET = "Churn"
STRATIFY = True 
#Valerus commune 

def load_data(path=CSV_PATH):
    return pd.read_csv(path)


def explore(df):
    print(f"Dimensions : {df.shape}")
    print("\nTypes :")
    print(df.dtypes.to_string())

    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    blank = df[text_cols].apply(lambda s: s.str.strip() == "").sum()
    print("\nChaînes vides par colonne (hors 0) :")
    print(blank[blank > 0].to_string())
    print("\nNaN par colonne (hors 0) :")
    na = df.isna().sum()
    print(na[na > 0].to_string() if (na > 0).any() else "aucun")
    print(f"\nDoublons (hors customerID) : {df.drop(columns='customerID').duplicated().sum()}")

    print("\nRépartition de la cible :")
    print(df[TARGET].value_counts(normalize=True).round(3).to_string())
    print("\nStatistiques numériques :")
    print(df.describe().round(2).to_string())


def clean(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df = df.drop(columns="customerID")  
    df[TARGET] = (df[TARGET] == "Yes").astype(int) 
    return df


def build_preprocessor(X):
    numeric = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    passthrough = [c for c in X.columns if c not in numeric + categorical]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), categorical),
            ("bin", "passthrough", passthrough),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def get_data(path=CSV_PATH):
    df = clean(load_data(path))
    X, y = df.drop(columns=TARGET), df[TARGET]

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=VAL_SIZE / (1 - TEST_SIZE), stratify=y_tmp, random_state=RANDOM_STATE
    )

    pre = build_preprocessor(X_train)
    X_train = pre.fit_transform(X_train)
    X_val = pre.transform(X_val)
    X_test = pre.transform(X_test)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "preprocessor": pre,
        "feature_names": list(X_train.columns),
    }


if __name__ == "__main__":
    raw = load_data()
    explore(raw)

    d = get_data()
    print("\n=== Après prétraitement ===")
    for name in ("train", "val", "test"):
        X, y = d[f"X_{name}"], d[f"y_{name}"]
        print(f"{name:5s}: X={X.shape}  taux de churn={y.mean():.3f}")
    print(f"Features ({len(d['feature_names'])}) : {d['feature_names']}")
    assert not d["X_train"].isna().any().any(), "NaN restants"
    print("OK : aucune valeur manquante, mêmes proportions de churn dans chaque ensemble.")
