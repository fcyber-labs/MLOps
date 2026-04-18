import pandas as pd
import numpy as np
import yaml
import os


# Load parameters from param.yaml
params = yaml.safe_load(open("params.yaml"))["preprocess"]


def preprocess(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Age categories
    interval = (18, 25, 35, 60, 120)
    cats = ['Student', 'Young', 'Adult', 'Senior']
    df["Age_cat"] = pd.cut(df.Age, interval, labels=cats)

    # Fill missing
    df['Saving accounts'] = df['Saving accounts'].fillna('no_inf')
    df['Checking account'] = df['Checking account'].fillna('no_inf')

    # Dummies
    df = df.merge(pd.get_dummies(df.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df.Risk, prefix='Risk'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df["Checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)
    df = df.merge(pd.get_dummies(df["Age_cat"], drop_first=True, prefix='Age_cat'), left_index=True, right_index=True)

    # Drop unused
    df.drop(columns=[
        "Saving accounts", "Checking account", "Purpose",
        "Sex", "Housing", "Age_cat", "Risk", "Risk_good"
    ], inplace=True)

    # Transform
    df['Credit amount'] = np.log(df['Credit amount'])

    # Save
    df.to_csv(output_path, index=False)

    print(f"Preprocessed data saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

if __name__=="__main__":
    preprocess(params["input"], params["output"])