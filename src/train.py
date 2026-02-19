"""
Training script for Telco Customer Churn prediction model.
Trains HistGradientBoostingClassifier with configurable hyperparameters.
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    
    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/extracted/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (0-1)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    # Model hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for gradient boosting",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum depth of trees",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum number of boosting iterations",
    )
    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        default=15,
        help="Maximum number of leaf nodes",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=20,
        help="Minimum samples required in a leaf node",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save model and metrics",
    )
    
    return parser.parse_args()

def load_and_prepare_data(data_path, test_size, random_state):
    """Load and prepare data for training."""
    print(f"Loading data from {data_path}...")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Clean data
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    
    print(f"Dataset shape after cleaning: {df.shape}")

    # FEATURE ENGINEERING AQUÍ (NUEVO)
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["IsNewCustomer"] = (df["tenure"] < 3).astype(int)
    df["LifetimeValueEstimate"] = df["MonthlyCharges"] * df["tenure"]
    df["TenureGroup"] = pd.qcut(df["tenure"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    df["MonthlyChargeTier"] = pd.qcut(df["MonthlyCharges"], q=4, labels=["Low", "Med", "High","Very High"])

    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def create_preprocessing_pipeline(X_train):
    """Create preprocessing pipeline based on training data."""
    
    # Detect column types
    #tmp = FeatureAdder().fit_transform(X_train)
    tmp = X_train
    
    num_attribs = tmp.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_attribs = tmp.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
    binary_num_attribs = [col for col in num_attribs if tmp[col].dropna().nunique() == 2]
    num_attribs = [col for col in num_attribs if col not in binary_num_attribs]
    
    print(f"\nFeature types detected:")
    print(f"  Numerical: {len(num_attribs)}")
    print(f"  Binary: {len(binary_num_attribs)}")
    print(f"  Categorical: {len(cat_attribs)}")
    
    # Create pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    binary_num_pipeline = "passthrough"
    
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    
    # Combine into preprocessor
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("binary_num", binary_num_pipeline, binary_num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])
    
    return preprocessor

def train_model(X_train, y_train, preprocessor, hyperparams):
    """Train the model with given hyperparameters."""
    print("\nTraining model...")
    print(f"Hyperparameters: {hyperparams}")
    
    # Create model
    model = HistGradientBoostingClassifier(
        learning_rate=hyperparams['learning_rate'],
        max_depth=hyperparams['max_depth'],
        max_iter=hyperparams['max_iter'],
        max_leaf_nodes=hyperparams['max_leaf_nodes'],
        min_samples_leaf=hyperparams['min_samples_leaf'],
        random_state=hyperparams['random_state'],
        
        class_weight='balanced',
        
    )
    
    # Create full pipeline
    pipeline = Pipeline([
        #("feature_adder", FeatureAdder()),
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    print("Training completed")
    
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    """Evaluate model and return metrics."""
    print("\nEvaluating model...")
    
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
    }
    
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    return metrics

def save_outputs(pipeline, metrics, hyperparams, output_dir):
    """Save model, metrics, and hyperparameters."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"\n Model saved to {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)        
    print(f"Metrics saved to {metrics_path}")
    
    # Save hyperparameters
    hyperparams_path = os.path.join(output_dir, "hyperparams.json")
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)        
    print(f" Hyperparameters saved to {hyperparams_path}")

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Start MLflow run
    # mlflow.start_run()
    
    # Log parameters
    hyperparams = {
        'learning_rate': args.learning_rate,
        'max_depth': args.max_depth,
        'max_iter': args.max_iter,
        'max_leaf_nodes': args.max_leaf_nodes,
        'min_samples_leaf': args.min_samples_leaf,
        'random_state': args.random_state,
    }
    
    # mlflow.log_params(hyperparams)
    # mlflow.log_param('test_size', args.test_size)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        args.data_path, args.test_size, args.random_state
    )
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(X_train)
    
    # Train model
    pipeline = train_model(X_train, y_train, preprocessor, hyperparams)
    
    # Evaluate model
    metrics = evaluate_model(pipeline, X_test, y_test)
    
    # Log metrics to MLflow
    # mlflow.log_metrics(metrics)
    
    # Save outputs
    save_outputs(pipeline, metrics, hyperparams, args.output_dir)
    
    # End MLflow run
    # # mlflow.end_run()
    
    print("\n Training completed successfully!")


if __name__ == "__main__":
    main()