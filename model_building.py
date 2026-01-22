"""
Model Building Module for Sentiment Analysis
=============================================
This module trains and evaluates multiple classification models:
1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest
4. Gradient Boosting

Uses features extracted from feature_extraction.py
"""

import pandas as pd
import numpy as np
from scipy import sparse
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_features(feature_type='tfidf'):
    """
    Load extracted features and labels.
    
    Parameters:
    -----------
    feature_type : str - 'bow', 'tfidf', 'word2vec', or 'bert'
    
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    print(f"\nLoading {feature_type.upper()} features...")
    
    # Load labels
    y_train = np.load(os.path.join(OUTPUT_DIR, "y_train.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"), allow_pickle=True)
    
    # Load features based on type
    if feature_type == 'bow':
        X_train = sparse.load_npz(os.path.join(OUTPUT_DIR, "X_train_bow.npz"))
        X_test = sparse.load_npz(os.path.join(OUTPUT_DIR, "X_test_bow.npz"))
    elif feature_type == 'tfidf':
        X_train = sparse.load_npz(os.path.join(OUTPUT_DIR, "X_train_tfidf.npz"))
        X_test = sparse.load_npz(os.path.join(OUTPUT_DIR, "X_test_tfidf.npz"))
    elif feature_type == 'word2vec':
        X_train = np.load(os.path.join(OUTPUT_DIR, "X_train_word2vec.npy"))
        X_test = np.load(os.path.join(OUTPUT_DIR, "X_test_word2vec.npy"))
    elif feature_type == 'bert':
        X_train = np.load(os.path.join(OUTPUT_DIR, "X_train_bert.npy"))
        X_test = np.load(os.path.join(OUTPUT_DIR, "X_test_bert.npy"))
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# Model Definitions
# =============================================================================

def get_models():
    """Return dictionary of models to train."""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'SVM': LinearSVC(
            dual='auto',
            random_state=RANDOM_STATE,
            max_iter=2000
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=RANDOM_STATE
        )
    }
    return models


def get_hyperparameter_grids():
    """Return hyperparameter grids for GridSearchCV."""
    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga']
        },
        'SVM': {
            'C': [0.1, 1.0, 10.0],
            'loss': ['hinge', 'squared_hinge']
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5]
        }
    }
    return param_grids


# =============================================================================
# Training Functions
# =============================================================================

def train_model(model, X_train, y_train, model_name):
    """Train a single model."""
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a trained model and return metrics."""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted')
    }
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1 Score (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 Score (weighted): {metrics['f1_weighted']:.4f}")
    
    return y_pred, metrics


def cross_validate_model(model, X_train, y_train, model_name, cv=5):
    """Perform cross-validation on a model."""
    print(f"\nCross-validating {model_name} (cv={cv})...")
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"  CV Accuracy: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
    return scores


def tune_hyperparameters(model, param_grid, X_train, y_train, model_name, cv=3):
    """Tune hyperparameters using GridSearchCV."""
    print(f"\nTuning {model_name} hyperparameters...")
    
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_confusion_matrix(y_test, y_pred, model_name, labels=['Negative', 'Neutral', 'Positive']):
    """Plot confusion matrix for a model."""
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.show()
    print(f"Confusion matrix saved to: {plot_path}")


def plot_model_comparison(results_df):
    """Plot comparison of model performances."""
    plt.figure(figsize=(12, 6))
    
    # Accuracy comparison
    plt.subplot(1, 2, 1)
    results_sorted = results_df.sort_values('accuracy', ascending=True)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(results_sorted)))
    plt.barh(results_sorted['model'], results_sorted['accuracy'], color=colors)
    plt.xlabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xlim(0, 1)
    
    # Add value labels
    for i, (_, row) in enumerate(results_sorted.iterrows()):
        plt.text(row['accuracy'] + 0.01, i, f"{row['accuracy']:.4f}", va='center')
    
    # F1 Score comparison
    plt.subplot(1, 2, 2)
    results_sorted = results_df.sort_values('f1_weighted', ascending=True)
    plt.barh(results_sorted['model'], results_sorted['f1_weighted'], color=colors)
    plt.xlabel('F1 Score (Weighted)')
    plt.title('Model F1 Score Comparison')
    plt.xlim(0, 1)
    
    # Add value labels
    for i, (_, row) in enumerate(results_sorted.iterrows()):
        plt.text(row['f1_weighted'] + 0.01, i, f"{row['f1_weighted']:.4f}", va='center')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.show()
    print(f"Model comparison plot saved to: {plot_path}")


def plot_detailed_metrics(results_df):
    """Plot detailed metrics comparison."""
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    
    x = np.arange(len(results_df))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, results_df[metric], width, label=metric.replace('_', ' ').title())
    
    ax.set_ylabel('Score')
    ax.set_title('Detailed Model Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['model'], rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, "detailed_metrics.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.show()
    print(f"Detailed metrics plot saved to: {plot_path}")


# =============================================================================
# Main Execution
# =============================================================================

def train_all_models(feature_type='tfidf', tune=False):
    """
    Train and evaluate all models.
    
    Parameters:
    -----------
    feature_type : str - Type of features to use ('bow', 'tfidf', 'word2vec', 'bert')
    tune : bool - Whether to perform hyperparameter tuning
    
    Returns:
    --------
    results_df : DataFrame with model performance
    trained_models : dict of trained model objects
    """
    print("="*60)
    print("MODEL BUILDING FOR SENTIMENT ANALYSIS")
    print("="*60)
    
    # Load features
    X_train, X_test, y_train, y_test = load_features(feature_type)
    
    # Get models
    models = get_models()
    param_grids = get_hyperparameter_grids()
    
    # Store results
    results = []
    trained_models = {}
    predictions = {}
    
    for name, model in models.items():
        print("\n" + "-"*50)
        
        if tune:
            # Hyperparameter tuning
            best_model, best_params = tune_hyperparameters(
                model, param_grids[name], X_train, y_train, name
            )
            model = best_model
        else:
            # Cross-validation
            cross_validate_model(model, X_train, y_train, name)
            
            # Train model
            model = train_model(model, X_train, y_train, name)
        
        # Evaluate model
        y_pred, metrics = evaluate_model(model, X_test, y_test, name)
        
        # Print classification report
        print(f"\nClassification Report - {name}:")
        print(classification_report(y_test, y_pred))
        
        # Store results
        results.append({
            'model': name,
            **metrics
        })
        trained_models[name] = model
        predictions[name] = y_pred
        
        # Save model
        model_path = os.path.join(OUTPUT_DIR, f"model_{name.lower().replace(' ', '_')}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to: {model_path}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_path = os.path.join(OUTPUT_DIR, "model_performance.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Save metrics to JSON
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    # Visualizations
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    # Plot model comparison
    plot_model_comparison(results_df)
    
    # Plot detailed metrics
    plot_detailed_metrics(results_df)
    
    # Plot confusion matrices for all models
    for name, y_pred in predictions.items():
        plot_confusion_matrix(y_test, y_pred, name)
    
    # Summary
    print("\n" + "="*60)
    print("MODEL BUILDING COMPLETE")
    print("="*60)
    print("\nModel Performance Summary:")
    print(results_df.sort_values('accuracy', ascending=False).to_string(index=False))
    
    best_model_name = results_df.loc[results_df['accuracy'].idxmax(), 'model']
    best_accuracy = results_df['accuracy'].max()
    print(f"\nBest performing model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    
    return results_df, trained_models


def main():
    """Main execution function."""
    # Check available features
    print("Checking available features...")
    
    available_features = []
    feature_files = {
        'tfidf': 'X_train_tfidf.npz',
        'bow': 'X_train_bow.npz',
        'word2vec': 'X_train_word2vec.npy',
        'bert': 'X_train_bert.npy'
    }
    
    for feat_type, filename in feature_files.items():
        if os.path.exists(os.path.join(OUTPUT_DIR, filename)):
            available_features.append(feat_type)
            print(f"  ✓ {feat_type.upper()} features found")
        else:
            print(f"  ✗ {feat_type.upper()} features not found")
    
    if not available_features:
        print("\nNo features found! Please run feature_extraction.py first.")
        return
    
    # Select feature type - default to TF-IDF
    print(f"\nAvailable feature types: {available_features}")
    
    if 'tfidf' in available_features:
        feature_type = 'tfidf'  # Default to TF-IDF
    else:
        feature_type = available_features[0]
    
    print(f"Using feature type: {feature_type}")
    
    # Skip hyperparameter tuning by default (faster)
    tune = False
    
    # Train models
    results_df, trained_models = train_all_models(feature_type=feature_type, tune=tune)
    
    return results_df, trained_models


if __name__ == "__main__":
    main()
