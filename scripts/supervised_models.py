"""
Module: supervised_models.py
Purpose: Train and evaluate supervised learning models on ESC-50 audio features
Authors: Armin Siavashi, Sepehr Farrokhi
Date: February 2025

This module provides functions for:
- Hyperparameter grid definitions for SVM, Logistic Regression, Random Forest, KNN
- Grid search with cross-validation using predefined folds
- Model comparison and selection based on test accuracy
- Intelligent tie-breaking (prefers simpler models when accuracy is tied)

Dependencies: scikit-learn, numpy
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler


def get_param_grids():
    return {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'l1_ratio': [0.0]

            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
    }

def get_svm_param_grids():
    return {
        "linear": {
            'kernel': ['linear'],
            'C': [0.1, 1, 10, 100]
        },
        "rbf": {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 'scale']
        },
        "poly": {
            'kernel': ['poly'],
            'C': [0.1, 1, 10],
            'degree': [2, 3, 4],
            'gamma': ['scale', 'auto']
        }
    }


def run_grid_search(X, y_encoded, df_filtered, feature_groups, param_grids, test_fold=5):
    # Create train/test split using predefined fold structure to avoid data leakage
    train_mask = df_filtered['fold'] != test_fold  # Folds 1-4 for training
    test_mask = df_filtered['fold'] == test_fold   # Fold 5 for final testing

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y_encoded[train_mask]
    y_test = y_encoded[test_mask]

    # Set up cross-validation using training folds (1-4)
    # Subtract 1 to convert fold numbers (1,2,3,4) to indices (0,1,2,3)
    train_folds = df_filtered.loc[train_mask, 'fold'].values
    ps = PredefinedSplit(test_fold=train_folds - 1)  # Use predefined folds for CV

    grid_search_all_results = {}

    for group_name, feature_indices in feature_groups.items():
        print(f"\n{'='*70}")
        print(f"FEATURE GROUP: {group_name} ({len(feature_indices)} features)")
        print(f"{'='*70}")

        X_train_group = X_train[:, feature_indices]
        X_test_group = X_test[:, feature_indices]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_group)
        X_test_scaled = scaler.transform(X_test_group)

        grid_search_all_results[group_name] = {}

        for model_name, config in param_grids.items():
            print(f"\n{'-'*60}")
            print(f"Model: {model_name}")
            print(f"{'-'*60}")

            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                cv=ps,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0,
                return_train_score=True
            )

            grid_search.fit(X_train_scaled, y_train)

            best_model = grid_search.best_estimator_

            y_pred = best_model.predict(X_test_scaled)
            y_pred_proba = best_model.predict_proba(X_test_scaled)

            grid_search_all_results[group_name][model_name] = {
                'grid_search': grid_search,
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
                'best_model': best_model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'test_accuracy': best_model.score(X_test_scaled, y_test),
                'feature_indices': feature_indices,
                'n_features': len(feature_indices),
                'scaler': scaler
            }

            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"Best CV Score: {grid_search.best_score_:.4f}")
            print(f"Test Accuracy: {grid_search_all_results[group_name][model_name]['test_accuracy']:.4f}")

    # Final Summary Table
    print("\n" + "="*80)
    print(f"FINAL SUMMARY: All Feature Groups & Models")
    print(f"{'='*80}")
    print(f"{'Feature Group':<20} {'Model':<20} {'CV Score':<12} {'Test Acc':<12}")
    print(f"{'-'*80}")

    for group_name, models in grid_search_all_results.items():
        for model_name, result in models.items():
            print(f"{group_name:<20} {model_name:<20} {result['best_cv_score']:<12.4f} {result['test_accuracy']:<12.4f}")

    return grid_search_all_results


def find_best_combination(grid_search_all_results):
    all_results_list = []
    for group_name, models in grid_search_all_results.items():
        for model_name, result in models.items():
            all_results_list.append({
                'group': group_name,
                'model': model_name,
                'cv_score': result['best_cv_score'],
                'test_acc': result['test_accuracy'],
                'n_features': result['n_features']
            })

    # Pick best with intelligent tie-breaking
    # Exclude Random Forest (tends to overfit to 100% training accuracy)
    filtered_results = [r for r in all_results_list if r['model'] != 'Random Forest']

    #To (prefer fewer features when tied):
    best_combination = max(filtered_results, 
                   key=lambda x: (x['test_acc'], -x['n_features'], x['cv_score']))

    print("Best combination overall:")
    print(f"{'='*80}")
    print(f"Feature Group:      {best_combination['group']}")
    print(f"Model:              {best_combination['model']}" )
    print(f"Number of Features: {best_combination['n_features']}")
    print(f"CV Score:           {best_combination['cv_score']:.4f}")
    print(f"Test Accuracy:      {best_combination['test_acc']:.4f}")
    print(f"\nBest Parameters:")
    print(grid_search_all_results[best_combination['group']][best_combination['model']]['best_params'])

    return best_combination
