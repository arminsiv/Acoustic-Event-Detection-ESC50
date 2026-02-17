"""
Module: performance.py
Purpose: Compute performance metrics for trained models
Authors: Armin Siavashi, Sepehr Farrokhi
Date: February 2025

This module provides functions for:
- Computing summary statistics (accuracy, AUC, overfitting metrics)
- Calculating TPR/FPR for sensitivity and false alarm analysis
- Extracting top N models for comparison

Dependencies: scikit-learn, pandas, numpy
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def compute_summary(grid_search_all_results, y_test, classes):
    summary_list = []

    for group_name, models in grid_search_all_results.items():
        for model_name, result in models.items():

            best_model = result['best_model']
            y_pred_proba = result['y_pred_proba']

            try:
                y_test_bin = label_binarize(y_test, classes=best_model.classes_)

                if hasattr(best_model, "predict_proba") and y_pred_proba.shape[1] == y_test_bin.shape[1]:
                    auc_score = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='weighted')
                else:
                    auc_score = 0.0
            except Exception as e:
                print(f"Warning: Could not calculate AUC for {model_name}: {e}")
                auc_score = 0.0

            overfitting = result['best_cv_score'] - result['test_accuracy']

            summary_list.append({
                'Feature Group': group_name,
                'Model': model_name,
                'CV Accuracy': result['best_cv_score'],
                'Test Accuracy': result['test_accuracy'],
                'AUC Score': auc_score,
                'Overfitting': overfitting
            })

    results_df = pd.DataFrame(summary_list)
    results_df = results_df.sort_values(by='Test Accuracy', ascending=False)
    return results_df


def compute_tpr_fpr(grid_search_all_results, y_test, results_df):
    model_metrics = []

    for group_name, models in grid_search_all_results.items():
        for model_name, result in models.items():
            predictions = result['y_pred']

            unique_classes = np.unique(y_test)
            sensitivities = np.zeros(len(unique_classes))
            false_alarm_rates = np.zeros(len(unique_classes))

            for k, c in enumerate(unique_classes):
                is_positive = (y_test == c)
                is_negative = ~is_positive

                true_pos = np.sum(predictions[is_positive] == c)
                false_neg = np.sum(predictions[is_positive] != c)
                false_pos = np.sum(predictions[is_negative] == c)
                true_neg = np.sum(predictions[is_negative] != c)

                sensitivities[k] = true_pos / max(true_pos + false_neg, 1)
                false_alarm_rates[k] = false_pos / max(false_pos + true_neg, 1)

            auc_val = results_df[
                (results_df['Model'] == model_name) &
                (results_df['Feature Group'] == group_name)
            ]['AUC Score'].values[0]

            model_metrics.append({
                'Model': model_name,
                'Feature Group': group_name,
                'Accuracy': result['test_accuracy'],
                'TPR (macro)': sensitivities.mean(),
                'FPR (macro)': false_alarm_rates.mean(),
                'AUC': auc_val
            })

    metrics_df = pd.DataFrame(model_metrics).sort_values('Accuracy', ascending=False)

    print("\n" + "="*100)
    print("SENSITIVITY AND FALSE ALARM ANALYSIS")
    print("="*100)
    print(metrics_df.head(10).to_string(index=False, float_format="%.4f"))
    print("\nTPR = sensitivity (fraction of positives correctly detected)")
    print("FPR = false alarm rate (fraction of negatives incorrectly flagged)")

    return metrics_df


def get_top_n_models(grid_search_all_results, n=5):
    all_models = []
    for group_name, models in grid_search_all_results.items():
        for model_name, result in models.items():
            all_models.append({
                'group': group_name,
                'model': model_name,
                'test_accuracy': result['test_accuracy'],
                'result': result
            })
    return sorted(all_models, key=lambda x: x['test_accuracy'], reverse=True)[:n]
