"""
Module: visualizations.py
Purpose: Generate plots and visualizations for model evaluation
Authors: Armin Siavashi, Sepehr Farrokhi
Date: February 2025

This module provides visualization functions for:
- Confusion matrices with heatmaps
- Coverage plots (TPR vs FPR comparison)
- ROC curves for multi-class classification
- Learning curves showing training progression
- Computational cost analysis (speed vs accuracy)
- Model stability analysis (cross-validation consistency)
- t-SNE embeddings for feature space visualization
- Per-class F1 score comparisons

Dependencies: matplotlib, seaborn, scikit-learn, librosa
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from collections import Counter
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.model_selection import learning_curve, PredefinedSplit, cross_val_score
from sklearn.base import clone


def plot_confusion_matrix(y_test, y_pred, classes, title, accuracy=None):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes,
                cbar_kws={'label': 'Count'})

    if accuracy is not None:
        plt.title(f'{title}\nTest Accuracy: {accuracy:.2%}', fontsize=14)
    else:
        plt.title(title, fontsize=14)

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



def plot_coverage(metrics_df, top_n=10):
    plt.figure(figsize=(12, 10))

    model_colors = {
        'Logistic Regression': 'blue',
        'Random Forest': 'green',
        'KNN': 'orange',
        'SVM': 'purple'
    }

    top_metrics = metrics_df.head(top_n)

    for idx, (_, row) in enumerate(top_metrics.iterrows()):
        color = model_colors.get(row['Model'], 'gray')
        marker_map = {'Logistic Regression': 'o', 'Random Forest': 's', 'KNN': '^', 'SVM': 'D'}
        marker = marker_map.get(row['Model'], 'o')

        plt.scatter(
            row['FPR (macro)'],
            row['TPR (macro)'],
            s=200,
            alpha=0.7,
            c=color,
            marker=marker,
            edgecolors='black',
            linewidth=1.5,
            label=f"{row['Model']} ({row['Feature Group'][:12]})" if idx < 5 else ""
        )

        if idx < 5:
            plt.annotate(
                f"{row['Model'][:8]}\n{row['Accuracy']:.3f}",
                (row['FPR (macro)'], row['TPR (macro)']),
                xytext=(8, 8),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )

    plt.scatter(0, 1, marker='*', s=800, c='gold', edgecolors='black',
                linewidth=2, label='Ideal Classifier', zorder=10)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=2, label='Random Guess')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Logistic Regression'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='Random Forest'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=10, label='KNN'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='purple', markersize=10, label='SVM'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=15, label='Ideal (TPR=1, FPR=0)'),
    ]

    plt.xlabel('False Positive Rate (FPR) -> Lower is Better', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate (TPR) -> Higher is Better', fontsize=13, fontweight='bold')
    plt.title('Coverage Plot - Model Comparison\n(Top 10 Models by Accuracy)',
              fontsize=15, fontweight='bold')
    plt.xlim(-0.01, 0.15)
    plt.ylim(0.40, 1.05)
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

    print("\nReading the coverage plot:")
    print("  Points near the top-left corner have high sensitivity with few false alarms.")
    print("  The gold star marks the theoretical perfect classifier.")
    best_row = metrics_df.iloc[0]
    print(f"  Best model: {best_row['Model']} on {best_row['Feature Group']} -- TPR={best_row['TPR (macro)']:.3f}, FPR={best_row['FPR (macro)']:.3f}")


def plot_accuracy_auc_bars(results_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    sns.barplot(
        data=results_df,
        x='Feature Group',
        y='Test Accuracy',
        hue='Model',
        palette='viridis',
        ax=ax1
    )

    ax1.set_title('Test Accuracy Comparison', fontsize=16)
    ax1.set_xlabel('Feature Group', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(title='Classifier', loc='lower right')

    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

    sns.barplot(
        data=results_df,
        x='Feature Group',
        y='AUC Score',
        hue='Model',
        palette='magma',
        ax=ax2
    )

    ax2.set_title('AUC Score Comparison', fontsize=16)
    ax2.set_xlabel('Feature Group', fontsize=12)
    ax2.set_ylabel('AUC Score', fontsize=12)
    ax2.set_ylim(0.5, 1.05)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend(title='Classifier', loc='lower right')

    plt.tight_layout()
    plt.show()


def plot_learning_curve_champion(best_model_config, best_combination, X_train, y_train, df_filtered, test_fold=5):
    print(f"Generating Learning Curve for: {best_combination['model']}")

    train_mask = df_filtered['fold'] != test_fold
    feature_indices = best_model_config['feature_indices']
    X_train_group = X_train[:, feature_indices]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_group)

    model_for_lc = clone(best_model_config['best_model'])

    train_folds = df_filtered.loc[train_mask, 'fold'].values
    ps = PredefinedSplit(test_fold=train_folds - 1)

    train_sizes, train_scores, val_scores = learning_curve(
        model_for_lc,
        X_train_scaled,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=ps,
        scoring='accuracy',
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(10, 6))

    plt.plot(train_sizes, train_mean, label='Training Score', marker='o', color='blue')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')

    plt.plot(train_sizes, val_mean, label='Cross-Validation Score', marker='s', color='green')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='green')

    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Learning Curve - {best_combination["model"]}\nFeature Set: {best_combination["group"]}', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_roc_curves(grid_search_all_results, y_test, top_n=5):
    all_models_list = []
    for group_name, models in grid_search_all_results.items():
        for model_name, res in models.items():
            all_models_list.append({
                'name': f"{model_name} ({group_name})",
                'model': res['best_model'],
                'y_pred_proba': res['y_pred_proba'],
                'accuracy': res['test_accuracy']
            })

    top_models = sorted(all_models_list, key=lambda x: x['accuracy'], reverse=True)[:top_n]

    print(f"Plotting ROC for the Top {len(top_models)} Models:")
    for m in top_models:
        print(f" -> {m['name']} (Acc: {m['accuracy']:.2%})")

    plt.figure(figsize=(12, 9))
    colors = sns.color_palette("bright", len(top_models))

    for idx, item in enumerate(top_models):
        try:
            model = item['model']
            y_score = item['y_pred_proba']

            y_test_bin = label_binarize(y_test, classes=model.classes_)

            if y_score.shape[1] != y_test_bin.shape[1]:
                print(f"Skipping {item['name']} due to dimension mismatch.")
                continue

            fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, color=colors[idx], lw=2.5,
                     label=f"{item['name']} (AUC = {roc_auc:.3f})")

        except Exception as e:
            print(f"Could not plot {item['name']}: {e}")

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve Comparison - Top {len(top_models)} Models', fontsize=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_per_class_f1(grid_search_all_results, X_test, y_test, le, top_n=3):
    from scripts.performance import get_top_n_models

    top_results = get_top_n_models(grid_search_all_results, n=top_n)

    class_data = []

    for item in top_results:
        group_name = item['group']
        model_name = item['model']
        result = item['result']

        feature_indices = result['feature_indices']
        X_test_group = X_test[:, feature_indices]

        scaler = result['scaler']
        X_test_scaled_group = scaler.transform(X_test_group)

        best_model = result['best_model']
        y_pred = best_model.predict(X_test_scaled_group)

        report = classification_report(y_test, y_pred, output_dict=True, target_names=le.classes_)

        for class_label in le.classes_:
            class_data.append({
                'Model': f"{model_name} ({group_name})",
                'Model_Short': model_name,
                'Feature_Group': group_name,
                'Class': class_label,
                'F1-Score': report[class_label]['f1-score'],
                'Precision': report[class_label]['precision'],
                'Recall': report[class_label]['recall']
            })

    df_class = pd.DataFrame(class_data)

    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_class, x='Class', y='F1-Score', hue='Model', palette='viridis')

    plt.title(f"Per-Class Detection Quality (F1-Score) - Top {top_n} Models", fontsize=16)
    plt.ylabel("F1 Score (Higher is better)", fontsize=12)
    plt.xlabel("Audio Class", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\nTop {top_n} Models:")
    for i, item in enumerate(top_results, 1):
        print(f"{i}. {item['model']} ({item['group']}): {item['test_accuracy']:.4f}")


def plot_tsne(grid_search_all_results, X_test, y_test, le, best_combination, feature_groups):
    winner_group = best_combination['group']
    feature_indices = feature_groups[winner_group]
    X_test_group = X_test[:, feature_indices]
    scaler = grid_search_all_results[winner_group][best_combination['model']]['scaler']
    X_test_best = scaler.transform(X_test_group)

    best_model = grid_search_all_results[winner_group][best_combination['model']]['best_model']
    y_pred = best_model.predict(X_test_best)

    print(f"Generating t-SNE visualization for feature set: {winner_group}")
    print("This may take a few seconds...")

    tsne = TSNE(n_components=2, random_state=42, perplexity=20, max_iter=1000)
    X_embedded = tsne.fit_transform(X_test_best)

    df_tsne = pd.DataFrame(X_embedded, columns=['Dim1', 'Dim2'])
    df_tsne['True Label'] = [le.classes_[i] for i in y_test]
    df_tsne['Predicted'] = [le.classes_[i] for i in y_pred]
    df_tsne['Correct'] = y_test == y_pred

    plt.figure(figsize=(12, 10))

    correct_mask = df_tsne['Correct']
    sns.scatterplot(
        data=df_tsne[correct_mask],
        x='Dim1', y='Dim2',
        hue='True Label',
        palette='tab10',
        s=100,
        alpha=0.8,
        marker='o',
        legend='full'
    )

    if not correct_mask.all():
        plt.scatter(
            df_tsne[~correct_mask]['Dim1'],
            df_tsne[~correct_mask]['Dim2'],
            marker='x',
            s=200,
            c='red',
            linewidths=3,
            label='Misclassified',
            zorder=5
        )

    plt.title(f"t-SNE Visualization of Audio Classes\n(Feature Set: {winner_group}, Model: {best_combination['model']})", fontsize=15)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Audio Class")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_confusion_pairs(y_test, y_pred, le, top_n=5):
    confusion_pairs = []
    for i, (true, pred) in enumerate(zip(y_test, y_pred)):
        if true != pred:
            true_label = le.classes_[true]
            pred_label = le.classes_[pred]
            confusion_pairs.append((true_label, pred_label))

    confusion_counts = Counter(confusion_pairs)
    print("\nMost Common Confusions:")
    for (true_class, pred_class), count in confusion_counts.most_common(top_n):
        print(f"  {true_class} -> {pred_class}: {count} times")

    return confusion_counts


def plot_computational_cost(grid_search_all_results, X_train, X_test, y_train, top_n=5):
    from scripts.performance import get_top_n_models

    print("Running Computational Cost Analysis...")
    performance_data = []

    top_results = get_top_n_models(grid_search_all_results, n=top_n)

    print(f"Analyzing top {top_n} models:\n")

    for item in top_results:
        group_name = item['group']
        model_name = item['model']
        result = item['result']
        accuracy = item['test_accuracy']

        feature_indices = result['feature_indices']

        X_train_group = X_train[:, feature_indices]
        X_test_group = X_test[:, feature_indices]

        scaler = result['scaler']
        X_train_scaled_group = scaler.fit_transform(X_train_group)
        X_test_scaled_group = scaler.transform(X_test_group)

        model = clone(result['best_model'])

        start_train = time.time()
        model.fit(X_train_scaled_group, y_train)
        train_time = time.time() - start_train

        start_pred = time.time()
        model.predict(X_test_scaled_group)
        pred_time = time.time() - start_pred

        performance_data.append({
            'Model': f"{model_name} ({group_name})",
            'Model_Short': model_name,
            'Feature_Group': group_name,
            'Training Time (s)': train_time,
            'Prediction Time (s)': pred_time,
            'Accuracy': accuracy,
            'N_Features': len(feature_indices)
        })
        print(f" {model_name} ({group_name})")

    df_perf = pd.DataFrame(performance_data)

    print(f"\n{'='*70}")
    print("Performance Summary:")
    print(f"{'='*70}")
    print(df_perf[['Model', 'Training Time (s)', 'Prediction Time (s)', 'Accuracy']].to_string(index=False))

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        df_perf['Prediction Time (s)'],
        df_perf['Accuracy'],
        s=df_perf['Training Time (s)'] * 1000,
        c=range(len(df_perf)),
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=1.5
    )

    for idx, row in df_perf.iterrows():
        plt.annotate(
            row['Model_Short'],
            (row['Prediction Time (s)'], row['Accuracy']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5)
        )

    plt.title("Trade-off: Speed vs. Accuracy\n(Bubble Size = Training Time)", fontsize=16, fontweight='bold')
    plt.xlabel("Inference/Prediction Time (seconds) -> Lower is Better", fontsize=12)
    plt.ylabel("Test Accuracy -> Higher is Better", fontsize=12)
    plt.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Model Index', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

    print("    TOP-LEFT corner = Best (High Accuracy, Fast Prediction)")
    print("    Larger bubbles = Longer training time")


def plot_model_stability(grid_search_all_results, X_train, y_train, df_filtered, test_fold=5, top_n=5):
    from scripts.performance import get_top_n_models

    cv_data = []

    top_results = get_top_n_models(grid_search_all_results, n=top_n)

    print(f"Analyzing stability of top {top_n} models...\n")

    train_mask = df_filtered['fold'] != test_fold
    train_folds = df_filtered.loc[train_mask, 'fold'].values
    ps = PredefinedSplit(test_fold=train_folds - 1)

    for item in top_results:
        group_name = item['group']
        model_name = item['model']
        result = item['result']

        feature_indices = result['feature_indices']
        X_train_group = X_train[:, feature_indices]

        scaler = result['scaler']
        X_train_scaled_group = scaler.fit_transform(X_train_group)

        model = clone(result['best_model'])

        scores = cross_val_score(model, X_train_scaled_group, y_train, cv=ps, scoring='accuracy')

        print(f"  {model_name} ({group_name})")
        print(f"   Fold scores: {scores}")
        print(f"   Mean: {scores.mean():.4f}, Std: {scores.std():.4f}\n")

        for fold_idx, score in enumerate(scores, 1):
            cv_data.append({
                'Model': f"{model_name} ({group_name})",
                'Model_Short': model_name,
                'Feature_Group': group_name,
                'Fold': fold_idx,
                'Accuracy': score
            })

    df_cv = pd.DataFrame(cv_data)

    print(f"{'='*70}")
    print("Stability Summary:")
    print(f"{'='*70}")
    stability_stats = df_cv.groupby('Model')['Accuracy'].agg(['mean', 'std', 'min', 'max'])
    print(stability_stats)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_cv, x='Model', y='Accuracy', palette='Set3', linewidth=1.5)

    sns.stripplot(data=df_cv, x='Model', y='Accuracy', color='black', alpha=0.5, size=8, jitter=True)

    plt.title("Model Stability Analysis\n(Cross-Validation Score Distribution Across Folds)", fontsize=15, fontweight='bold')
    plt.ylabel("Accuracy per Fold", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(y=df_cv['Accuracy'].mean(), color='red', linestyle='--', linewidth=1, alpha=0.5, label='Overall Mean')
    plt.legend()
    plt.tight_layout()
    plt.show()
