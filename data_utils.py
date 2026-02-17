"""
Module: data_utils.py
Purpose: Load and prepare ESC-50 audio data for machine learning
Authors: Armin Siavashi, Sepehr Farrokhi
Date: February 2025

This module provides functions for:
- Loading audio files from disk
- Building feature matrices using extraction functions
- Encoding class labels
- Splitting data into train/test sets using predefined folds

Dependencies: librosa, scikit-learn, pandas, numpy
"""

import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder


def load_audio_data(df_filtered, audio_path):
    audio_data = []
    for file in df_filtered['filename']:
        file_path = os.path.join(audio_path, file)
        y, sr = librosa.load(file_path, sr=None)
        audio_data.append((file, y, sr))
    print(f"Loaded {len(audio_data)} audio files.")
    return audio_data


def build_feature_matrix(df_filtered, audio_path, extract_fn):
    X = []
    y = []
    filenames = []

    for idx, row in df_filtered.iterrows():
        file_path = os.path.join(audio_path, row['filename'])
        try:
            features = extract_fn(file_path)
            X.append(features)
            y.append(row['category'])
            filenames.append(row['filename'])
        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")

    X = np.array(X)
    y = np.array(y)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of samples: {len(y)}")
    print(f"Number of features per sample: {X.shape[1]}")
    print(f"\nClass distribution:")
    print(pd.Series(y).value_counts().sort_index())

    return X, y, filenames


def encode_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return y_encoded, le


def split_data(X, y_encoded, df_filtered, test_fold=5):
    train_mask = df_filtered['fold'] != test_fold
    test_mask = df_filtered['fold'] == test_fold

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y_encoded[train_mask]
    y_test = y_encoded[test_mask]

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"\nClass distribution in train set:")
    print(pd.Series(y_train).value_counts().sort_index())
    print(f"\nClass distribution in test set:")
    print(pd.Series(y_test).value_counts().sort_index())

    return X_train, X_test, y_train, y_test, train_mask, test_mask
