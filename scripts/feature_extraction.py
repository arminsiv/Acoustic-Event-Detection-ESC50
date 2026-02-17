"""
Module: feature_extraction.py
Purpose: Extract 116 audio features from ESC-50 sound files
Authors: Armin Siavashi, Sepehr Farrokhi
Date: February 2025

This module extracts comprehensive audio features including:
- MFCCs and Delta MFCCs (timbral features)
- Spectral features (centroid, rolloff, bandwidth, flatness, contrast)
- Temporal features (ZCR, RMS, envelope)
- Pitch and harmonic features (HNR, pitch statistics)
- Musical features (Chroma STFT, Tonnetz)

Total: 116 features per audio clip

Dependencies: librosa, numpy
"""

import numpy as np
import librosa
import warnings


def extract_features(audio_path):
    """
    Extract comprehensive audio features:
    - EXISTING (80 features): MFCC, Spectral, ZCR, RMS, Pitch, HNR...
    - NEW (36 features): Chroma STFT, Tonnetz (Harmonic features)

    Total: 116 features
    """
    y, sr = librosa.load(audio_path, sr=22050)

    features = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)

    features.extend(mfcc.mean(axis=1))
    features.extend(mfcc.std(axis=1))
    features.extend(delta_mfcc.mean(axis=1))
    features.extend(delta_mfcc.std(axis=1))

    # Spectral
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(sc.mean())
    features.append(sc.std())

    ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(ro.mean())
    features.append(ro.std())

    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(bw.mean())
    features.append(bw.std())

    # Temporal (ZCR, RMS)
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(zcr.mean())
    features.append(zcr.std())

    rms = librosa.feature.rms(y=y)
    features.append(rms.mean())
    features.append(rms.std())

    # Spectral flatness & Contrast
    flat = librosa.feature.spectral_flatness(y=y)
    features.append(flat.mean())
    features.append(flat.std())

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(contrast.mean(axis=1))  # 7

    # Pitch Features
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)

    if len(pitch_values) > 0:
        features.append(np.mean(pitch_values))
        features.append(np.std(pitch_values))
        features.append(np.min(pitch_values))
        features.append(np.max(pitch_values))
    else:
        features.extend([0, 0, 0, 0])

    # Envelope
    envelope = np.abs(librosa.util.normalize(y))
    features.append(envelope.mean())
    features.append(envelope.std())
    features.append(envelope.max())

    # HNR
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_energy = np.sum(y_harmonic ** 2)
    percussive_energy = np.sum(y_percussive ** 2)

    if percussive_energy > 0:
        hnr = harmonic_energy / percussive_energy
        features.append(hnr)
        features.append(np.log1p(hnr))
    else:
        features.extend([0, 0])

    # Chroma STFT (Indices 80 to 103)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(chroma.mean(axis=1))
    features.extend(chroma.std(axis=1))

    # Tonnetz (Indices 104 to 115)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        features.extend(tonnetz.mean(axis=1))
        features.extend(tonnetz.std(axis=1))
    except:
        features.extend(np.zeros(12))

    return np.array(features)


def get_feature_names():
    feature_names = []
    # MFCCs (26)
    feature_names.extend([f'mfcc_mean_{i}' for i in range(13)])
    feature_names.extend([f'mfcc_std_{i}' for i in range(13)])
    # Delta MFCCs (26)
    feature_names.extend([f'delta_mfcc_mean_{i}' for i in range(13)])
    feature_names.extend([f'delta_mfcc_std_{i}' for i in range(13)])
    # Spectral (14)
    feature_names.extend(['spectral_centroid_mean', 'spectral_centroid_std'])
    feature_names.extend(['spectral_rolloff_mean', 'spectral_rolloff_std'])
    feature_names.extend(['spectral_bandwidth_mean', 'spectral_bandwidth_std'])
    feature_names.extend(['zcr_mean', 'zcr_std'])
    feature_names.extend(['rms_mean', 'rms_std'])
    feature_names.extend(['spectral_flatness_mean', 'spectral_flatness_std'])
    # Contrast (7)
    feature_names.extend([f'spectral_contrast_{i}' for i in range(7)])
    # Pitch (4)
    feature_names.extend(['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max'])
    # Envelope (3)
    feature_names.extend(['envelope_mean', 'envelope_std', 'envelope_max'])
    # HNR (2)
    feature_names.extend(['hnr', 'hnr_log'])
    # Chroma (24)
    feature_names.extend([f'chroma_mean_{i}' for i in range(12)])
    feature_names.extend([f'chroma_std_{i}' for i in range(12)])
    # Tonnetz (12)
    feature_names.extend([f'tonnetz_mean_{i}' for i in range(6)])
    feature_names.extend([f'tonnetz_std_{i}' for i in range(6)])
    return feature_names


def get_feature_groups(feature_names):
    feature_groups = {
        'temporal': [i for i, name in enumerate(feature_names)
                     if 'delta_mfcc' in name or 'zcr' in name or 'envelope' in name],
        'harmonic': [i for i, name in enumerate(feature_names)
                     if 'mfcc' in name or 'pitch' in name or 'hnr' in name],
        'spectral_brightness': [i for i, name in enumerate(feature_names)
                                if 'mfcc' in name or 'spectral_centroid' in name or 'spectral_contrast' in name],
        'noise_based': [i for i, name in enumerate(feature_names)
                        if 'spectral_contrast' in name or 'hnr' in name or 'spectral_flatness' in name],
        'general_set_1': [i for i, name in enumerate(feature_names)
                          if 'mfcc' in name or 'delta_mfcc' in name or 'zcr' in name or 'envelope' in name],
        'general_set_2': [i for i, name in enumerate(feature_names)
                          if 'mfcc' in name or 'spectral_centroid' in name or 'spectral_contrast' in name or 'pitch' in name],
        'last_set_chroma_tonnetz': [i for i, name in enumerate(feature_names)
                                     if 'chroma' in name or 'tonnetz' in name],
        'all_features': list(range(len(feature_names)))
    }
    return feature_groups
