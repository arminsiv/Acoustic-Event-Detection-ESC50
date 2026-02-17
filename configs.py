"""
Module: configs.py
Purpose: Configuration constants for ESC-50 sound classification
Authors: Armin Siavashi, Sepehr Farrokhi
Date: February 2025

This module defines:
- CATEGORIES: Dictionary mapping category names to selected sound classes
- SELECTED_CLASSES: Flattened list of all 10 sound classes
- FINAL_TEST_FOLD: The fold number reserved for final testing (fold 5)

Usage: Import these constants in main notebooks and scripts
"""

CATEGORIES = {
    'Animals': ['dog', 'rooster'],
    'Natural soundscapes & water sounds': ['thunderstorm', 'sea_waves'],
    'Human sounds': ['snoring', 'sneezing'],
    'Interior/domestic sounds': ['clock_alarm', 'vacuum_cleaner'],
    'Exterior/urban noises': ['siren', 'helicopter']
}

SELECTED_CLASSES = sum(CATEGORIES.values(), [])

FINAL_TEST_FOLD = 5