"""
testing_urdu.py
===============
Generate All Urdu SHAP Plots for a Specific User Tweet

This script generates all RTL SHAP plots in Urdu using a specific tweet.
Automatically detects the label from the dataset.

Usage:
    python tests/testing_urdu.py
"""

import os
import sys
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import shap_rtl
from shap_rtl.plots import (
    waterfall, bar, beeswarm, summary_plot,
    heatmap, scatter, decision_plot, dependence_plot,
    force_plot
)

print("=" * 80)
print("GENERATING URDU SHAP PLOTS FOR USER TWEET")
print("=" * 80)


# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Urdu_language_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset path
DATA_FILE = r'C:\Users\USER\OneDrive - Higher Education Commission\Dr. shahid project\SHAP_package\Preprocessed_violent_views_dataset .xlsx'

# User tweet for testing
USER_TWEET = "سب کافر دشمنوں کا سر تن سے جدا کر دو"

print(f"User Tweet: {USER_TWEET}")
print(f"Output directory: {OUTPUT_DIR}")


# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/6] Loading Urdu dataset...")
df = pd.read_excel(DATA_FILE)
texts = df['cleaned_text'].astype(str).fillna('')
labels = df['label']
print(f"  Samples: {len(texts)}")
print(f"  Class distribution: {np.bincount(labels)}")


# ============================================================
# TRAIN MODEL
# ============================================================
print("\n[2/6] Training model...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=300)
X = vectorizer.fit_transform(texts).toarray()
model = LogisticRegression(max_iter=1000, class_weight='balanced').fit(X, labels)
print(f"  Features: {X.shape[1]}")


# ============================================================
# GET PREDICTION FOR USER TWEET
# ============================================================
print("\n[3/6] Predicting user tweet...")
user_instance = vectorizer.transform([USER_TWEET]).toarray()[0]
user_pred = model.predict(user_instance.reshape(1, -1))[0]
user_prob = model.predict_proba(user_instance.reshape(1, -1))[0]

# Automatically determine label
user_label = "تشدد" if user_pred == 1 else "غیر تشدد"
print(f"  Prediction: {user_pred} (Violent: {user_prob[1]:.3f})")
print(f"  Label: {user_label}")


# ============================================================
# COMPUTE SHAP VALUES FOR USER TWEET
# ============================================================
print("\n[4/6] Computing SHAP values for user tweet...")

# Get SHAP values for the user tweet
masker = shap.maskers.Independent(data=X, max_samples=100)
explainer_shap = shap.LinearExplainer(model, masker=masker)
user_shap_vals = explainer_shap(user_instance.reshape(1, -1))

# Get present features in the user tweet
present_idx = np.where(user_instance > 0)[0]
urdu_feature_names = list(vectorizer.get_feature_names_out())
user_feature_names = [urdu_feature_names[i] for i in present_idx]
user_shap_values = user_shap_vals.values[0][present_idx]
user_data = user_instance[present_idx]

# Create SHAP Explanation object for the user tweet
shap_exp_user = shap.Explanation(
    values=user_shap_values,
    base_values=user_shap_vals.base_values[0],
    data=user_data,
    feature_names=user_feature_names
)

print(f"  Features present: {len(user_feature_names)}")
print(f"  Top features: {user_feature_names[:5]}")
print(f"  Base value: {user_shap_vals.base_values[0]:.4f}")

# Also compute for random samples (for summary, heatmap, etc.)
n_samples = 50
random_indices = np.random.choice(len(df), n_samples, replace=False)
X_subset = X[random_indices]
shap_values = explainer_shap.shap_values(X_subset)

if isinstance(shap_values, list):
    shap_values_class = shap_values[1]
    base_value = explainer_shap.expected_value[1] if isinstance(
        explainer_shap.expected_value, list
    ) else explainer_shap.expected_value
else:
    shap_values_class = shap_values
    base_value = explainer_shap.expected_value

# Create combined explanation for multi-sample plots
shap_exp_combined = shap.Explanation(
    values=shap_values_class,
    base_values=base_value,
    data=X_subset,
    feature_names=urdu_feature_names
)

print(f"  Combined SHAP shape: {shap_exp_combined.values.shape}")


# ============================================================
# GENERATE PLOTS (NO DISPLAY, ONLY SAVE)
# ============================================================
print("\n[5/6] Generating and saving plots...")

def save_plot(filename, dpi=300, bbox_inches='tight'):
    """Helper to save plot with consistent settings (no display)."""
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=dpi, bbox_inches=bbox_inches)
    plt.close()
    print(f"  ✅ Saved: {filename}")


# 1. Waterfall Plot (User Tweet)
print("\n  Generating Waterfall Plot...")
plt.figure(figsize=(12, 8))
waterfall(
    shap_values=shap_exp_user,
    max_display=10,
    show=False
)
save_plot("1_waterfall_urdu.png")


# 2. Bar Plot (User Tweet)
print("  Generating Bar Plot...")
plt.figure(figsize=(12, 8))
bar(
    shap_values=shap_exp_user,
    max_display=10,
    show=False
)
save_plot("2_bar_urdu.png")


# 3. Beeswarm Plot
print("  Generating Beeswarm Plot...")
plt.figure(figsize=(14, 10))
beeswarm(
    shap_values=shap_exp_combined,
    max_display=15,
    show=False
)
save_plot("3_beeswarm_urdu.png")


# 4. Summary Plot
print("  Generating Summary Plot...")
plt.figure(figsize=(14, 10))
summary_plot(
    shap_values=shap_exp_combined,
    max_display=15,
    show=False
)
save_plot("4_summary_urdu.png")


# 5. Heatmap
print("  Generating Heatmap...")
plt.figure(figsize=(14, 10))
heatmap(
    shap_values=shap_exp_combined,
    max_display=15,
    show=False,
    instance_order=None
)
save_plot("5_heatmap_urdu.png")


# 6. Scatter Plot
print("  Generating Scatter Plot...")
plt.figure(figsize=(12, 8))
scatter(
    shap_values=shap_exp_combined,
    max_display=10,
    show=False
)
save_plot("6_scatter_urdu.png")


# 7. Decision Plot (User Tweet)
print("  Generating Decision Plot...")
plt.figure(figsize=(12, 8))
decision_plot(
    base_value=shap_exp_user.base_values,
    shap_values=shap_exp_user.values.reshape(1, -1),
    feature_names=user_feature_names,
    show=False
)
save_plot("7_decision_urdu.png")


# 8. Dependence Plot
print("  Generating Dependence Plot...")
mean_abs = np.abs(shap_exp_combined.values).mean(axis=0)
top_idx = np.argmax(mean_abs)
top_feature = shap_exp_combined.feature_names[top_idx]

print(f"  Top feature: {top_feature} (index: {top_idx})")

plt.figure(figsize=(14, 10))
dependence_plot(
    shap_values=shap_exp_combined,
    features=X_subset,
    feature_index=top_idx,
    feature_names=urdu_feature_names,
    show=False
)
save_plot("8_dependence_urdu.png")


# 9. Force Plot (User Tweet) - NEW!
print("  Generating Force Plot...")
plt.figure(figsize=(20, 4))
force_plot(
    shap_values=shap_exp_user,
    show=False
)
save_plot("9_force_urdu.png")


# ============================================================
# GENERATE INDEX FILE
# ============================================================
print("\n[6/6] Generating index file...")

