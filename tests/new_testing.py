"""
Real-Data RTL-Aware Full SHAP Suite (Terminal-Friendly)
─────────────────────────────────────────────────────
Trains a Logistic Regression model on a real Urdu text dataset, computes 
optimized SHAP values for a targeted tweet string, filters out zero-impact 
absent features, and generates all 9 clean, RTL-ready structural plots saved as PNGs.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Dynamic Path Setup for Repositories 
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
PARENT_PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, ".."))

sys.path.append(os.path.join(PARENT_PROJECT_ROOT, "src"))  
sys.path.insert(0, PARENT_PROJECT_ROOT) 

# 2. Import Native & RTL-Aware Visualization Libraries
import matplotlib
matplotlib.use("Agg") # Non-GUI terminal rendering backend
import matplotlib.pyplot as plt

import shap
import shap_rtl
from shap_rtl.plots import (
    waterfall, bar, beeswarm, summary_plot, heatmap, decision_plot, scatter, dependence_plot, force_plot
)
from shap_rtl.rtl_utils import _find_rtl_font, add_rtl_title

# -----------------------------------------------------------------
# 3. Handle Matplotlib Urdu Font Registration
# -----------------------------------------------------------------
plt.rcdefaults()
font_path, font_name = _find_rtl_font()
print(f"Using font : {font_name}")
print(f"Font path  : {font_path}")
print(f"shap_rtl   : v{shap_rtl.__version__}")

from matplotlib import font_manager
try:
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = prop.get_name()
    print(f"Successfully registered font name: {prop.get_name()}")
except Exception as e:
    print(f"Font injection step skipped/failed: {e}")

plt.rcParams['axes.unicode_minus'] = False  

# -----------------------------------------------------------------
# 4. Load Data & Train Model
# -----------------------------------------------------------------
file_name = r"C:\Users\USER\OneDrive - Higher Education Commission\Dr. shahid project\SHAP_package\Preprocessed_violent_views_dataset .xlsx"

if not os.path.exists(file_name):
    print(f"❌ Excel file '{file_name}' not found! Please check path configuration.")
    sys.exit()

print(" -> Loading Dataset...")
df = pd.read_excel(file_name)
texts = df['cleaned_text'].astype(str).fillna('')
labels = df['label']

print(" -> Generating Vocabulary & Tokenizing Feature Array...")
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), max_features=1000)
X = vectorizer.fit_transform(texts).toarray()

print(" -> Training Logistic Regression Classifier Model...")
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X, labels)

# -----------------------------------------------------------------
# 5. SHAP Explainer Setup & Filter Out Absent Features
# -----------------------------------------------------------------
print(" -> Initializing Linear SHAP Explainer Engine...")
masker = shap.maskers.Independent(data=X, max_samples=100)
explainer = shap.LinearExplainer(model, masker=masker)

# Target sentence for analysis
user_tweet = "سب کافر دشمنوں کا سر تن سے جدا کر دو"
instance_vectorized = vectorizer.transform([user_tweet]).toarray()[0]
all_feature_names = list(vectorizer.get_feature_names_out())

print(" -> Running SHAP valuations across tokens...")
shap_values_raw = explainer(instance_vectorized.reshape(1, -1))

# --- THE FILTER TRICK: Extract indices where the word is actually in the text ---
present_indices = np.where(instance_vectorized > 0)[0]

# Filter arrays down strictly to the tokens present in your user_tweet
filtered_values = shap_values_raw.values[0][present_indices]
filtered_data = instance_vectorized[present_indices]
filtered_names = [all_feature_names[idx] for idx in present_indices]

# Rebuild single-row Explanation wrapper containing ONLY the active present tokens
formatted_explanation = shap.Explanation(
    values=filtered_values,
    base_values=shap_values_raw.base_values[0],
    data=filtered_data,
    feature_names=filtered_names
)

# For multi-row dependent plots (Summary, Heatmap, Decision), build an active 2D slice
multi_instance_raw = X[:30] 
multi_shap_raw = explainer(multi_instance_raw)

multi_explanation_filtered = shap.Explanation(
    values=multi_shap_raw.values[:, present_indices],
    base_values=multi_shap_raw.base_values,
    data=multi_instance_raw[:, present_indices],
    feature_names=filtered_names
)

# Configuration Helpers
num_active_features = len(filtered_names)
print(f" -> Generating {num_active_features} text feature plot configurations...")

# Target save directory tracking fixed exactly to the tests folder
SAVE_TARGET_DIR = r"C:\Users\USER\OneDrive - Higher Education Commission\Dr. shahid project\SHAP_package\tests"
os.makedirs(SAVE_TARGET_DIR, exist_ok=True)

def _save_plot(filename, title_text):
    # Sanitize and force fallback fonts for tick labels to avoid X-axis tofu blocks globally
    try:
        ax = plt.gca()
        for tick in ax.get_xticklabels():
            tick.set_fontfamily("DejaVu Sans")
    except Exception:
        pass

    add_rtl_title(title_text)
    plt.tight_layout()
    out_path = os.path.join(SAVE_TARGET_DIR, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"   Saved -> {out_path}")
    plt.close()

# -----------------------------------------------------------------
# 6. Generate Complete RTL-Aware SHAP Plots Suite (All 9 Plots)
# -----------------------------------------------------------------

# Plot 1: Clean Waterfall Plot (Single Instance)
print(" -> Plotting 1/9: Waterfall...")
plt.figure(figsize=(10, 6))
waterfall(formatted_explanation, max_display=num_active_features, show=False)
_save_plot("shap_1_waterfall.png", "واٹرفال پلاٹ — متشدد خیالات کی درجہ بندی")

# Plot 2: Clean Bar Plot (Single Instance)
print(" -> Plotting 2/9: Bar...")
plt.figure(figsize=(10, 6))
bar(formatted_explanation, max_display=num_active_features, show=False)
_save_plot("shap_2_bar.png", "بار پلاٹ — منڈل کی خصوصیات کی اہمیت")

# Plot 3: Summary / Beeswarm Plot (Distribution across samples)
print(" -> Plotting 3/9: Beeswarm...")
plt.figure(figsize=(10, 6))
beeswarm(multi_explanation_filtered, max_display=num_active_features, show=False)
_save_plot("shap_3_beeswarm.png", "سمری بی سوارم پلاٹ — خصوصیات کی تقسیم")

# Plot 4: Classic Summary Plot
print(" -> Plotting 4/9: Summary...")
plt.figure(figsize=(10, 6))
summary_plot(multi_explanation_filtered.values, features=multi_explanation_filtered.data, 
             feature_names=filtered_names, max_display=num_active_features, show=False)
_save_plot("shap_4_summary.png", "خصوصیات کا خلاصہ پلاٹ")

# Plot 5: Heatmap Plot (Sample patterns vs Feature impact)
print(" -> Plotting 5/9: Heatmap...")
plt.figure(figsize=(10, 6))
heatmap(multi_explanation_filtered, max_display=num_active_features, show=False)
_save_plot("shap_5_heatmap.png", "ہیٹ میپ پلاٹ — خصوصیات کے اثرات کا میٹرکس")

# Plot 6: Decision Plot (Visualizing trajectory towards prediction)
print(" -> Plotting 6/9: Decision...")
plt.figure(figsize=(10, 6))
decision_plot(
    float(formatted_explanation.base_values), 
    formatted_explanation.values, 
    feature_names=filtered_names,
    show=False
)
_save_plot("shap_6_decision.png", "ڈیسیژن پلاٹ — ماڈل کا فیصلہ سازی کا سفر")

# Plot 7: Scatter Plot (Focusing on a specific word's distribution)
print(" -> Plotting 7/9: Scatter...")
plt.figure(figsize=(10, 6))
scatter(multi_explanation_filtered[:, 0], show=False)
_save_plot("shap_7_scatter.png", f"اسکیٹر پلاٹ — لفظ '{filtered_names[0]}' کا اثر تغیر")

# Plot 8: Dependence Plot (Interaction between primary and secondary features)
print(" -> Plotting 8/9: Dependence...")
plt.figure(figsize=(10, 6))
dependence_plot(
    0, 
    multi_explanation_filtered.values, 
    multi_explanation_filtered.data, 
    feature_names=filtered_names, 
    show=False
)
_save_plot("shap_8_dependence.png", f"ڈپینڈینس پلاٹ — تعامل لفظ '{filtered_names[0]}'")

# Plot 9: Force Plot (Visual force breakdown accumulation)
print(" -> Plotting 9/9: Force...")
# Construct expected dictionary data block for matplotlib force plotting explicitly
force_data_dict = {
    "baseValue": float(formatted_explanation.base_values),
    "outValue": float(formatted_explanation.base_values + np.sum(formatted_explanation.values)),
    "outNames": ["Prediction outcome"],
    "featureNames": filtered_names,
    "features": {i: {"value": float(v), "effect": float(e)} for i, (v, e) in enumerate(zip(formatted_explanation.data, formatted_explanation.values))},
    "link": "identity"
}

force_plot(
    force_data_dict,
    figsize=(14, 3),
    show=False
)
_save_plot("shap_9_force.png", "فورس پلاٹ — متبادل خصوصیات کا مجموعی دباؤ")

print("\n" + "=" * 60)
print(" -> Complete SHAP analysis suite finished! 9/9 plots generated successfully.")
print("=" * 60)