"""
shap_rtl.explainer
──────────────────
RTL-aware SHAP explanation generation with LLM-powered feature analysis.
"""

import os
import sys
import copy
import time
import re
import numpy as np
from IPython.display import display, HTML

# RTL text processing
import arabic_reshaper
from bidi.algorithm import get_display

# SHAP and visualization
import shap
import matplotlib.pyplot as plt

# LLM integration
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class RTLSHAPExplainer:
    """
    RTL-aware SHAP explainer with LLM-generated natural language explanations.
    """
    
    def __init__(
        self,
        language: str = "Urdu",
        model=None,
        vectorizer=None,
        api_type: str = "groq"
    ):
        self.language = language.strip().capitalize()
        self.model = model
        self.vectorizer = vectorizer
        self.api_type = api_type
        self._llm = None
        
        # RTL display width
        self.RW = 120
        
        # Supported RTL languages
        self.rtl_languages = ["Urdu", "Arabic", "Persian", "Hebrew"]
        
        # Store translated sample text to avoid re-translating
        self._eng_sample_text = None
        
        # Store features for dependence plot
        self._cached_features = None
    
    @property
    def llm(self):
        """Lazy-initialize the LLM based on API type."""
        if self._llm is None:
            if self.api_type == "groq":
                api_key = os.getenv("GROQ_API_KEY")
                self._llm = ChatOpenAI(
                    model="llama-3.3-70b-versatile",
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1",
                    temperature=0.1,
                )
            else:  # github / azure
                api_key = os.getenv("GITHUB_TOKEN")
                self._llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=api_key,
                    base_url="https://models.inference.ai.azure.com",
                    temperature=0.1,
                )
        return self._llm
    
    @llm.setter
    def llm(self, value):
        """Allow tests to inject a custom LLM or mock."""
        self._llm = value
    
    def _invoke_llm(self, prompt: str, retries: int = 2, delay: float = 3.0) -> str:
        """Call the LLM with automatic retry on rate-limit errors."""
        last_exc = None
        for attempt in range(retries + 1):
            try:
                return self.llm.invoke(prompt).content.strip()
            except Exception as exc:
                last_exc = exc
                if attempt < retries:
                    time.sleep(delay)
        raise last_exc
    
    # ── RTL Text Helpers ──────────────────────────────────────────────────────
    
    def _fix_for_terminal(self, text: str) -> str:
        """Reshape RTL text so joined letters display correctly in terminals."""
        try:
            return arabic_reshaper.reshape(str(text))
        except Exception:
            return str(text)
    
    def _fix_for_plot(self, text: str) -> str:
        """Reshape + BiDi RTL text for Matplotlib."""
        try:
            reshaped = arabic_reshaper.reshape(str(text))
            return get_display(reshaped)
        except Exception:
            return str(text)
    
    # ── Translation Methods ──────────────────────────────────────────────────
    
    def _translate_text(self, text: str) -> str:
        """Translate text using LLM."""
        if not text:
            return text
        
        try:
            prompt = f"""
            Translate this {self.language} text to English.
            Return ONLY the translation, no extra text.
            
            Text: {text}
            """
            
            response = self._invoke_llm(prompt)
            return response.replace('"', '').strip()
            
        except Exception as e:
            print(f"[WARNING] Translation failed: {e}")
            return text
    
    def _translate_sample(self, sample_text: str) -> str:
        """
        Translate sample text and cache it.
        This prevents re-translating multiple times.
        """
        if self._eng_sample_text is not None:
            return self._eng_sample_text
        
        if not sample_text:
            return None
        
        print("\n[TRANSLATING SAMPLE TEXT TO ENGLISH...]")
        self._eng_sample_text = self._translate_text(sample_text)
        
        if self._eng_sample_text and self._eng_sample_text != sample_text:
            print(f"\nOriginal {self.language} Text:")
            print(f"  {self._fix_for_terminal(sample_text)}")
            print(f"\nEnglish Translation:")
            print(f"  {self._eng_sample_text}")
            print("-" * 80)
        else:
            print("[WARNING] Could not translate sample text.")
            self._eng_sample_text = sample_text
        
        return self._eng_sample_text
    
    def _translate_features(self, feature_names: list) -> list:
        """Translate feature names to English."""
        if not feature_names:
            return []
        
        try:
            name_list = ", ".join([str(f) for f in feature_names])
            prompt = f"""
            Translate these {self.language} feature names to English.
            Return ONLY a comma-separated list in the same order, no extra text:
            {name_list}
            """
            response = self._invoke_llm(prompt)
            eng_names = [n.strip() for n in response.split(",")]
            
            while len(eng_names) < len(feature_names):
                eng_names.append(feature_names[len(eng_names)])
            eng_names = eng_names[:len(feature_names)]
            return eng_names
            
        except Exception:
            return [str(f) for f in feature_names]
    
    def _translate_label(self, label: str) -> str:
        """Translate prediction label to English."""
        try:
            prompt = f"""
            Translate '{label}' from {self.language} to English.
            Return ONLY the translation, no extra text.
            """
            return self._invoke_llm(prompt).replace('"', '')
        except Exception:
            return label
    
    # ── Feature Extraction ──────────────────────────────────────────────────
    
    def get_top_features(self, shap_values, num_features: int = 5) -> list:
        """Extract top N features ranked by absolute SHAP value."""
        values = shap_values.values
        feature_names = shap_values.feature_names
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(values.shape[-1])]
        
        if values.ndim == 2:
            mean_abs = np.mean(np.abs(values), axis=0)
            mean_vals = np.mean(values, axis=0)
        else:
            mean_abs = np.abs(values)
            mean_vals = values
        
        sorted_idx = np.argsort(mean_abs)[::-1][:num_features]
        
        return [(str(feature_names[i]), float(mean_vals[i])) for i in sorted_idx]
    
    # ── Plot Function Handler ──────────────────────────────────────────────────
    
    def _call_plot_function(self, plot_fn, shap_values):
        """
        Call a plot function with the appropriate arguments.
        Some plots (like dependence_plot) need extra parameters.
        """
        import inspect
        
        # Get the function signature
        sig = inspect.signature(plot_fn)
        
        # Check if the function needs 'features' parameter
        if 'features' in sig.parameters:
            # For dependence_plot, we need features
            if hasattr(shap_values, 'data') and shap_values.data is not None:
                features = shap_values.data
            else:
                # Fallback to cached features
                features = self._cached_features
            
            if features is not None:
                return plot_fn(shap_values=shap_values, features=features)
            else:
                print("[WARNING] Features not available for dependence plot.")
                print("[INFO] Please pass features to run_full_analysis()")
                return plot_fn(shap_values=shap_values)
        else:
            # For other plots (bar, summary, waterfall, etc.)
            return plot_fn(shap_values)
    
    # ── Prompt Templates ─────────────────────────────────────────────────────
    
    def _rtl_prompt(self, feat_name: str, shap_val: float, prediction_label: str) -> str:
        """Generate RTL-language explanation prompt."""
        impact = "positively" if shap_val > 0 else "negatively"
        
        return f"""
        SYSTEM: You are an AI Explainability Assistant.
        LANGUAGE: {self.language}
        PREDICTION: {prediction_label}
        FEATURE: "{feat_name}"
        SHAP VALUE: {shap_val:+.4f}
        TASK: Explain why the feature "{feat_name}" {impact} impacts the prediction "{prediction_label}" in {self.language} only.
        INSTRUCTIONS: Respond in ONE concise paragraph in {self.language} only.
        Domain and linguistic analysis only. No English, no raw numeric scores.
        """
    
    def _eng_prompt(self, feat_name: str, shap_val: float, prediction_label: str, eng_translation: str) -> str:
        """Generate English explanation prompt."""
        impact = "positively" if shap_val > 0 else "negatively"
        
        return f"""
        SHAP Analysis: In the context of the prediction '{prediction_label}' 
        (English: '{eng_translation}'), explain in one concise English paragraph 
        why the {self.language} feature '{feat_name}' 
        (SHAP value: {shap_val:+.4f}) {impact} impacts the model output.
        """
    
    # ── Main Workflow ──────────────────────────────────────────────────────
    
    def run_full_analysis(
        self,
        shap_values,
        prediction_label: str = "prediction",
        plot_fn=None,
        max_features: int = 10,
        allow_positive_negative_split: bool = True,
        sample_text: str = None,
        features=None  # NEW: For dependence plot
    ):
        """
        Run the complete RTL SHAP explanation workflow.
        
        Parameters
        ----------
        shap_values : shap.Explanation
            SHAP explanation object
        prediction_label : str
            Prediction label in the RTL language
        plot_fn : callable
            Plot function to use (bar, summary_plot, dependence_plot, etc.)
        max_features : int
            Maximum features to display
        allow_positive_negative_split : bool
            Whether to show positive/negative split (deprecated - kept for compatibility)
        sample_text : str, optional
            Sample text in RTL language for translation
        features : array-like, optional
            Feature values for dependence plot (required for dependence_plot)
        
        Workflow:
        1. Display RTL plot
        2. Ask: How many features to explain in RTL?
        3. Generate RTL explanations
        4. Ask: Would you like English plot?
        5. If Yes → Translate sample (once), display English plot
        6. Ask: Would you like English explanations?
        7. If Yes → Ask: How many features? → Generate English explanations
        """
        # Store features for dependence plot
        if features is not None:
            self._cached_features = features
        elif hasattr(shap_values, 'data') and shap_values.data is not None:
            self._cached_features = shap_values.data
        
        print("\n" + "=" * 80)
        print(f"{self.language.upper()} SHAP EXPLANATIONS".center(80))
        print(f"Prediction: {prediction_label}".center(80))
        print("=" * 80)
        
        # ── Step 1: Display RTL Plot ──────────────────────────────────────
        if plot_fn is not None:
            print(f"\n[GENERATING {self.language.upper()} SHAP PLOT...]")
            self._call_plot_function(plot_fn, shap_values)
        else:
            print("\n[INFO] No plot_fn provided. Skipping visualization.")
        
        # ── Step 2: Ask for RTL Explanation ──────────────────────────────
        print(f"\n[{self.language} Explanation Mode]")
        
        n = None
        while True:
            try:
                user_input = input(f"How many features would you like explained in {self.language}? (Enter 0 or press ESC to skip): ")
                
                if user_input.strip() == "" or user_input.lower() == "escape" or user_input == "0":
                    print("\n[INFO] Skipping RTL explanation.")
                    self._ask_english_plot(shap_values, prediction_label, plot_fn, sample_text)
                    return
                
                n = int(user_input)
                if n <= 0:
                    print("[INFO] Skipping RTL explanation.")
                    self._ask_english_plot(shap_values, prediction_label, plot_fn, sample_text)
                    return
                
                n = min(n, max_features)
                break
                
            except ValueError:
                print("Please enter a valid number (or press ESC/Enter to skip).")
                continue
            except KeyboardInterrupt:
                print("\n[INFO] Operation cancelled by user.")
                return
        
        # ── Step 3: Generate RTL Explanations ─────────────────────────────
        top_features = self.get_top_features(shap_values, n)
        
        print("-" * self.RW)
        print(f"PREDICTION: {prediction_label}".rjust(self.RW))
        print("-" * self.RW)
        
        print(f"\nAnalysis of Top {n} SHAP Features ({self.language})".rjust(self.RW))
        print("-" * 50)
        
        for feat_name, shap_val in top_features:
            try:
                display_name = self._fix_for_terminal(feat_name)
                print(f"\nFeature: {display_name} | Value: {shap_val:+.4f}")
                print("-" * 40)
                
                prompt = self._rtl_prompt(feat_name, shap_val, prediction_label)
                response = self._invoke_llm(prompt)
                
                fixed_response = self._fix_for_terminal(response)
                print(fixed_response.rjust(self.RW))
                print("-" * 40)
                
            except Exception as e:
                print(f"Feature: {feat_name} (API unavailable: {str(e)})".rjust(self.RW))
        
        print("\n" + "=" * 80)
        
        # ── Step 4: Ask for English ──────────────────────────────────────
        if self.language in self.rtl_languages:
            self._ask_english_plot(shap_values, prediction_label, plot_fn, sample_text, top_features)
    
    # ── English Plot and Explanation Methods ────────────────────────────────
    
    def _ask_english_plot(self, shap_values, prediction_label, plot_fn, sample_text, top_features=None):
        """Ask user if they want to see the plot in English."""
        while True:
            try:
                choice = input("\nWould you like to see the plot in English? (Yes/No): ").strip().lower()
                
                if choice in ["yes", "y"]:
                    print("\n[INFO] Generating English plot...")
                    self._display_english_plot(
                        shap_values=shap_values,
                        prediction_label=prediction_label,
                        plot_fn=plot_fn,
                        sample_text=sample_text,
                        top_features=top_features
                    )
                    return
                elif choice in ["no", "n", ""]:
                    print("\n[INFO] Exiting. Thank you!")
                    return
                else:
                    print("Please enter Yes or No.")
                    
            except KeyboardInterrupt:
                print("\n[INFO] Exiting. Thank you!")
                return
            except Exception as e:
                print(f"[ERROR] {e}")
                return
    
    def _display_english_plot(self, shap_values, prediction_label, plot_fn, sample_text, top_features=None):
        """Display the English plot."""
        print("\n[INFO] Translating features to English...")
        
        # Translate sample text once (cached)
        eng_sample_text = None
        if sample_text:
            eng_sample_text = self._translate_sample(sample_text)
        
        # Translate feature names
        feature_names = list(shap_values.feature_names or [])
        eng_names = self._translate_features(feature_names)
        
        # Translate prediction label
        eng_label = self._translate_label(prediction_label)
        print(f"Translated Prediction Label: {eng_label}")
        
        # Rebuild explanation with English names
        eng_shap = self._rebuild_explanation_with_names(shap_values, eng_names)
        
        # Display English plot - use the handler for plots that need features
        if plot_fn is not None:
            print(f"\n[GENERATING ENGLISH SHAP PLOT...]")
            self._call_plot_function(plot_fn, eng_shap)
        else:
            print("\n[INFO] No plot_fn provided. Skipping English visualization.")
        
        # Ask if user wants explanation in English
        self._ask_english_explanation(eng_shap, prediction_label, eng_label, eng_sample_text, plot_fn, top_features)
    
    def _ask_english_explanation(self, eng_shap, prediction_label, eng_label, eng_sample_text, plot_fn, top_features=None):
        """Ask user if they want explanation in English."""
        while True:
            try:
                choice = input("\nWould you like explanation in English? (Yes/No): ").strip().lower()
                
                if choice in ["yes", "y"]:
                    self._run_english_explanations(
                        eng_shap=eng_shap,
                        prediction_label=prediction_label,
                        eng_label=eng_label,
                        eng_sample_text=eng_sample_text,
                        plot_fn=plot_fn,
                        top_features=top_features
                    )
                    return
                elif choice in ["no", "n", ""]:
                    print("\n[INFO] Exiting. Thank you!")
                    return
                else:
                    print("Please enter Yes or No.")
                    
            except KeyboardInterrupt:
                print("\n[INFO] Exiting. Thank you!")
                return
            except Exception as e:
                print(f"[ERROR] {e}")
                return
    
    def _run_english_explanations(self, eng_shap, prediction_label, eng_label, eng_sample_text, plot_fn, top_features=None, max_features=10):
        """Generate English explanations."""
        print("\n" + "=" * 80)
        print("ENGLISH EXPLANATIONS".center(80))
        print("=" * 80)
        
        # Display translated sample text (already translated)
        if eng_sample_text:
            print(f"\nTranslated Sample: {eng_sample_text}")
            print("-" * 80)
        
        # Ask how many features to explain in English
        print(f"\n[English Explanation Mode]")
        
        while True:
            try:
                user_input = input(
                    "How many features would you like explained in English? "
                    "(Press Enter or type 0 to exit): "
                ).strip()

                if user_input == "" or user_input == "0":
                    print("Exiting explanation mode.")
                    return

                n = int(user_input)

                if n <= 0:
                    print("Exiting explanation mode.")
                    return

                n = min(n, max_features)
                break

            except ValueError:
                print("Please enter a valid number.")
        
        # Get top features
        eng_top_features = self.get_top_features(eng_shap, n)
        
        print("-" * 80)
        print(f"PREDICTION: {eng_label}".center(80))
        print("-" * 80)
        
        print(f"\nAnalysis of Top {n} SHAP Features (English)")
        print("-" * 50)
        
        for feat_name, shap_val in eng_top_features:
            try:
                print(f"\nFeature: {feat_name} | Value: {shap_val:+.4f}")
                print("-" * 40)
                
                prompt = self._eng_prompt(feat_name, shap_val, prediction_label, eng_label)
                response = self._invoke_llm(prompt)
                print(response)
                print("-" * 40)
                
            except Exception as e:
                print(f"Feature: {feat_name} (API unavailable: {str(e)})")
        
        print("\n" + "=" * 80)
        print("Analysis complete.")
    
    def _rebuild_explanation_with_names(self, shap_values, new_names):
        """Rebuild SHAP Explanation with new feature names."""
        try:
            return shap.Explanation(
                values=shap_values.values,
                base_values=shap_values.base_values,
                data=shap_values.data,
                feature_names=new_names,
                display_data=shap_values.display_data,
            )
        except Exception:
            sv_eng = copy.copy(shap_values)
            try:
                object.__setattr__(sv_eng, "feature_names", new_names)
            except Exception:
                sv_eng.__dict__["feature_names"] = new_names
            return sv_eng


# ── Convenience Functions ──────────────────────────────────────────────

def create_explainer(language: str = "Urdu", model=None, vectorizer=None, api_type: str = "groq"):
    """Factory function to create an RTLSHAPExplainer instance."""
    return RTLSHAPExplainer(
        language=language,
        model=model,
        vectorizer=vectorizer,
        api_type=api_type
    )


if __name__ == "__main__":
    print("=" * 80)
    print("SHAP RTL EXPLAINER - Ready to use!".center(80))
    print("=" * 80)
    print("\n[INFO] Using LLM for translations")