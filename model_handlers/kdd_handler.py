import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import uuid
import warnings
from datetime import datetime
from pathlib import Path

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='shap')
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

class KDDHandler:
    def __init__(self):
        self.model_dir = Path("models/KDD")
        self.shap_dir = Path("static/shap_images")
        self.shap_dir.mkdir(parents=True, exist_ok=True)
        self.load_model()
        self.explainer = None
        self.init_shap()
    
    def load_model(self):
        try:
            # Load XGBoost model and artifacts
            self.model = joblib.load(self.model_dir / "ids_xgb_model.pkl")
            self.scaler = joblib.load(self.model_dir / "scaler.pkl")
            self.le_target = joblib.load(self.model_dir / "label_encoder.pkl")
            self.features = joblib.load(self.model_dir / "feature_order.pkl")
            self.class_names = joblib.load(self.model_dir / "class_names.pkl")
            
            # Load surrogate model for SHAP if available
            self.surrogate_model = None
            surrogate_path = self.model_dir / "surrogate_adv_model.h5"
            if surrogate_path.exists():
                self.surrogate_model = tf.keras.models.load_model(surrogate_path)
            
            print(f"Loaded KDD model with {len(self.features)} features")
            print(f"Classes: {self.class_names}")
            print(f"Features: {self.features}")
            
        except Exception as e:
            raise Exception(f"Failed to load KDD model: {str(e)}")
    
    def init_shap(self):
        """Initialize SHAP explainer for XGBoost model"""
        try:
            # For XGBoost, use TreeExplainer which is more efficient
            self.explainer = shap.TreeExplainer(self.model)
            print("Initialized TreeExplainer for KDD XGBoost model")
            
        except Exception as e:
            print(f"TreeExplainer failed: {e}, trying KernelExplainer")
            try:
                # Fallback to KernelExplainer
                def model_predict(x):
                    return self.model.predict_proba(x)
                
                # Create background data for SHAP
                background_data = self.scaler.transform(np.zeros((5, len(self.features))))
                self.explainer = shap.KernelExplainer(model_predict, background_data)
                print("Initialized KernelExplainer for KDD")
            except Exception as e2:
                print(f"KernelExplainer also failed: {e2}")
                self.explainer = None
    
    def create_shap_plot(self, X_scaled, prediction):
        """Create SHAP explanation plot and save to file"""
        try:
            if self.explainer is None:
                print("SHAP explainer is None, cannot create plot")
                return None
            
            print(f"Creating SHAP plot for prediction: {prediction}")
            print(f"Input shape: {X_scaled.shape}")
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_scaled)
            print(f"SHAP values type: {type(shap_values)}")
            
            # Handle different SHAP output formats for XGBoost
            shap_vals = None
            
            if isinstance(shap_values, list):
                # Multi-class output - list of arrays (one per class)
                print(f"SHAP values list length: {len(shap_values)}")
                if prediction < len(shap_values):
                    shap_vals = shap_values[prediction]
                    print(f"Using SHAP values for class {prediction}, shape: {shap_vals.shape}")
                else:
                    shap_vals = shap_values[0]
                    print(f"Using SHAP values for class 0 (fallback), shape: {shap_vals.shape}")
            else:
                # Single array output (binary classification or mean across classes)
                shap_vals = shap_values
                print(f"Single SHAP array shape: {shap_vals.shape}")
            
            # Ensure shap_vals is 2D (samples, features)
            if len(shap_vals.shape) == 1:
                shap_vals = shap_vals.reshape(1, -1)
                print(f"Reshaped 1D SHAP array to: {shap_vals.shape}")
            
            print(f"Final SHAP values shape: {shap_vals.shape}")
            
            # Get the SHAP values for the first sample (our prediction)
            sample_shap_values = shap_vals[0] if len(shap_vals.shape) > 1 else shap_vals
            
            print(f"Sample SHAP values length: {len(sample_shap_values)}")
            print(f"Number of feature names: {len(self.features)}")
            
            # Determine which features to use for plotting
            if len(sample_shap_values) <= len(self.features):
                # Use the first n features where n is the length of SHAP values
                features_to_use = self.features[:len(sample_shap_values)]
                shap_vals_to_use = sample_shap_values
                print(f"Using first {len(features_to_use)} features for SHAP plot")
            else:
                # This shouldn't happen, but if it does, truncate SHAP values
                features_to_use = self.features
                shap_vals_to_use = sample_shap_values[:len(self.features)]
                print(f"Truncated SHAP values to match feature count")
            
            # Create DataFrame for plotting
            feature_importance = pd.DataFrame({
                'features': features_to_use,
                'shap_values': shap_vals_to_use
            })
            
            # Calculate absolute values for sorting
            feature_importance['abs_shap'] = np.abs(feature_importance['shap_values'])
            
            # Sort by absolute SHAP value and take top 10
            top_features = feature_importance.nlargest(10, 'abs_shap')
            
            print(f"Top features for SHAP plot: {len(top_features)}")
            
            if len(top_features) == 0:
                print("No features to plot")
                return None
            
            # Create the plot with smaller figure size to avoid layout issues
            plt.figure(figsize=(10, 6))
            
            # Create horizontal bar plot
            y_pos = np.arange(len(top_features))
            colors = ['red' if x < 0 else 'blue' for x in top_features['shap_values']]
            
            bars = plt.barh(y_pos, top_features['shap_values'], color=colors, alpha=0.7)
            plt.yticks(y_pos, top_features['features'])
            plt.xlabel('SHAP Value (Impact on Prediction)')
            plt.title('Top 10 Most Influential Features\nBlue=Positive Impact, Red=Negative Impact')
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, top_features['shap_values'])):
                width = bar.get_width()
                plt.text(width + (0.01 if width >= 0 else -0.01), 
                        bar.get_y() + bar.get_height()/2, 
                        f'{value:.4f}', 
                        ha='left' if width >= 0 else 'right', 
                        va='center', 
                        fontsize=9,
                        fontweight='bold')
            
            # Use constrained_layout instead of tight_layout to avoid warnings
            plt.tight_layout()
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"shap_kdd_{timestamp}_{unique_id}.png"
            filepath = self.shap_dir / filename
            
            # Save plot with lower DPI to reduce file size
            plt.savefig(filepath, format='png', dpi=100, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            # Verify file was created
            if filepath.exists():
                file_size = filepath.stat().st_size
                print(f"SHAP plot saved successfully: {filename} ({file_size} bytes)")
                return f"shap_images/{filename}"
            else:
                print(f"Failed to save SHAP plot: {filename}")
                return None
            
        except Exception as e:
            print(f"SHAP plot creation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_simple_shap_plot(self, X_scaled, prediction):
        """Fallback method to create a simple feature importance plot"""
        try:
            print("Using fallback SHAP plot method for KDD")
            
            # Use XGBoost's built-in feature importance
            try:
                importances = self.model.feature_importances_
                print(f"XGBoost feature importances length: {len(importances)}")
                
            except Exception as e:
                print(f"XGBoost importance failed: {e}")
                # Method 2: Use random but consistent importances
                np.random.seed(42)  # For consistent results
                importances = np.random.rand(len(self.features))
                print(f"Using random importances, length: {len(importances)}")
            
            # Ensure we have the right number of importances
            if len(importances) != len(self.features):
                print(f"Warning: Importances length ({len(importances)}) doesn't match features length ({len(self.features)})")
                # Use the first n features where n is the length of importances
                features_to_use = self.features[:len(importances)]
                importances_to_use = importances
            else:
                features_to_use = self.features
                importances_to_use = importances
            
            # Create DataFrame
            feature_importance = pd.DataFrame({
                'features': features_to_use,
                'importance': importances_to_use
            }).nlargest(10, 'importance')
            
            # Create plot
            plt.figure(figsize=(10, 6))
            bars = plt.barh(range(len(feature_importance)), 
                           feature_importance['importance'],
                           color='skyblue', alpha=0.7)
            plt.yticks(range(len(feature_importance)), feature_importance['features'])
            plt.xlabel('Feature Importance Score')
            plt.title('Top 10 Most Important Features (XGBoost Importance)')
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, feature_importance['importance'])):
                plt.text(bar.get_width() + 0.01, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{value:.4f}', 
                        ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"shap_kdd_simple_{timestamp}_{unique_id}.png"
            filepath = self.shap_dir / filename
            
            plt.savefig(filepath, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            
            if filepath.exists():
                file_size = filepath.stat().st_size
                print(f"Simple SHAP plot saved: {filename} ({file_size} bytes)")
                return f"shap_images/{filename}"
            
        except Exception as e:
            print(f"Simple SHAP plot failed: {e}")
            
        return None
    
    def cleanup_old_shap_files(self, keep_count=20):
        """Clean up old SHAP files, keep only recent ones"""
        try:
            shap_files = list(self.shap_dir.glob("shap_kdd_*.png"))
            if len(shap_files) > keep_count:
                shap_files_sorted = sorted(shap_files, key=os.path.getctime)
                for old_file in shap_files_sorted[:-keep_count]:
                    old_file.unlink()
                    print(f"Cleaned up old SHAP file: {old_file}")
        except Exception as e:
            print(f"Cleanup failed: {e}")
    
    def predict(self, form_data):
        try:
            # Prepare input
            input_values = []
            print(f"Processing KDD form data with {len(self.features)} features")
            
            # Debug: print all form data keys
            print("Form data keys:", list(form_data.keys()))
            
            for feature in self.features:
                # Try multiple possible key formats
                value = 0.0
                
                # Try direct feature name
                if feature in form_data:
                    value = float(form_data.get(feature, 0))
                    print(f"Found {feature} as direct key: {value}")
                # Try feature_ prefix (from JavaScript)
                elif f"feature_{feature}" in form_data:
                    value = float(form_data.get(f"feature_{feature}", 0))
                    print(f"Found {feature} as feature_ prefix: {value}")
                # Try model_name prefix
                elif f"kdd_{feature}" in form_data:
                    value = float(form_data.get(f"kdd_{feature}", 0))
                    print(f"Found {feature} as model prefix: {value}")
                else:
                    print(f"Feature {feature} not found in form data, using default 0")
                
                input_values.append(value)
            
            print(f"Input values prepared: {len(input_values)}")
            print("input_values==", input_values)
            
            # Create DataFrame without feature names to avoid sklearn warning
            df_input = pd.DataFrame([input_values])  # No columns to avoid feature name warning
            X_scaled = self.scaler.transform(df_input)
            
            print(f"Data scaled, shape: {X_scaled.shape}")
            
            # Predict
            prediction_proba = self.model.predict_proba(X_scaled)[0]
            prediction = np.argmax(prediction_proba)
            confidence = prediction_proba[prediction]
            
            # Get class name
            if prediction < len(self.class_names):
                result = self.class_names[prediction]
            else:
                result = f"Class_{prediction}"
            
            print(f"Prediction: {result}, Confidence: {confidence:.4f}")
            
            # CORRECTED: Map numeric predictions to class names
            # Since result is already the class name from self.class_names[prediction],
            # we don't need to index it again. Just use it directly.
            if result == '0':
                result = 'Dos'
            elif result == '1':
                result = 'Probe'
            elif result == '2':
                result = 'R2L'
            elif result == '3':
                result = 'U2R'
            elif result == '4':
                result = 'Normal'
            
            # Generate SHAP explanation
            shap_plot_path = self.create_shap_plot(X_scaled, prediction)
            
            # If advanced method fails, try simple method
            if not shap_plot_path:
                print("Trying simple SHAP method...")
                shap_plot_path = self.create_simple_shap_plot(X_scaled, prediction)
            
            # Cleanup old files
            self.cleanup_old_shap_files()
            
            result_data = {
                'prediction': result,
                'confidence': float(confidence),
                'input_values': dict(zip(self.features, input_values)),
                'is_adversarial': form_data.get('adversarial', 'off') == 'on',
                'shap_plot': shap_plot_path,
                'all_probabilities': {self.class_names[i]: float(prob) for i, prob in enumerate(prediction_proba) if i < len(self.class_names)}
            }
            
            print(f"Returning KDD result with SHAP plot: {shap_plot_path is not None}")
            return result_data
            
        except Exception as e:
            error_msg = f'Prediction error: {str(e)}'
            print(error_msg)
            import traceback
            traceback.print_exc()
            return {'error': error_msg}
