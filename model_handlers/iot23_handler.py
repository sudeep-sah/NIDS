import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import uuid
from datetime import datetime
from pathlib import Path

class IoT23Handler:
    def __init__(self):
        self.model_dir = Path("models/iot23")
        self.shap_dir = Path("static/shap_images")
        self.shap_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        self.load_model()
        self.explainer = None
        self.init_shap()
    
    def load_model(self):
        try:
            self.model = joblib.load(self.model_dir / "ids_xgb_model.pkl")
            self.scaler = joblib.load(self.model_dir / "scaler.pkl")
            self.le_target = joblib.load(self.model_dir / "label_encoder.pkl")
            self.features = joblib.load(self.model_dir / "feature_order.pkl")
            self.class_names = joblib.load(self.model_dir / "class_names.pkl")
            print(f"Loaded IoT23 model with {len(self.features)} features")
        except Exception as e:
            raise Exception(f"Failed to load IoT-23 model: {str(e)}")
    
    def init_shap(self):
        """Initialize SHAP explainer"""
        try:
            # Create a small background dataset for SHAP
            background_data = self.scaler.transform(
                np.zeros((10, len(self.features)))
            )
            self.explainer = shap.TreeExplainer(self.model, background_data)
            print("SHAP explainer initialized successfully")
        except Exception as e:
            print(f"SHAP explainer initialization failed: {e}")
            # Try alternative approach
            try:
                self.explainer = shap.TreeExplainer(self.model)
                print("SHAP explainer initialized with alternative method")
            except Exception as e2:
                print(f"Alternative SHAP initialization also failed: {e2}")
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
            print(f"SHAP values shape: {np.array(shap_values).shape if hasattr(shap_values, '__len__') else 'scalar'}")
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class output
                shap_vals = shap_values[prediction][0] if prediction < len(shap_values) else shap_values[0][0]
            elif len(shap_values.shape) == 3:
                # 3D array for multi-class
                shap_vals = shap_values[0, :, prediction] if prediction < shap_values.shape[2] else shap_values[0, :, 0]
            elif len(shap_values.shape) == 2:
                # 2D array
                shap_vals = shap_values[0, :]
            else:
                # 1D array or scalar
                shap_vals = shap_values[0] if hasattr(shap_values, '__len__') else shap_values
            
            print(f"Processed SHAP values length: {len(shap_vals) if hasattr(shap_vals, '__len__') else 1}")
            
            # Ensure we have the right number of features
            if hasattr(shap_vals, '__len__') and len(shap_vals) != len(self.features):
                print(f"Warning: SHAP values length ({len(shap_vals)}) doesn't match features length ({len(self.features)})")
                # Truncate or pad to match features
                if len(shap_vals) > len(self.features):
                    shap_vals = shap_vals[:len(self.features)]
                else:
                    shap_vals = np.pad(shap_vals, (0, len(self.features) - len(shap_vals)))
            
            # Create DataFrame for plotting
            feature_importance = pd.DataFrame({
                'features': self.features,
                'shap_values': shap_vals if hasattr(shap_vals, '__len__') else [shap_vals] * len(self.features)
            })
            
            # Calculate absolute values for sorting
            feature_importance['abs_shap'] = np.abs(feature_importance['shap_values'])
            
            # Sort by absolute SHAP value and take top 10
            top_features = feature_importance.nlargest(10, 'abs_shap')
            
            print(f"Top features for SHAP plot: {len(top_features)}")
            
            if len(top_features) == 0:
                print("No features to plot")
                return None
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
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
            
            plt.tight_layout()
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"shap_iot23_{timestamp}_{unique_id}.png"
            filepath = self.shap_dir / filename
            
            # Save plot
            plt.savefig(filepath, format='png', dpi=150, bbox_inches='tight', 
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
            print("Using fallback SHAP plot method")
            
            # Use feature importances from the model
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            else:
                # Create dummy importances based on feature index
                importances = np.arange(len(self.features)) / len(self.features)
            
            # Create DataFrame
            feature_importance = pd.DataFrame({
                'features': self.features,
                'importance': importances
            }).nlargest(10, 'importance')
            
            # Create plot
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(feature_importance)), 
                           feature_importance['importance'])
            plt.yticks(range(len(feature_importance)), feature_importance['features'])
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Most Important Features (Fallback Method)')
            
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
            filename = f"shap_iot23_fallback_{timestamp}_{unique_id}.png"
            filepath = self.shap_dir / filename
            
            plt.savefig(filepath, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            if filepath.exists():
                print(f"Fallback SHAP plot saved: {filename}")
                return f"shap_images/{filename}"
            
        except Exception as e:
            print(f"Fallback SHAP plot also failed: {e}")
            
        return None
    
    def cleanup_old_shap_files(self, keep_count=20):
        """Clean up old SHAP files, keep only recent ones"""
        try:
            shap_files = list(self.shap_dir.glob("shap_iot23_*.png"))
            if len(shap_files) > keep_count:
                # Sort by creation time and remove oldest
                shap_files_sorted = sorted(shap_files, key=os.path.getctime)
                for old_file in shap_files_sorted[:-keep_count]:
                    old_file.unlink()
                    print(f"Cleaned up old SHAP file: {old_file}")
        except Exception as e:
            print(f"Cleanup failed: {e}")
    
    def predict(self, form_data):
        try:
            # Prepare input features
            input_values = []
            print(f"Processing form data with {len(self.features)} features")
            
            for feature in self.features:
                value = float(form_data.get(feature, 0))
                input_values.append(value)
            
            print(f"Input values prepared: {len(input_values)}")
            print("input_values==",input_values)
            
            # Create DataFrame and scale
            df_input = pd.DataFrame([input_values], columns=self.features)
            X_scaled = self.scaler.transform(df_input)
            
            print(f"Data scaled, shape: {X_scaled.shape}")
            
            # Predict
            prediction = self.model.predict(X_scaled)[0]
            prediction_proba = self.model.predict_proba(X_scaled)[0]
            confidence = max(prediction_proba)
            result = self.le_target.inverse_transform([prediction])[0]
            
            print(f"Prediction: {result}, Confidence: {confidence:.4f}")
            
            # Generate SHAP explanation
            shap_plot_path = self.create_shap_plot(X_scaled, prediction)
            
            # If SHAP plot failed, try fallback method
            if not shap_plot_path:
                print("Primary SHAP method failed, trying fallback...")
                shap_plot_path = self.create_simple_shap_plot(X_scaled, prediction)
            
            # Cleanup old files
            self.cleanup_old_shap_files()
            
            result_data = {
                'prediction': result,
                'confidence': float(confidence),
                'input_values': dict(zip(self.features, input_values)),
                'is_adversarial': form_data.get('adversarial', 'off') == 'on',
                'shap_plot': shap_plot_path
            }
            
            print(f"Returning result with SHAP plot: {shap_plot_path is not None}")
            return result_data
            
        except Exception as e:
            error_msg = f'Prediction error: {str(e)}'
            print(error_msg)
            import traceback
            traceback.print_exc()
            return {'error': error_msg}