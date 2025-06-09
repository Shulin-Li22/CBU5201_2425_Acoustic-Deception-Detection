import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import re
import traceback
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

def parse_array_string(value):
    """Extract numeric values from string and return their mean"""
    try:
        if isinstance(value, str):
            value = re.sub(r'[\[\]]', '', value)
            numbers = re.findall(r'[-+]?\d*\.\d+|\d+', value)
            floats = [float(num) for num in numbers]
            return np.mean(floats) if floats else np.nan
        elif isinstance(value, (int, float)):
            return float(value)
        return np.nan
    except:
        return np.nan

class AudioFeatureTransformer(BaseEstimator, TransformerMixin):
    """Transform audio features for deception detection"""
    def __init__(self, language):
        self.language = language
        self.scaler = StandardScaler()
        self.top_n_features = 20
        self.selected_features = None

    def _clean_dataframe(self, df):
        """Clean numeric columns in dataframe"""
        df_cleaned = df.copy()
        
        numeric_columns = df_cleaned.select_dtypes(include=['object', 'float64', 'int64']).columns
        numeric_columns = [col for col in numeric_columns 
                           if col not in ['filename', 'Language', 'Story_type']]

        for col in numeric_columns:
            df_cleaned[col] = df_cleaned[col].apply(parse_array_string)
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

        for col in numeric_columns:
            if df_cleaned[col].isna().any():
                col_mean = df_cleaned[col].mean()
                df_cleaned[col].fillna(col_mean, inplace=True)
        
        return df_cleaned
    
    def _safe_ratio(self, numerator, denominator, epsilon=1e-8):
        """Calculate ratio with protection against zero division"""
        return np.where(denominator > epsilon, numerator / denominator, 0)
    
    def _preprocess_features(self, X):
        """Process and combine features"""
        X_processed = X.copy()
        
        # MFCC features
        mfcc_cols = [col for col in X_processed.columns if 'mfcc' in col]
        mfcc_means = [col for col in mfcc_cols if 'mean' in col]
        mfcc_stds = [col for col in mfcc_cols if 'std' in col]
        
        if mfcc_means:
            X_processed['mfcc_mean_avg'] = X_processed[mfcc_means].mean(axis=1)
        if mfcc_stds:
            X_processed['mfcc_std_avg'] = X_processed[mfcc_stds].mean(axis=1)
            if 'mfcc_mean_avg' in X_processed.columns:
                X_processed['mfcc_variation'] = self._safe_ratio(
                    X_processed['mfcc_std_avg'],
                    X_processed['mfcc_mean_avg']
                )
        
        # Spectral features
        spectral_cols = [col for col in X_processed.columns if 'spectral' in col]
        spectral_means = [col for col in spectral_cols if 'mean' in col]
        spectral_stds = [col for col in spectral_cols if 'std' in col]
        
        if spectral_means:
            X_processed['spectral_mean_avg'] = X_processed[spectral_means].mean(axis=1)
        if spectral_stds:
            X_processed['spectral_std_avg'] = X_processed[spectral_stds].mean(axis=1)
        
        # Rhythm features
        if all(col in X_processed.columns for col in ['beat_interval_mean', 'beat_interval_std']):
            X_processed['rhythm_variation'] = self._safe_ratio(
                X_processed['beat_interval_std'],
                X_processed['beat_interval_mean']
            )
        
        # Language-specific features
        if self.language == 'English':
            if all(col in X_processed.columns for col in ['zero_crossing_rate_mean', 'energy_variability']):
                X_processed['speech_clarity'] = X_processed['zero_crossing_rate_mean'] * X_processed['energy_variability']
        else:  # Chinese
            if all(col in X_processed.columns for col in ['pitch_mean', 'pitch_std']):
                X_processed['tonal_variation'] = self._safe_ratio(
                    X_processed['pitch_std'],
                    X_processed['pitch_mean']
                )
        
        return X_processed
    
    def _select_stable_features(self, X, y):
        """Select stable features using random forest"""
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        
        n_runs = 5
        feature_importance_matrix = np.zeros((n_runs, len(X.columns)))
        
        for i in range(n_runs):
            rf.fit(X, y)
            feature_importance_matrix[i] = rf.feature_importances_
        
        mean_importance = feature_importance_matrix.mean(axis=0)
        std_importance = feature_importance_matrix.std(axis=0)
        
        stability_scores = mean_importance / (std_importance + 1e-10)
        
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'importance': mean_importance,
            'stability': stability_scores,
            'score': mean_importance * stability_scores
        }).sort_values('score', ascending=False)
        
        selected_features = feature_scores['feature'].head(self.top_n_features).tolist()
        
        print(f"\nTop {self.top_n_features} features:")
        print(feature_scores.head(self.top_n_features))
        
        return selected_features
    
    def fit(self, X, y=None):
        if y is not None:
            y = y.astype(int)
        
        X_cleaned = self._clean_dataframe(X)
        X_processed = self._preprocess_features(X_cleaned)
        self.selected_features = self._select_stable_features(X_processed, y)
        
        return self
    
    def transform(self, X):
        X_cleaned = self._clean_dataframe(X)
        X_processed = self._preprocess_features(X_cleaned)
        if self.selected_features is not None:
            X_processed = X_processed[self.selected_features]
        return self.scaler.fit_transform(X_processed)

class DeceptionDetectionModel:
    """Model for detecting deception in audio features"""
    def __init__(self, language):
        self.language = language
        self.pipeline = self._create_pipeline()
        self.cv_results = None
        self.classification_report = None
    
    def _create_pipeline(self):
        """Create model pipeline with language-specific parameters"""
        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'random_state': 42
        }
        
        if self.language == 'English':
            model_params = {
                'max_depth': 3,
                'learning_rate': 0.01,
                'n_estimators': 300,
                'min_child_weight': 5,
                'gamma': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'scale_pos_weight': 1
            }
        else:  # Chinese
            model_params = {
                'max_depth': 4,
                'learning_rate': 0.01,
                'n_estimators': 250,
                'min_child_weight': 3,
                'gamma': 0.2,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.2,
                'reg_lambda': 1.5,
                'scale_pos_weight': 1
            }
        
        params = {**base_params, **model_params}
        
        return Pipeline([
            ('feature_transformer', AudioFeatureTransformer(self.language)),
            ('classifier', xgb.XGBClassifier(**params))
        ])
    
    def train_and_evaluate(self, X, y):
        """Train and evaluate model using cross-validation with debugging"""
        print(f"\nTraining and evaluating {self.language} model...")
        
        y = y.astype(int)
        
        # Debug prints
        print(f"Input X shape: {X.shape}")
        print(f"Input y shape: {y.shape}")
        print(f"Class distribution: \n{pd.Series(y).value_counts(normalize=True)}")
        
        cv = RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=3,
            random_state=42
        )
        
        # Store all predictions and true values
        all_predictions = []
        all_true_values = []
        all_probabilities = []
        fold_scores = []
        
        # Process each fold
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"\nFold {fold_idx + 1}:")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            try:
                # Fit and predict
                self.pipeline.fit(X_train, y_train)
                y_pred = self.pipeline.predict(X_val)
                y_prob = self.pipeline.predict_proba(X_val)[:, 1]
                
                # Store predictions
                all_predictions.extend(y_pred)
                all_true_values.extend(y_val)
                all_probabilities.extend(y_prob)
                
                # Calculate fold metrics
                fold_acc = accuracy_score(y_val, y_pred)
                fold_scores.append(fold_acc)
                
                print(f"Fold {fold_idx + 1} accuracy: {fold_acc:.3f}")
                print(f"Class distribution - Train: \n{pd.Series(y_train).value_counts(normalize=True)}")
                print(f"Class distribution - Val: \n{pd.Series(y_val).value_counts(normalize=True)}")
                
            except Exception as e:
                print(f"Error in fold {fold_idx + 1}:")
                print(str(e))
                traceback.print_exc()
        
        # Calculate overall CV metrics
        all_predictions = np.array(all_predictions)
        all_true_values = np.array(all_true_values)
        all_probabilities = np.array(all_probabilities)
        
        cv_scores = {
            'accuracy': np.mean(fold_scores),
            'accuracy_std': np.std(fold_scores),
            'precision': precision_score(all_true_values, all_predictions),
            'recall': recall_score(all_true_values, all_predictions),
            'f1': f1_score(all_true_values, all_predictions),
            'auc_roc': roc_auc_score(all_true_values, all_probabilities)
        }
        
        print("\nCross-validation results:")
        print(f"Accuracy: {cv_scores['accuracy']:.3f} (±{cv_scores['accuracy_std']*2:.3f})")
        print(f"Precision: {cv_scores['precision']:.3f}")
        print(f"Recall: {cv_scores['recall']:.3f}")
        print(f"F1: {cv_scores['f1']:.3f}")
        print(f"AUC-ROC: {cv_scores['auc_roc']:.3f}")
        
        # Store CV results
        self.cv_results = cv_scores
        
        # Fit final model
        try:
            self.pipeline.fit(X, y)
            y_pred = self.pipeline.predict(X)
            y_prob = self.pipeline.predict_proba(X)[:, 1]
            
            # Store and print classification report
            self.classification_report = classification_report(y, y_pred)
            
            print("\nFinal model evaluation:")
            print(self.classification_report)
            print(f"AUC-ROC: {roc_auc_score(y, y_prob):.3f}")
            
        except Exception as e:
            print("Error in final model evaluation:")
            print(str(e))
            traceback.print_exc()
        
        return self.cv_results

def main():
    print("Loading data...")
    df = pd.read_csv('audio_features.csv')
    
    results = {}
    for lang in ['Chinese', 'English']:
        print(f"\nProcessing {lang} data...")
        
        lang_df = df[df['Language'] == lang].copy()
        X = lang_df.drop(['Story_type', 'filename', 'Language'], axis=1)
        y = (lang_df['Story_type'] == 'Deceptive Story').astype(int)
        
        print("\nDataset info:")
        print(f"Number of samples: {len(X)}")
        print(f"Number of features: {X.shape[1]}")
        print("\nClass distribution:")
        print(pd.Series(y).value_counts(normalize=True))
        
        model = DeceptionDetectionModel(lang)
        results[lang] = model.train_and_evaluate(X, y)
    
    return results

if __name__ == "__main__":
    results = main()