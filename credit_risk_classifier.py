"""
Credit Risk Classifier Project
Hybrid model combining structured data with NLP features using SentenceTransformers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import xgboost as xgb

# NLP Libraries
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
import joblib

# Set random seed for reproducibility
np.random.seed(42)

class CreditRiskClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.xgb_model = None
        self.lr_model = None
        self.sentence_model = None
        self.label_encoders = {}
        
    def load_and_sample_data(self, filepath, sample_size=100000):
        """Load and sample data for faster processing"""
        print("Loading and sampling data...")
        
        # Try different encodings to handle the file
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                print(f"Trying encoding: {encoding}")
                # For large files, just sample directly without counting total rows
                df = pd.read_csv(filepath, encoding=encoding, nrows=sample_size)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error with {encoding}: {e}")
                continue
        
        if df is None:
            raise ValueError("Could not load the file with any supported encoding")
            
        print(f"Loaded sample size: {len(df):,} rows")
        return df
    
    def create_target_variable(self, df):
        """Create binary target variable for loan default"""
        print("Creating target variable...")
        
        # Define default statuses
        default_statuses = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off',
                          'Late (31-120 days)', 'Late (16-30 days)']
        
        df['is_default'] = df['loan_status'].isin(default_statuses).astype(int)
        
        print(f"Default rate: {df['is_default'].mean():.2%}")
        return df
    
    def preprocess_structured_features(self, df):
        """Preprocess structured features"""
        print("Preprocessing structured features...")
        
        # Select important numerical features (expanded)
        numerical_features = [
            'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'annual_inc',
            'dti', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
            'collections_12_mths_ex_med', 'acc_now_delinq', 'tot_coll_amt',
            'tot_cur_bal', 'total_rev_hi_lim', 'delinq_2yrs', 'inq_last_6mths',
            'mths_since_last_delinq', 'mths_since_last_record', 'pub_rec_bankruptcies',
            'tax_liens', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy',
            'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct',
            'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
            'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_inq',
            'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
            'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
            'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m',
            'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',
            'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'tot_hi_cred_lim', 'total_bal_ex_mort',
            'total_bc_limit', 'total_il_high_credit_limit'
        ]
        
        # Select important categorical features (expanded)
        categorical_features = [
            'term', 'grade', 'sub_grade', 'emp_length', 'home_ownership',
            'verification_status', 'purpose', 'addr_state', 'initial_list_status',
            'application_type', 'hardship_flag', 'debt_settlement_flag'
        ]
        
        # Keep only available features
        available_numerical = [col for col in numerical_features if col in df.columns]
        available_categorical = [col for col in categorical_features if col in df.columns]
        
        print(f"Using {len(available_numerical)} numerical and {len(available_categorical)} categorical features")
        
        # Handle numerical features
        df_processed = df[available_numerical + available_categorical + ['desc', 'is_default']].copy()
        
        # Impute missing values for numerical features
        for col in available_numerical:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # Encode categorical features
        for col in available_categorical:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
            else:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
        
        # Create additional engineered features
        if 'loan_amnt' in df_processed.columns and 'annual_inc' in df_processed.columns:
            df_processed['loan_to_income_ratio'] = df_processed['loan_amnt'] / (df_processed['annual_inc'] + 1)
        
        if 'int_rate' in df_processed.columns and 'loan_amnt' in df_processed.columns:
            df_processed['total_interest'] = df_processed['int_rate'] * df_processed['loan_amnt'] / 100
        
        # Additional ratio features
        if 'revol_bal' in df_processed.columns and 'revol_util' in df_processed.columns:
            df_processed['revol_bal_to_util_ratio'] = df_processed['revol_bal'] / (df_processed['revol_util'] + 0.01)
        
        if 'open_acc' in df_processed.columns and 'total_acc' in df_processed.columns:
            df_processed['open_to_total_acc_ratio'] = df_processed['open_acc'] / (df_processed['total_acc'] + 1)
        
        if 'pub_rec' in df_processed.columns and 'delinq_2yrs' in df_processed.columns:
            df_processed['total_negative_records'] = df_processed['pub_rec'] + df_processed['delinq_2yrs']
        
        # Credit utilization features
        if 'total_rev_hi_lim' in df_processed.columns and 'revol_bal' in df_processed.columns:
            df_processed['credit_util_ratio'] = df_processed['revol_bal'] / (df_processed['total_rev_hi_lim'] + 1)
        
        return df_processed
    
    def extract_nlp_features(self, df, max_samples=50000):
        """Extract NLP features using SentenceTransformers (optimized)
        Combines desc and emp_title (if available) and averages embeddings to keep dim fixed.
        """
        print("Extracting NLP features...")
        
        # Initialize a lighter, faster model
        if self.sentence_model is None:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # Faster model
        
        # Build comprehensive combined text field
        text_fields = ['desc', 'emp_title', 'purpose', 'title']
        combined_parts = []
        
        for field in text_fields:
            if field in df.columns:
                series = df[field].fillna('').astype(str)
                combined_parts.append(series)
            else:
                combined_parts.append(pd.Series([''] * len(df), index=df.index))
        
        combined_text = (combined_parts[0] + '. ' + combined_parts[1] + 
                        '. Purpose: ' + combined_parts[2] + 
                        '. Title: ' + combined_parts[3]).str.strip()
        combined_text = combined_text.str.replace(r'\. \. ', '. ', regex=True)
        combined_text = combined_text.str.replace(r'^\. ', '', regex=True)
        combined_text = combined_text.replace('', 'No description provided')
        
        # Limit samples for speed
        if len(combined_text) > max_samples:
            print(f"Sampling {max_samples} combined texts for NLP processing to optimize speed")
            sample_idx = np.random.choice(len(combined_text), max_samples, replace=False)
            text_sample = combined_text.iloc[sample_idx]
            full_text = combined_text
        else:
            text_sample = combined_text
            full_text = combined_text
            sample_idx = np.arange(len(combined_text))
        
        # Extract embeddings for sample
        print("Computing sentence embeddings...")
        embeddings_sample = self.sentence_model.encode(
            text_sample.tolist(), 
            batch_size=64, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create full embeddings array
        embeddings_full = np.zeros((len(full_text), embeddings_sample.shape[1]))
        
        if len(combined_text) > max_samples:
            # Use sample embeddings for sampled indices
            embeddings_full[sample_idx] = embeddings_sample
            
            # For remaining indices, use mean embedding
            mean_embedding = np.mean(embeddings_sample, axis=0)
            remaining_idx = np.setdiff1d(np.arange(len(full_text)), sample_idx)
            embeddings_full[remaining_idx] = mean_embedding
        else:
            embeddings_full = embeddings_sample
        
        # Create DataFrame with embedding features
        embedding_df = pd.DataFrame(embeddings_full, 
                                  columns=[f'nlp_feature_{i}' for i in range(embeddings_full.shape[1])])
        
        print(f"Extracted {embeddings_full.shape[1]} NLP features")
        return embedding_df
    
    def create_visualizations(self, df, X_train, X_test, y_train, y_test, 
                            xgb_pred_proba, lr_pred_proba, feature_importance=None):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Target Distribution
        plt.subplot(3, 4, 1)
        target_counts = df['is_default'].value_counts()
        plt.pie(target_counts.values, labels=['No Default', 'Default'], autopct='%1.1f%%', 
                colors=['lightgreen', 'lightcoral'])
        plt.title('Loan Default Distribution')
        
        # 2. Interest Rate Distribution by Default
        plt.subplot(3, 4, 2)
        if 'int_rate' in df.columns:
            sns.boxplot(data=df, x='is_default', y='int_rate')
            plt.title('Interest Rate by Default Status')
        
        # 3. Loan Amount Distribution
        plt.subplot(3, 4, 3)
        if 'loan_amnt' in df.columns:
            plt.hist(df[df['is_default']==0]['loan_amnt'], alpha=0.7, label='No Default', bins=50)
            plt.hist(df[df['is_default']==1]['loan_amnt'], alpha=0.7, label='Default', bins=50)
            plt.xlabel('Loan Amount')
            plt.ylabel('Frequency')
            plt.title('Loan Amount Distribution')
            plt.legend()
        
        # 4. Grade Distribution
        plt.subplot(3, 4, 4)
        if 'grade' in df.columns:
            grade_default = df.groupby('grade')['is_default'].mean().reset_index()
            plt.bar(grade_default['grade'], grade_default['is_default'])
            plt.title('Default Rate by Grade')
            plt.ylabel('Default Rate')
        
        # 5. ROC Curve
        plt.subplot(3, 4, 5)
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_pred_proba)
        fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_pred_proba)
        
        plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(y_test, xgb_pred_proba):.3f})')
        plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, lr_pred_proba):.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # 6. Feature Importance (if available)
        if feature_importance is not None:
            plt.subplot(3, 4, 6)
            top_features = feature_importance.head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.title('Top 10 Feature Importance (XGBoost)')
            plt.xlabel('Importance')
        
        # 7. Prediction Distribution
        plt.subplot(3, 4, 7)
        plt.hist(xgb_pred_proba[y_test==0], alpha=0.7, label='No Default', bins=30)
        plt.hist(xgb_pred_proba[y_test==1], alpha=0.7, label='Default', bins=30)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('XGBoost Prediction Distribution')
        plt.legend()
        
        # 8. Confusion Matrix
        plt.subplot(3, 4, 8)
        xgb_pred = (xgb_pred_proba > 0.5).astype(int)
        cm = confusion_matrix(y_test, xgb_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (XGBoost)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 9. Model Performance Comparison
        plt.subplot(3, 4, 9)
        models = ['XGBoost', 'Logistic Regression']
        auc_scores = [roc_auc_score(y_test, xgb_pred_proba), roc_auc_score(y_test, lr_pred_proba)]
        f1_scores = [f1_score(y_test, (xgb_pred_proba > 0.5).astype(int)),
                    f1_score(y_test, (lr_pred_proba > 0.5).astype(int))]
        
        x = np.arange(len(models))
        width = 0.35
        plt.bar(x - width/2, auc_scores, width, label='ROC-AUC', alpha=0.8)
        plt.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models)
        plt.legend()
        plt.ylim(0, 1)
        
        # 10. Annual Income vs Default (if available)
        plt.subplot(3, 4, 10)
        if 'annual_inc' in df.columns:
            df_plot = df[df['annual_inc'] < df['annual_inc'].quantile(0.95)]  # Remove outliers
            sns.boxplot(data=df_plot, x='is_default', y='annual_inc')
            plt.title('Annual Income by Default Status')
        
        # 11. DTI Distribution (if available)
        plt.subplot(3, 4, 11)
        if 'dti' in df.columns:
            df_plot = df[df['dti'] < df['dti'].quantile(0.95)]  # Remove outliers
            plt.hist(df_plot[df_plot['is_default']==0]['dti'], alpha=0.7, label='No Default', bins=30)
            plt.hist(df_plot[df_plot['is_default']==1]['dti'], alpha=0.7, label='Default', bins=30)
            plt.xlabel('Debt-to-Income Ratio')
            plt.ylabel('Frequency')
            plt.title('DTI Distribution')
            plt.legend()
        
        # 12. Calibration Plot
        plt.subplot(3, 4, 12)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (xgb_pred_proba > bin_lower) & (xgb_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_test[in_bin].mean()
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(accuracy_in_bin)
        
        plt.plot(bin_centers, bin_accuracies, 'o-', label='XGBoost')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('credit_risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'credit_risk_analysis.png'")
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train XGBoost and Logistic Regression models with optimization and tuning
        Uses early stopping and a small parameter search to maximize ROC-AUC under time constraints.
        """
        print("Training models...")
        
        # Prepare validation split for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Handle class imbalance for XGBoost
        pos = y_tr.sum()
        neg = len(y_tr) - pos
        scale_pos_weight = max((neg / max(pos, 1)), 1)
        print(f"scale_pos_weight: {scale_pos_weight:.2f}")
        
        # Randomized search for better coverage
        import random
        random.seed(42)
        
        param_options = {
            'max_depth': [3, 4, 5, 6, 7, 8],
            'min_child_weight': [1, 2, 3, 5],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'learning_rate': [0.03, 0.05, 0.07, 0.1, 0.15],
            'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0],
            'reg_alpha': [0.0, 0.1, 0.5, 1.0]
        }
        
        # Generate 120 random combinations
        param_grid = []
        for _ in range(120):
            params = {}
            for key, values in param_options.items():
                params[key] = random.choice(values)
            param_grid.append(params)
        
        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'n_estimators': 2000,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'scale_pos_weight': scale_pos_weight
        }
        
        best_auc = -np.inf
        best_model = None
        best_params = None
        start_time = time.time()
        
        print("Tuning XGBoost (early stopping)...")
        
        # Adaptive search - start with promising regions
        sorted_grid = sorted(param_grid, key=lambda x: -x['learning_rate'] * (1/x['max_depth']))
        
        for i, p in enumerate(sorted_grid, 1):
            params = {**base_params, **p, 'early_stopping_rounds': 30}  # Faster stopping
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            val_pred = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_pred)
            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_params = params
                print(f"New best AUC: {best_auc:.4f} at iteration {i}")
            
            if i % 15 == 0:
                elapsed = time.time() - start_time
                print(f"Checked {i}/{len(sorted_grid)} params. Best AUC so far: {best_auc:.4f}. Elapsed: {elapsed/60:.1f}m")
            
            # Early exit if no improvement in last 30 trials and we have a good model
            if i > 30 and best_auc > 0.745:
                recent_trials = sorted_grid[max(0, i-30):i]
                recent_aucs = []
                for trial_p in recent_trials[-10:]:
                    trial_params = {**base_params, **trial_p, 'early_stopping_rounds': 20}
                    trial_model = xgb.XGBClassifier(**trial_params)
                    trial_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                    trial_pred = trial_model.predict_proba(X_val)[:, 1]
                    recent_aucs.append(roc_auc_score(y_val, trial_pred))
                
                if max(recent_aucs) < best_auc:
                    print(f"Early stopping: no improvement in recent trials. Best AUC: {best_auc:.4f}")
                    break
            
            # Safety break if tuning exceeds ~40 minutes (to keep under 1 hour total)
            if time.time() - start_time > 40 * 60:
                print("Tuning time limit reached; stopping parameter search.")
                break
        
        self.xgb_model = best_model
        print(f"Best XGBoost val AUC: {best_auc:.4f}")
        print(f"Best params: {{'max_depth': {best_params['max_depth']}, 'min_child_weight': {best_params['min_child_weight']}, 'subsample': {best_params['subsample']}, 'colsample_bytree': {best_params['colsample_bytree']}, 'learning_rate': {best_params['learning_rate']}, 'reg_lambda': {best_params['reg_lambda']}}}")
        xgb_time = time.time() - start_time
        
        # Logistic Regression (baseline)
        print("Training Logistic Regression...")
        start_time_lr = time.time()
        self.lr_model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
        
        # Scale features for logistic regression
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.lr_model.fit(X_train_scaled, y_train)
        lr_time = time.time() - start_time_lr
        
        print(f"XGBoost tuning+training time: {xgb_time/60:.2f} minutes")
        print(f"Logistic Regression training time: {lr_time:.2f} seconds")
        
        # Make predictions
        xgb_pred_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        lr_pred_proba = self.lr_model.predict_proba(X_test_scaled)[:, 1]
        
        return xgb_pred_proba, lr_pred_proba, X_test_scaled
    
    def evaluate_models(self, y_test, xgb_pred_proba, lr_pred_proba):
        """Evaluate model performance"""
        print("Evaluating models...")
        
        # Calculate metrics
        xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
        lr_auc = roc_auc_score(y_test, lr_pred_proba)
        
        xgb_pred = (xgb_pred_proba > 0.5).astype(int)
        lr_pred = (lr_pred_proba > 0.5).astype(int)
        
        xgb_f1 = f1_score(y_test, xgb_pred)
        lr_f1 = f1_score(y_test, lr_pred)
        
        print("=" * 50)
        print("MODEL PERFORMANCE RESULTS")
        print("=" * 50)
        print(f"XGBoost:")
        print(f"  ROC-AUC: {xgb_auc:.4f}")
        print(f"  F1-Score: {xgb_f1:.4f}")
        print()
        print(f"Logistic Regression:")
        print(f"  ROC-AUC: {lr_auc:.4f}")
        print(f"  F1-Score: {lr_f1:.4f}")
        print("=" * 50)
        
        # Detailed classification report for best model
        best_model = "XGBoost" if xgb_auc > lr_auc else "Logistic Regression"
        best_pred = xgb_pred if xgb_auc > lr_auc else lr_pred
        
        print(f"\nDetailed Classification Report ({best_model}):")
        print(classification_report(y_test, best_pred))
        
        return {
            'xgb_auc': xgb_auc, 'lr_auc': lr_auc,
            'xgb_f1': xgb_f1, 'lr_f1': lr_f1,
            'best_model': best_model
        }
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from XGBoost model"""
        if self.xgb_model is None:
            return None
            
        importance = self.xgb_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def run_complete_pipeline(self):
        """Run the complete credit risk classification pipeline"""
        print("Starting Credit Risk Classification Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load and prepare data
        df = self.load_and_sample_data('accepted_2007_to_2018Q4.csv', sample_size=200000)
        df = self.create_target_variable(df)
        
        # Remove loans that are still current (can't determine final outcome)
        df = df[df['loan_status'] != 'Current'].copy()
        print(f"Dataset after removing current loans: {len(df):,} rows")
        
        # Preprocess structured features
        df_processed = self.preprocess_structured_features(df)
        
        # Extract NLP features
        nlp_features = self.extract_nlp_features(df_processed, max_samples=60000)
        
        # Ensure consistent indices
        nlp_features.index = df_processed.index
        
        # Combine all features
        feature_cols = [col for col in df_processed.columns if col not in ['desc', 'is_default']]
        X_structured = df_processed[feature_cols]
        X_combined = pd.concat([X_structured, nlp_features], axis=1)
        y = df_processed['is_default']
        
        # Feature selection to remove noise and improve performance
        print("Performing feature selection...")
        n_features_to_select = min(300, X_combined.shape[1])  # Keep top 300 features
        
        # Use mutual information for feature selection (better for non-linear relationships)
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features_to_select)
        X_selected = selector.fit_transform(X_combined, y)
        
        # Get selected feature names
        selected_features = X_combined.columns[selector.get_support()]
        X_combined = pd.DataFrame(X_selected, columns=selected_features, index=X_combined.index)
        self.selected_features = selected_features
        
        print(f"Selected {len(selected_features)} features out of {X_combined.shape[1]} total")
        
        # Ensure consistent shapes
        print(f"X_structured shape: {X_structured.shape}")
        print(f"nlp_features shape: {nlp_features.shape}")
        print(f"X_combined shape: {X_combined.shape}")
        print(f"y shape: {y.shape}")
        
        # If there's still a mismatch, align the data
        if X_combined.shape[0] != y.shape[0]:
            print(f"Shape mismatch detected. Aligning data...")
            min_length = min(X_combined.shape[0], y.shape[0])
            X_combined = X_combined.iloc[:min_length]
            y = y.iloc[:min_length]
        
        print(f"Final dataset shape: {X_combined.shape}")
        print(f"Features: {X_combined.shape[1]} total ({len(feature_cols)} structured + {nlp_features.shape[1]} NLP)")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        
        # Train models
        xgb_pred_proba, lr_pred_proba, X_test_scaled = self.train_models(X_train, X_test, y_train, y_test)
        
        # Evaluate models
        results = self.evaluate_models(y_test, xgb_pred_proba, lr_pred_proba)
        
        # Get feature importance
        feature_importance = self.get_feature_importance(X_combined.columns)
        
        # Create visualizations
        self.create_visualizations(df_processed, X_train, X_test, y_train, y_test,
                                 xgb_pred_proba, lr_pred_proba, feature_importance)
        
        # Save models and preprocessing artifacts
        joblib.dump(self.xgb_model, 'xgboost_model.pkl')
        joblib.dump(self.lr_model, 'logistic_regression_model.pkl')
        joblib.dump(self.scaler, 'feature_scaler.pkl')
        # Save feature selector and selected feature names for prediction alignment
        try:
            if hasattr(self, 'selected_features'):
                joblib.dump(self.selected_features.tolist() if hasattr(self.selected_features, 'tolist') 
                           else list(self.selected_features), 'xgb_feature_list.pkl')
                print("Saved selected feature list: xgb_feature_list.pkl")
        except Exception as e:
            print(f"Warning: could not save feature list: {e}")
        
        total_time = time.time() - start_time
        print(f"\nTotal pipeline execution time: {total_time/60:.2f} minutes")
        
        # Print final summary
        print("\n" + "=" * 60)
        print("PROJECT SUMMARY")
        print("=" * 60)
        print(f"Dataset: Lending Club Loan Data")
        print(f"Sample size: {len(df):,} loans")
        print(f"Features: {X_combined.shape[1]} (structured + NLP)")
        print(f"Best model: {results['best_model']}")
        print(f"Best ROC-AUC: {max(results['xgb_auc'], results['lr_auc']):.4f}")
        print(f"Best F1-Score: {max(results['xgb_f1'], results['lr_f1']):.4f}")
        print(f"Total execution time: {total_time/60:.2f} minutes")
        print("=" * 60)
        
        return results, feature_importance


if __name__ == "__main__":
    # Initialize and run the classifier
    classifier = CreditRiskClassifier()
    results, feature_importance = classifier.run_complete_pipeline()
    
    # Display top features
    if feature_importance is not None:
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
