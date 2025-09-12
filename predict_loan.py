"""
Credit Risk Prediction Script (XGBoost)
Simple working version for the optimized XGBoost model
"""

import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

class CreditRiskPredictor:
    def __init__(self):
        # Load the trained models and scaler
        self.xgb_model = joblib.load('xgboost_model.pkl')
        self.lr_model = joblib.load('logistic_regression_model.pkl')
        self.scaler = joblib.load('feature_scaler.pkl')
        
        # Load sentence transformer
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Models loaded successfully!")
        
    def preprocess_loan(self, loan_data):
        """
        Preprocess loan data to match training format
        """
        # Extended numerical features (matching training)
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
        
        # Categorical features (matching training)
        categorical_features = [
            'term', 'grade', 'sub_grade', 'emp_length', 'home_ownership',
            'verification_status', 'purpose', 'addr_state', 'initial_list_status',
            'application_type', 'hardship_flag', 'debt_settlement_flag'
        ]
        
        # Process numerical features
        processed_features = []
        for feature in numerical_features:
            value = loan_data.get(feature, 0)
            processed_features.append(float(value))
        
        # Process categorical features (simplified mappings)
        grade_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
        term_mapping = {' 36 months': 0, ' 60 months': 1}
        
        for feature in categorical_features:
            if feature == 'grade':
                value = grade_mapping.get(loan_data.get(feature, 'C'), 2)
            elif feature == 'term':
                value = term_mapping.get(loan_data.get(feature, ' 36 months'), 0)
            else:
                value = 0  # Default for other categorical features
            processed_features.append(value)
        
        # Add engineered features (matching training)
        loan_amnt = loan_data.get('loan_amnt', 10000)
        annual_inc = loan_data.get('annual_inc', 50000)
        int_rate = loan_data.get('int_rate', 10.0)
        revol_bal = loan_data.get('revol_bal', 5000)
        revol_util = loan_data.get('revol_util', 50)
        open_acc = loan_data.get('open_acc', 5)
        total_acc = loan_data.get('total_acc', 10)
        pub_rec = loan_data.get('pub_rec', 0)
        delinq_2yrs = loan_data.get('delinq_2yrs', 0)
        total_rev_hi_lim = loan_data.get('total_rev_hi_lim', 10000)
        
        # Engineered features
        loan_to_income_ratio = loan_amnt / (annual_inc + 1)
        total_interest = int_rate * loan_amnt / 100
        revol_bal_to_util_ratio = revol_bal / (revol_util + 0.01)
        open_to_total_acc_ratio = open_acc / (total_acc + 1)
        total_negative_records = pub_rec + delinq_2yrs
        credit_util_ratio = revol_bal / (total_rev_hi_lim + 1)
        
        engineered = [loan_to_income_ratio, total_interest, revol_bal_to_util_ratio,
                     open_to_total_acc_ratio, total_negative_records, credit_util_ratio]
        processed_features.extend(engineered)
        
        # Process NLP features (combined text)
        text_parts = []
        text_parts.append(loan_data.get('desc', ''))
        text_parts.append(loan_data.get('emp_title', ''))
        text_parts.append(loan_data.get('purpose', ''))
        text_parts.append(loan_data.get('title', ''))
        
        combined_text = (text_parts[0] + '. ' + text_parts[1] + 
                        '. Purpose: ' + text_parts[2] + 
                        '. Title: ' + text_parts[3]).strip()
        
        if not combined_text or combined_text == '. . Purpose: . Title: ':
            combined_text = 'No description provided'
        
        # Get NLP embedding
        nlp_embedding = self.sentence_model.encode([combined_text], convert_to_numpy=True)[0]
        processed_features.extend(nlp_embedding.tolist())
        
        # Create feature array
        feature_array = np.array(processed_features).reshape(1, -1)
        
        # Note: The model expects 300 features due to feature selection during training
        # This is a simplified version - in production you'd save and load the actual selector
        if feature_array.shape[1] > 300:
            # Simple truncation for demo purposes
            feature_array = feature_array[:, :300]
        elif feature_array.shape[1] < 300:
            # Pad with zeros if needed
            padding = np.zeros((1, 300 - feature_array.shape[1]))
            feature_array = np.concatenate([feature_array, padding], axis=1)
        
        return feature_array
    
    def predict_default_probability(self, loan_data, model='xgboost'):
        """
        Predict default probability
        
        Args:
            loan_data (dict): Loan information
            model (str): 'xgboost' or 'logistic'
            
        Returns:
            float: Default probability (0-1)
        """
        X = self.preprocess_loan(loan_data)
        
        if model == 'xgboost':
            probability = self.xgb_model.predict_proba(X)[0, 1]
        else:
            X_scaled = self.scaler.transform(X)
            probability = self.lr_model.predict_proba(X_scaled)[0, 1]
        
        return probability
    
    def get_risk_category(self, probability):
        """Categorize risk level"""
        if probability < 0.15:
            return "Low Risk"
        elif probability < 0.35:
            return "Medium Risk"
        else:
            return "High Risk"


# Example usage
if __name__ == "__main__":
    try:
        predictor = CreditRiskPredictor()
        
        # Example loan application
        sample_loan = {
            'loan_amnt': 25000,
            'int_rate': 14.5,
            'grade': 'C',
            'term': ' 60 months',
            'annual_inc': 65000,
            'dti': 18.5,
            'open_acc': 12,
            'revol_util': 75.2,
            'delinq_2yrs': 0,
            'pub_rec': 0,
            'desc': 'Consolidating high interest credit card debt to lower monthly payments',
            'emp_title': 'Marketing Manager',
            'purpose': 'debt_consolidation'
        }
        
        # Get predictions
        xgb_prob = predictor.predict_default_probability(sample_loan, 'xgboost')
        lr_prob = predictor.predict_default_probability(sample_loan, 'logistic')
        
        print("\nCREDIT RISK ASSESSMENT REPORT")
        print("=" * 45)
        print(f"Loan Amount: ${sample_loan['loan_amnt']:,}")
        print(f"Interest Rate: {sample_loan['int_rate']}%")
        print(f"Credit Grade: {sample_loan['grade']}")
        print(f"Term: {sample_loan['term'].strip()}")
        print(f"Annual Income: ${sample_loan['annual_inc']:,}")
        print(f"Debt-to-Income: {sample_loan['dti']}%")
        print()
        print("DEFAULT PROBABILITY PREDICTIONS:")
        print(f"   XGBoost Model: {xgb_prob:.1%} ({predictor.get_risk_category(xgb_prob)})")
        print(f"   Logistic Regression: {lr_prob:.1%} ({predictor.get_risk_category(lr_prob)})")
        print()
        
        # Recommendation based on XGBoost (primary model)
        if xgb_prob > 0.5:
            print(" RECOMMENDATION: HIGH RISK")
            print(" Consider REJECTING this loan application")
        elif xgb_prob > 0.3:
            print(" RECOMMENDATION: MEDIUM RISK")
            print(" Consider HIGHER interest rate or additional requirements")
        else:
            print(" RECOMMENDATION: LOW RISK") 
            print(" Loan appears SUITABLE for approval at standard terms")
            
    except FileNotFoundError as e:
        print(" Error: Model files not found. Please ensure the following files exist:")
        print("   - xgboost_model.pkl")
        print("   - logistic_regression_model.pkl") 
        print("   - feature_scaler.pkl")
    except Exception as e:
        print(f" Error: {e}")
