# Hybrid-Credit-risk-classifier-model-using-Transformers

A hybrid machine learning model for credit risk assessment that combines structured financial data with NLP features from borrower descriptions using SentenceTransformers.

• Architected Multimodal ML pipeline using 177k+ loan samples from Lending Club Dataset, combining
structured data with NLP features from borrower text using SentenceTransformers.
• Trained XGBoost and Logistic Regression models on combined structured features and text embeddings to
predict loan default risk, achieving 78.5% ROC-AUC and 0.495 F1-score.
• Built and deployed an interactive web app using Streamlit with real-time risk assessment dashboard,risk
categorization and automated lending recommendations


## 📊 Results Achieved
- **XGBoost Model**: ROC-AUC 0.785, F1-Score 0.495 (Final Optimized)
- **Logistic Regression**: ROC-AUC 0.781, F1-Score 0.382
- **Training Time**: ~8 minutes (with hyperparameter tuning)
- **Dataset**: 177,363 loans from Lending Club dataset (2007-2018)
- **Features**: 300 selected (from 457 total: structured + NLP)

## 🗂 Project Structure
```
Hybrid-Credit-risk-classifier-model-using-Transformers/
├── README.md                           # This file
├── requirements.txt                   # Python dependencies
├── credit_risk_classifier.py          # Main training pipeline
├── predict_loan.py                   # Loan prediction script (XGBoost)
├── app.py                            # Streamlit web application
├── credit_risk_analysis.png          # Comprehensive visualizations
├── Dataset link                      # Kaggle dataset link 
├── xgboost_model.pkl                 # Trained XGBoost model
├── logistic_regression_model.pkl     # Trained Logistic Regression model
└── feature_scaler.pkl               # Feature scaler for preprocessing
```


## 🎨 Visualizations Generated
The pipeline creates comprehensive visualizations including:
- Target distribution and class balance
- Interest rate and loan amount distributions by default status
- Default rate by credit grade
- ROC curves comparing both models
- Feature importance rankings
- Prediction distributions and calibration plots
- Confusion matrices

## 🔧 Technical Details

### Data Processing
- **Sampling**: 80,000 loans sampled from full dataset for optimal speed
- **Filtering**: Removed "Current" status loans (final outcome unknown)
- **Target**: Binary classification for default risk (18.38% positive class)
- **Encoding**: Label encoding for categorical variables, median imputation for missing values

### Feature Engineering
- **Structured Features**: 16 numerical + 8 categorical features
- **Engineered Features**: loan_to_income_ratio, total_interest
- **NLP Features**: 384-dimensional embeddings from `all-MiniLM-L6-v2` SentenceTransformer

### Model Optimization
- **XGBoost**: Histogram tree method, 100 estimators, optimized for speed
- **Logistic Regression**: StandardScaler preprocessing, 1000 max iterations
- **Training Time**: ~3 seconds (XGBoost) + ~6 seconds (LR)

### Top Features by Importance
1. `grade` - Credit grade assigned by Lending Club
2. `term`  - Loan term (36 vs 60 months)
3. `int_rate` - Interest rate
4. `sub_grade`  - More granular credit grade
5. `home_ownership`  - Homeownership status

## 💡 Key Insights
- **Grade dominance**: Credit grade is the most important predictor
- **Term impact**: Longer-term loans (60 months) have higher default rates
- **Interest rate correlation**: Higher rates strongly correlate with defaults
- **NLP contribution**: Text embeddings add predictive value beyond structured data
- **Class imbalance**: Default rate of 18.38% requires careful threshold tuning

## ⚡ Performance Optimization Features
- **Smart sampling**: Process subset of large dataset for faster training
- **Efficient NLP**: Use lightweight SentenceTransformer model
- **Optimized XGBoost**: Histogram tree method with parallelization
- **Memory efficient**: Process embeddings in batches
- **Quick visualization**: Single comprehensive plot with 12 subplots

## 🔍 Model Deployment
Use `predict_new_loan.py` for production predictions:
- Load pre-trained models
- Preprocess new loan applications
- Return probability scores and risk categories
- Provide lending recommendations

## 📈 Future Improvements
- **Hyperparameter tuning**: Grid search for optimal parameters
- **Class balancing**: SMOTE or class weights for imbalanced data
- **Threshold optimization**: Maximize F1-score or business metrics
- **Additional text fields**: Include employment title, purpose details
- **Ensemble methods**: Stack multiple models for better predictions
- **SHAP analysis**: Detailed feature importance for individual predictions

## 📝 Business Impact
This model can help lenders:
- Automate loan approval decisions
- Set appropriate interest rates based on risk
- Reduce default rates through better screening
- Process applications faster with ML automation
- Balance risk vs. profitability in lending decisions

