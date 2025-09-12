"""
Credit Risk Assessment Web App
Streamlit interface for the hybrid credit risk classifier
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from predict_loan import CreditRiskPredictor
import time

# Page configuration
st.set_page_config(
    page_title="Credit Risk Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .risk-high {
        background-color: #fff5f5;
        border: 3px solid #e53e3e;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #2d3748;
        box-shadow: 0 4px 6px rgba(229, 62, 62, 0.1);
    }
    .risk-medium {
        background-color: #fffaf0;
        border: 3px solid #dd6b20;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #2d3748;
        box-shadow: 0 4px 6px rgba(221, 107, 32, 0.1);
    }
    .risk-low {
        background-color: #f0fff4;
        border: 3px solid #38a169;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #2d3748;
        box-shadow: 0 4px 6px rgba(56, 161, 105, 0.1);
    }
    .risk-high h3 {
        color: #c53030;
        margin-top: 0;
    }
    .risk-medium h3 {
        color: #c05621;
        margin-top: 0;
    }
    .risk-low h3 {
        color: #2f855a;
        margin-top: 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)"""
    try:
        return CreditRiskPredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def create_gauge_chart(probability):
    """Create a gauge chart for risk probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Default Risk Probability (%)"},
        delta = {'reference': 20},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 15], 'color': "lightgreen"},
                {'range': [15, 35], 'color': "yellow"},
                {'range': [35, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_comparison_chart(xgb_prob, lr_prob):
    """Create a comparison chart between models"""
    models = ['XGBoost', 'Logistic Regression']
    probabilities = [xgb_prob * 100, lr_prob * 100]
    
    colors = ['#1f77b4', '#ff7f0e']
    
    fig = go.Figure(data=[
        go.Bar(x=models, y=probabilities, marker_color=colors)
    ])
    
    fig.update_layout(
        title="Model Comparison",
        yaxis_title="Default Probability (%)",
        xaxis_title="Model",
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">🏦 Credit Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Loan Default Risk Prediction")
    st.markdown("---")
    
    # Load model
    predictor = load_model()
    if not predictor:
        st.stop()
    
    # Sidebar for input
    st.sidebar.header("📋 Loan Application Details")
    
    # Basic loan information
    st.sidebar.subheader("Loan Information")
    loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=1000, max_value=100000, 
                                       value=st.session_state.get('loan_amnt', 25000), step=1000)
    int_rate = st.sidebar.slider("Interest Rate (%)", min_value=3.0, max_value=30.0, 
                                value=st.session_state.get('int_rate', 14.5), step=0.1)
    
    term_options = [" 36 months", " 60 months"]
    term_default = st.session_state.get('term', ' 60 months')
    term_index = term_options.index(term_default) if term_default in term_options else 1
    term = st.sidebar.selectbox("Loan Term", term_options, index=term_index)
    
    grade_options = ["A", "B", "C", "D", "E", "F", "G"]
    grade_default = st.session_state.get('grade', 'C')
    grade_index = grade_options.index(grade_default) if grade_default in grade_options else 2
    grade = st.sidebar.selectbox("Credit Grade", grade_options, index=grade_index)
    
    # Personal information
    st.sidebar.subheader("Personal Information")
    annual_inc = st.sidebar.number_input("Annual Income ($)", min_value=10000, max_value=500000, 
                                        value=st.session_state.get('annual_inc', 65000), step=5000)
    dti = st.sidebar.slider("Debt-to-Income Ratio (%)", min_value=0.0, max_value=50.0, 
                           value=st.session_state.get('dti', 18.5), step=0.1)
    emp_title = st.sidebar.text_input("Job Title", value=st.session_state.get('emp_title', 'Software Engineer'))
    
    # Credit information
    st.sidebar.subheader("Credit History")
    open_acc = st.sidebar.number_input("Open Credit Accounts", min_value=0, max_value=50, 
                                      value=st.session_state.get('open_acc', 12))
    revol_util = st.sidebar.slider("Credit Utilization (%)", min_value=0.0, max_value=100.0, 
                                  value=st.session_state.get('revol_util', 75.2), step=0.1)
    delinq_2yrs = st.sidebar.number_input("Delinquencies (2 years)", min_value=0, max_value=10, 
                                         value=st.session_state.get('delinq_2yrs', 0))
    pub_rec = st.sidebar.number_input("Public Records", min_value=0, max_value=10, 
                                     value=st.session_state.get('pub_rec', 0))
    
    # Additional fields with defaults
    default_revol_bal = st.session_state.get('revol_bal', int(revol_util/100 * 10000))
    revol_bal = st.sidebar.number_input("Revolving Balance ($)", min_value=0, max_value=100000, value=default_revol_bal)
    
    default_total_acc = st.session_state.get('total_acc', max(15, open_acc + 3))
    total_acc = st.sidebar.number_input("Total Credit Accounts", min_value=1, max_value=100, value=default_total_acc)
    
    # Loan purpose and description
    st.sidebar.subheader("Loan Purpose")
    purpose_options = ["debt_consolidation", "home_improvement", "major_purchase", "medical", 
                      "vacation", "car", "other"]
    purpose_default = st.session_state.get('purpose', 'debt_consolidation')
    purpose_index = purpose_options.index(purpose_default) if purpose_default in purpose_options else 0
    purpose = st.sidebar.selectbox("Purpose", purpose_options, index=purpose_index)
    
    desc_default = st.session_state.get('desc', 
        "Consolidating high interest credit card debt to lower monthly payments and improve financial situation.")
    desc = st.sidebar.text_area("Loan Description", value=desc_default, height=100)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Risk Assessment Results")
        
        if st.button("🔍 Analyze Loan Risk", type="primary"):
            with st.spinner("Analyzing loan application..."):
                # Prepare loan data
                loan_data = {
                    'loan_amnt': loan_amnt,
                    'int_rate': int_rate,
                    'grade': grade,
                    'term': term,
                    'annual_inc': annual_inc,
                    'dti': dti,
                    'open_acc': open_acc,
                    'revol_util': revol_util,
                    'delinq_2yrs': delinq_2yrs,
                    'pub_rec': pub_rec,
                    'revol_bal': revol_bal,
                    'total_acc': total_acc,
                    'desc': desc,
                    'emp_title': emp_title,
                    'purpose': purpose
                }
                
                # Get predictions
                time.sleep(1)  # Add slight delay for better UX
                xgb_prob = predictor.predict_default_probability(loan_data, 'xgboost')
                lr_prob = predictor.predict_default_probability(loan_data, 'logistic')
                
                # Store results in session state
                st.session_state.xgb_prob = xgb_prob
                st.session_state.lr_prob = lr_prob
                st.session_state.analyzed = True
        
        # Display results if analysis has been done
        if hasattr(st.session_state, 'analyzed') and st.session_state.analyzed:
            xgb_prob = st.session_state.xgb_prob
            lr_prob = st.session_state.lr_prob
            
            # Risk categorization
            risk_level = predictor.get_risk_category(xgb_prob)
            
            # Display main metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("XGBoost Prediction", f"{xgb_prob:.1%}", 
                         help="Primary model prediction")
            
            with col_b:
                st.metric("Logistic Regression", f"{lr_prob:.1%}",
                         help="Secondary model prediction")
            
            with col_c:
                st.metric("Risk Category", risk_level)
            
            # Gauge chart
            st.plotly_chart(create_gauge_chart(xgb_prob), use_container_width=True)
            
            # Risk assessment box
            if xgb_prob > 0.5:
                st.markdown(f"""
                <div class="risk-high">
                    <h3>🔴 HIGH RISK</h3>
                    <p><strong>Recommendation:</strong> Consider REJECTING this loan application</p>
                    <p>Default probability of {xgb_prob:.1%} is above acceptable threshold.</p>
                </div>
                """, unsafe_allow_html=True)
            elif xgb_prob > 0.3:
                st.markdown(f"""
                <div class="risk-medium">
                    <h3>🟡 MEDIUM RISK</h3>
                    <p><strong>Recommendation:</strong> Consider HIGHER interest rate or additional requirements</p>
                    <p>Default probability of {xgb_prob:.1%} suggests elevated risk.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>🟢 LOW RISK</h3>
                    <p><strong>Recommendation:</strong> Loan appears SUITABLE for approval</p>
                    <p>Default probability of {xgb_prob:.1%} is within acceptable range.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Model comparison
            st.subheader("Model Comparison")
            st.plotly_chart(create_comparison_chart(xgb_prob, lr_prob), use_container_width=True)
            
            # Detailed breakdown
            with st.expander("📈 Detailed Analysis"):
                st.write("**Key Risk Factors:**")
                
                risk_factors = []
                if int_rate > 15:
                    risk_factors.append(f"High interest rate ({int_rate}%)")
                if dti > 25:
                    risk_factors.append(f"High debt-to-income ratio ({dti}%)")
                if revol_util > 80:
                    risk_factors.append(f"High credit utilization ({revol_util}%)")
                if delinq_2yrs > 0:
                    risk_factors.append(f"Recent delinquencies ({delinq_2yrs})")
                if grade in ['E', 'F', 'G']:
                    risk_factors.append(f"Lower credit grade ({grade})")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.write(f"⚠️ {factor}")
                else:
                    st.write("✅ No major risk factors identified")
                
                st.write("**Loan Summary:**")
                st.write(f"- Amount: ${loan_amnt:,}")
                st.write(f"- Monthly payment (est): ${(loan_amnt * (int_rate/100/12) * (1 + int_rate/100/12)**(int(term.split()[0]))) / ((1 + int_rate/100/12)**(int(term.split()[0])) - 1):,.2f}")
                st.write(f"- Total interest (est): ${(loan_amnt * int_rate/100 * int(term.split()[0])/12):,.2f}")
    
    with col2:
        st.subheader("ℹ️ About This Model")
        
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Model Performance</h4>
            <ul>
                <li>ROC-AUC: <strong>78.5%</strong></li>
                <li>F1-Score: <strong>49.5%</strong></li>
                <li>Training Data: <strong>177K+ loans</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>🔬 Technology Stack</h4>
            <ul>
                <li><strong>XGBoost:</strong> Primary model</li>
                <li><strong>NLP:</strong> SentenceTransformers</li>
                <li><strong>Features:</strong> 300 selected features</li>
                <li><strong>Data:</strong> Lending Club (2007-2018)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Feature Types</h4>
            <ul>
                <li><strong>Structured:</strong> Financial ratios, credit history</li>
                <li><strong>Text Analysis:</strong> Loan description sentiment</li>
                <li><strong>Engineered:</strong> Debt ratios, utilization metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample applications
        st.subheader("📋 Quick Test Cases")
        
        col_low, col_high = st.columns(2)
        
        with col_low:
            if st.button("🟢 Low Risk Test", use_container_width=True):
                # Set low risk example values in session state
                st.session_state.update({
                    'low_risk_clicked': True,
                    'loan_amnt': 15000,
                    'int_rate': 7.5,
                    'grade': 'A',
                    'term': ' 36 months',
                    'annual_inc': 80000,
                    'dti': 10.0,
                    'open_acc': 8,
                    'revol_util': 20.0,
                    'delinq_2yrs': 0,
                    'pub_rec': 0,
                    'revol_bal': 2000,
                    'total_acc': 12,
                    'emp_title': 'Senior Software Engineer',
                    'purpose': 'home_improvement',
                    'desc': 'Home renovation project with stable income and excellent credit history.'
                })
                st.success("✅ Low risk example loaded! Check the sidebar and click Analyze.")
        
        with col_high:
            if st.button("🔴 High Risk Test", use_container_width=True):
                # Set high risk example values in session state
                st.session_state.update({
                    'high_risk_clicked': True,
                    'loan_amnt': 35000,
                    'int_rate': 24.5,
                    'grade': 'F',
                    'term': ' 60 months',
                    'annual_inc': 35000,
                    'dti': 35.0,
                    'open_acc': 15,
                    'revol_util': 95.0,
                    'delinq_2yrs': 3,
                    'pub_rec': 2,
                    'revol_bal': 9500,
                    'total_acc': 20,
                    'emp_title': 'Part-time Worker',
                    'purpose': 'debt_consolidation',
                    'desc': 'Need urgent cash for multiple debts, recent job loss, struggling financially.'
                })
                st.error("⚠️ High risk example loaded! Check the sidebar and click Analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏦 Credit Risk Assessment Tool | Built with Streamlit & XGBoost</p>
        <p>⚠️ For educational purposes only. Not for actual lending decisions.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
