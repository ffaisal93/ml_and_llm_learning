"""
Customer Churn Prediction: Complete Solution
Business use case with detailed implementation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class ChurnPredictionSystem:
    """
    End-to-end churn prediction system
    
    Business Problem:
    - Predict which customers will cancel subscription in next 30 days
    - Target top K at-risk customers for retention campaigns
    - Measure business impact (churn reduction, ROI)
    """
    
    def __init__(self):
        self.model = None
        self.feature_importance = None
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Engineering for Churn Prediction
        
        Key Features:
        1. Usage patterns: login frequency, feature usage
        2. Engagement: support tickets, feedback
        3. Payment: failed payments, delays
        4. Behavioral: time since last login, feature adoption
        """
        features = df.copy()
        
        # Time-based features
        features['days_since_last_login'] = (
            pd.Timestamp.now() - pd.to_datetime(features['last_login_date'])
        ).dt.days
        
        features['login_frequency_7d'] = features['logins_last_7_days']
        features['login_frequency_30d'] = features['logins_last_30_days']
        
        # Engagement features
        features['support_ticket_count'] = features['tickets_last_month']
        features['satisfaction_score'] = features['avg_feedback_score']
        features['feature_adoption_rate'] = (
            features['features_used'] / features['total_features']
        )
        
        # Payment features
        features['failed_payment_count'] = features['failed_payments']
        features['payment_delay_avg'] = features['avg_payment_delay_days']
        features['subscription_age_days'] = features['days_since_signup']
        
        # Behavioral patterns
        features['session_duration_avg'] = features['avg_session_minutes']
        features['usage_trend'] = (
            features['logins_last_7_days'] / 
            (features['logins_last_30_days'] + 1)
        )  # Recent vs historical usage
        
        return features
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train churn prediction model
        
        Model Selection:
        - XGBoost: High accuracy, feature importance
        - Can ensemble with logistic regression for interpretability
        """
        from xgboost import XGBClassifier
        
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Feature importance for interpretability
        self.feature_importance = dict(zip(
            X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1]),
            self.model.feature_importances_
        ))
    
    def predict_churn_risk(self, customer_features: np.ndarray) -> np.ndarray:
        """
        Predict churn risk score (0-1)
        
        Returns probability of churning
        """
        return self.model.predict_proba(customer_features)[:, 1]
    
    def get_top_at_risk_customers(self, customers: pd.DataFrame, 
                                 k: int = 1000) -> pd.DataFrame:
        """
        Get top K customers at risk of churning
        
        Business Action:
        - Send retention offers to these customers
        - Prioritize by risk score
        """
        # Engineer features
        features = self.engineer_features(customers)
        
        # Predict risk
        risk_scores = self.predict_churn_risk(features.values)
        
        # Add risk scores
        customers_with_risk = customers.copy()
        customers_with_risk['churn_risk'] = risk_scores
        
        # Sort by risk (highest first)
        top_at_risk = customers_with_risk.nlargest(k, 'churn_risk')
        
        return top_at_risk
    
    def evaluate_business_impact(self, intervention_group: pd.DataFrame,
                                control_group: pd.DataFrame) -> Dict:
        """
        Evaluate business impact of churn prediction system
        
        A/B Test Results:
        - Intervention group: Received retention offers
        - Control group: No intervention
        
        Metrics:
        - Churn rate reduction
        - ROI: (Savings - Costs) / Costs
        - Customer lifetime value saved
        """
        # Churn rates
        intervention_churn_rate = intervention_group['churned'].mean()
        control_churn_rate = control_group['churned'].mean()
        
        # Reduction
        churn_reduction = control_churn_rate - intervention_churn_rate
        reduction_percentage = (churn_reduction / control_churn_rate) * 100
        
        # Business impact
        avg_customer_value = 1000  # Example: $1000 per customer
        customers_intervened = len(intervention_group)
        customers_saved = customers_intervened * churn_reduction
        revenue_saved = customers_saved * avg_customer_value
        
        # Costs
        intervention_cost_per_customer = 10  # Example: $10 per offer
        total_intervention_cost = customers_intervened * intervention_cost_per_customer
        
        # ROI
        roi = ((revenue_saved - total_intervention_cost) / total_intervention_cost) * 100
        
        return {
            'churn_reduction': churn_reduction,
            'reduction_percentage': reduction_percentage,
            'customers_saved': customers_saved,
            'revenue_saved': revenue_saved,
            'intervention_cost': total_intervention_cost,
            'roi': roi
        }


def churn_prediction_pipeline():
    """
    Complete pipeline for churn prediction
    
    Steps:
    1. Data collection and preparation
    2. Feature engineering
    3. Model training
    4. Prediction and action
    5. Evaluation and iteration
    """
    print("Churn Prediction Pipeline")
    print("=" * 60)
    
    # Step 1: Data preparation (simulated)
    print("\n1. Data Collection:")
    print("   - Customer demographics")
    print("   - Usage patterns (logins, features used)")
    print("   - Engagement metrics (support tickets, feedback)")
    print("   - Payment history")
    print("   - Churn labels (canceled in next 30 days)")
    
    # Step 2: Feature engineering
    print("\n2. Feature Engineering:")
    print("   - days_since_last_login")
    print("   - login_frequency_7d, login_frequency_30d")
    print("   - feature_adoption_rate")
    print("   - support_ticket_count")
    print("   - failed_payment_count")
    print("   - usage_trend (recent vs historical)")
    
    # Step 3: Model training
    print("\n3. Model Training:")
    print("   - Algorithm: XGBoost (high accuracy)")
    print("   - Evaluation: Precision@K, Recall, AUC-ROC")
    print("   - Cross-validation: 5-fold")
    
    # Step 4: Prediction
    print("\n4. Prediction & Action:")
    print("   - Score all customers daily")
    print("   - Identify top K at-risk customers")
    print("   - Send retention offers (discounts, features)")
    
    # Step 5: Evaluation
    print("\n5. Business Impact Evaluation:")
    print("   - A/B test: Intervention vs Control")
    print("   - Metrics: Churn rate, ROI, Customer lifetime value")
    print("   - Iterate based on results")


# Usage Example
if __name__ == "__main__":
    churn_prediction_pipeline()
    
    print("\n" + "=" * 60)
    print("Example Business Impact:")
    print("=" * 60)
    
    # Simulated results
    system = ChurnPredictionSystem()
    
    # Example: 10,000 customers, 5% churn rate
    intervention_group = pd.DataFrame({
        'churned': np.random.binomial(1, 0.03, 5000)  # 3% after intervention
    })
    control_group = pd.DataFrame({
        'churned': np.random.binomial(1, 0.05, 5000)  # 5% baseline
    })
    
    impact = system.evaluate_business_impact(intervention_group, control_group)
    
    print(f"\nChurn Reduction: {impact['reduction_percentage']:.1f}%")
    print(f"Customers Saved: {impact['customers_saved']:.0f}")
    print(f"Revenue Saved: ${impact['revenue_saved']:,.0f}")
    print(f"ROI: {impact['roi']:.1f}%")

