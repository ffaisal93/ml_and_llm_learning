"""
Fraud Detection: Complete Business Solution
Real-time fraud detection for payment processing
"""
import numpy as np
import pandas as pd
from typing import Dict, List

class FraudDetectionSystem:
    """
    End-to-end fraud detection system
    
    Business Problem:
    - Detect fraudulent transactions in real-time
    - Minimize false positives (<0.1%)
    - Process transactions in <50ms
    - Reduce fraud losses by $X million
    """
    
    def __init__(self):
        self.model = None
        self.threshold = 0.5
    
    def engineer_features(self, transaction: Dict) -> np.ndarray:
        """
        Feature Engineering for Fraud Detection
        
        Key Features:
        1. Transaction: Amount, merchant, location, time
        2. User: Historical behavior, device, IP
        3. Pattern: Velocity (transactions/hour), unusual patterns
        """
        features = []
        
        # Transaction features
        features.append(transaction['amount'])
        features.append(transaction['merchant_category_code'])
        features.append(transaction.get('is_weekend', 0))
        features.append(transaction.get('is_night', 0))
        
        # User behavior features
        features.append(transaction.get('user_transaction_count_24h', 0))
        features.append(transaction.get('user_transaction_count_7d', 0))
        features.append(transaction.get('avg_transaction_amount_30d', 0))
        features.append(transaction.get('days_since_last_transaction', 0))
        
        # Pattern features
        features.append(transaction.get('amount_deviation_from_avg', 0))
        features.append(transaction.get('merchant_first_time', 0))
        features.append(transaction.get('location_change', 0))
        features.append(transaction.get('device_change', 0))
        
        # Velocity features
        features.append(transaction.get('transactions_per_hour', 0))
        features.append(transaction.get('amount_per_hour', 0))
        
        return np.array(features)
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train fraud detection model
        
        Model Selection:
        - Isolation Forest: Good for anomaly detection
        - Gradient Boosting: High accuracy, handles imbalance
        - Ensemble: Combine both
        """
        from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
        from imblearn.over_sampling import SMOTE
        
        # Handle class imbalance (99.9% legitimate, 0.1% fraud)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_resampled, y_resampled)
        
        # Optimize threshold for business metric
        # Want: High precision (minimize false positives)
        self.optimize_threshold(X_train, y_train)
    
    def optimize_threshold(self, X: np.ndarray, y: np.ndarray):
        """
        Optimize threshold for business metric
        
        Goal: Minimize false positives while catching fraud
        """
        from sklearn.metrics import precision_recall_curve
        
        y_scores = self.model.predict_proba(X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y, y_scores)
        
        # Find threshold that gives precision > 0.99 (false positive rate < 0.01)
        target_precision = 0.99
        idx = np.where(precision >= target_precision)[0]
        
        if len(idx) > 0:
            self.threshold = thresholds[idx[0]]
        else:
            self.threshold = 0.5
    
    def predict_fraud(self, transaction: Dict) -> Dict:
        """
        Predict if transaction is fraudulent
        
        Returns:
        - is_fraud: Boolean
        - risk_score: Probability (0-1)
        - decision_time: Processing time
        """
        import time
        start_time = time.time()
        
        # Engineer features
        features = self.engineer_features(transaction)
        features = features.reshape(1, -1)
        
        # Predict
        risk_score = self.model.predict_proba(features)[0, 1]
        is_fraud = risk_score >= self.threshold
        
        decision_time = (time.time() - start_time) * 1000  # ms
        
        return {
            'is_fraud': is_fraud,
            'risk_score': risk_score,
            'decision_time_ms': decision_time,
            'threshold': self.threshold
        }
    
    def evaluate_system(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate fraud detection system
        
        Metrics:
        - Precision: Minimize false positives (critical)
        - Recall: Catch fraud (important)
        - False Positive Rate: Must be <0.1%
        - Latency: Must be <50ms
        """
        from sklearn.metrics import precision_score, recall_score, confusion_matrix
        
        y_scores = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_scores >= self.threshold).astype(int)
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'false_positive_rate': false_positive_rate,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }


def fraud_detection_pipeline():
    """
    Complete pipeline for fraud detection
    
    Business Requirements:
    - Real-time processing (<50ms)
    - Low false positive rate (<0.1%)
    - High fraud detection rate (>90%)
    - Scalable to millions of transactions/day
    """
    print("Fraud Detection Pipeline")
    print("=" * 60)
    
    print("\n1. Problem Definition:")
    print("   - Type: Anomaly detection + Classification")
    print("   - Imbalance: 99.9% legitimate, 0.1% fraud")
    print("   - Latency: <50ms per transaction")
    print("   - False positive rate: <0.1%")
    
    print("\n2. Feature Engineering:")
    print("   - Transaction: Amount, merchant, location, time")
    print("   - User: Historical behavior, device, IP")
    print("   - Pattern: Velocity, unusual patterns")
    print("   - Time-based: Hour, day of week, time since last")
    
    print("\n3. Model Selection:")
    print("   - Primary: Gradient Boosting (high accuracy)")
    print("   - Secondary: Isolation Forest (anomaly detection)")
    print("   - Ensemble: Combine both")
    print("   - Handle imbalance: SMOTE, cost-sensitive learning")
    
    print("\n4. Threshold Optimization:")
    print("   - Optimize for precision (minimize false positives)")
    print("   - Balance with recall (catch fraud)")
    print("   - Business metric: Cost of false positive vs missed fraud")
    
    print("\n5. Deployment:")
    print("   - Real-time API: <50ms latency")
    print("   - Caching: Common patterns")
    print("   - Rule-based fallback: For edge cases")
    print("   - Human review: Top risk transactions")
    
    print("\n6. Monitoring:")
    print("   - Fraud detection rate")
    print("   - False positive rate")
    print("   - Latency (p50, p95, p99)")
    print("   - Model drift detection")


# Usage Example
if __name__ == "__main__":
    fraud_detection_pipeline()
    
    print("\n" + "=" * 60)
    print("Example Transaction")
    print("=" * 60)
    
    # Example transaction
    transaction = {
        'amount': 1500.0,
        'merchant_category_code': 5411,  # Grocery
        'is_weekend': 1,
        'is_night': 0,
        'user_transaction_count_24h': 5,  # High velocity
        'user_transaction_count_7d': 20,
        'avg_transaction_amount_30d': 50.0,
        'amount_deviation_from_avg': 1450.0,  # Large deviation
        'merchant_first_time': 1,  # First time at this merchant
        'location_change': 1,  # Different location
        'device_change': 0,
        'transactions_per_hour': 3,  # High velocity
        'amount_per_hour': 4500.0
    }
    
    print("\nTransaction Features:")
    for key, value in transaction.items():
        print(f"  {key}: {value}")
    
    print("\nRed Flags:")
    print("  - High velocity (5 transactions in 24h)")
    print("  - Large amount deviation ($1500 vs $50 avg)")
    print("  - First time at merchant")
    print("  - Location change")
    print("  - High transactions per hour")
    
    print("\n→ This transaction would likely be flagged as high risk")

