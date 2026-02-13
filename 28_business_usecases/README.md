# Topic 28: Business Use Cases & Solutions

## What You'll Learn

This topic covers real-world business problems with detailed solutions:
- Customer Churn Prediction
- Recommendation Systems
- Fraud Detection
- Price Optimization
- Demand Forecasting
- Click-Through Rate (CTR) Prediction
- Customer Lifetime Value (CLV)
- A/B Testing for ML Models

## Why We Need This

### Interview Importance
- **Common questions**: "How would you solve X business problem?"
- **Practical knowledge**: Shows real-world experience
- **Problem-solving**: Demonstrates end-to-end thinking

### Real-World Application
- **Production systems**: These are real problems
- **Business impact**: ML drives business value
- **System design**: Need to consider scalability, monitoring

## Business Use Cases

### 1. Customer Churn Prediction

**Problem Statement:**
A subscription-based company wants to predict which customers will cancel their subscription in the next month.

**Business Impact:**
- Reduce churn by 10% → Save $X million annually
- Identify at-risk customers early
- Target retention campaigns effectively

**Solution Approach:**

**Step 1: Define Problem**

**Target Variable:**
We need to define what "churn" means. For a subscription service, churn typically means a customer canceled their subscription. We create a binary target:
- **Churn = 1**: Customer canceled subscription in the next 30 days
- **Churn = 0**: Customer did not cancel in the next 30 days

**Time Window:**
We predict churn in the next 30 days. This gives the business time to intervene. The window should be:
- Long enough to allow intervention (30 days is reasonable)
- Short enough to be actionable (predicting 1 year ahead is less useful)
- Aligned with business cycles (monthly subscriptions → 30 days)

**Evaluation Metric:**
We use **Precision@K** because:
- Business has limited resources for retention campaigns
- Can only reach out to top K customers (e.g., top 1000)
- Want to maximize the proportion of those K who actually churn
- Precision@K = (Number of churned customers in top K) / K

**Why not just accuracy?**
- Churn is imbalanced (maybe 5% churn rate)
- Accuracy would be high (95%) even if we predict "no churn" for everyone
- Precision@K focuses on the actionable customers

**Step 2: Data Collection**

**Feature Categories:**

**Demographics:**
- Age, location, subscription tier, account age
- **Why important**: Different segments have different churn patterns (e.g., students vs professionals)

**Usage Patterns:**
- Login frequency (daily, weekly, monthly)
- Feature usage (which features used, how often)
- Session duration (average time per session)
- **Why important**: Declining usage often precedes churn. If someone stops logging in, they're likely to churn.

**Engagement Metrics:**
- Support ticket count (more tickets might indicate dissatisfaction)
- Feedback scores (low satisfaction → higher churn)
- Response to emails/notifications (engagement with communications)
- **Why important**: Engagement is a strong indicator of satisfaction and retention

**Payment History:**
- Payment method, payment frequency
- Failed payment count (payment issues → churn)
- Payment delays (late payments might indicate financial stress)
- **Why important**: Payment problems are a direct cause of churn

**Behavioral Patterns:**
- Time since last login (inactive users → churn)
- Feature adoption rate (tried new features? engaged with product?)
- Usage trend (increasing or decreasing?)
- **Why important**: Behavioral changes signal intent to churn

**Target Variable Creation:**
For each customer at time T, we look forward 30 days:
- If they canceled between T and T+30 days → churn = 1
- Otherwise → churn = 0

**Important**: We must only use information available at time T. No future information (data leakage).

**Step 3: Feature Engineering**

**Time-Based Features:**
These capture recency and frequency patterns:

```python
# Days since last login - critical feature
# If someone hasn't logged in for 30 days, high churn risk
features['days_since_last_login'] = (today - last_login_date).days

# Login frequency - measures engagement
features['login_frequency_7d'] = count_logins_last_7_days
features['login_frequency_30d'] = count_logins_last_30_days

# Trend: Is usage increasing or decreasing?
features['usage_trend'] = login_frequency_7d / (login_frequency_30d + 1)
# If trend < 1, usage is declining → churn risk
```

**Why these matter:** Inactive users are at high risk. If someone stops using the product, they'll likely cancel.

**Engagement Features:**
```python
# Support tickets - more tickets might indicate problems
features['support_ticket_count'] = count_tickets_last_month
features['avg_ticket_resolution_time'] = average_resolution_hours

# Satisfaction - direct measure of happiness
features['satisfaction_score'] = average_feedback_score
features['satisfaction_trend'] = recent_score - historical_score

# Feature adoption - engaged users try new features
features['feature_adoption_rate'] = features_used / total_features
features['new_features_tried_last_month'] = count_new_features_tried
```

**Why these matter:** Dissatisfaction and lack of engagement are strong predictors of churn.

**Payment Features:**
```python
# Payment problems - direct cause of churn
features['failed_payment_count'] = count_failed_payments_last_3_months
features['payment_delay_days'] = average_delay_days
features['payment_method_changes'] = count_payment_method_changes

# Subscription details
features['subscription_age_days'] = days_since_signup
features['price_per_month'] = monthly_subscription_cost
features['discount_applied'] = is_using_discount
```

**Why these matter:** Payment issues are a direct cause of involuntary churn. Also, newer customers and those on discounts might churn more.

**Behavioral Patterns:**
```python
# Feature usage patterns
features['most_used_feature'] = feature_with_highest_usage
features['feature_diversity'] = number_of_different_features_used

# Session patterns
features['avg_session_duration'] = average_minutes_per_session
features['sessions_per_week'] = average_sessions_per_week

# Communication engagement
features['email_open_rate'] = emails_opened / emails_sent
features['notification_response_rate'] = notifications_clicked / notifications_sent
```

**Why these matter:** Declining engagement across multiple dimensions indicates churn risk.

**Step 4: Model Selection**
- **Baseline**: Logistic regression (interpretable)
- **Production**: Gradient Boosting (XGBoost) for accuracy
- **Ensemble**: Combine multiple models

**Step 5: Evaluation Metrics**
- **Primary**: Precision@K (for top K at-risk customers)
- **Secondary**: Recall, F1-score, AUC-ROC
- **Business**: Cost of intervention vs cost of churn

**Step 6: Deployment**
- **Real-time**: Score customers daily
- **Action**: Send retention offers to top K at-risk customers
- **Monitoring**: Track actual churn rate, model drift

**Step 7: Business Impact Measurement**
- **A/B Test**: Random sample gets no intervention (control)
- **Measure**: Churn rate reduction in treatment group
- **ROI**: (Churn reduction × Customer value) - Intervention cost

**Code Structure:**
```python
# 1. Data preparation
def prepare_churn_data(df):
    # Feature engineering
    # Target creation
    return features, target

# 2. Model training
def train_churn_model(X_train, y_train):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    return model

# 3. Prediction
def predict_churn_risk(model, customer_features):
    risk_score = model.predict_proba(customer_features)[:, 1]
    return risk_score

# 4. Action
def get_top_at_risk_customers(model, customers, k=1000):
    scores = predict_churn_risk(model, customers)
    top_k_indices = np.argsort(scores)[-k:]
    return top_k_indices
```

### 2. Recommendation System

**Problem Statement:**
E-commerce platform wants to recommend products to users to increase sales.

**Business Impact:**
- Increase conversion rate by 15%
- Improve user engagement
- Increase average order value

**Solution Approach:**

**Step 1: Problem Definition**
- **Type**: Collaborative filtering + Content-based
- **Evaluation**: Precision@K, Recall@K, NDCG@K
- **Real-time**: Need <100ms latency

**Step 2: Data**
- **User-item interactions**: Views, purchases, ratings
- **Item features**: Category, price, brand, description
- **User features**: Demographics, purchase history

**Step 3: Model Architecture**
- **Hybrid approach**:
  1. Collaborative filtering (matrix factorization)
  2. Content-based (item similarity)
  3. Deep learning (neural collaborative filtering)
- **Ensemble**: Combine all three

**Step 4: Implementation**
```python
# Matrix factorization
def train_collaborative_filtering(interactions):
    # User-item matrix
    # Factorize: R ≈ P @ Q^T
    model = MatrixFactorization()
    model.fit(interactions)
    return model

# Content-based
def content_based_recommendations(user_history, item_features):
    # Find items similar to user's past purchases
    similarities = cosine_similarity(user_history, item_features)
    return top_k_items(similarities)

# Hybrid
def hybrid_recommendations(user_id, collaborative_scores, content_scores):
    # Weighted combination
    final_scores = 0.6 * collaborative_scores + 0.4 * content_scores
    return top_k_items(final_scores)
```

**Step 5: Evaluation**
- **Offline**: Train/test split, cross-validation
- **Online**: A/B test with real users
- **Metrics**: CTR, conversion rate, revenue

**Step 6: Deployment**
- **Caching**: Pre-compute recommendations
- **Real-time**: Update for new users/items
- **Cold start**: Use content-based for new users

### 3. Fraud Detection

**Problem Statement:**
Payment processor needs to detect fraudulent transactions in real-time.

**Business Impact:**
- Reduce fraud losses by $X million
- Maintain low false positive rate (<0.1%)
- Process transactions in <50ms

**Solution Approach:**

**Step 1: Problem Definition**
- **Type**: Anomaly detection + Classification
- **Imbalance**: 99.9% legitimate, 0.1% fraud
- **Latency**: Real-time (<50ms)

**Step 2: Features**
- **Transaction**: Amount, merchant, location, time
- **User**: Historical behavior, device, IP
- **Pattern**: Velocity (transactions/hour), unusual patterns

**Step 3: Model**
- **Primary**: Isolation Forest (anomaly detection)
- **Secondary**: Gradient Boosting (classification)
- **Ensemble**: Combine both

**Step 4: Handling Imbalance**
- **Sampling**: SMOTE (oversample minority)
- **Cost-sensitive**: Higher penalty for false negatives
- **Threshold tuning**: Optimize for business metric

**Step 5: Evaluation**
- **Primary**: Precision (minimize false positives)
- **Secondary**: Recall (catch fraud)
- **Business**: Fraud detection rate, false positive rate

**Step 6: Deployment**
- **Real-time scoring**: Model API
- **Rule-based fallback**: For edge cases
- **Human review**: Top risk transactions

### 4. Price Optimization

**Problem Statement:**
Ride-sharing company wants to optimize pricing to maximize revenue.

**Business Impact:**
- Increase revenue by 5-10%
- Balance supply and demand
- Maintain customer satisfaction

**Solution Approach:**

**Step 1: Problem**
- **Objective**: Maximize revenue = price × demand(price)
- **Constraints**: Price bounds, competitor prices
- **Dynamic**: Update prices in real-time

**Step 2: Data**
- **Historical**: Past prices, demand, revenue
- **Context**: Time, location, weather, events
- **Competitor**: Competitor prices

**Step 3: Model**
- **Demand model**: Predict demand as function of price
- **Price elasticity**: How demand changes with price
- **Optimization**: Find price that maximizes revenue

**Step 4: Implementation**
```python
def predict_demand(price, features):
    # Demand = f(price, time, location, ...)
    model = GradientBoostingRegressor()
    demand = model.predict([[price] + features])
    return demand

def optimize_price(features, price_bounds):
    # Revenue = price × demand(price)
    # Find price that maximizes revenue
    best_price = None
    best_revenue = 0
    
    for price in np.linspace(price_bounds[0], price_bounds[1], 100):
        demand = predict_demand(price, features)
        revenue = price * demand
        if revenue > best_revenue:
            best_revenue = revenue
            best_price = price
    
    return best_price
```

**Step 5: Evaluation**
- **A/B test**: Compare optimized vs fixed pricing
- **Metrics**: Revenue, customer satisfaction, supply/demand balance

### 5. Demand Forecasting

**Problem Statement:**
Retailer needs to forecast product demand to optimize inventory.

**Business Impact:**
- Reduce inventory costs by 20%
- Minimize stockouts
- Improve customer satisfaction

**Solution Approach:**

**Step 1: Problem**
- **Type**: Time series forecasting
- **Horizon**: Next 7, 30, 90 days
- **Granularity**: Product × Store × Day

**Step 2: Features**
- **Historical**: Past sales, trends, seasonality
- **External**: Holidays, promotions, weather
- **Product**: Category, price, lifecycle stage

**Step 3: Models**
- **Baseline**: Moving average, exponential smoothing
- **Advanced**: ARIMA, Prophet, LSTM
- **Ensemble**: Combine multiple models

**Step 4: Evaluation**
- **Metrics**: MAPE, RMSE, MAE
- **Business**: Stockout rate, overstock rate

## General Problem-Solving Framework

### 1. Understand Business Problem
- What is the business goal?
- What are the constraints?
- What is success?

### 2. Define ML Problem
- Classification, regression, ranking?
- What is the target variable?
- What are the features?

### 3. Data Collection
- What data is available?
- What data is needed?
- How to label data?

### 4. Feature Engineering
- Domain knowledge
- Time-based features
- Interactions

### 5. Model Selection
- Start simple (baseline)
- Try multiple models
- Ensemble if needed

### 6. Evaluation
- Offline metrics
- Online A/B test
- Business metrics

### 7. Deployment
- Real-time vs batch
- Latency requirements
- Monitoring

### 8. Iteration
- Monitor performance
- Collect feedback
- Improve model

## Exercises

1. Design solution for a new business problem
2. Implement end-to-end pipeline
3. Evaluate business impact
4. Design A/B test

## Next Steps

- **Topic 29**: System design for ML
- **Topic 30**: A/B testing and experimentation

