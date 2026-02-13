# Topic 29: System Design for ML

## What You'll Learn

This topic covers system design for ML systems:
- Scalable ML pipelines
- Model serving architecture
- Real-time vs batch inference
- Feature stores
- Model versioning
- Monitoring and alerting
- A/B testing infrastructure
- Cost optimization

## Why We Need This

### Interview Importance
- **Common questions**: "Design a system to serve 1M predictions/second"
- **System design**: Critical for ML engineer roles
- **Production knowledge**: Shows real-world experience

### Real-World Application
- **Production systems**: Need to scale
- **Reliability**: Systems must be robust
- **Cost**: Efficient systems save money

## System Design Topics

### 1. Scalable ML Pipeline

**Components:**
- Data ingestion
- Feature engineering
- Model training
- Model serving
- Monitoring

**Architecture:**
```
Data Sources → Feature Store → Training Pipeline → Model Registry
                                                      ↓
User Requests → Feature Store → Model Serving → Predictions
```

### 2. Model Serving

**Options:**
- **Real-time**: REST API, gRPC
- **Batch**: Scheduled jobs
- **Streaming**: Kafka, Flink

**Considerations:**
- Latency requirements
- Throughput requirements
- Cost constraints

### 3. Feature Stores

**Purpose:**
- Centralized feature storage
- Consistent features across training/serving
- Feature versioning
- Real-time feature computation

**Benefits:**
- Prevent training-serving skew
- Reuse features
- Faster development

### 4. Model Versioning

**Strategy:**
- Version models (v1, v2, ...)
- Track metadata (metrics, data, hyperparameters)
- Easy rollback

**Tools:**
- MLflow
- Weights & Biases
- Custom solutions

### 5. Monitoring

**What to monitor:**
- Prediction latency
- Throughput
- Error rates
- Data drift
- Model performance (A/B test)

**Alerting:**
- Set thresholds
- Alert on anomalies
- Dashboard for visualization

## Design Patterns

### Pattern 1: Real-Time Serving

**Requirements:**
- <100ms latency
- 10K requests/second
- 99.9% uptime

**Architecture:**
```
Load Balancer → API Gateway → Feature Service → Model Service → Cache
                                                      ↓
                                                 Database (for logging)
```

**Components:**
- **Load Balancer**: Distribute traffic
- **API Gateway**: Rate limiting, authentication
- **Feature Service**: Get features (from store or compute)
- **Model Service**: Run inference
- **Cache**: Store predictions for common requests

### Pattern 2: Batch Inference

**Requirements:**
- Process millions of records
- Run daily/weekly
- Cost-efficient

**Architecture:**
```
Scheduled Job → Data Pipeline → Feature Engineering → Batch Inference → Results Storage
```

**Tools:**
- Airflow (orchestration)
- Spark (processing)
- S3/GCS (storage)

### Pattern 3: A/B Testing Infrastructure

**Requirements:**
- Route traffic to different models
- Track metrics per variant
- Statistical significance testing

**Architecture:**
```
Request → Experiment Service → Model A (50%) / Model B (50%)
                                    ↓
                            Metrics Collection → Analysis
```

## Cost Optimization

### Strategies:
1. **Caching**: Cache predictions for common inputs
2. **Batching**: Process multiple requests together
3. **Model optimization**: Quantization, pruning
4. **Right-sizing**: Use appropriate instance types
5. **Spot instances**: For batch jobs

## Exercises

1. Design system for 1M predictions/second
2. Design feature store
3. Design A/B testing infrastructure
4. Optimize costs

## Next Steps

- **Topic 30**: A/B testing and experimentation
- Review all system design patterns

