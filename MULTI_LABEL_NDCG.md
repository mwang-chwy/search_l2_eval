# Multi-Label NDCG Evaluation System

## 📊 Overview

The search evaluation framework now supports **multiple evaluation metrics** to capture different aspects of search relevance:

## NDCG Metrics
1. **Engagement NDCG** - Measures ranking quality based on user engagement levels
2. **Purchase NDCG** - Measures ranking quality based on actual purchases  
3. **Autoship NDCG** - Measures ranking quality based on subscription conversions

## CTR/CVR Metrics  
4. **CTR@k** - Click-through rate (engagement threshold-based)
5. **CVR@k** - Conversion rate using purchase labels

## 🗂️ Data Structure

### Input Data Format (`sample_eval_data.csv`)
```csv
query_id,sku,engagement,purchase,autoship
dog_treats_q1,246188,3,1,1
dog_treats_q1,370298,2,1,0
dog_treats_q1,58579,1,0,0
dog_treats_q1,993862,0,0,0
```

### Label Definitions
- **`engagement`**: 4-level engagement score (0=none, 1=view, 2=click, 3=high engagement)
- **`purchase`**: Binary purchase indicator (0=no purchase, 1=purchased) 
- **`autoship`**: Binary autoship indicator (0=no autoship, 1=autoship subscription)

## 🎯 NDCG Metrics

### 1. Engagement NDCG (`ndcg_engagement@k`)
```python
@register_metric("ndcg_engagement")
def ndcg_engagement(group_df, k=10):
    return _calculate_ndcg_for_column(group_df, 'engagement', k)
```
- **Purpose**: Measures how well the ranking captures user engagement patterns
- **Use Case**: Optimizing for user engagement and interaction quality
- **Values**: Uses engagement scores (0,1,2,3) directly as relevance grades

### 2. Purchase NDCG (`ndcg_purchase@k`)
```python
@register_metric("ndcg_purchase") 
def ndcg_purchase(group_df, k=10):
    return _calculate_ndcg_for_column(group_df, 'purchase', k)
```
- **Purpose**: Measures how well the ranking drives actual purchases
- **Use Case**: Optimizing for conversion and sales performance
- **Values**: Uses binary purchase labels (0,1) as relevance grades

### 3. Autoship NDCG (`ndcg_autoship@k`)
```python
@register_metric("ndcg_autoship")
def ndcg_autoship(group_df, k=10):
    return _calculate_ndcg_for_column(group_df, 'autoship', k)
```
- **Purpose**: Measures how well the ranking drives subscription conversions
- **Use Case**: Optimizing for customer lifetime value (LTV)
- **Values**: Uses binary autoship labels (0,1) as relevance grades

## 🔧 Revenue Calculation

Revenue is calculated dynamically using the `calculate_revenue` function:

```python
def calculate_revenue(df: pd.DataFrame, purchase_label: int) -> pd.DataFrame:
    # If purchase == 1: use price from PDM table
    # If purchase == 0: revenue = 0
```

## 📈 Configuration

The system is configured in `config/eval_config.yaml`:

```yaml
schema_mapping:
  engagement: engagement    # Engagement labels (0,1,2,3)
  purchase: purchase        # Purchase labels (0,1)  
  autoship: autoship       # Autoship labels (0,1)

metrics:
  - name: ndcg_engagement
    params: { k: [10, 36, 72] }
  - name: ndcg_purchase
    params: { k: [10, 36, 72] }
  - name: ndcg_autoship
    params: { k: [10, 36, 72] }
```

## 🎯 Business Applications

### Engagement vs Purchase Analysis
Compare `ndcg_engagement` and `ndcg_purchase` to understand:
- Are engaging products actually purchased?
- Is there a gap between engagement and conversion?

### Subscription Optimization  
Use `ndcg_autoship` to:
- Optimize for high-LTV subscription customers
- Identify products that drive recurring revenue

### Multi-Objective Optimization
Combine all three NDCG metrics to:
- Optimize for multiple business goals simultaneously
- Create balanced search experiences

## 🚀 Usage

### Step-by-Step Notebook
Open `evaluator_step_by_step.ipynb` to:
- Run each NDCG metric individually  
- Compare results across different relevance definitions
- Debug and customize the evaluation pipeline

### Production Evaluation
```python
from evaluator.evaluator import Evaluator
evaluator = Evaluator('config/eval_config.yaml')
results = evaluator.run('data/sample_model_preds.csv', 'data/sample_eval_data.csv')

# Access different NDCG results
engagement_ndcg = results['per_query']['ndcg_engagement@10']
purchase_ndcg = results['per_query']['ndcg_purchase@10']  
autoship_ndcg = results['per_query']['ndcg_autoship@10']
```

## 📊 Expected Output

Results will include all NDCG variants:
```
ndcg_engagement@10: 0.8234
ndcg_engagement@36: 0.7891  
ndcg_engagement@72: 0.7654

ndcg_purchase@10: 0.7123
ndcg_purchase@36: 0.6876
ndcg_purchase@72: 0.6543

ndcg_autoship@10: 0.6234
ndcg_autoship@36: 0.5987
ndcg_autoship@72: 0.5756
```

This provides comprehensive insights into search quality from three key business perspectives! 🎉