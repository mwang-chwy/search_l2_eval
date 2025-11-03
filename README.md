# Search L2 Evaluation Framework

A comprehensive evaluation pipeline for search ranking models with PDM (Product Data Management) integration and revenue calculation capabilities.

## 🎯 Overview

This framework provides a complete solution for evaluating search ranking models in e-commerce environments. It integrates product catalog data, calculates revenue-based metrics, and provides detailed performance analysis at both query and merchandise category levels.

## ✨ Key Features

- **Multi-Label NDCG**: Three specialized NDCG metrics (engagement, purchase, autoship)
- **Business-Aligned Metrics**: Revenue per Search, Average Price, Revenue Recall
- **PDM Integration**: Automatic product enrichment with catalog data
- **Revenue Calculation**: Dynamic revenue computation from purchase labels and PDM prices
- **MC-Level Analysis**: Performance breakdown by merchandise categories
- **Real Data Support**: Works with actual Chewy SKU and product data
- **Multi-Objective Evaluation**: Compare engagement vs conversion vs LTV optimization
- **Flexible Configuration**: YAML-based configuration system
- **Data Type Consistency**: Robust handling of various pandas data types
- **Debugging Tools**: Step-by-step Jupyter notebook for development

## 📁 Project Structure

```
search_l2_eval/
├── config/
│   └── eval_config.yaml          # Main configuration file
├── data/
│   ├── sample_eval_data.csv      # Ground truth evaluation data
│   ├── sample_model_preds.csv    # Model prediction rankings
│   ├── pdm_subset.csv           # PDM product data subset
│   └── PDM Product_2025-10-01-2037.csv  # Full PDM catalog
├── evaluator/
│   ├── __init__.py
│   └── evaluator.py             # Main evaluation orchestrator
├── metrics/
│   ├── __init__.py
│   ├── registry.py              # Metric registration system
│   ├── ndcg.py                  # NDCG implementation
│   ├── revenue_per_search.py    # Revenue calculation
│   └── avg_price.py             # Average price metrics
├── utils/
│   ├── __init__.py
│   ├── io.py                    # Data loading utilities
│   ├── validation.py            # Schema validation
│   └── pdm_utils.py             # PDM integration utilities
├── run_eval.py                  # CLI evaluation runner
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/mwang-chwy/search_l2_eval.git
cd search_l2_eval

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Evaluation

```bash
python run_eval.py \
    --sample_model_preds data/sample_model_preds.csv \
    --sample_eval_data data/sample_eval_data.csv \
    --config config/eval_config.yaml
```

### 3. Programmatic Usage

```python
from evaluator.evaluator import Evaluator

# Initialize evaluator
evaluator = Evaluator('config/eval_config.yaml')

# Run evaluation
results = evaluator.run(
    model_output='data/sample_model_preds.csv',
    eval_dataset='data/sample_eval_data.csv'
)

# Access results
print("Aggregate Metrics:", results['aggregate'])
print("Per-Query Results:", results['per_query'])
print("MC1 Analysis:", results['mc1_analysis'])

# Example output - multiple metric variants per query
# Per-query results include:
# - ndcg_engagement@10: 0.8234
# - ndcg_purchase@10: 0.7123  
# - ndcg_autoship@10: 0.6234
# - ctr@10: 0.85
# - cvr@10: 0.45
# - revenue_per_search@10: 45.67
```

## 📊 Data Formats

### Model Predictions (`sample_model_preds.csv`)
```csv
query_id,sku,rank,prediction_score
dog_treats_q1,246188,1,0.95
dog_treats_q1,258374,2,0.87
cat_products_q2,370298,1,0.92
```

### Evaluation Data (`sample_eval_data.csv`)
```csv
query_id,sku,engagement,purchase,autoship
dog_treats_q1,246188,3,1,1
dog_treats_q1,258374,2,1,0
cat_products_q2,370298,1,0,0
```

**Multi-Label Structure:**
- **`engagement`**: User engagement level (0=none, 1=view, 2=click, 3=high)
- **`purchase`**: Binary purchase indicator (0=no, 1=yes)
- **`autoship`**: Binary subscription indicator (0=no, 1=yes)

### PDM Data
Contains product catalog information including:
- `PART_NUMBER`: Product SKU
- `MC1_NAME`, `MC2_NAME`: Merchandise classifications
- `PRICE`: Product price
- Additional product metadata

## ⚙️ Configuration

Edit `config/eval_config.yaml` to customize:

```yaml
# Schema mapping for multi-label data
schema_mapping:
  search_id: query_id
  part_number: sku
  engagement: engagement    # Engagement labels (0,1,2,3)
  purchase: purchase        # Purchase labels (0,1)
  autoship: autoship       # Autoship labels (0,1)

# Multi-label NDCG metrics
metrics:
  - name: ndcg_engagement
    params: { k: [10, 36, 72] }
    
  - name: ndcg_purchase
    params: { k: [10, 36, 72] }
    
  - name: ndcg_autoship
    params: { k: [10, 36, 72] }
  
  # CTR/CVR metrics    
  - name: ctr
    params: { k: [5, 10, 20], ctr_label: 1 }
  - name: cvr
    params: { k: [5, 10, 20], cvr_label: 1 }
      
  - name: revenue_per_search
    params: { k: [8, 16, 32] }

# PDM configuration
pdm:
  file_path: "data/PDM Product_2025-10-01-2037.csv"
  part_number_col: "PART_NUMBER"
  mc1_col: "MERCH_CLASSIFICATION1"
  mc2_col: "MERCH_CLASSIFICATION2"
  price_col: "PRICE"
  purchase_label: 1  # purchase column value indicating purchase

# Output configuration
output:
  aggregate: "results/aggregate_metrics.csv"
  per_query: "results/per_query_metrics.csv"
  mc1_analysis: "results/mc1_analysis.csv"
  mc2_analysis: "results/mc2_analysis.csv"
```

## 📈 Output Files

The framework generates several output files:

- **`aggregate_metrics.csv`**: Overall performance averages
- **`per_query_metrics.csv`**: Detailed metrics for each search query
- **`mc1_analysis.csv`**: Performance breakdown by merchandise category level 1
- **`mc2_analysis.csv`**: Performance breakdown by merchandise category level 2

## 🧪 Development & Debugging

For development and debugging, use the step-by-step notebook:
- `evaluator_step_by_step.ipynb` - Interactive debugging with detailed explanations

## 💡 Key Features Deep Dive

### Revenue Calculation Priority
1. **Primary**: Use revenue from evaluation data if available
2. **Fallback**: Use price from PDM catalog for purchased items (relevance = purchase_label)
3. **Default**: Revenue = 0 for non-purchased items

### Data Type Consistency
- Automatic string type conversion for merge operations
- Handles mixed data types from different sources
- Prevents pandas merge errors between int64 and string columns

### Missing Label Handling
- Fills null relevance values with 0 (unlabeled items)
- Realistic production scenario support
- Graceful handling of incomplete evaluation datasets

## � Metrics Definitions

The framework supports multiple evaluation metrics designed for e-commerce search ranking assessment:

### Multi-Label NDCG Metrics

The framework supports **three specialized NDCG metrics** for different business objectives:

#### 1. Engagement NDCG (`ndcg_engagement@k`)
```python
# Uses engagement labels (0,1,2,3)
ndcg_engagement@10: 0.8234
```
- **Purpose**: Measures ranking quality based on user engagement levels
- **Use Case**: Optimizing for user interaction and content discovery
- **Label Scale**: 
  - `0`: No engagement
  - `1`: View/impression
  - `2`: Click/interaction  
  - `3`: High engagement (add to cart, detailed view)
- **Business Value**: Higher engagement indicates better user experience and content relevance

#### 2. Purchase NDCG (`ndcg_purchase@k`)
```python
# Uses binary purchase labels (0,1)
ndcg_purchase@10: 0.7123
```
- **Purpose**: Measures ranking quality for actual purchase conversions
- **Use Case**: Optimizing for sales performance and conversion rates
- **Label Scale**:
  - `0`: No purchase
  - `1`: Purchase completed
- **Business Value**: Direct correlation with sales revenue and conversion optimization

#### 3. Autoship NDCG (`ndcg_autoship@k`)
```python  
# Uses binary autoship labels (0,1)
ndcg_autoship@10: 0.6234
```
- **Purpose**: Measures ranking quality for subscription conversions
- **Use Case**: Optimizing for customer lifetime value (LTV)
- **Label Scale**:
  - `0`: No autoship subscription
  - `1`: Autoship subscription created
- **Business Value**: Recurring revenue and customer retention optimization

### Revenue-Based Metrics

#### Revenue per Search (`revenue_per_search@k`)
```python
revenue_per_search@10: 45.67  # Average revenue per search query
```
- **Purpose**: Measures the average revenue generated per search query
- **Calculation**: `sum(revenue for purchased items in top k) / total_queries`
- **Use Case**: Direct business impact measurement
- **Business Value**: Quantifies the monetary value of search result quality

#### Revenue Recall (`revenue_recall@k`) 
```python
revenue_recall@10: 0.78  # Fraction of potential revenue captured
```
- **Purpose**: Measures what fraction of potential revenue is captured in top k results
- **Calculation**: `revenue_captured@k / total_potential_revenue`
- **Use Case**: Understanding revenue opportunity coverage
- **Business Value**: Identifies revenue loss due to poor ranking

### Click-Through Rate & Conversion Metrics

#### CTR@k (Click-Through Rate)
```python
ctr@10: 0.85  # 85% of queries have at least one click in top 10
```
- **Purpose**: Measures fraction of queries with at least one engagement >= CTR threshold in top k
- **Calculation**: Binary per-query metric (1 if any item in top k has engagement >= ctr_label, 0 otherwise)
- **Default Threshold**: `ctr_label=1` (view/impression counts as click)
- **Use Case**: Understanding search result discoverability and user engagement
- **Business Value**: Measures whether users find something interesting to click on

#### CVR@k (Conversion Rate - Purchase)
```python
cvr@10: 0.45  # 45% of queries result in at least one purchase in top 10
```
- **Purpose**: Measures fraction of queries with at least one purchase in top k
- **Calculation**: Binary per-query metric (1 if any item in top k has purchase >= cvr_label, 0 otherwise)
- **Default Threshold**: `cvr_label=1` (purchase=1 counts as conversion)
- **Use Case**: Understanding purchase conversion rates
- **Business Value**: Direct measure of search effectiveness for sales

### Price-Based Metrics

#### Average Price (`avg_price@k`)
```python
avg_price@10: 23.45  # Average price of products in top k
```
- **Purpose**: Measures the average price of products in top k results
- **Use Case**: Understanding price distribution in search results
- **Business Value**: Price positioning and margin analysis

### Traditional Ranking Metrics

#### NDCG (Normalized Discounted Cumulative Gain)
- **Formula**: `DCG@k / IDCG@k`
- **DCG**: `Σ(2^rel - 1) / log2(rank + 1)` for items 1 to k
- **IDCG**: DCG of the ideal ranking (sorted by relevance)
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: 
  - 1.0 = Perfect ranking
  - 0.8+ = Excellent ranking
  - 0.6+ = Good ranking
  - <0.5 = Poor ranking

### Metric Selection Guidelines

#### For User Experience Optimization
- **Primary**: `ndcg_engagement@10`, `ctr@10`
- **Secondary**: `ndcg_engagement@36`, Traditional `ndcg@10`
- **Focus**: Content discovery and user satisfaction

#### For Sales Performance Optimization  
- **Primary**: `ndcg_purchase@10`, `cvr@10`, `revenue_per_search@10`
- **Secondary**: `revenue_recall@10`, `avg_price@10`
- **Focus**: Conversion rates and immediate revenue

#### For Customer Lifetime Value
- **Primary**: `ndcg_autoship@10`, `ndcg_autoship@36` 
- **Secondary**: `revenue_per_search@32` (longer-term view)
- **Focus**: Subscription acquisition and retention

#### Multi-Objective Optimization
Use all three NDCG variants together to balance:
- User engagement (content quality)
- Purchase conversion (immediate revenue)  
- Autoship conversion (long-term value)

### Interpreting Results

#### Correlation Analysis
Compare metrics to understand business relationships:
```python
# High CTR but low CVR may indicate:
ctr@10: 0.85              # Users click on results
cvr@10: 0.25              # But rarely convert to purchase

# High engagement but low purchase may indicate:
ndcg_engagement@10: 0.85  # Users find content interesting  
ndcg_purchase@10: 0.45    # But don't convert to purchases

# High purchase but low autoship may indicate:
ndcg_purchase@10: 0.78    # Good immediate conversions
ndcg_autoship@10: 0.32    # But poor subscription retention
```

## � Extending the Framework

### Adding New Metrics

1. Create a new metric file in `metrics/`
2. Register it in `metrics/registry.py`
3. Follow the standard metric function signature:

```python
def my_metric(df: pd.DataFrame, **params) -> float:
    # Your metric calculation
    return result
```

### Custom PDM Integration

Modify `utils/pdm_utils.py` to support different product catalog formats or additional enrichment logic.

## 📝 Requirements

- **Python**: 3.8+
- **Core Dependencies**: See `requirements.txt`
  - pandas >= 2.0.0
  - numpy >= 1.21.0  
  - PyYAML >= 6.0
- **Development**: jupyter, ipykernel (for notebook debugging)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 📞 Support

For questions or issues, please create an issue in the GitHub repository.

---

**Note**: This framework is designed for internal Chewy search evaluation workflows and includes real product data for authentic testing scenarios.