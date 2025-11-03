# Search L2 Evaluation Framework

A multi-label evaluation framework for search ranking models with engagement, purchase, and autoship metrics.

## 🎯 Overview

This framework evaluates search ranking quality using three distinct label types representing different stages of the user funnel: engagement (clicks/views), purchase (conversions), and autoship (subscriptions). It includes PDM integration for product enrichment and revenue analysis.

## ✨ Key Features

- **Multi-Label NDCG**: Separate NDCG metrics for engagement, purchase, and autoship
- **CTR/CVR Metrics**: Click-through rate and conversion rate evaluation  
- **Revenue Metrics**: Revenue per search, revenue recall, and average price analysis
- **PDM Integration**: Product catalog enrichment with merchandise categories and pricing
- **MC Analysis**: Performance breakdown by merchandise categories (MC1/MC2)
- **Step-by-Step Debugging**: Interactive Jupyter notebook for development
- **YAML Configuration**: Flexible metric and data source configuration

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
│   ├── ctr_cvr.py               # CTR and CVR metrics
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
pip install -r requirements.txt
```

### 2. Run Evaluation

```bash
python run_eval.py \
    --model_preds data/sample_model_preds.csv \
    --eval_data data/sample_eval_data.csv \
    --config config/eval_config.yaml
```

### 3. Programmatic Usage

```python
from evaluator.evaluator import Evaluator

evaluator = Evaluator('config/eval_config.yaml')
results = evaluator.run(
    'data/sample_model_preds.csv',
    'data/sample_eval_data.csv'
)

# Results include multiple metric variants
print(results['aggregate'])  # Overall averages
print(results['per_query'])  # Per-query details
```

## 📊 Data Formats

### Model Predictions (`sample_model_preds.csv`)
```csv
query_id,sku,rank,prediction_score
dog_treats_q1,246188,1,0.95
dog_treats_q1,258374,2,0.87
```

### Evaluation Data (`sample_eval_data.csv`)  
```csv
query_id,sku,engagement,purchase,autoship
dog_treats_q1,246188,3,1,1
dog_treats_q1,258374,2,1,0
```

**Label Types:**
- **`engagement`**: 0=none, 1=view, 2=click, 3=purchase
- **`purchase`**: 0=no purchase, 1=purchased  
- **`autoship`**: 0=no subscription, 1=autoship subscription

### PDM Data (Optional)
Product catalog with SKU, merchandise categories (MC1/MC2), and pricing for revenue analysis.

## ⚙️ Available Metrics

### Multi-Label NDCG
- **`ndcg_engagement@k`**: Ranking quality for user engagement (0-3 scale)
- **`ndcg_purchase@k`**: Ranking quality for purchase conversions (binary)
- **`ndcg_autoship@k`**: Ranking quality for subscription conversions (binary)

### CTR/CVR Metrics  
- **`ctr@k`**: Fraction of queries with engagement ≥ threshold in top k
- **`cvr@k`**: Fraction of queries with purchase in top k

### Revenue Metrics
- **`revenue_per_search@k`**: Average revenue per query from top k results
- **`revenue_recall@k`**: Fraction of total revenue captured in top k
- **`avg_price@k`**: Average price of products in top k

## ⚙️ Configuration

Key settings in `config/eval_config.yaml`:

```yaml
metrics:
  - name: ndcg_engagement
    params: { k: [10, 36, 72] }
  - name: ctr  
    params: { k: [5, 10, 20], ctr_label: 1 }
    
# PDM integration (optional)
pdm:
  file_path: "data/PDM Product_2025-10-01-2037.csv"
  purchase_label: 1
```

## 📈 Results

The evaluation produces:
- **`aggregate_metrics.csv`**: Overall performance averages across all metrics
- **`per_query_metrics.csv`**: Detailed per-query results for all metrics
- **`mc1_analysis.csv`** / **`mc2_analysis.csv`**: Performance by merchandise category

## 🔧 Extending the Framework

### Adding New Metrics

1. Create a new function in `metrics/` directory
2. Register with `@register_metric("metric_name")` decorator  
3. Follow signature: `def metric_name(group_df: pd.DataFrame, **params) -> float`

## 📝 Requirements

- **Python**: 3.8+
- **Dependencies**: pandas, numpy, PyYAML (see `requirements.txt`)

## 📞 Support

Create an issue in the GitHub repository for questions or problems.
