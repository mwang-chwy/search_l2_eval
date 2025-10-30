# Search L2 Evaluation Framework

A comprehensive evaluation pipeline for search ranking models with PDM (Product Data Management) integration and revenue calculation capabilities.

## 🎯 Overview

This framework provides a complete solution for evaluating search ranking models in e-commerce environments. It integrates product catalog data, calculates revenue-based metrics, and provides detailed performance analysis at both query and merchandise category levels.

## ✨ Key Features

- **Multi-Metric Evaluation**: NDCG, Revenue per Search, Average Price
- **PDM Integration**: Automatic product enrichment with catalog data
- **Revenue Calculation**: Priority-based revenue computation (eval data → PDM fallback)
- **MC-Level Analysis**: Performance breakdown by merchandise categories
- **Real Data Support**: Works with actual Chewy SKU and product data
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

# Or for minimal installation (core framework only)
pip install -r requirements-minimal.txt
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
query_id,sku,relevance,revenue
dog_treats_q1,246188,3,15.99
dog_treats_q1,258374,2,0.0
cat_products_q2,370298,1,28.49
```

### PDM Data
Contains product catalog information including:
- `PART_NUMBER`: Product SKU
- `MC1_NAME`, `MC2_NAME`: Merchandise classifications
- `PRICE`: Product price
- Additional product metadata

## ⚙️ Configuration

Edit `config/eval_config.yaml` to customize:

```yaml
metrics:
  - name: "ndcg"
    params:
      k: [5, 10, 20]
      
  - name: "revenue_per_search"
    params: {}

pdm:
  file_path: "data/PDM Product_2025-10-01-2037.csv"
  part_number_col: "PART_NUMBER"
  price_col: "PRICE"
  mc1_col: "MC1_NAME"
  mc2_col: "MC2_NAME"
  purchase_label: 1

output:
  aggregate: "results/aggregate_metrics.csv"
  per_query: "results/per_query_metrics.csv"
  mc1_analysis: "results/mc1_analysis.csv"
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

## 🔧 Extending the Framework

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
- **Optional ML Components**: lightgbm, scikit-learn (for revenue analysis)
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