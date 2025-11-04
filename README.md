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
# Or install dependencies directly:
# pip install pandas numpy pyyaml pyarrow
```

### 2. Run Evaluation

The framework automatically detects whether you're using separate files or a merged file. **Version parameter is required** for automatic output organization:

**Separate Files (different file paths):**
```bash
python run_eval.py \
    --model_preds data/sample_model_preds.csv \
    --eval_data data/sample_eval_data.csv \
    --config config/eval_config.yaml \
    --version "baseline_v1"
```

**Merged File (same file path for both parameters):**
```bash
python run_eval.py \
    --model_preds data/sample_unified_data.csv \
    --eval_data data/sample_unified_data.csv \
    --config config/eval_config.yaml \
    --version "p13n_cvr_only"
```

**Mixed Formats with Custom Results Directory:**
```bash
python run_eval.py \
    --model_preds data/large_predictions.parquet \
    --eval_data data/labels.csv \
    --config config/eval_config.yaml \
    --version "p13n_cvr07_price03" \
    --results_dir "experiments"
```

**Multiple Model Evaluation:**
```bash
# Model 1: CVR Only
python run_eval.py \
    --model_preds data/cvr_only_model.parquet \
    --config config/eval_config.yaml \
    --version "p13n_cvr_only"

# Model 2: CVR + Price  
python run_eval.py \
    --model_preds data/cvr_price_model.parquet \
    --config config/eval_config.yaml \
    --version "p13n_cvr07_price03"
```

### 3. Programmatic Usage

```python
from evaluator.evaluator import Evaluator

# Version parameter is required for automatic output organization
evaluator = Evaluator('config/eval_config.yaml', version='baseline_v1')

# Separate files
results = evaluator.run(
    'data/sample_model_preds.csv',
    'data/sample_eval_data.csv'
)

# Unified file
results = evaluator.run(
    'data/sample_unified_data.csv',
    'data/sample_unified_data.csv'  # Same file for both parameters
)

# Results include multiple metric variants
print(results['aggregate'])  # Overall averages
print(results['per_query'])  # Per-query details
print(f"Results saved to: {evaluator.version_dir}")

# Multiple model comparison
models = [
    {'version': 'p13n_cvr_only', 'file': 'data/cvr_model.parquet'},
    {'version': 'p13n_cvr07_price03', 'file': 'data/cvr_price_model.parquet'}
]

all_results = {}
for model in models:
    evaluator = Evaluator('config/eval_config.yaml', version=model['version'])
    results = evaluator.run(model['file'], None)
    all_results[model['version']] = results
    print(f"✅ {model['version']} completed - saved to {evaluator.version_dir}")
```

## 📊 Data Formats

**File Format Support**: The framework automatically supports both CSV and Parquet files based on file extension (`.csv` or `.parquet`). You can mix formats - for example, use Parquet predictions with CSV labels.

### Model Predictions (`sample_model_preds.csv` or `.parquet`)
```csv
query_id,sku,rank,prediction_score
dog_treats_q1,246188,1,0.95
dog_treats_q1,258374,2,0.87
```

### Evaluation Data (`sample_eval_data.csv` or `.parquet`)  
```csv
query_id,sku,engagement,purchase,autoship
dog_treats_q1,246188,3,1,1
dog_treats_q1,258374,2,1,0
```

**Label Types:**
- **`engagement`**: 0=none, 1=click, 2=atc, 3=purchase
- **`purchase`**: 0=no purchase, 1=purchased  
- **`autoship`**: 0=no subscription, 1=autoship subscription

### PDM Data (Optional)
Product catalog with SKU, merchandise categories (MC1/MC2), and pricing for revenue analysis.

### Unified File Support
You can provide a single file (CSV or Parquet) containing both predictions and labels. The evaluator automatically detects when the same file path is used for both parameters:

```csv
query_id,sku,rank,prediction_score,engagement,purchase,autoship
dog_treats_q1,246188,1,0.95,3,1,1
dog_treats_q1,258374,2,0.87,2,1,0
```

**Smart Detection**: When `--model_preds` and `--eval_data` point to the same file, the framework automatically treats it as a merged file and skips the merge step. Example: `data/sample_unified_data.csv` or `data/sample_unified_data.parquet`.

## 🏷️ Version Management

The framework requires a version identifier for each evaluation to organize results and prevent overwrites:

### Version Naming Conventions
- **Model variants**: `p13n_cvr_only`, `p13n_cvr07_price03`, `hybrid_v2`
- **Experiments**: `baseline_v1`, `experiment_2024_11_04`, `ablation_study_a`  
- **A/B tests**: `treatment_group_a`, `control_group_b`

### Automatic Organization
- Each version gets a timestamped directory: `{version}_{YYYYMMDD_HHMMSS}/`
- **No overwrites** - multiple runs of the same version create separate directories
- **Metadata tracking** - automatic logging of evaluation parameters and results
- **Progress tracking** - tqdm progress bars during evaluation

### Version Benefits
```bash
# Each run creates separate results
python run_eval.py --model_preds model_v1.parquet --version "baseline_v1"
# → results/baseline_v1_20241104_143022/

python run_eval.py --model_preds model_v2.parquet --version "improved_v2"  
# → results/improved_v2_20241104_143045/

# Same version, different timestamp (no overwrites)
python run_eval.py --model_preds model_v1_retrain.parquet --version "baseline_v1"
# → results/baseline_v1_20241104_150312/
```

## 🛠️ Command Line Options

```bash
python run_eval.py [OPTIONS]

Required Arguments:
  --model_preds PATH    Model predictions file (CSV/Parquet)
  --version STRING      Version identifier for the evaluation

Optional Arguments:  
  --eval_data PATH      Evaluation data file (optional for merged files)
  --config PATH         Configuration file (default: config/eval_config.yaml)
  --results_dir PATH    Base results directory (default: results)

Examples:
  # Basic usage
  python run_eval.py --model_preds data/model.parquet --version "baseline_v1"
  
  # Custom config and results directory
  python run_eval.py \
    --model_preds data/model.parquet \
    --version "experiment_a" \
    --config configs/custom.yaml \
    --results_dir experiments
    
  # Separate prediction and evaluation files
  python run_eval.py \
    --model_preds data/predictions.parquet \
    --eval_data data/labels.csv \
    --version "hybrid_model_v2"
```

## ⚙️ Available Metrics

### Multi-Label NDCG
- **`ndcg_engagement@k`**: Ranking quality for user engagement (multiple scale, e.g., 0-3)
- **`ndcg_purchase@k`**: Ranking quality for purchase conversions (binary)
- **`ndcg_autoship@k`**: Ranking quality for subscription conversions (binary)

**⚠️ Important NDCG Behavior**: When all labels for a query are zero (no relevant items), NDCG returns `None` (null) because the metric is mathematically undefined. These null values are automatically excluded from:
- Aggregate metric calculations (overall averages)
- MC1/MC2 analysis (merchandise category breakdowns)
- Statistical summaries

This prevents bias from queries with no relevant items and provides more accurate performance metrics.

### CTR/CVR Metrics  
- **`ctr@k`**: Fraction of queries with engagement ≥ click in top k
- **`cvr@k`**: Fraction of queries with purchase in top k

### Revenue Metrics
- **`revenue_per_search@k`**: Average revenue per query from top k results
- **`revenue_recall@k`**: Fraction of total revenue captured in top k
- **`avg_price@k`**: Average price of products in top k

## ⚙️ Configuration

Key settings in `config/eval_config.yaml`:

```yaml
# Schema mapping to match different column names
schema_mapping:
  search_id: query_id
  part_number: sku
  engagement: engagement
  purchase: purchase
  autoship: autoship

# Metrics configuration
metrics:
  - name: ndcg_engagement
    params: { k: [10, 36, 72] }
  - name: ctr  
    params: { k: [5, 10, 20], ctr_label: 1 }
  - name: cvr
    params: { k: [5, 10, 20], cvr_label: 1 }
    
# Output file paths
output:
  aggregate: "results/aggregate_metrics.csv"
  per_query: "results/per_query_metrics.csv"
  mc1_analysis: "results/mc1_analysis.csv"
  mc2_analysis: "results/mc2_analysis.csv"
    
# PDM integration (optional)
pdm:
  file_path: "data/PDM Product_2025-10-01-2037.csv"
  purchase_label: 1
```

## 📈 Results Structure

**Version-Organized Output**: Each evaluation automatically creates a timestamped directory to prevent overwrites:

```
results/
├── baseline_v1_20241104_143022/
│   ├── aggregate.csv              # Overall performance averages
│   ├── per_query.csv             # Detailed per-query results  
│   ├── mc1_analysis.csv          # Performance by MC1 categories
│   ├── mc2_analysis.csv          # Performance by MC2 categories
│   └── evaluation_metadata.csv   # Evaluation details and metadata
├── p13n_cvr_only_20241104_143045/
│   ├── aggregate.csv
│   ├── per_query.csv
│   ├── mc1_analysis.csv
│   ├── mc2_analysis.csv
│   └── evaluation_metadata.csv
└── model_comparison_latest.csv   # Optional comparison summary
```

**Result Files:**
- **`aggregate.csv`**: Overall performance averages across all metrics
- **`per_query.csv`**: Detailed per-query results for all metrics
- **`mc1_analysis.csv`** / **`mc2_analysis.csv`**: Performance by merchandise category
- **`evaluation_metadata.csv`**: Evaluation metadata including version, timestamps, and file paths

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
