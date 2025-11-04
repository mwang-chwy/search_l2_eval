import pandas as pd
import numpy as np
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from utils.io import load_data, save_csv, load_config
from utils.validation import rename_columns, validate_schema
from utils.pdm_utils import load_pdm_data, enrich_with_pdm, assign_search_mc, calculate_revenue
# Import metrics package to register all metrics
import metrics
from metrics.registry import METRIC_REGISTRY

class Evaluator:
    def __init__(self, config_path, version=None, results_base_dir="results"):
        """
        Initialize the evaluator with version-specific output handling
        
        Args:
            config_path: Path to configuration file
            version: Version identifier (required for automatic output versioning)
            results_base_dir: Base directory for results (default: "results")
        """
        if version is None:
            raise ValueError("❌ Version parameter is required! Please specify a version name (e.g., 'p13n_cvr_only')")
        
        self.config = load_config(config_path)
        self.version = version
        self.results_base_dir = results_base_dir
        
        # Create version-specific directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.version_dir = os.path.join(results_base_dir, f"{version}_{timestamp}")
        
        print(f"🚀 Initializing evaluator for version: {version}")
        print(f"📁 Results will be saved to: {self.version_dir}")
        
        # Create version directory
        os.makedirs(self.version_dir, exist_ok=True)
        
        # Update output paths in config to use version-specific directory
        if 'output' in self.config:
            for output_type, original_path in self.config['output'].items():
                filename = os.path.basename(original_path)
                new_path = os.path.join(self.version_dir, filename)
                self.config['output'][output_type] = new_path
        
        self.metrics = self.config["metrics"]
        self.group_by = self.config.get("group_by", "search_id")
        self.schema_mapping = self.config.get("schema_mapping", {})
        self.output_paths = self.config.get("output", {})
        self.pdm_config = self.config.get("pdm", {})
        self.analyze_by_mc = self.config.get("analyze_by_mc", False)
        
        # Initialize filtered metrics (will be set during data preparation)
        self.filtered_metrics = self.metrics
        
        # Load PDM data if configured
        self.pdm_df = None
        if self.pdm_config.get("file_path"):
            self.pdm_df = load_pdm_data(
                self.pdm_config["file_path"], 
                self.pdm_config
            )

    def _get_required_fields(self):
        """Get all required fields based on enabled metrics"""
        required = set()
        for metric_config in self.metrics:
            required.update(metric_config.get("required_fields", []))
        return list(required)

    def _check_data_availability(self, df):
        """Check which label columns have valid (non-null, non-zero) data"""
        availability = {}
        label_columns = ['engagement', 'purchase', 'autoship']
        
        for col in label_columns:
            if col in df.columns:
                # Check if column has any non-null, non-zero values
                valid_data = df[col].notna() & (df[col] != 0)
                availability[col] = {
                    'exists': True,
                    'has_data': valid_data.any(),
                    'count': valid_data.sum(),
                    'total': len(df)
                }
            else:
                availability[col] = {
                    'exists': False,
                    'has_data': False,
                    'count': 0,
                    'total': len(df)
                }
        
        return availability

    def _filter_metrics_by_availability(self, data_availability):
        """Filter metrics based on data availability"""
        filtered_metrics = []
        skipped_metrics = []
        
        for metric_config in self.metrics:
            metric_name = metric_config['name']
            required_fields = metric_config.get('required_fields', [])
            
            # Check if metric can be calculated with available data
            can_calculate = True
            missing_reason = ""
            
            # Check label column requirements
            if 'engagement' in required_fields and not data_availability['engagement']['has_data']:
                can_calculate = False
                missing_reason = "engagement column missing or has no valid data"
            elif 'purchase' in required_fields and not data_availability['purchase']['has_data']:
                can_calculate = False
                missing_reason = "purchase column missing or has no valid data"
            elif 'autoship' in required_fields and not data_availability['autoship']['has_data']:
                can_calculate = False
                missing_reason = "autoship column missing or has no valid data"
            elif 'revenue' in required_fields and not data_availability['purchase']['has_data']:
                can_calculate = False
                missing_reason = "revenue calculation requires purchase column data"
            
            if can_calculate:
                filtered_metrics.append(metric_config)
            else:
                skipped_metrics.append({
                    'name': metric_name,
                    'reason': missing_reason
                })
        
        return filtered_metrics, skipped_metrics

    def _prepare_data(self, model_output, eval_dataset):
        # Check if this is a merged file case:
        # 1. eval_dataset is None (not provided)
        # 2. same file path for both parameters
        is_merged_file = (eval_dataset is None) or (model_output == eval_dataset)
        
        if is_merged_file:
            # Load single merged file
            if eval_dataset is None:
                print(f"Using merged file (eval_data not provided): {model_output}")
            else:
                print(f"Using merged file (same file path): {model_output}")
            df = load_data(model_output)
            
            # Verify it contains at least some prediction columns
            pred_columns = ['rank', 'prediction_score']  # Common prediction columns
            has_preds = any(col in df.columns for col in pred_columns)
            
            if not has_preds:
                raise ValueError(f"Merged file must contain at least one prediction column: {pred_columns}")
            
            print(f"Merged file contains {df.shape[0]} rows with both predictions and labels")
            
        else:
            # Load separate files and merge them
            print(f"Loading separate files - Predictions: {model_output}, Labels: {eval_dataset}")
            preds = load_data(model_output)
            labels = load_data(eval_dataset)
            
            merge_cols = [
                self.schema_mapping.get("search_id", "query_id"),
                self.schema_mapping.get("part_number", "sku")
            ]
            
            # Ensure consistent data types for merge columns to prevent type mismatch errors
            for col in merge_cols:
                if col in preds.columns:
                    preds[col] = preds[col].astype('string')
                if col in labels.columns:
                    labels[col] = labels[col].astype('string')
            
            df = preds.merge(labels, how='left', on=merge_cols, suffixes=("_pred", ""))
            print(f"Merged {preds.shape[0]} predictions with {labels.shape[0]} labels, resulting in {df.shape[0]} rows")
        
        df = rename_columns(df, self.schema_mapping)
        
        # Handle missing labels for the three label types (fill with 0 for unlabeled items)
        label_columns = ['engagement', 'purchase', 'autoship']
        for col in label_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                
        # Check data availability first
        data_availability = self._check_data_availability(df)
        
        # Enrich with PDM data if available
        if self.pdm_df is not None:
            print("Enriching data with PDM information...")
            df = enrich_with_pdm(df, self.pdm_df, self.pdm_config)
            df = assign_search_mc(df)
            
            # Calculate revenue only if purchase data is available
            if data_availability['purchase']['has_data']:
                purchase_label = self.pdm_config.get("purchase_label", 1)
                df = calculate_revenue(df, purchase_label)
                print(f"Data enriched. Added MC categories and calculated revenue.")
            else:
                print(f"Data enriched. Added MC categories (revenue calculation skipped - no purchase data).")
        print("\n=== Data Availability Report ===")
        for col, info in data_availability.items():
            if info['exists']:
                if info['has_data']:
                    print(f"✅ {col}: {info['count']}/{info['total']} rows have valid data")
                else:
                    print(f"⚠️  {col}: Column exists but all values are 0 or null")
            else:
                print(f"❌ {col}: Column not found")
        
        # Filter metrics based on data availability
        self.filtered_metrics, skipped_metrics = self._filter_metrics_by_availability(data_availability)
        
        if skipped_metrics:
            print(f"\n⚠️  Skipped {len(skipped_metrics)} metrics due to missing data:")
            for skipped in skipped_metrics:
                print(f"   - {skipped['name']}: {skipped['reason']}")
        
        print(f"✅ Will calculate {len(self.filtered_metrics)} metrics with available data")
        
        # Get required fields only from filtered metrics
        required_fields = set()
        for metric_config in self.filtered_metrics:
            required_fields.update(metric_config.get("required_fields", []))
        required_fields = list(required_fields)
        
        validate_schema(df, required_fields)
        return df
    
    def _compute_metrics_for_group(self, group_df):
        results = {
            "search_id": group_df['search_id'].iloc[0]
        }
        
        # Add search_term if available (optional field)
        if 'search_term' in group_df.columns:
            results["search_term"] = group_df['search_term'].iloc[0]
        
        # Add MC information if available
        if 'search_mc1' in group_df.columns:
            results["search_mc1"] = group_df['search_mc1'].iloc[0]
        if 'search_mc2' in group_df.columns:
            results["search_mc2"] = group_df['search_mc2'].iloc[0]
        
        for mconf in self.filtered_metrics:
            name = mconf["name"]
            if name not in METRIC_REGISTRY:
                continue
                
            fn = METRIC_REGISTRY[name]
            params = mconf.get("params", {})
            
            # Handle multiple k values
            k_values = params.get("k", [])
            if isinstance(k_values, list):
                for k in k_values:
                    metric_params = params.copy()
                    metric_params["k"] = k
                    try:
                        results[f"{name}@{k}"] = fn(group_df, **metric_params)
                    except Exception as e:
                        print(f"Error calculating {name}@{k}: {e}")
                        results[f"{name}@{k}"] = None
            else:
                try:
                    results[name] = fn(group_df, **params)
                except Exception as e:
                    print(f"Error calculating {name}: {e}")
                    results[name] = None
        return results

    def _analyze_by_mc_level(self, results_df, mc_col: str) -> pd.DataFrame:
        """Analyze metrics by merchandise classification level"""
        if mc_col not in results_df.columns:
            return pd.DataFrame()
        
        # Filter out unknown MCs
        mc_df = results_df[results_df[mc_col] != 'UNKNOWN'].copy()
        if mc_df.empty:
            return pd.DataFrame()
        
        # Exclude non-metric columns
        cols_to_exclude = ['search_id', 'search_term', 'search_mc1', 'search_mc2']
        metric_cols = [col for col in mc_df.columns if col not in cols_to_exclude]
        
        mc_metrics_df = mc_df[metric_cols + [mc_col]]
        
        # Calculate averages by MC category (automatically excludes None values from NDCG)
        avg_metrics_mc = mc_metrics_df.groupby(mc_col).agg({
            col: 'mean' for col in metric_cols if col != mc_col  # mean() excludes None/NaN
        }).reset_index()
        
        return avg_metrics_mc

    def run(self, model_output, eval_dataset):
        """Main evaluation method"""
        df = self._prepare_data(model_output, eval_dataset)
        
        # Get unique search IDs for progress tracking
        unique_search_ids = df['search_id'].unique()
        total_queries = len(unique_search_ids)
        
        print(f"\n🔄 Processing {total_queries} search queries...")
        
        # Compute per-query metrics with progress bar
        per_query_results = []
        
        # Create progress bar with descriptive format
        with tqdm(
            df.groupby('search_id'), 
            total=total_queries,
            desc="Computing metrics",
            unit="query",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} queries [{elapsed}<{remaining}]"
        ) as pbar:
            for search_id, group_df in pbar:
                # Update progress bar description with current query
                pbar.set_postfix({"Current": str(search_id)[:20]})  # Truncate long query IDs
                
                group_results = self._compute_metrics_for_group(group_df)
                per_query_results.append(group_results)
        
        print(f"✅ Completed processing {total_queries} queries")
        
        results_df = pd.DataFrame(per_query_results)
        
        # Calculate aggregate metrics (means, excluding NaN/None values)
        # Note: NDCG metrics return None for queries with all-zero labels, these are excluded from aggregation
        aggregate_results = {}
        for col in results_df.columns:
            if col not in ['search_id', 'search_term', 'search_mc1', 'search_mc2'] and pd.api.types.is_numeric_dtype(results_df[col]):
                aggregate_results[col] = results_df[col].mean()  # Automatically excludes None values
        
        evaluation_results = {
            'per_query': results_df,
            'aggregate': pd.DataFrame([aggregate_results])
        }
        
        # MC-level analysis if enabled
        if self.analyze_by_mc and self.pdm_df is not None:
            mc1_analysis = self._analyze_by_mc_level(results_df, 'search_mc1')
            mc2_analysis = self._analyze_by_mc_level(results_df, 'search_mc2')
            
            evaluation_results['mc1_analysis'] = mc1_analysis
            evaluation_results['mc2_analysis'] = mc2_analysis
        
        # Save results with model information for metadata
        self.save_results(evaluation_results, model_output, eval_dataset)
        
        return evaluation_results

    def save_results(self, results, model_output=None, eval_dataset=None):
        """Save evaluation results to configured output paths with version metadata"""
        
        # Save standard results
        if 'aggregate' in self.output_paths:
            save_csv(results['aggregate'], self.output_paths['aggregate'])
            print(f"✅ Aggregate results saved: {self.output_paths['aggregate']}")
            
        if 'per_query' in self.output_paths:
            save_csv(results['per_query'], self.output_paths['per_query'])
            print(f"✅ Per-query results saved: {self.output_paths['per_query']}")
            
        if 'mc1_analysis' in results and 'mc1_analysis' in self.output_paths:
            save_csv(results['mc1_analysis'], self.output_paths['mc1_analysis'])
            print(f"✅ MC1 analysis saved: {self.output_paths['mc1_analysis']}")
            
        if 'mc2_analysis' in results and 'mc2_analysis' in self.output_paths:
            save_csv(results['mc2_analysis'], self.output_paths['mc2_analysis'])
            print(f"✅ MC2 analysis saved: {self.output_paths['mc2_analysis']}")
        
        # Save version metadata
        metadata = {
            'version': self.version,
            'timestamp': datetime.now().isoformat(),
            'model_output': model_output,
            'eval_dataset': eval_dataset,
            'results_directory': self.version_dir,
            'total_metrics': results['aggregate'].shape[1] if 'aggregate' in results else 0,
            'total_queries': results['per_query'].shape[0] if 'per_query' in results else 0,
            'has_mc1_analysis': 'mc1_analysis' in results,
            'has_mc2_analysis': 'mc2_analysis' in results
        }
        
        metadata_df = pd.DataFrame([metadata])
        metadata_path = os.path.join(self.version_dir, 'evaluation_metadata.csv')
        metadata_df.to_csv(metadata_path, index=False)
        print(f"📋 Metadata saved: {metadata_path}")
        
        print(f"\n🎉 All results for version '{self.version}' saved to: {self.version_dir}")
