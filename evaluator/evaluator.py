import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from utils.io import load_csv, save_csv, load_config
from utils.validation import rename_columns, validate_schema
from utils.pdm_utils import load_pdm_data, enrich_with_pdm, assign_search_mc, calculate_revenue
# Import metrics package to register all metrics
import metrics
from metrics.registry import METRIC_REGISTRY

class Evaluator:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.metrics = self.config["metrics"]
        self.group_by = self.config.get("group_by", "search_id")
        self.schema_mapping = self.config.get("schema_mapping", {})
        self.output_paths = self.config.get("output", {})
        self.pdm_config = self.config.get("pdm", {})
        self.analyze_by_mc = self.config.get("analyze_by_mc", False)
        
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

    def _prepare_data(self, model_output, eval_dataset):
        preds = load_csv(model_output)
        labels = load_csv(eval_dataset)
        
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
        df = rename_columns(df, self.schema_mapping)
        
        # Handle missing relevance labels (fill with 0 for unlabeled items)
        relevance_col = 'relevance' if 'relevance' in df.columns else 'label'
        if relevance_col in df.columns:
            df[relevance_col] = df[relevance_col].fillna(0)
        
        # Enrich with PDM data if available
        if self.pdm_df is not None:
            print("Enriching data with PDM information...")
            df = enrich_with_pdm(df, self.pdm_df, self.pdm_config)
            df = assign_search_mc(df)
            
            # Calculate revenue based on purchase labels
            purchase_label = self.pdm_config.get("purchase_label", 1)
            df = calculate_revenue(df, purchase_label)
            
            print(f"Data enriched. Added MC categories and calculated revenue.")
                
        required_fields = self._get_required_fields()
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
        
        for mconf in self.metrics:
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
        
        # Calculate averages by MC category
        avg_metrics_mc = mc_metrics_df.groupby(mc_col).agg({
            col: 'mean' for col in metric_cols if col != mc_col
        }).reset_index()
        
        return avg_metrics_mc

    def run(self, model_output, eval_dataset):
        """Main evaluation method"""
        df = self._prepare_data(model_output, eval_dataset)
        
        # Compute per-query metrics
        per_query_results = []
        for search_id, group_df in df.groupby('search_id'):
            group_results = self._compute_metrics_for_group(group_df)
            per_query_results.append(group_results)
        
        results_df = pd.DataFrame(per_query_results)
        
        # Calculate aggregate metrics (means, excluding NaN)
        aggregate_results = {}
        for col in results_df.columns:
            if col not in ['search_id', 'search_term', 'search_mc1', 'search_mc2'] and pd.api.types.is_numeric_dtype(results_df[col]):
                aggregate_results[col] = results_df[col].mean()
        
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
        
        # Save results
        self.save_results(evaluation_results)
        
        return evaluation_results

    def save_results(self, results):
        """Save evaluation results to configured output paths"""
        if 'aggregate' in self.output_paths:
            save_csv(results['aggregate'], self.output_paths['aggregate'])
            
        if 'per_query' in self.output_paths:
            save_csv(results['per_query'], self.output_paths['per_query'])
            
        if 'mc1_analysis' in results and 'mc1_analysis' in self.output_paths:
            save_csv(results['mc1_analysis'], self.output_paths['mc1_analysis'])
            
        if 'mc2_analysis' in results and 'mc2_analysis' in self.output_paths:
            save_csv(results['mc2_analysis'], self.output_paths['mc2_analysis'])
