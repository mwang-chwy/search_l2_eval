import argparse
from evaluator.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description='Search L2 Evaluation Framework')
    parser.add_argument("--model_preds", required=True, 
                       help="Model predictions file (for merged files, this contains both predictions and labels)")
    parser.add_argument("--eval_data", required=False, default=None,
                       help="Evaluation data file (optional for merged files)")
    parser.add_argument("--config", default="config/eval_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--version", required=True,
                       help="Version identifier for the evaluation run (e.g., 'p13n_cvr_only', 'baseline_v1')")
    parser.add_argument("--results_dir", default="results",
                       help="Base directory for results (default: 'results')")
    args = parser.parse_args()

    print(f"🚀 Starting evaluation for version: {args.version}")
    print(f"📂 Model predictions: {args.model_preds}")
    print(f"📂 Evaluation data: {args.eval_data or 'Merged file'}")
    print(f"⚙️ Config file: {args.config}")
    
    evaluator = Evaluator(config_path=args.config, version=args.version, results_base_dir=args.results_dir)
    results = evaluator.run(args.model_preds, args.eval_data)

    print("\n🎉 Evaluation completed successfully!")
    print(f"📁 Results saved to: {evaluator.version_dir}")
    
    print("\n📊 === Aggregate Metrics Summary ===")
    if "aggregate" in results and not results["aggregate"].empty:
        print(results["aggregate"].head())
    else:
        print("No aggregate metrics available")
    
    print(f"\n📋 Detailed results available in:")
    for output_type, path in evaluator.output_paths.items():
        print(f"   {output_type}: {path}")
    
    print(f"\n📄 Evaluation metadata: {evaluator.version_dir}/evaluation_metadata.csv")

if __name__ == "__main__":
    main()
