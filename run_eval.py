import argparse
from evaluator.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_preds", required=True)
    parser.add_argument("--eval_data", required=True)
    parser.add_argument("--config", default="config/eval_config.yaml")
    args = parser.parse_args()

    evaluator = Evaluator(config_path=args.config)
    results = evaluator.run(args.model_preds, args.eval_data)

    print("\n=== Aggregate Metrics ===")
    print(results["aggregate"].head())
    print("\nSaved detailed results to:", evaluator.output_paths)

if __name__ == "__main__":
    main()
