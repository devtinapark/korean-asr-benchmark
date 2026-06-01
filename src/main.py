"""
Korean ASR Benchmark - Main Entry Point
Systematically evaluates and ranks ASR models for Korean language
"""
import argparse
import sys
from pathlib import Path

from .benchmark import ASRBenchmark
from .ranking import ModelRanker, RankingCriteria
from .reporter import BenchmarkReporter


def run_full_benchmark(
    config_path: str = "config.yaml",
    top_n: int = 5,
    output_dir: str = "results"
):
    """
    Run full benchmark pipeline

    Args:
        config_path: Path to configuration file
        top_n: Number of top models to select
        output_dir: Output directory for results
    """
    print("\n" + "="*80)
    print("KOREAN ASR BENCHMARK - Spoken Kitchen Use Case".center(80))
    print("="*80 + "\n")

    # Initialize benchmark
    benchmark = ASRBenchmark(config_path)

    # Setup data
    benchmark.setup_data()

    # Run benchmark on all models
    results = benchmark.run_all_models()

    # Convert results to dict format
    results_dict = {name: result.to_dict() for name, result in results.items()}

    # Rank models
    print("\n" + "="*80)
    print("Ranking Models")
    print("="*80 + "\n")

    ranker = ModelRanker()
    ranked_df = ranker.rank_models(results_dict)

    # Get top N
    top_df, top_models = ranker.get_top_n(results_dict, n=top_n)

    # Print rankings
    ranker.print_ranking_table(ranked_df)
    ranker.print_top_n_summary(top_df, top_models, n=top_n)

    # Generate reports
    reporter = BenchmarkReporter(output_dir)

    metadata = {
        "dataset": "FLEURS Korean (ko_kr)",
        "num_samples": len(benchmark.data_loader) if benchmark.data_loader else 0,
        "config_file": config_path
    }

    reporter.generate_all_reports(ranked_df, top_models, metadata)

    # Save detailed predictions for top models
    if benchmark.config.get('output', {}).get('save_predictions', False):
        from .reporter import PredictionExporter
        exporter = PredictionExporter()

        for model_name in top_models[:top_n]:
            result = results[model_name]
            if result.predictions and result.references:
                exporter.export_predictions(
                    model_name,
                    result.references,
                    result.predictions
                )

    print("\n" + "="*80)
    print("Benchmark Complete!".center(80))
    print("="*80)
    print(f"\nTop {top_n} models saved to: {output_dir}/top_5_models.txt")
    print(f"Full report: {output_dir}/benchmark_report.md")
    print(f"Results: {output_dir}/results.csv\n")


def run_single_model(
    model_name: str,
    config_path: str = "config.yaml",
    output_dir: str = "results"
):
    """
    Run benchmark on a single model

    Args:
        model_name: Name of model from config
        config_path: Path to configuration file
        output_dir: Output directory for results
    """
    print(f"\n{'='*80}")
    print(f"Running single model benchmark: {model_name}".center(80))
    print("="*80 + "\n")

    benchmark = ASRBenchmark(config_path)
    benchmark.setup_data()

    result = benchmark.run_single_model(model_name)

    print("\n" + "="*80)
    print("Single Model Benchmark Complete")
    print("="*80 + "\n")


def list_models(config_path: str = "config.yaml"):
    """
    List all available models in config

    Args:
        config_path: Path to configuration file
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    models = config.get('models', {})

    print("\n" + "="*80)
    print("Available Models".center(80))
    print("="*80 + "\n")

    for i, (model_name, model_config) in enumerate(models.items(), 1):
        print(f"{i}. {model_name}")
        print(f"   HuggingFace ID: {model_config['name']}")
        if 'language' in model_config:
            print(f"   Language: {model_config['language']}")
        print()

    print(f"Total: {len(models)} models\n")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Korean ASR Benchmark - Systematic evaluation of ASR models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark on all models
  python -m src.main

  # Run benchmark with custom config
  python -m src.main --config my_config.yaml

  # Select top 10 models instead of 5
  python -m src.main --top-n 10

  # Run single model
  python -m src.main --model whisper-large-v3

  # List available models
  python -m src.main --list-models
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Run benchmark on a single model'
    )

    parser.add_argument(
        '--top-n',
        type=int,
        default=5,
        help='Number of top models to select (default: 5)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models in config'
    )

    args = parser.parse_args()

    try:
        # Check if config file exists
        if not Path(args.config).exists():
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)

        # Handle commands
        if args.list_models:
            list_models(args.config)

        elif args.model:
            run_single_model(
                args.model,
                args.config,
                args.output_dir
            )

        else:
            run_full_benchmark(
                args.config,
                args.top_n,
                args.output_dir
            )

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(0)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
