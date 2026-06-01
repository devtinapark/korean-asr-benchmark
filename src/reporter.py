"""
Results Reporter
Generates reports and visualizations for benchmark results
"""
from typing import Dict, List
import pandas as pd
from pathlib import Path
import json
import yaml


class BenchmarkReporter:
    """
    Creates comprehensive reports from benchmark results
    """

    def __init__(self, output_dir: str = "results"):
        """
        Initialize reporter

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(
        self,
        ranked_df: pd.DataFrame,
        top_models: List[str],
        metadata: Dict = None
    ) -> str:
        """
        Generate markdown report

        Args:
            ranked_df: DataFrame with ranked models
            top_models: List of top model names
            metadata: Additional metadata to include

        Returns:
            Markdown report string
        """
        md = []

        # Header
        md.append("# Korean ASR Benchmark Results")
        md.append("")
        md.append("## Objective Ranking for Kitchen Use Case")
        md.append("")

        # Metadata
        if metadata:
            md.append("### Configuration")
            md.append("")
            md.append(f"- **Dataset**: {metadata.get('dataset', 'FLEURS Korean')}")
            md.append(f"- **Samples**: {metadata.get('num_samples', 'N/A')}")
            md.append(f"- **Models Tested**: {len(ranked_df)}")
            md.append("")

        # Summary
        md.append("## Summary")
        md.append("")
        md.append("This benchmark systematically evaluates ASR models for Korean language support,")
        md.append("specifically for the 'Spoken Kitchen' use case with casual Korean, kitchen")
        md.append("background noise simulation, and English loanwords (Konglish).")
        md.append("")

        # Top 5 Models
        md.append("## Top 5 Models")
        md.append("")

        for i, model_name in enumerate(top_models[:5], 1):
            row = ranked_df[ranked_df['model'] == model_name].iloc[0]

            md.append(f"### {i}. {model_name}")
            md.append("")
            md.append("| Metric | Value |")
            md.append("|--------|-------|")
            md.append(f"| Composite Score | {row['composite_score']:.4f} |")
            md.append(f"| CER (Character Error Rate) | {row['cer']:.4f} |")
            md.append(f"| WER (Word Error Rate) | {row['wer']:.4f} |")
            md.append(f"| CER/WER Ratio | {row['cer_wer_ratio']:.2f}x |")
            md.append(f"| Loanword Accuracy | {row['loanword_accuracy']:.4f} |")
            md.append(f"| Speed (samples/sec) | {row['samples_per_second']:.2f} |")
            md.append("")

        # Full Rankings
        md.append("## Full Rankings")
        md.append("")
        md.append("| Rank | Model | Score | CER | WER | CER/WER | Loanword | Speed |")
        md.append("|------|-------|-------|-----|-----|---------|----------|-------|")

        for _, row in ranked_df.iterrows():
            md.append(
                f"| {row['rank']} | {row['model']} | {row['composite_score']:.4f} | "
                f"{row['cer']:.4f} | {row['wer']:.4f} | {row['cer_wer_ratio']:.2f}x | "
                f"{row['loanword_accuracy']:.4f} | {row['samples_per_second']:.2f} |"
            )

        md.append("")

        # Metrics Explanation
        md.append("## Metrics Explanation")
        md.append("")
        md.append("### Primary Metrics")
        md.append("")
        md.append("- **CER (Character Error Rate)**: Primary metric for Korean due to agglutinative nature")
        md.append("  - Lower is better (0 = perfect, 1 = complete failure)")
        md.append("  - More granular than WER for Korean text")
        md.append("")
        md.append("- **WER (Word Error Rate)**: Secondary metric")
        md.append("  - Lower is better")
        md.append("  - Word-level accuracy")
        md.append("")
        md.append("- **CER/WER Ratio**: Korean language characteristic")
        md.append("  - Should be >2x for proper Korean handling")
        md.append("  - Higher ratio indicates better handling of agglutinative structure")
        md.append("")
        md.append("- **Loanword Accuracy**: English loanword (Konglish) transcription")
        md.append("  - Critical for kitchen context (e.g., 'oven', 'pasta', 'recipe')")
        md.append("  - Higher is better")
        md.append("")
        md.append("- **Speed**: Inference speed in samples/second")
        md.append("  - Important for real-time applications")
        md.append("  - Higher is better")
        md.append("")

        # Composite Score
        md.append("### Composite Score")
        md.append("")
        md.append("Weighted combination of all metrics:")
        md.append("- CER: 40%")
        md.append("- WER: 20%")
        md.append("- Loanword Accuracy: 20%")
        md.append("- Speed: 10%")
        md.append("- CER/WER Ratio: 10%")
        md.append("")

        return "\n".join(md)

    def save_markdown_report(
        self,
        ranked_df: pd.DataFrame,
        top_models: List[str],
        metadata: Dict = None,
        filename: str = "benchmark_report.md"
    ):
        """
        Save markdown report to file

        Args:
            ranked_df: DataFrame with ranked models
            top_models: List of top model names
            metadata: Additional metadata
            filename: Output filename
        """
        md = self.generate_markdown_report(ranked_df, top_models, metadata)
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            f.write(md)

        print(f"Markdown report saved to {output_path}")

    def save_csv_results(
        self,
        ranked_df: pd.DataFrame,
        filename: str = "results.csv"
    ):
        """
        Save results to CSV

        Args:
            ranked_df: DataFrame with ranked models
            filename: Output filename
        """
        output_path = self.output_dir / filename
        ranked_df.to_csv(output_path, index=False)
        print(f"CSV results saved to {output_path}")

    def save_json_results(
        self,
        ranked_df: pd.DataFrame,
        metadata: Dict = None,
        filename: str = "results.json"
    ):
        """
        Save results to JSON

        Args:
            ranked_df: DataFrame with ranked models
            metadata: Additional metadata
            filename: Output filename
        """
        output_path = self.output_dir / filename

        results = {
            "metadata": metadata or {},
            "rankings": ranked_df.to_dict('records')
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"JSON results saved to {output_path}")

    def save_yaml_results(
        self,
        ranked_df: pd.DataFrame,
        metadata: Dict = None,
        filename: str = "results.yaml"
    ):
        """
        Save results to YAML

        Args:
            ranked_df: DataFrame with ranked models
            metadata: Additional metadata
            filename: Output filename
        """
        output_path = self.output_dir / filename

        results = {
            "metadata": metadata or {},
            "rankings": ranked_df.to_dict('records')
        }

        with open(output_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)

        print(f"YAML results saved to {output_path}")

    def save_top_models_list(
        self,
        top_models: List[str],
        filename: str = "top_5_models.txt"
    ):
        """
        Save list of top models to text file

        Args:
            top_models: List of top model names
            filename: Output filename
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            f.write("Top 5 Korean ASR Models for Kitchen Use Case\n")
            f.write("=" * 50 + "\n\n")

            for i, model in enumerate(top_models[:5], 1):
                f.write(f"{i}. {model}\n")

        print(f"Top models list saved to {output_path}")

    def generate_all_reports(
        self,
        ranked_df: pd.DataFrame,
        top_models: List[str],
        metadata: Dict = None
    ):
        """
        Generate all report formats

        Args:
            ranked_df: DataFrame with ranked models
            top_models: List of top model names
            metadata: Additional metadata
        """
        print("\n" + "="*60)
        print("Generating Reports")
        print("="*60 + "\n")

        self.save_markdown_report(ranked_df, top_models, metadata)
        self.save_csv_results(ranked_df)
        self.save_json_results(ranked_df, metadata)
        self.save_yaml_results(ranked_df, metadata)
        self.save_top_models_list(top_models)

        print("\n" + "="*60)
        print("All reports generated successfully!")
        print(f"Output directory: {self.output_dir}")
        print("="*60 + "\n")


class PredictionExporter:
    """
    Exports detailed predictions for analysis
    """

    def __init__(self, output_dir: str = "results/predictions"):
        """
        Initialize exporter

        Args:
            output_dir: Directory to save predictions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_predictions(
        self,
        model_name: str,
        references: List[str],
        predictions: List[str],
        sample_ids: List[str] = None
    ):
        """
        Export predictions to file

        Args:
            model_name: Name of the model
            references: Reference transcriptions
            predictions: Model predictions
            sample_ids: Sample IDs (optional)
        """
        if sample_ids is None:
            sample_ids = [str(i) for i in range(len(references))]

        # Create DataFrame
        df = pd.DataFrame({
            'sample_id': sample_ids,
            'reference': references,
            'prediction': predictions
        })

        # Save to CSV
        filename = f"{model_name.replace('/', '_')}_predictions.csv"
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)

        print(f"Predictions saved to {output_path}")
