"""
Generate Figures 2 and 3 from Evaluation Results

This script reads the evaluation results and creates bar charts
matching Figures 2 and 3 from the paper.

Usage:
    # Using separate result files (recommended):
    python plot_results.py \
        --baseline_results ./results_baseline.json \
        --ih_results ./results_ih.json \
        --output_dir ./figures

    # Using combined result file (legacy):
    python plot_results.py \
        --results_file ./results.json \
        --output_dir ./figures
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict


# ====================
# Plotting Functions
# ====================

def plot_figure_1(baseline_results: Dict, ih_results: Dict, output_path: str):
    """Generate Figure 1: Main Results"""

    # Test names for Figure 2
    tests = [
        'prompt_injection_hijacking',
        'prompt_injection_new',
        'user_conflicting',
        'indirect_browsing',
        'system_extraction'
    ]

    # Display names (shortened for plot)
    display_names = [
        'Prompt Injection\n(Hijacking)',
        'Prompt Injection\n(New Instructions)',
        'User Conflicting\nInstructions',
        'Prompt Injection\n(Indirect via Browsing)',
        'System Message\nExtraction'
    ]

    # Extract robustness values
    baseline_values = []
    ih_values = []

    for test in tests:
        if test in baseline_results:
            baseline_values.append(baseline_results[test]['robustness'])
        else:
            baseline_values.append(0)

        if test in ih_results:
            ih_values.append(ih_results[test]['robustness'])
        else:
            ih_values.append(0)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Set up bar positions
    x = np.arange(len(tests))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width / 2, baseline_values, width,
                   label='Baseline LM', color='#E57373', alpha=0.8)
    bars2 = ax.bar(x + width / 2, ih_values, width,
                   label='+ Instruction Hierarchy', color='#64B5F6', alpha=0.8)

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Customize plot
    ax.set_ylabel('Robustness (%)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 2: Main Results', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add horizontal line at 50%
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 2 to: {output_path}")
    plt.close()


def plot_figure_3(baseline_results: Dict, ih_results: Dict, output_path: str):
    """Generate Figure 3: Generalization Results"""

    # Test names for Figure 3
    tests = [
        'indirect_tools',
        'tensortrust_password',
        'gandalf_password',
        'jailbreak_unsafe',
        'chatgpt_jailbreaks'
    ]

    # Display names
    display_names = [
        'Prompt Injection\n(Indirect via Tools)',
        'Tensortrust\nPassword Extraction',
        'Gandalf Game\nPassword Extraction',
        'Jailbreakchat\nw/Unsafe Prompts',
        'ChatGPT Jailbreaks\nw/Unsafe Prompts'
    ]

    # Extract robustness values
    baseline_values = []
    ih_values = []

    for test in tests:
        if test in baseline_results:
            baseline_values.append(baseline_results[test]['robustness'])
        else:
            baseline_values.append(0)

        if test in ih_results:
            ih_values.append(ih_results[test]['robustness'])
        else:
            ih_values.append(0)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Set up bar positions
    x = np.arange(len(tests))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width / 2, baseline_values, width,
                   label='Baseline LM', color='#E57373', alpha=0.8)
    bars2 = ax.bar(x + width / 2, ih_values, width,
                   label='+ Instruction Hierarchy', color='#64B5F6', alpha=0.8)

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Customize plot
    ax.set_ylabel('Robustness (%)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 3: Generalization Results', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=10)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add horizontal line at 50%
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 3 to: {output_path}")
    plt.close()


def plot_combined_figure(baseline_results: Dict, ih_results: Dict, output_path: str):
    """Generate combined figure with both Figure 2 and 3"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Figure 2 data
    tests_fig2 = [
        'prompt_injection_hijacking',
        'prompt_injection_new',
        'user_conflicting',
        'indirect_browsing',
        'system_extraction'
    ]

    display_names_fig2 = [
        'Prompt Injection\n(Hijacking)',
        'Prompt Injection\n(New Instructions)',
        'User Conflicting\nInstructions',
        'Indirect\nBrowsing',
        'System Message\nExtraction'
    ]

    # Figure 3 data
    tests_fig3 = [
        'indirect_tools',
        'tensortrust_password',
        'gandalf_password',
        'jailbreak_unsafe',
        'chatgpt_jailbreaks'
    ]

    display_names_fig3 = [
        'Indirect\nTools',
        'Tensortrust\nPassword',
        'Gandalf\nPassword',
        'Jailbreakchat\nUnsafe',
        'ChatGPT\nJailbreaks'
    ]

    # Plot Figure 2
    baseline_values_fig2 = [baseline_results.get(t, {}).get('robustness', 0) for t in tests_fig2]
    ih_values_fig2 = [ih_results.get(t, {}).get('robustness', 0) for t in tests_fig2]

    x1 = np.arange(len(tests_fig2))
    width = 0.35

    bars1 = ax1.bar(x1 - width / 2, baseline_values_fig2, width,
                    label='Baseline LM', color='#E57373', alpha=0.8)
    bars2 = ax1.bar(x1 + width / 2, ih_values_fig2, width,
                    label='+ Instruction Hierarchy', color='#64B5F6', alpha=0.8)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylabel('Robustness (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Figure 2: Main Results', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(display_names_fig2, fontsize=9)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    # Plot Figure 3
    baseline_values_fig3 = [baseline_results.get(t, {}).get('robustness', 0) for t in tests_fig3]
    ih_values_fig3 = [ih_results.get(t, {}).get('robustness', 0) for t in tests_fig3]

    x2 = np.arange(len(tests_fig3))

    bars3 = ax2.bar(x2 - width / 2, baseline_values_fig3, width,
                    label='Baseline LM', color='#E57373', alpha=0.8)
    bars4 = ax2.bar(x2 + width / 2, ih_values_fig3, width,
                    label='+ Instruction Hierarchy', color='#64B5F6', alpha=0.8)

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_ylabel('Robustness (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Figure 3: Generalization Results', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(display_names_fig3, fontsize=9)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined figure to: {output_path}")
    plt.close()


def print_results_table(baseline_results: Dict, ih_results: Dict):
    """Print results in table format"""

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print("\nFigure 2: Main Results")
    print("-" * 80)
    print(f"{'Test Name':<40} {'Baseline':>10} {'+ IH':>10} {'Improvement':>15}")
    print("-" * 80)

    tests_fig2 = [
        ('Prompt Injection (Hijacking)', 'prompt_injection_hijacking'),
        ('Prompt Injection (New Instructions)', 'prompt_injection_new'),
        ('User Conflicting Instructions', 'user_conflicting'),
        ('Indirect Browsing', 'indirect_browsing'),
        ('System Message Extraction', 'system_extraction'),
    ]

    for display_name, test_key in tests_fig2:
        baseline_val = baseline_results.get(test_key, {}).get('robustness', 0)
        ih_val = ih_results.get(test_key, {}).get('robustness', 0)
        improvement = ih_val - baseline_val

        print(f"{display_name:<40} {baseline_val:>9.1f}% {ih_val:>9.1f}% {improvement:>+14.1f}%")

    print("\n" + "=" * 80)
    print("\nFigure 3: Generalization Results")
    print("-" * 80)
    print(f"{'Test Name':<40} {'Baseline':>10} {'+ IH':>10} {'Improvement':>15}")
    print("-" * 80)

    tests_fig3 = [
        ('Indirect Tools', 'indirect_tools'),
        ('TensorTrust Password Extraction', 'tensortrust_password'),
        ('Gandalf Password Extraction', 'gandalf_password'),
        ('Jailbreakchat w/Unsafe Prompts', 'jailbreak_unsafe'),
        ('ChatGPT Jailbreaks w/Unsafe Prompts', 'chatgpt_jailbreaks'),
    ]

    for display_name, test_key in tests_fig3:
        baseline_val = baseline_results.get(test_key, {}).get('robustness', 0)
        ih_val = ih_results.get(test_key, {}).get('robustness', 0)
        improvement = ih_val - baseline_val

        print(f"{display_name:<40} {baseline_val:>9.1f}% {ih_val:>9.1f}% {improvement:>+14.1f}%")

    print("=" * 80)

    # Calculate averages
    all_baseline = []
    all_ih = []

    for _, test_key in tests_fig2 + tests_fig3:
        baseline_val = baseline_results.get(test_key, {}).get('robustness', 0)
        ih_val = ih_results.get(test_key, {}).get('robustness', 0)
        all_baseline.append(baseline_val)
        all_ih.append(ih_val)

    avg_baseline = np.mean(all_baseline)
    avg_ih = np.mean(all_ih)
    avg_improvement = avg_ih - avg_baseline

    print(f"\n{'AVERAGE':<40} {avg_baseline:>9.1f}% {avg_ih:>9.1f}% {avg_improvement:>+14.1f}%")
    print("=" * 80)


# ====================
# Main Function
# ====================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Figures 2 and 3 from evaluation results"
    )

    # Option 1: Separate files (new recommended way)
    parser.add_argument(
        "--baseline_results",
        default=None,
        help="Path to baseline results JSON file"
    )
    parser.add_argument(
        "--ih_results",
        default=None,
        help="Path to instruction hierarchy results JSON file"
    )

    # Option 2: Combined file (legacy support)
    parser.add_argument(
        "--results_file",
        default=None,
        help="Path to combined evaluation results JSON file (legacy)"
    )

    parser.add_argument(
        "--output_dir",
        default="./figures",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--format",
        choices=['png', 'pdf', 'svg'],
        default='png',
        help="Output format"
    )

    args = parser.parse_args()

    # Determine which mode to use
    if args.baseline_results and args.ih_results:
        # Mode 1: Separate files
        print("\n" + "=" * 80)
        print("GENERATING FIGURES 2 & 3")
        print("=" * 80)
        print(f"Baseline results: {args.baseline_results}")
        print(f"IH results: {args.ih_results}")
        print(f"Output directory: {args.output_dir}")
        print("=" * 80)

        # Load baseline results
        if not os.path.exists(args.baseline_results):
            raise FileNotFoundError(f"Baseline results file not found: {args.baseline_results}")

        with open(args.baseline_results, 'r') as f:
            baseline_data = json.load(f)

        # Extract baseline results (handle both formats)
        if 'baseline' in baseline_data:
            baseline_results = baseline_data['baseline']
        else:
            baseline_results = baseline_data

        # Load instruction hierarchy results
        if not os.path.exists(args.ih_results):
            raise FileNotFoundError(f"IH results file not found: {args.ih_results}")

        with open(args.ih_results, 'r') as f:
            ih_data = json.load(f)

        # Extract IH results (handle both formats)
        if 'instruction_hierarchy' in ih_data:
            ih_results = ih_data['instruction_hierarchy']
        else:
            ih_results = ih_data

    elif args.results_file:
        # Mode 2: Combined file (legacy)
        print("\n" + "=" * 80)
        print("GENERATING FIGURES 2 & 3")
        print("=" * 80)
        print(f"Combined results file: {args.results_file}")
        print(f"Output directory: {args.output_dir}")
        print("=" * 80)

        if not os.path.exists(args.results_file):
            raise FileNotFoundError(
                f"Results file not found: {args.results_file}\n"
                f"Please run: python evaluate_models.py"
            )

        with open(args.results_file, 'r') as f:
            results = json.load(f)

        # Extract baseline and instruction hierarchy results
        baseline_results = results.get('baseline', {})
        ih_results = results.get('instruction_hierarchy', {})

        if not baseline_results or not ih_results:
            raise ValueError(
                "Results file must contain both 'baseline' and 'instruction_hierarchy' results.\n"
                f"Found: {list(results.keys())}"
            )
    else:
        # No valid input provided
        parser.error(
            "Must provide either:\n"
            "  1. Both --baseline_results and --ih_results, OR\n"
            "  2. --results_file (combined results)\n\n"
            "Examples:\n"
            "  python plot_results.py --baseline_results results_baseline.json --ih_results results_ih.json\n"
            "  python plot_results.py --results_file results.json"
        )

    # Validate we have the required data
    if not baseline_results:
        raise ValueError("No baseline results found in input file(s)")
    if not ih_results:
        raise ValueError("No instruction hierarchy results found in input file(s)")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Print results table
    print_results_table(baseline_results, ih_results)

    # Generate figures
    print("\n" + "=" * 80)
    print("Generating figures...")
    print("=" * 80)

    # Individual figures
    fig2_path = os.path.join(args.output_dir, f'figure_2.{args.format}')
    fig3_path = os.path.join(args.output_dir, f'figure_3.{args.format}')
    combined_path = os.path.join(args.output_dir, f'combined_figures.{args.format}')

    plot_figure_2(baseline_results, ih_results, fig2_path)
    plot_figure_3(baseline_results, ih_results, fig3_path)
    plot_combined_figure(baseline_results, ih_results, combined_path)

    print("\n" + "=" * 80)
    print("✓ FIGURES GENERATED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nFigures saved to: {args.output_dir}/")
    print(f"  - figure_2.{args.format} (Main Results)")
    print(f"  - figure_3.{args.format} (Generalization Results)")
    print(f"  - combined_figures.{args.format} (Both figures side-by-side)")
    print("=" * 80)


if __name__ == "__main__":
    main()