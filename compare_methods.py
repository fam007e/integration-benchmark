#!/usr/bin/env python3
"""
Targeted Method Comparison Tool

Compares specific methods or categories with detailed analysis.
Useful for deep-diving into particular comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def load_data(filename='integration_results.csv'):
    """Load benchmark results"""
    try:
        df = pd.read_csv(filename)
        df = df.replace([np.inf, -np.inf], np.nan)
        return df
    except FileNotFoundError:
        print(f"ERROR: {filename} not found!")
        print("Run the benchmark first: make run")
        sys.exit(1)

def compare_methods(df, method_list, problem=None):
    """
    Compare specific methods

    Args:
        df: DataFrame with results
        method_list: List of method names to compare
        problem: Specific problem to analyze (None = all problems)
    """
    print("\n" + "="*80)
    print(f"COMPARING METHODS: {', '.join(method_list)}")
    print("="*80 + "\n")

    # Filter data
    data = df[df['Method'].isin(method_list)].copy()
    if problem:
        data = data[data['Problem'] == problem]
        print(f"Problem: {problem}")
    else:
        print("Across all problems")

    print(f"\nTotal comparisons: {len(data)}")
    print("\nSummary Statistics:")
    print("-" * 80)

    for method in method_list:
        method_data = data[data['Method'] == method]
        if len(method_data) == 0:
            continue

        avg_error = method_data['Error'].mean()
        avg_time = method_data['Time_Mean_us'].mean()
        avg_eval = method_data['N_Eval'].mean()

        # Safe FOM calculation
        valid_fom = method_data['FOM_Eval'][method_data['FOM_Eval'] > 0]
        avg_fom = np.log10(valid_fom.mean()) if len(valid_fom) > 0 else -999

        print(f"\n{method}:")
        print(f"  Avg Error:      {avg_error:.3e}")
        print(f"  Avg Time:       {avg_time:.2f} μs")
        print(f"  Avg N_eval:     {avg_eval:.0f}")
        print(f"  Avg log₁₀(FOM): {avg_fom:.2f}")
        print(f"  Category:       {method_data['Category'].iloc[0]}")

    # Statistical test
    if len(method_list) == 2:
        from scipy import stats
        m1_data = data[data['Method'] == method_list[0]]
        m2_data = data[data['Method'] == method_list[1]]

        # Match problems for paired test
        common_problems = set(m1_data['Problem']) & set(m2_data['Problem'])
        if len(common_problems) > 0:
            m1_errors = []
            m2_errors = []
            for prob in common_problems:
                m1_errors.append(m1_data[m1_data['Problem'] == prob]['Error'].values[0])
                m2_errors.append(m2_data[m2_data['Problem'] == prob]['Error'].values[0])

            t_stat, p_value = stats.ttest_rel(m1_errors, m2_errors)
            print("\n" + "-" * 80)
            print("Paired t-test on errors:")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value:     {p_value:.4f}")
            if p_value < 0.05:
                winner = method_list[0] if t_stat < 0 else method_list[1]
                print(f"  Result:      {winner} is significantly better (α=0.05)")
            else:
                print(f"  Result:      No significant difference (α=0.05)")

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Error comparison
    ax = axes[0, 0]
    for method in method_list:
        method_data = data[data['Method'] == method]
        problems = method_data['Problem'].unique()
        errors = [method_data[method_data['Problem'] == p]['Error'].values[0]
                 for p in problems]
        ax.semilogy(range(len(problems)), errors, 'o-', label=method, markersize=8)

    ax.set_xlabel('Problem Index')
    ax.set_ylabel('Absolute Error (log scale)')
    ax.set_title('Error Comparison Across Problems')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # FOM comparison
    ax = axes[0, 1]
    data_fom = data[data['FOM_Eval'] > 0].copy()
    data_fom['log_FOM'] = np.log10(data_fom['FOM_Eval'])
    sns.boxplot(data=data_fom, x='Method', y='log_FOM', ax=ax)
    ax.set_ylabel('log₁₀(FOM)')
    ax.set_title('FOM Distribution')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Time comparison
    ax = axes[1, 0]
    sns.boxplot(data=data, x='Method', y='Time_Mean_us', ax=ax)
    ax.set_ylabel('Time (microseconds)')
    ax.set_title('Execution Time Distribution')
    ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    # Efficiency scatter
    ax = axes[1, 1]
    for method in method_list:
        method_data = data[data['Method'] == method]
        mask = (method_data['Error'] > 0) & (method_data['N_Eval'] > 0)
        ax.loglog(method_data[mask]['N_Eval'], method_data[mask]['Error'],
                 'o', label=method, markersize=8, alpha=0.7)

    ax.set_xlabel('Function Evaluations')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Efficiency: Error vs. Cost')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Method Comparison: {", ".join(method_list)}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f"comparison_{'_'.join(method_list[:3])}.png"
    plt.savefig(filename)
    print(f"\n✓ Comparison plot saved: {filename}")
    plt.close()

def compare_categories(df, cat_list):
    """Compare method categories"""
    print("\n" + "="*80)
    print(f"COMPARING CATEGORIES: {', '.join(cat_list)}")
    print("="*80 + "\n")

    data = df[df['Category'].isin(cat_list)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # FOM distribution
    ax = axes[0, 0]
    data_fom = data[data['FOM_Eval'] > 0].copy()
    data_fom['log_FOM'] = np.log10(data_fom['FOM_Eval'])
    sns.violinplot(data=data_fom, x='Category', y='log_FOM', ax=ax)
    ax.set_ylabel('log₁₀(FOM)')
    ax.set_title('FOM Distribution by Category')
    ax.tick_params(axis='x', rotation=45)

    # Error by difficulty
    ax = axes[0, 1]
    data_err = data[data['Error'] > 0].copy()
    data_err['log_Error'] = np.log10(data_err['Error'])

    for cat in cat_list:
        cat_data = data_err[data_err['Category'] == cat]
        difficulties = ['Easy', 'Medium', 'Hard', 'Very_Hard']
        avg_errors = [cat_data[cat_data['Difficulty'] == d]['log_Error'].mean()
                     for d in difficulties if d in cat_data['Difficulty'].values]
        valid_diffs = [d for d in difficulties if d in cat_data['Difficulty'].values]
        ax.plot(valid_diffs, avg_errors, 'o-', label=cat, markersize=8)

    ax.set_xlabel('Problem Difficulty')
    ax.set_ylabel('log₁₀(Error)')
    ax.set_title('Performance vs. Difficulty')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Computational cost
    ax = axes[1, 0]
    sns.boxplot(data=data, x='Category', y='N_Eval', ax=ax)
    ax.set_yscale('log')
    ax.set_ylabel('Function Evaluations (log scale)')
    ax.set_title('Computational Cost')
    ax.tick_params(axis='x', rotation=45)

    # Overall ranking
    ax = axes[1, 1]
    cat_scores = []
    for cat in cat_list:
        cat_data = data[data['Category'] == cat]
        valid_fom = cat_data['FOM_Eval'][cat_data['FOM_Eval'] > 0]
        if len(valid_fom) > 0:
            score = np.log10(valid_fom.mean())
            cat_scores.append(score)
        else:
            cat_scores.append(-10)

    colors = plt.cm.viridis(np.linspace(0, 1, len(cat_list)))
    bars = ax.barh(range(len(cat_list)), cat_scores, color=colors)
    ax.set_yticks(range(len(cat_list)))
    ax.set_yticklabels(cat_list)
    ax.set_xlabel('Average log₁₀(FOM)')
    ax.set_title('Overall Category Ranking')
    ax.grid(axis='x', alpha=0.3)

    for i, score in enumerate(cat_scores):
        ax.text(score + 0.1, i, f'{score:.2f}', va='center')

    plt.suptitle(f'Category Comparison: {", ".join(cat_list)}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    filename = f"category_comparison_{'_'.join(cat_list[:3])}.png"
    plt.savefig(filename)
    print(f"✓ Category comparison plot saved: {filename}")
    plt.close()

def find_best_for_problem(df, problem_name):
    """Find best methods for a specific problem"""
    print("\n" + "="*80)
    print(f"BEST METHODS FOR: {problem_name}")
    print("="*80 + "\n")

    data = df[df['Problem'] == problem_name].copy()
    if len(data) == 0:
        print(f"Problem '{problem_name}' not found!")
        print(f"Available problems: {', '.join(df['Problem'].unique())}")
        return

    # Sort by FOM
    data = data[data['FOM_Eval'] > 0].sort_values('FOM_Eval', ascending=False)

    print("Top 10 Methods:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Method':<30} {'Category':<15} {'Error':>12} {'FOM':>12}")
    print("-" * 80)

    for rank, (idx, row) in enumerate(data.head(10).iterrows(), 1):
        print(f"{rank:<5} {row['Method']:<30} {row['Category']:<15} "
              f"{row['Error']:>12.3e} {np.log10(row['FOM_Eval']):>12.2f}")

    difficulty = data['Difficulty'].iloc[0]
    exact = data['Exact'].iloc[0]
    print(f"\nProblem Details:")
    print(f"  Difficulty: {difficulty}")
    print(f"  Exact value: {exact:.10e}")

    # Quick visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Top 10 by FOM
    ax = axes[0]
    top10 = data.head(10)
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    bars = ax.barh(range(10), np.log10(top10['FOM_Eval'].values), color=colors)
    ax.set_yticks(range(10))
    ax.set_yticklabels(top10['Method'].values, fontsize=9)
    ax.set_xlabel('log₁₀(FOM)')
    ax.set_title(f'Top 10 Methods for {problem_name}')
    ax.grid(axis='x', alpha=0.3)

    # Error vs cost for all methods
    ax = axes[1]
    for cat in data['Category'].unique():
        cat_data = data[data['Category'] == cat]
        ax.loglog(cat_data['N_Eval'], cat_data['Error'],
                 'o', label=cat, alpha=0.7, markersize=6)

    # Highlight top method
    best = data.iloc[0]
    ax.loglog(best['N_Eval'], best['Error'], '*',
             color='red', markersize=20, label=f'Best: {best["Method"]}')

    ax.set_xlabel('Function Evaluations')
    ax.set_ylabel('Absolute Error')
    ax.set_title('All Methods on This Problem')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"best_for_{problem_name.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"\n✓ Plot saved: {filename}")
    plt.close()

def interactive_menu():
    """Interactive comparison tool"""
    df = load_data()

    while True:
        print("\n" + "="*80)
        print("INTEGRATION BENCHMARK - COMPARISON TOOL")
        print("="*80)
        print("\nOptions:")
        print("  1. Compare specific methods")
        print("  2. Compare method categories")
        print("  3. Find best methods for a problem")
        print("  4. List all methods")
        print("  5. List all problems")
        print("  6. Exit")

        choice = input("\nSelect option (1-6): ").strip()

        if choice == '1':
            print("\nAvailable methods:")
            for i, method in enumerate(df['Method'].unique(), 1):
                print(f"  {i}. {method}")

            method_input = input("\nEnter method names (comma-separated): ").strip()
            methods = [m.strip() for m in method_input.split(',')]

            problem = input("Specific problem (or press Enter for all): ").strip()
            problem = problem if problem else None

            compare_methods(df, methods, problem)

        elif choice == '2':
            print("\nAvailable categories:")
            for i, cat in enumerate(df['Category'].unique(), 1):
                print(f"  {i}. {cat}")

            cat_input = input("\nEnter categories (comma-separated): ").strip()
            categories = [c.strip() for c in cat_input.split(',')]

            compare_categories(df, categories)

        elif choice == '3':
            print("\nAvailable problems:")
            problems = df['Problem'].unique()
            for i, prob in enumerate(problems, 1):
                diff = df[df['Problem'] == prob]['Difficulty'].iloc[0]
                print(f"  {i}. {prob} ({diff})")

            prob = input("\nEnter problem name: ").strip()
            find_best_for_problem(df, prob)

        elif choice == '4':
            print("\nAll Methods:")
            methods = df.groupby('Method')['Category'].first().reset_index()
            for _, row in methods.iterrows():
                print(f"  {row['Method']:<40} ({row['Category']})")

        elif choice == '5':
            print("\nAll Problems:")
            for prob in df['Problem'].unique():
                diff = df[df['Problem'] == prob]['Difficulty'].iloc[0]
                print(f"  {prob:<30} (Difficulty: {diff})")

        elif choice == '6':
            print("\nExiting...")
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command-line mode
        if sys.argv[1] == '--compare-methods' and len(sys.argv) > 2:
            df = load_data()
            methods = sys.argv[2].split(',')
            problem = sys.argv[3] if len(sys.argv) > 3 else None
            compare_methods(df, methods, problem)
        elif sys.argv[1] == '--compare-categories' and len(sys.argv) > 2:
            df = load_data()
            categories = sys.argv[2].split(',')
            compare_categories(df, categories)
        elif sys.argv[1] == '--best-for' and len(sys.argv) > 2:
            df = load_data()
            find_best_for_problem(df, sys.argv[2])
        else:
            print("Usage:")
            print("  python compare_methods.py  (interactive mode)")
            print("  python compare_methods.py --compare-methods Method1,Method2")
            print("  python compare_methods.py --compare-categories Cat1,Cat2")
            print("  python compare_methods.py --best-for ProblemName")
    else:
        # Interactive mode
        interactive_menu()
