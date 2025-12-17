#!/usr/bin/env python3
"""
Advanced Numerical Integration Benchmark Visualization Suite
Generates comprehensive plots for figure of merit analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10


class IntegrationVisualizer:
    def __init__(
        self,
        results_file="integration_results.csv",
        convergence_file="convergence_analysis.csv",
    ):
        """Load and prepare data"""
        self.df = pd.read_csv(results_file)
        try:
            self.conv_df = pd.read_csv(convergence_file)
        except:
            self.conv_df = None
        # Clean data
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        # Create output directory
        os.makedirs("plots", exist_ok=True)
        print(f"✓ Loaded {len(self.df)} benchmark results")
        print(f" Problems: {self.df['Problem'].nunique()}")
        print(f" Methods: {self.df['Method'].nunique()}")
        print(f" Categories: {', '.join(self.df['Category'].unique())}")

    def plot_fom_scatter(self):
        """Scatter plot: Error vs FOM (Pareto frontier)"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        categories = self.df["Category"].unique()

        # FOM vs Error (log-log)
        ax = axes[0]
        for cat in categories:
            data = self.df[self.df["Category"] == cat]
            mask = (data["FOM_Eval"] > 0) & (data["Error"] > 0)
            ax.scatter(
                data[mask]["Error"], data[mask]["FOM_Eval"], label=cat, alpha=0.6, s=50
            )
        ax.set_xlabel("Absolute Error", fontsize=12, fontweight="bold")
        ax.set_ylabel(
            r"FOM (eval) = $1/(\epsilon^{2} \times N_{\mathrm{eval}})$",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(
            r"Figure of Merit vs. Error" "\n" r"(Higher FOM = Better Efficiency)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(title="Method Category", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # FOM vs Time
        ax = axes[1]
        for cat in categories:
            data = self.df[self.df["Category"] == cat]
            mask = (data["FOM_Time"] > 0) & (data["Time_Mean_us"] > 0)
            ax.scatter(
                data[mask]["Time_Mean_us"],
                data[mask]["FOM_Time"],
                label=cat,
                alpha=0.6,
                s=50,
            )
        ax.set_xlabel("Time (microseconds)", fontsize=12, fontweight="bold")
        ax.set_ylabel(
            r"FOM (time) = $1/(\epsilon^{2} \times t)$", fontsize=12, fontweight="bold"
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(
            "Time-based Figure of Merit\n(Wall-clock Performance)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(title="Method Category", framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("plots/01_fom_scatter.png", bbox_inches="tight")
        print("✓ Generated: plots/01_fom_scatter.png")
        plt.close()

    def plot_error_vs_neval(self):
        """Error vs Number of Evaluations (efficiency curves)"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        problems = self.df["Problem"].unique()[:6]
        for idx, problem in enumerate(problems):
            ax = axes[idx]
            data = self.df[self.df["Problem"] == problem]
            for cat in data["Category"].unique():
                cat_data = data[data["Category"] == cat]
                mask = (cat_data["Error"] > 0) & (cat_data["N_Eval"] > 0)
                ax.scatter(
                    cat_data[mask]["N_Eval"],
                    cat_data[mask]["Error"],
                    label=cat,
                    alpha=0.7,
                    s=40,
                )
            difficulty = data["Difficulty"].iloc[0]
            ax.set_xlabel("Function Evaluations", fontsize=10)
            ax.set_ylabel("Absolute Error", fontsize=10)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(
                f"{problem}\n(Difficulty: {difficulty})", fontsize=11, fontweight="bold"
            )
            ax.legend(fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            "Efficiency Curves: Error vs. Function Evaluations",
            fontsize=16,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()
        plt.savefig("plots/02_error_vs_neval.png", bbox_inches="tight")
        print("✓ Generated: plots/02_error_vs_neval.png")
        plt.close()

    def plot_heatmap_fom(self):
        """Heatmap: Methods vs Problems (FOM values)"""
        pivot = self.df.pivot_table(
            values="FOM_Eval", index="Method", columns="Problem", aggfunc="mean"
        )
        pivot_log = np.log10(pivot.clip(lower=1e-10))
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            pivot_log,
            annot=False,
            fmt=".1f",
            cmap="RdYlGn",
            cbar_kws={"label": r"$\log_{10}(\mathrm{FOM})$"},
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(
            r"Heat Map: Method Performance Across Problems"
            "\n"
            r"(Color = $\log_{10}(\mathrm{FOM})$, Green = Better)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Problem", fontsize=12, fontweight="bold")
        ax.set_ylabel("Integration Method", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("plots/03_heatmap_fom.png", bbox_inches="tight")
        print("✓ Generated: plots/03_heatmap_fom.png")
        plt.close()

    def plot_convergence_rates(self):
        """Convergence rate analysis from convergence_analysis.csv"""
        if self.conv_df is None:
            print("⚠ Skipping convergence plots (file not found)")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        methods = self.conv_df["Method"].unique()
        n_values = np.array([10, 20, 40, 80, 160, 320, 640, 1280], dtype=float)

        for idx, method in enumerate(methods):
            if idx >= 6:
                break
            ax = axes[idx]
            data = self.conv_df[self.conv_df["Method"] == method]
            errors = data.iloc[0, 1:].values.astype(float)

            ax.loglog(n_values, errors, "o-", linewidth=2, markersize=8, label=method)

            valid = errors > 0
            if np.sum(valid) > 2:
                log_n = np.log(n_values[valid])
                log_err = np.log(errors[valid])
                slope, intercept = np.polyfit(log_n, log_err, 1)
                fit = np.exp(intercept) * n_values**slope
                ax.loglog(n_values, fit, "--", alpha=0.6, label=f"O(n^{slope:.2f})")

            ax.loglog(
                n_values, 1e-2 * n_values ** (-2), "k:", alpha=0.3, label=r"$O(n^{-2})$"
            )
            ax.loglog(
                n_values,
                1e-4 * n_values ** (-4),
                "k--",
                alpha=0.3,
                label=r"$O(n^{-4})$",
            )

            ax.set_xlabel("Number of Intervals (n)", fontsize=10)
            ax.set_ylabel("Absolute Error", fontsize=10)
            ax.set_title(f"{method} Convergence", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            r"Convergence Rate Analysis" "\n" r"(Slope indicates order of accuracy)",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig("plots/04_convergence_rates.png", bbox_inches="tight")
        print("✓ Generated: plots/04_convergence_rates.png")
        plt.close()

    def plot_category_comparison(self):
        """Box plots comparing method categories"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # FOM by category
        ax = axes[0, 0]
        data_fom = self.df[self.df["FOM_Eval"] > 0].copy()
        data_fom["log_FOM"] = np.log10(data_fom["FOM_Eval"])
        sns.boxplot(data=data_fom, x="Category", y="log_FOM", ax=ax)
        ax.set_ylabel(r"$\log_{10}(\mathrm{FOM})$", fontsize=11, fontweight="bold")
        ax.set_xlabel("Method Category", fontsize=11, fontweight="bold")
        ax.set_title("FOM Distribution by Category", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)

        # Error by category
        ax = axes[0, 1]
        data_err = self.df[self.df["Error"] > 0].copy()
        data_err["log_Error"] = np.log10(data_err["Error"])
        sns.boxplot(data=data_err, x="Category", y="log_Error", ax=ax)
        ax.set_ylabel(r"$\log_{10}(\mathrm{Error})$", fontsize=11, fontweight="bold")
        ax.set_xlabel("Method Category", fontsize=11, fontweight="bold")
        ax.set_title("Error Distribution by Category", fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)

        # Time by category
        ax = axes[1, 0]
        data_time = self.df[self.df["Time_Mean_us"] > 0].copy()
        sns.boxplot(data=data_time, x="Category", y="Time_Mean_us", ax=ax)
        ax.set_ylabel("Time (microseconds)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Method Category", fontsize=11, fontweight="bold")
        ax.set_title("Execution Time by Category", fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=45)

        # N_eval by category
        ax = axes[1, 1]
        sns.boxplot(data=self.df, x="Category", y="N_Eval", ax=ax)
        ax.set_ylabel("Function Evaluations", fontsize=11, fontweight="bold")
        ax.set_xlabel("Method Category", fontsize=11, fontweight="bold")
        ax.set_title("Computational Cost by Category", fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=45)

        plt.suptitle(
            "Statistical Comparison of Method Categories",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig("plots/05_category_comparison.png", bbox_inches="tight")
        print("✓ Generated: plots/05_category_comparison.png")
        plt.close()

    def plot_pareto_frontier(self):
        """Pareto frontier: optimal trade-off between error and cost"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        problems = self.df["Problem"].unique()[:2]
        for idx, problem in enumerate(problems):
            ax = axes[idx]
            data = self.df[self.df["Problem"] == problem].copy()
            data = data[(data["Error"] > 0) & (data["N_Eval"] > 0)]
            for cat in data["Category"].unique():
                cat_data = data[data["Category"] == cat]
                ax.scatter(
                    cat_data["N_Eval"], cat_data["Error"], label=cat, alpha=0.6, s=60
                )

            points = data[["N_Eval", "Error"]].values
            pareto_front = []
            for i, point in enumerate(points):
                dominated = False
                for other in points:
                    if (other[0] <= point[0] and other[1] < point[1]) or (
                        other[0] < point[0] and other[1] <= point[1]
                    ):
                        dominated = True
                        break
                if not dominated:
                    pareto_front.append(point)

            if pareto_front:
                pareto_front = np.array(pareto_front)
                pareto_front = pareto_front[pareto_front[:, 0].argsort()]
                ax.plot(
                    pareto_front[:, 0],
                    pareto_front[:, 1],
                    "r--",
                    linewidth=2,
                    label="Pareto Frontier",
                    zorder=10,
                )
                ax.scatter(
                    pareto_front[:, 0],
                    pareto_front[:, 1],
                    color="red",
                    s=100,
                    marker="*",
                    zorder=11,
                    edgecolors="black",
                )

            difficulty = data["Difficulty"].iloc[0]
            ax.set_xlabel("Function Evaluations", fontsize=11, fontweight="bold")
            ax.set_ylabel("Absolute Error", fontsize=11, fontweight="bold")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(
                f"{problem} (Difficulty: {difficulty})\nPareto-Optimal Methods",
                fontsize=12,
                fontweight="bold",
            )
            ax.legend(fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            r"Pareto Frontier Analysis" "\n" r"(Red stars = optimal trade-offs)",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig("plots/06_pareto_frontier.png", bbox_inches="tight")
        print("✓ Generated: plots/06_pareto_frontier.png")
        plt.close()

    def plot_difficulty_analysis(self):
        """How method performance varies with problem difficulty"""
        difficulties = self.df["Difficulty"].unique()
        fig, axes = plt.subplots(
            1, len(difficulties), figsize=(6 * len(difficulties), 5)
        )
        if len(difficulties) == 1:
            axes = [axes]

        for idx, diff in enumerate(difficulties):
            ax = axes[idx]
            data = self.df[self.df["Difficulty"] == diff]
            avg_fom = data.groupby("Category")["FOM_Eval"].apply(
                lambda x: np.log10(x[x > 0].mean()) if len(x[x > 0]) > 0 else np.nan
            )
            avg_fom = avg_fom.dropna().sort_values(ascending=False)
            colors = plt.cm.viridis(np.linspace(0, 1, len(avg_fom)))
            bars = ax.barh(range(len(avg_fom)), avg_fom.values, color=colors)
            ax.set_yticks(range(len(avg_fom)))
            ax.set_yticklabels(avg_fom.index)
            ax.set_xlabel(
                r"Average $\log_{10}(\mathrm{FOM})$", fontsize=11, fontweight="bold"
            )
            ax.set_title(
                f"Difficulty: {diff}\nMethod Ranking", fontsize=12, fontweight="bold"
            )
            ax.grid(axis="x", alpha=0.3)
            for i, (bar, val) in enumerate(zip(bars, avg_fom.values)):
                ax.text(val + 0.1, i, f"{val:.2f}", va="center", fontsize=9)

        plt.suptitle(
            "Method Performance vs. Problem Difficulty", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig("plots/07_difficulty_analysis.png", bbox_inches="tight")
        print("✓ Generated: plots/07_difficulty_analysis.png")
        plt.close()

    def plot_timing_analysis(self):
        """Timing statistics and variability"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Time vs Error scatter
        ax = axes[0, 0]
        for cat in self.df["Category"].unique():
            data = self.df[self.df["Category"] == cat]
            mask = (data["Time_Mean_us"] > 0) & (data["Error"] > 0)
            ax.scatter(
                data[mask]["Time_Mean_us"],
                data[mask]["Error"],
                label=cat,
                alpha=0.6,
                s=50,
            )
        ax.set_xlabel("Time (microseconds)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Absolute Error", fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Time-Accuracy Trade-off", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Timing variability
        ax = axes[0, 1]
        data_cv = self.df[self.df["Time_Mean_us"] > 0].copy()
        data_cv["CV"] = data_cv["Time_Std_us"] / data_cv["Time_Mean_us"] * 100
        data_cv = data_cv[data_cv["CV"] < 50]
        sns.boxplot(data=data_cv, x="Category", y="CV", ax=ax)
        ax.set_ylabel("Coefficient of Variation (%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Method Category", fontsize=11, fontweight="bold")
        ax.set_title(
            "Timing Stability\n(Lower = More Consistent)",
            fontsize=12,
            fontweight="bold",
        )
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

        # Top 10 fastest
        ax = axes[1, 0]
        fastest = self.df.nsmallest(10, "Time_Mean_us")
        colors = plt.cm.plasma(np.linspace(0, 1, 10))
        bars = ax.barh(range(10), fastest["Time_Mean_us"].values, color=colors)
        ax.set_yticks(range(10))
        ax.set_yticklabels(fastest["Method"].values, fontsize=9)
        ax.set_xlabel("Time (microseconds)", fontsize=11, fontweight="bold")
        ax.set_title("Top 10 Fastest Methods", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Top time-efficient
        ax = axes[1, 1]
        data_eff = self.df[
            (self.df["FOM_Time"] > 0) & (self.df["FOM_Time"] < 1e15)
        ].copy()
        data_eff["log_FOM_Time"] = np.log10(data_eff["FOM_Time"])
        top_efficient = data_eff.nlargest(10, "log_FOM_Time")
        colors = plt.cm.viridis(np.linspace(0, 1, 10))
        bars = ax.barh(range(10), top_efficient["log_FOM_Time"].values, color=colors)
        ax.set_yticks(range(10))
        ax.set_yticklabels(top_efficient["Method"].values, fontsize=9)
        ax.set_xlabel(
            r"$\log_{10}(\mathrm{FOM}_{\mathrm{Time}})$", fontsize=11, fontweight="bold"
        )
        ax.set_title(
            "Top 10 Most Time-Efficient Methods", fontsize=12, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)

        plt.suptitle("Timing and Performance Analysis", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig("plots/08_timing_analysis.png", bbox_inches="tight")
        print("✓ Generated: plots/08_timing_analysis.png")
        plt.close()

    def generate_summary_report(self):
        """Generate a text summary report"""
        with open("plots/summary_report.txt", "w") as f:
            f.write("=" * 80 + "\n")
            f.write("NUMERICAL INTEGRATION BENCHMARK - SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total benchmarks run: {len(self.df)}\n")
            f.write(f"Problems tested: {self.df['Problem'].nunique()}\n")
            f.write(f"Methods evaluated: {self.df['Method'].nunique()}\n")
            f.write(f"Method categories: {', '.join(self.df['Category'].unique())}\n\n")

            f.write("TOP 10 METHODS BY AVERAGE FOM\n")
            f.write("-" * 40 + "\n")
            avg_fom = (
                self.df.groupby("Method")
                .agg(
                    {
                        "FOM_Eval": lambda x: np.log10(x[x > 0].mean())
                        if len(x[x > 0]) > 0
                        else -999,
                        "Category": "first",
                    }
                )
                .sort_values("FOM_Eval", ascending=False)
                .head(10)
            )
            for idx, (method, row) in enumerate(avg_fom.iterrows(), 1):
                f.write(
                    f"{idx:2d}. {method:30s} (log₁₀FOM: {row['FOM_Eval']:7.2f}, {row['Category']})\n"
                )

            f.write("\n\nBEST METHOD IN EACH CATEGORY\n")
            f.write("-" * 40 + "\n")
            for cat in self.df["Category"].unique():
                cat_data = self.df[self.df["Category"] == cat]
                best = cat_data.loc[cat_data["FOM_Eval"].idxmax()]
                f.write(
                    f"{cat:20s}: {best['Method']:30s} (log₁₀FOM: {np.log10(best['FOM_Eval']):7.2f})\n"
                )

            f.write("\n\nBEST METHOD FOR EACH PROBLEM\n")
            f.write("-" * 40 + "\n")
            for prob in self.df["Problem"].unique():
                prob_data = self.df[self.df["Problem"] == prob]
                best = prob_data.loc[prob_data["FOM_Eval"].idxmax()]
                diff = best["Difficulty"]
                f.write(f"{prob:25s} ({diff:10s}): {best['Method']:30s}\n")

            if self.conv_df is not None:
                f.write("\n\nCONVERGENCE RATE ESTIMATES\n")
                f.write("-" * 40 + "\n")
                n_values = np.array([10, 20, 40, 80, 160, 320, 640, 1280], dtype=float)
                for _, row in self.conv_df.iterrows():
                    method = row["Method"]
                    errors = row[1:].values.astype(float)
                    valid = errors > 0
                    if np.sum(valid) > 2:
                        log_n = np.log(n_values[valid])
                        log_err = np.log(errors[valid])
                        slope, _ = np.polyfit(log_n, log_err, 1)
                        f.write(f"{method:20s}: O(n^{slope:6.2f})\n")

            f.write("\n" + "=" * 80 + "\n")
        print("✓ Generated: plots/summary_report.txt")

    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATION SUITE")
        print("=" * 70 + "\n")
        self.plot_fom_scatter()
        self.plot_error_vs_neval()
        self.plot_heatmap_fom()
        self.plot_convergence_rates()
        self.plot_category_comparison()
        self.plot_pareto_frontier()
        self.plot_difficulty_analysis()
        self.plot_timing_analysis()
        self.generate_summary_report()
        print("\n" + "=" * 70)
        print("✓ ALL VISUALIZATIONS COMPLETE!")
        print("=" * 70)
        print(f"\nGenerated files in ./plots/ directory:")
        print(" 01_fom_scatter.png - FOM vs Error/Time scatter plots")
        print(" 02_error_vs_neval.png - Efficiency curves by problem")
        print(" 03_heatmap_fom.png - Method-Problem performance matrix")
        print(" 04_convergence_rates.png - Convergence rate analysis")
        print(" 05_category_comparison.png - Statistical category comparison")
        print(" 06_pareto_frontier.png - Optimal trade-off analysis")
        print(" 07_difficulty_analysis.png - Performance vs difficulty")
        print(" 08_timing_analysis.png - Timing and efficiency metrics")
        print(" summary_report.txt - Text summary of results")
        print()


if __name__ == "__main__":
    if not os.path.exists("integration_results.csv"):
        print("ERROR: integration_results.csv not found!")
        print("Please run the C benchmark program first to generate data.")
        exit(1)

    viz = IntegrationVisualizer()
    viz.generate_all_plots()
    print("🎉 Visualization complete! Open the plots/ directory to view results.")
