# Advanced Numerical Integration Benchmark Suite

A comprehensive benchmarking framework for evaluating and comparing numerical integration methods using Figure of Merit (FOM) analysis.

## 🎯 Overview

This project provides a rigorous, publication-quality benchmark suite for numerical integration methods. It evaluates **19 different integration methods** across **10+ test problems** with varying characteristics, measuring both accuracy and computational efficiency.

### Figure of Merit (FOM)

The benchmark uses two FOM metrics:

```
FOM_eval = 1 / (ε² × N_evaluations)
FOM_time = 1 / (ε² × CPU_time)
```

Where:
- **ε** = absolute error vs. exact solution
- **Higher FOM** = better efficiency

## ✨ Features

### Integration Methods Implemented

**Newton-Cotes Family:**
- Midpoint Rule
- Trapezium Rule (Trapezoidal)
- Simpson's Rule (1/3)
- Simpson's 3/8 Rule
- Boole's Rule

**Gaussian Quadrature:**
- 2, 3, 5, 7, and 10-point Gauss-Legendre
- Multiple resolution settings

**Advanced Methods:**
- Romberg Integration (Richardson extrapolation)
- Adaptive Simpson's with recursive refinement
- Clenshaw-Curtis (Chebyshev-based)
- Tanh-Sinh (Double Exponential)
- Monte Carlo (uniform sampling)
- Quasi-Monte Carlo (Van der Corput sequence)

### Test Problem Suite

**Easy:** Smooth exponentials, low-degree polynomials, simple trigonometric
**Medium:** Oscillatory functions, rational functions, Gaussian peaks
**Hard:** Runge's function, weak singularities
**Very Hard:** High-frequency oscillations, near-singularities

### Comprehensive Metrics

- Function evaluation counts
- High-resolution timing (nanosecond precision)
- Recursion depth tracking
- Statistical analysis (mean, std deviation)
- Convergence rate estimation
- Category-based comparisons

### Visualization Suite

**8 Publication-Quality Plots:**
1. **FOM Scatter** - Error vs. efficiency trade-offs
2. **Efficiency Curves** - Error vs. function evaluations
3. **Heat Map** - Method performance across all problems
4. **Convergence Rates** - Order of accuracy analysis
5. **Category Comparison** - Statistical distributions
6. **Pareto Frontier** - Optimal method selection
7. **Difficulty Analysis** - Performance vs. problem complexity
8. **Timing Analysis** - Wall-clock performance metrics

## 🚀 Quick Start

### Prerequisites

```bash
# C compiler
gcc --version  # or clang

# Python 3.8+ with packages
pip install pandas numpy matplotlib seaborn scipy

# Optional: Performance analysis tools
sudo apt install linux-perf valgrind  # Linux
```

### Build and Run

```bash
# Complete workflow (recommended for first run)
make full

# Or step by step:
make              # Compile
make run          # Execute benchmark
make visualize    # Generate plots
```

Results will be in:
- `integration_results.csv` - Full benchmark data
- `convergence_analysis.csv` - Convergence rates
- `plots/` - All visualization images
- `plots/summary_report.txt` - Text summary

## 📊 Usage Examples

### Basic Benchmarking

```bash
# Standard run
make run

# Quick test (fewer iterations)
make test

# Debug mode
make debug
./integration_bench_debug
```

### Performance Profiling

```bash
# CPU profiling with gprof
make profile
./integration_bench_prof
gprof integration_bench_prof gmon.out > analysis.txt

# Cache analysis with valgrind
make cache-analysis

# Performance counters with perf (Linux)
make perf-stat
```

### Re-generate Visualizations

If you already have CSV data and want to update plots:

```bash
make visualize
```

## 📈 Understanding the Output

### Console Output

```
Problem: Exponential (Difficulty: Easy)
  Domain: [0.0, 1.0], Exact: 1.718281828459e+00
───────────────────────────────────────────────────────────────
Method              Result       Error      N_eval     FOM(eval)      Time(μs)
───────────────────────────────────────────────────────────────
Gauss5_n50      1.718282e+00  4.95e-13        300     6.73e+09       12.450
Simpson_n200    1.718282e+00  8.21e-11      40200     3.01e+08       89.234
...
```

### CSV Format

```csv
Problem,Method,Category,N_Points,Result,Exact,Error,RelError,N_Eval,Time_Mean_us,...
Exponential,Gauss5_n50,Gaussian,50,1.71828182846,1.71828182846,4.95e-13,2.88e-13,300,12.45,...
```

### Key Insights from Plots

**01_fom_scatter.png** - Identify best methods overall
**02_error_vs_neval.png** - See efficiency for specific problems
**03_heatmap_fom.png** - Quick visual comparison (green = good)
**06_pareto_frontier.png** - Find optimal methods for your needs

## 🔧 Customization

### Adding New Methods

Edit `integration_benchmark.c`:

```c
// 1. Implement your method
double my_new_method(double (*f)(double), double a, double b, int n) {
    // Your implementation
    return result;
}

// 2. Add to methods array in main()
Method1D methods[] = {
    // ... existing methods ...
    {"MyMethod_n100", my_new_method, 100, "Custom"},
};
```

### Adding Test Problems

```c
// 1. Define function
double f_my_test(double x) {
    g_instr.function_evals++;
    return /* your function */;
}

// 2. Add to problems array
TestProblem problems[] = {
    // ... existing problems ...
    {"MyTest", f_my_test, a, b, exact_value, "Medium"},
};
```

### Adjusting Parameters

In `integration_benchmark.c`:

```c
#define NUM_TIMING_RUNS 10     // Runs per method (increase for stability)
#define ADAPTIVE_TOLERANCE 1e-10  // Adaptive method tolerance
#define MC_SAMPLES 100000      // Monte Carlo sample count
```

## 📊 Example Results

### Typical Performance Rankings

**Smooth Functions (Easy):**
1. Gauss-Legendre (5-10 point) - Best FOM
2. Tanh-Sinh - High accuracy
3. Simpson's Rule - Good balance

**Oscillatory Functions (Hard):**
1. Adaptive methods - Handles variation
2. High-order Gaussian - Many points
3. Clenshaw-Curtis - Stable

**Singular Functions (Very Hard):**
1. Tanh-Sinh - Handles endpoints
2. Adaptive Simpson's - Refinement
3. Lower-order Newton-Cotes - Robust

### Convergence Orders

- Midpoint/Trapezium: O(n⁻²)
- Simpson's: O(n⁻⁴)
- Gauss-Legendre (5pt): O(n⁻¹⁰)
- Adaptive: Problem-dependent

## 🧪 Scientific Applications

This benchmark is useful for:

- **Algorithm Selection** - Choose best method for your problem class
- **Research** - Compare new methods against established baselines
- **Education** - Understand trade-offs in numerical methods
- **Engineering** - Optimize computational workflows
- **Publications** - Generate high-quality comparison plots

## 📚 Theory Background

### Why Different Methods Exist

- **Newton-Cotes**: Use equally-spaced points (good for tabulated data)
- **Gaussian**: Optimize point placement (best for smooth functions)
- **Adaptive**: Refine where needed (efficient for variable functions)
- **Monte Carlo**: High dimensions (curse of dimensionality)

### The FOM Concept

FOM balances accuracy and cost:
- Low error + low cost → High FOM ✓
- Low error + high cost → Medium FOM
- High error → Low FOM ✗

### Pareto Optimality

A method is Pareto-optimal if no other method achieves:
- Lower error with same cost, OR
- Same error with lower cost

These are the "frontier" methods worth considering.

## 🛠️ Troubleshooting

### Build Issues

```bash
# Missing math library
# Add -lm flag: gcc ... -lm

# Timing issues on macOS
# Use CLOCK_MONOTONIC_RAW instead of CLOCK_MONOTONIC

# Optimization issues
# Try -O2 instead of -O3 if -O3 causes issues
```

### Python Issues

```bash
# Missing packages
pip install -r requirements.txt

# Or individually
pip install pandas numpy matplotlib seaborn scipy

# Import errors
python3 -c "import pandas; import matplotlib"
```

### Performance Issues

```bash
# Reduce runs for faster testing
make test  # Uses NUM_RUNS=3 instead of 10

# Reduce methods
# Comment out methods in main() you don't need

# Reduce problems
# Comment out problems in main()
```

## 📖 References

### Key Papers

1. **Numerical Recipes** (Press et al.) - Comprehensive coverage
2. **Quadrature Methods** (Davis & Rabinowitz) - Theory
3. **Adaptive Quadrature** (Gander & Gautschi) - Modern techniques

### Relevant Theory

- **Newton-Cotes**: Based on Lagrange interpolation
- **Gaussian**: Orthogonal polynomial theory
- **Romberg**: Richardson extrapolation
- **Monte Carlo**: Law of large numbers

## 🤝 Contributing

Contributions welcome! Areas of interest:

- Additional integration methods (Gauss-Kronrod, Filon, etc.)
- Multi-dimensional integration
- Parallel implementations
- GPU acceleration
- Additional test problems
- Performance optimizations

## 📄 License

MIT License - Free for academic and commercial use.

## 🙏 Acknowledgments

Based on classical numerical analysis literature and modern computational methods research.

---

## 📞 Support

For questions or issues:
1. Check `plots/summary_report.txt` for results interpretation
2. Review convergence plots for method behavior
3. Compare with literature for validation

**Happy Benchmarking! 🚀**
