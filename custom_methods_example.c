/*
 * EXAMPLE: Adding Custom Integration Methods to the Benchmark Suite
 *
 * This file shows how to implement and add your own integration methods
 * to the benchmark framework.
 */

#include <stdio.h>
#include <math.h>

// External instrumentation (from main benchmark)
extern long long g_function_evals;

// ============================================================================
// EXAMPLE 1: Simple Custom Method - Left Riemann Sum
// ============================================================================

/**
 * Left Riemann Sum: Uses left endpoint of each interval
 * This is similar to Midpoint rule but less accurate
 * Useful as a simple baseline for comparison
 */
double left_riemann_sum(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;

    for (int i = 0; i < n; i++) {
        sum += f(a + i * h);
    }

    return h * sum;
}

// ============================================================================
// EXAMPLE 2: Advanced Custom Method - Adaptive Trapezoid
// ============================================================================

/**
 * Recursive helper for adaptive trapezoid
 */
double adaptive_trapezoid_recursive(double (*f)(double), double a, double b,
                                    double fa, double fb, double tol, int depth) {
    double c = (a + b) / 2.0;
    double fc = f(c);

    // Trapezoid estimates
    double T1 = (b - a) * (fa + fb) / 2.0;
    double T2 = (c - a) * (fa + fc) / 2.0 + (b - c) * (fc + fb) / 2.0;

    // Check convergence
    if (depth > 20 || fabs(T2 - T1) < tol) {
        return T2;
    }

    // Recursively refine
    return adaptive_trapezoid_recursive(f, a, c, fa, fc, tol/2.0, depth+1) +
           adaptive_trapezoid_recursive(f, c, b, fc, fb, tol/2.0, depth+1);
}

/**
 * Adaptive Trapezoid Rule with automatic refinement
 * Good for functions with localized features
 */
double adaptive_trapezoid(double (*f)(double), double a, double b, int max_depth) {
    double fa = f(a);
    double fb = f(b);
    double tol = 1e-10;
    return adaptive_trapezoid_recursive(f, a, b, fa, fb, tol, 0);
}

// ============================================================================
// EXAMPLE 3: Specialized Method - Trigonometric Integration
// ============================================================================

/**
 * Specialized method for periodic functions using FFT-like approach
 * Efficient for smooth periodic integrands
 *
 * Note: This is a simplified version - full implementation would use FFT
 */
double periodic_trapezoid(double (*f)(double), double a, double b, int n) {
    // Ensure n is even for symmetry
    if (n % 2 != 0) n++;

    double h = (b - a) / n;
    double sum = 0.0;

    // Standard trapezoid but optimized for periodic boundary
    // For true periodic functions: f(a) = f(b), so we can optimize
    sum = f(a);  // Only count once (not 0.5*(f(a) + f(b)))

    for (int i = 1; i < n; i++) {
        sum += f(a + i * h);
    }

    return h * sum;
}

// ============================================================================
// EXAMPLE 4: Novel Research Method - Hybrid Gauss-Newton-Cotes
// ============================================================================

/**
 * Hybrid approach: Use Gaussian quadrature in smooth regions,
 * switch to Newton-Cotes near potential discontinuities
 *
 * This demonstrates how you might implement adaptive method selection
 */
double hybrid_integration(double (*f)(double), double a, double b, int n_panels) {
    double total = 0.0;
    double h = (b - a) / n_panels;

    for (int i = 0; i < n_panels; i++) {
        double a_sub = a + i * h;
        double b_sub = a + (i + 1) * h;

        // Sample function to detect smoothness
        double samples[5];
        for (int j = 0; j < 5; j++) {
            samples[j] = f(a_sub + j * h / 4.0);
        }

        // Compute variation (simple measure of smoothness)
        double variation = 0.0;
        for (int j = 1; j < 5; j++) {
            variation += fabs(samples[j] - samples[j-1]);
        }

        // Choose method based on variation
        if (variation < 0.1) {
            // Smooth region: use 3-point Gauss-Legendre
            double w[] = {5.0/9.0, 8.0/9.0, 5.0/9.0};
            double x[] = {-sqrt(3.0/5.0), 0.0, sqrt(3.0/5.0)};
            double mid = (a_sub + b_sub) / 2.0;
            double half = (b_sub - a_sub) / 2.0;

            double sub_total = 0.0;
            for (int j = 0; j < 3; j++) {
                sub_total += w[j] * f(mid + half * x[j]);
            }
            total += half * sub_total;
        } else {
            // Non-smooth region: use robust Simpson's rule
            double c = (a_sub + b_sub) / 2.0;
            total += (h / 6.0) * (f(a_sub) + 4.0*f(c) + f(b_sub));
        }
    }

    return total;
}

// ============================================================================
// EXAMPLE 5: Machine Learning Inspired - Weighted Sampling
// ============================================================================

/**
 * Use importance sampling based on function gradient
 * Allocates more samples where function varies rapidly
 *
 * This is a simplified ML-inspired approach
 */
double importance_sampling_integration(double (*f)(double), double a, double b, int n_samples) {
    // Phase 1: Pilot run to estimate importance
    int n_pilot = n_samples / 10;
    double* importance = malloc(n_pilot * sizeof(double));
    double total_importance = 0.0;

    for (int i = 0; i < n_pilot; i++) {
        double x = a + (b - a) * i / (n_pilot - 1.0);
        double dx = (b - a) / (n_pilot * 100.0);

        // Estimate local gradient (importance)
        double grad = fabs(f(x + dx) - f(x - dx)) / (2.0 * dx);
        importance[i] = grad + 1e-10;  // Avoid zero
        total_importance += importance[i];
    }

    // Phase 2: Allocate samples proportionally to importance
    double sum = 0.0;
    int samples_used = 0;

    for (int i = 0; i < n_pilot && samples_used < n_samples; i++) {
        // Number of samples for this region
        int local_samples = (int)((importance[i] / total_importance) * n_samples);
        if (local_samples < 1) local_samples = 1;

        double x_start = a + (b - a) * i / (n_pilot - 1.0);
        double x_end = a + (b - a) * (i + 1) / (n_pilot - 1.0);
        double local_h = (x_end - x_start) / local_samples;

        for (int j = 0; j < local_samples && samples_used < n_samples; j++) {
            sum += f(x_start + (j + 0.5) * local_h) * local_h;
            samples_used++;
        }
    }

    free(importance);
    return sum;
}

// ============================================================================
// HOW TO ADD THESE TO THE MAIN BENCHMARK
// ============================================================================

/*

To integrate these methods into the main benchmark suite, add them to the
methods array in main():

Method1D methods[] = {
    // ... existing methods ...

    // Your custom methods
    {"LeftRiemann_n100", left_riemann_sum, 100, "Custom"},
    {"AdaptiveTrap", adaptive_trapezoid, 20, "Adaptive-Custom"},
    {"PeriodicTrap_n100", periodic_trapezoid, 100, "Specialized"},
    {"Hybrid_n50", hybrid_integration, 50, "Hybrid"},
    {"ImportanceSample_1e4", importance_sampling_integration, 10000, "ML-Inspired"},
};

Then recompile and run:
    make clean
    make run
    make visualize

Your methods will appear in all plots and analysis!

*/

// ============================================================================
// TESTING YOUR METHOD BEFORE ADDING TO BENCHMARK
// ============================================================================

/**
 * Standalone test function to verify your method works correctly
 */
void test_custom_method() {
    printf("Testing custom integration methods...\n\n");

    // Test function: ∫₀¹ x² dx = 1/3 = 0.333...
    double (*test_func)(double) = test_x_squared;
    double a = 0.0, b = 1.0;
    double exact = 1.0 / 3.0;

    // Test each method
    double result;

    result = left_riemann_sum(test_func, a, b, 100);
    printf("Left Riemann (n=100): %.10f (error: %.2e)\n",
           result, fabs(result - exact));

    result = adaptive_trapezoid(test_func, a, b, 20);
    printf("Adaptive Trapezoid:   %.10f (error: %.2e)\n",
           result, fabs(result - exact));

    result = hybrid_integration(test_func, a, b, 50);
    printf("Hybrid (n=50):        %.10f (error: %.2e)\n",
           result, fabs(result - exact));

    printf("\nIf errors are reasonable, your methods are ready!\n");
}

// Helper test function
double test_x_squared(double x) {
    return x * x;
}

// ============================================================================
// DESIGN TIPS FOR CUSTOM METHODS
// ============================================================================

/*

1. FUNCTION EVALUATION COUNTING
   Always use the provided function pointer f(), not direct calls.
   The framework automatically counts evaluations.

2. PARAMETER USAGE
   The 'int n' parameter is your method's complexity control.
   Use it consistently (n intervals, n points, etc.)

3. ERROR HANDLING
   Return reasonable values even for edge cases.
   Don't crash on n=0 or a=b.

4. MEMORY MANAGEMENT
   Free any allocated memory before returning.
   Use malloc/free, not VLAs for portability.

5. NUMERICAL STABILITY
   Watch out for:
   - Division by zero
   - Catastrophic cancellation
   - Overflow/underflow

6. DOCUMENTATION
   Comment your method's:
   - Mathematical basis
   - Expected accuracy (order)
   - Best use cases
   - Limitations

7. TESTING
   Test on multiple function types:
   - Polynomial (exact for some orders)
   - Smooth (exponential, sin)
   - Oscillatory
   - Near-singular

8. COMPARISON
   Run against known methods first.
   If your "novel" method performs worse than Simpson's on smooth
   functions, something might be wrong!

*/

// ============================================================================
// EXAMPLE MAIN (for standalone testing)
// ============================================================================

#ifdef STANDALONE_TEST
int main() {
    test_custom_method();
    return 0;
}

// Compile standalone test:
// gcc -DSTANDALONE_TEST custom_methods_example.c -o test_custom -lm
// ./test_custom
#endif
