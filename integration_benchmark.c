#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdint.h>

// ============================================================================
// CONFIGURATION
// ============================================================================
#define MAX_ROMBERG_LEVELS 10
#define NUM_TIMING_RUNS 10
#define ADAPTIVE_MAX_ITERATIONS 10000
#define ADAPTIVE_TOLERANCE 1e-10
#define MC_SAMPLES 100000

// ============================================================================
// GLOBAL INSTRUMENTATION
// ============================================================================
typedef struct {
    long long function_evals;
    long long cache_hits;
    long long recursion_depth;
    double accumulated_error;
} InstrumentationData;

static InstrumentationData g_instr = {0, 0, 0, 0.0};

void reset_instrumentation() {
    g_instr.function_evals = 0;
    g_instr.cache_hits = 0;
    g_instr.recursion_depth = 0;
    g_instr.accumulated_error = 0.0;
}

// ============================================================================
// HIGH-RESOLUTION TIMING
// ============================================================================
typedef struct {
    struct timespec start;
    struct timespec end;
} PrecisionTimer;

void timer_start(PrecisionTimer *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

double timer_stop(PrecisionTimer *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->end);
    return (t->end.tv_sec - t->start.tv_sec) + 
           (t->end.tv_nsec - t->start.tv_nsec) / 1e9;
}

// ============================================================================
// RANDOM NUMBER GENERATION (for Monte Carlo)
// ============================================================================
static uint64_t rng_state = 0x4d595df4d0f33173;

double rand_uniform() {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (rng_state >> 11) * 0x1.0p-53;
}

void seed_rng(uint64_t seed) {
    rng_state = seed;
}

// Van der Corput sequence for quasi-Monte Carlo
double van_der_corput(uint64_t n, uint64_t base) {
    double vdc = 0.0, denom = 1.0;
    while (n) {
        denom *= base;
        vdc += (n % base) / denom;
        n /= base;
    }
    return vdc;
}

// ============================================================================
// TEST FUNCTIONS (Comprehensive Suite)
// ============================================================================

// 1. Smooth exponential
double f_exp(double x) {
    g_instr.function_evals++;
    return exp(x);
}
double exact_exp(double a, double b) { return exp(b) - exp(a); }

// 2. Polynomial x^7
double f_poly7(double x) {
    g_instr.function_evals++;
    return pow(x, 7);
}
double exact_poly7(double a, double b) { return (pow(b,8) - pow(a,8)) / 8.0; }

// 3. Low frequency oscillatory
double f_sin10(double x) {
    g_instr.function_evals++;
    return sin(10.0 * x);
}
double exact_sin10(double a, double b) { return (cos(10.0*a) - cos(10.0*b)) / 10.0; }

// 4. Gaussian
double f_gaussian(double x) {
    g_instr.function_evals++;
    return exp(-x * x);
}
double exact_gaussian() { return sqrt(M_PI) * erf(3.0); }

// 5. Runge's function (pathological for high-order polynomials)
double f_runge(double x) {
    g_instr.function_evals++;
    return 1.0 / (1.0 + 25.0 * x * x);
}
double exact_runge() { return 2.0 * atan(5.0) / 5.0; }

// 6. Weak singularity
double f_weak_sing(double x) {
    g_instr.function_evals++;
    return pow(x + 1e-10, -0.3);
}
double exact_weak_sing() { return (pow(1.0 + 1e-10, 0.7) - pow(1e-10, 0.7)) / 0.7; }

// 7. High frequency oscillatory
double f_cos50pi(double x) {
    g_instr.function_evals++;
    return cos(50.0 * M_PI * x);
}
double exact_cos50pi() { return 0.0; }

// 8. Simple sine
double f_sin(double x) {
    g_instr.function_evals++;
    return sin(x);
}
double exact_sin_0_pi() { return 2.0; }

// 9. Product of trig functions
double f_sin_cos(double x) {
    g_instr.function_evals++;
    return sin(x) * cos(x);
}
double exact_sin_cos() { return 0.5; }

// 10. Rational function
double f_rational(double x) {
    g_instr.function_evals++;
    return 1.0 / (1.0 + x*x);
}
double exact_rational() { return M_PI / 4.0; }

// 11. Log function (singular at 0)
double f_log(double x) {
    g_instr.function_evals++;
    return log(x + 1e-10);
}
double exact_log() { return -1.0; }

// 12. Discontinuous (absolute value)
double f_abs_sin(double x) {
    g_instr.function_evals++;
    return fabs(sin(x));
}
double exact_abs_sin_0_2pi() { return 4.0; }

// 13. Highly peaked function
double f_peaked(double x) {
    g_instr.function_evals++;
    return exp(50.0 * (x - 0.5) * (x - 0.5));
}

// 14. Bessel-like oscillation
double f_bessel_like(double x) {
    g_instr.function_evals++;
    if (x < 1e-10) return 1.0;
    return sin(20.0 * x) / x;
}

// 15. Multi-modal
double f_multimodal(double x) {
    g_instr.function_evals++;
    return sin(10*x) * exp(-x) + cos(5*x);
}

// ============================================================================
// 2D TEST FUNCTIONS
// ============================================================================

double f2d_gaussian(double x, double y) {
    g_instr.function_evals++;
    return exp(-(x*x + y*y));
}

double f2d_polynomial(double x, double y) {
    g_instr.function_evals++;
    return x*x + y*y + x*y;
}

double f2d_oscillatory(double x, double y) {
    g_instr.function_evals++;
    return sin(5*x) * cos(5*y);
}

// ============================================================================
// BASIC INTEGRATION METHODS
// ============================================================================

double midpoint_rule(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += f(a + (i + 0.5) * h);
    }
    return h * sum;
}

double trapezium_rule(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.5 * (f(a) + f(b));
    for (int i = 1; i < n; i++) {
        sum += f(a + i * h);
    }
    return h * sum;
}

double simpson_rule(double (*f)(double), double a, double b, int n) {
    if (n % 2 != 0) n++;
    double h = (b - a) / n;
    double sum = f(a) + f(b);
    
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        sum += (i % 2 == 0) ? 2.0 * f(x) : 4.0 * f(x);
    }
    return (h / 3.0) * sum;
}

double simpson_38_rule(double (*f)(double), double a, double b, int n) {
    while (n % 3 != 0) n++;
    double h = (b - a) / n;
    double sum = f(a) + f(b);
    
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        sum += (i % 3 == 0) ? 2.0 * f(x) : 3.0 * f(x);
    }
    return (3.0 * h / 8.0) * sum;
}

double boole_rule(double (*f)(double), double a, double b, int n) {
    while (n % 4 != 0) n++;
    double h = (b - a) / n;
    double sum = 7.0 * (f(a) + f(b));
    
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        int mod = i % 4;
        if (mod == 0) sum += 14.0 * f(x);
        else if (mod == 1 || mod == 3) sum += 32.0 * f(x);
        else sum += 12.0 * f(x);
    }
    return (2.0 * h / 45.0) * sum;
}

// ============================================================================
// GAUSSIAN QUADRATURE (Multiple Orders)
// ============================================================================

double gauss_legendre_2(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / n, sum = 0.0;
    double w = 1.0, x1 = -1.0/sqrt(3.0), x2 = 1.0/sqrt(3.0);
    
    for (int i = 0; i < n; i++) {
        double mid = a + (i + 0.5) * h;
        double half = h / 2.0;
        sum += w * (f(mid + half * x1) + f(mid + half * x2));
    }
    return (h / 2.0) * sum;
}

double gauss_legendre_3(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / n, sum = 0.0;
    double w[] = {5.0/9.0, 8.0/9.0, 5.0/9.0};
    double x[] = {-sqrt(3.0/5.0), 0.0, sqrt(3.0/5.0)};
    
    for (int i = 0; i < n; i++) {
        double mid = a + (i + 0.5) * h;
        double half = h / 2.0;
        for (int j = 0; j < 3; j++) {
            sum += w[j] * f(mid + half * x[j]);
        }
    }
    return (h / 2.0) * sum;
}

double gauss_legendre_5(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / n, sum = 0.0;
    double w[] = {
        0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
        0.4786286704993665, 0.2369268850561891
    };
    double x[] = {
        -0.9061798459386640, -0.5384693101056831, 0.0,
        0.5384693101056831, 0.9061798459386640
    };
    
    for (int i = 0; i < n; i++) {
        double mid = a + (i + 0.5) * h;
        double half = h / 2.0;
        for (int j = 0; j < 5; j++) {
            sum += w[j] * f(mid + half * x[j]);
        }
    }
    return (h / 2.0) * sum;
}

double gauss_legendre_7(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / n, sum = 0.0;
    double w[] = {
        0.1294849661688697, 0.2797053914892767, 0.3818300505051189, 0.4179591836734694,
        0.3818300505051189, 0.2797053914892767, 0.1294849661688697
    };
    double x[] = {
        -0.9491079123427585, -0.7415311855993944, -0.4058451513773972, 0.0,
        0.4058451513773972, 0.7415311855993944, 0.9491079123427585
    };
    
    for (int i = 0; i < n; i++) {
        double mid = a + (i + 0.5) * h;
        double half = h / 2.0;
        for (int j = 0; j < 7; j++) {
            sum += w[j] * f(mid + half * x[j]);
        }
    }
    return (h / 2.0) * sum;
}

double gauss_legendre_10(double (*f)(double), double a, double b, int n) {
    double h = (b - a) / n, sum = 0.0;
    double w[] = {
        0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963,
        0.2955242247147529, 0.2955242247147529, 0.2692667193099963, 0.2190863625159820,
        0.1494513491505806, 0.0666713443086881
    };
    double x[] = {
        -0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472,
        -0.1488743389816312, 0.1488743389816312, 0.4333953941292472, 0.6794095682990244,
        0.8650633666889845, 0.9739065285171717
    };
    
    for (int i = 0; i < n; i++) {
        double mid = a + (i + 0.5) * h;
        double half = h / 2.0;
        for (int j = 0; j < 10; j++) {
            sum += w[j] * f(mid + half * x[j]);
        }
    }
    return (h / 2.0) * sum;
}

// ============================================================================
// ROMBERG INTEGRATION
// ============================================================================

double romberg_integration(double (*f)(double), double a, double b, int max_k) {
    double R[MAX_ROMBERG_LEVELS][MAX_ROMBERG_LEVELS];
    double h = b - a;
    R[0][0] = 0.5 * h * (f(a) + f(b));
    
    for (int i = 1; i < max_k && i < MAX_ROMBERG_LEVELS; i++) {
        h /= 2.0;
        double sum = 0.0;
        int num_new = (1 << (i-1));
        for (int k = 1; k <= num_new; k++) {
            sum += f(a + (2*k - 1) * h);
        }
        R[i][0] = 0.5 * R[i-1][0] + h * sum;
        
        for (int j = 1; j <= i; j++) {
            double factor = pow(4.0, j);
            R[i][j] = (factor * R[i][j-1] - R[i-1][j-1]) / (factor - 1.0);
        }
    }
    return R[max_k-1][max_k-1];
}

// ============================================================================
// ADAPTIVE SIMPSON'S METHOD
// ============================================================================

double adaptive_simpson_recursive(double (*f)(double), double a, double b, 
                                   double tol, double S, double fa, double fb, double fc,
                                   int depth) {
    if (depth > g_instr.recursion_depth) {
        g_instr.recursion_depth = depth;
    }
    
    double c = (a + b) / 2.0, h = b - a;
    double d = (a + c) / 2.0, e = (c + b) / 2.0;
    double fd = f(d), fe = f(e);
    
    double Sleft = (h/12.0) * (fa + 4.0*fd + fc);
    double Sright = (h/12.0) * (fc + 4.0*fe + fb);
    double S2 = Sleft + Sright;
    
    if (depth >= ADAPTIVE_MAX_ITERATIONS || fabs(S2 - S) <= 15.0*tol) {
        return S2 + (S2 - S) / 15.0;
    }
    
    return adaptive_simpson_recursive(f, a, c, tol/2.0, Sleft, fa, fc, fd, depth+1) +
           adaptive_simpson_recursive(f, c, b, tol/2.0, Sright, fc, fb, fe, depth+1);
}

double adaptive_simpson(double (*f)(double), double a, double b, double tol) {
    g_instr.recursion_depth = 0;
    double c = (a + b) / 2.0, h = b - a;
    double fa = f(a), fb = f(b), fc = f(c);
    double S = (h/6.0) * (fa + 4.0*fc + fb);
    return adaptive_simpson_recursive(f, a, b, tol, S, fa, fb, fc, 0);
}

// ============================================================================
// CLENSHAW-CURTIS QUADRATURE
// ============================================================================

double clenshaw_curtis(double (*f)(double), double a, double b, int n) {
    if (n < 2) n = 2;
    double sum = 0.0;
    double* fvals = malloc((n+1) * sizeof(double));
    
    // Evaluate at Chebyshev nodes
    for (int j = 0; j <= n; j++) {
        double theta = j * M_PI / n;
        double x = cos(theta);
        double t = 0.5 * ((b - a) * x + (b + a));
        fvals[j] = f(t);
    }
    
    // Compute weights and sum
    for (int j = 0; j <= n; j++) {
        double w = 1.0;
        if (j == 0 || j == n) w = 0.5;
        
        double theta_j = j * M_PI / n;
        for (int k = 1; k < n/2; k++) {
            w -= 2.0 * cos(2*k*theta_j) / (4.0*k*k - 1.0);
        }
        if (n % 2 == 0) {
            w -= cos(n*theta_j) / (n*n - 1.0);
        }
        sum += w * fvals[j];
    }
    
    free(fvals);
    return (b - a) * sum / n;
}

// ============================================================================
// TANH-SINH (DOUBLE EXPONENTIAL) QUADRATURE
// ============================================================================

double tanh_sinh(double (*f)(double), double a, double b, int level) {
    double h = pow(2.0, -level);
    double sum = 0.0;
    int n_max = (int)(10.0 / h);
    
    for (int k = -n_max; k <= n_max; k++) {
        double t = k * h;
        double u = tanh(M_PI * sinh(t) / 2.0);
        double du = M_PI * cosh(t) / (2.0 * pow(cosh(M_PI * sinh(t) / 2.0), 2));
        
        if (fabs(u) < 1.0 - 1e-15) {
            double x = 0.5 * ((b - a) * u + (b + a));
            sum += f(x) * du;
        }
    }
    
    return (b - a) * h * sum / 2.0;
}

// ============================================================================
// MONTE CARLO INTEGRATION
// ============================================================================

double monte_carlo_uniform(double (*f)(double), double a, double b, int n_samples) {
    double sum = 0.0;
    for (int i = 0; i < n_samples; i++) {
        double x = a + (b - a) * rand_uniform();
        sum += f(x);
    }
    return (b - a) * sum / n_samples;
}

double monte_carlo_quasi(double (*f)(double), double a, double b, int n_samples) {
    double sum = 0.0;
    for (int i = 0; i < n_samples; i++) {
        double x = a + (b - a) * van_der_corput(i, 2);
        sum += f(x);
    }
    return (b - a) * sum / n_samples;
}

// ============================================================================
// 2D INTEGRATION METHODS
// ============================================================================

double integrate_2d_product(double (*f)(double, double), 
                            double ax, double bx, double ay, double by,
                            int nx, int ny) {
    double hx = (bx - ax) / nx;
    double hy = (by - ay) / ny;
    double sum = 0.0;
    
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double x = ax + (i + 0.5) * hx;
            double y = ay + (j + 0.5) * hy;
            sum += f(x, y);
        }
    }
    return hx * hy * sum;
}

double monte_carlo_2d(double (*f)(double, double),
                     double ax, double bx, double ay, double by,
                     int n_samples) {
    double sum = 0.0;
    for (int i = 0; i < n_samples; i++) {
        double x = ax + (bx - ax) * rand_uniform();
        double y = ay + (by - ay) * rand_uniform();
        sum += f(x, y);
    }
    return (bx - ax) * (by - ay) * sum / n_samples;
}

// ============================================================================
// BENCHMARK STRUCTURE
// ============================================================================

typedef struct {
    char name[100];
    double (*func)(double);
    double a, b;
    double exact;
    char difficulty[20];
} TestProblem;

typedef struct {
    char name[50];
    double (*method)(double (*)(double), double, double, int);
    int n_points;
    char category[30];
} Method1D;

typedef struct {
    double result;
    double error;
    double rel_error;
    long long n_eval;
    double time_mean;
    double time_std;
    long long recursion_depth;
    double fom_eval;
    double fom_time;
    int convergence_order;  // Estimated
} BenchmarkResult;

// ============================================================================
// CSV EXPORT
// ============================================================================

void export_to_csv(const char* filename, TestProblem* problems, int n_problems,
                   Method1D* methods, int n_methods, BenchmarkResult** results) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open %s\n", filename);
        return;
    }
    
    // Header
    fprintf(fp, "Problem,Method,Category,N_Points,Result,Exact,Error,RelError,");
    fprintf(fp, "N_Eval,Time_Mean_us,Time_Std_us,RecursionDepth,FOM_Eval,FOM_Time,");
    fprintf(fp, "Difficulty\n");
    
    // Data rows
    for (int p = 0; p < n_problems; p++) {
        for (int m = 0; m < n_methods; m++) {
            BenchmarkResult* r = &results[p][m];
            fprintf(fp, "%s,%s,%s,%d,%.15e,%.15e,%.3e,%.3e,",
                    problems[p].name, methods[m].name, methods[m].category,
                    methods[m].n_points, r->result, problems[p].exact,
                    r->error, r->rel_error);
            fprintf(fp, "%lld,%.6f,%.6f,%lld,%.6e,%.6e,%s\n",
                    r->n_eval, r->time_mean * 1e6, r->time_std * 1e6,
                    r->recursion_depth, r->fom_eval, r->fom_time,
                    problems[p].difficulty);
        }
    }
    
    fclose(fp);
    printf("✓ Results exported to %s\n", filename);
}

// ============================================================================
// CONVERGENCE RATE ANALYSIS
// ============================================================================

void analyze_convergence(double (*f)(double), double a, double b, double exact,
                        double (*method)(double (*)(double), double, double, int),
                        const char* method_name, FILE* conv_file) {
    int n_values[] = {10, 20, 40, 80, 160, 320, 640, 1280};
    int n_tests = sizeof(n_values) / sizeof(n_values[0]);
    
    fprintf(conv_file, "%s", method_name);
    for (int i = 0; i < n_tests; i++) {
        reset_instrumentation();
        double result = method(f, a, b, n_values[i]);
        double error = fabs(result - exact);
        fprintf(conv_file, ",%.15e", error);
    }
    fprintf(conv_file, "\n");
}

// ============================================================================
// MAIN BENCHMARK FUNCTION
// ============================================================================

BenchmarkResult benchmark_method_1d(Method1D* method, TestProblem* problem) {
    BenchmarkResult res = {0};
    PrecisionTimer t;
    double times[NUM_TIMING_RUNS];
    double results[NUM_TIMING_RUNS];
    
    // Multiple runs for statistical significance
    for (int run = 0; run < NUM_TIMING_RUNS; run++) {
        reset_instrumentation();
        timer_start(&t);
        
        results[run] = method->method(problem->func, problem->a, problem->b, method->n_points);
        
        times[run] = timer_stop(&t);
        
        if (run == 0) {
            res.n_eval = g_instr.function_evals;
            res.recursion_depth = g_instr.recursion_depth;
        }
    }
    
    // Compute statistics
    double time_sum = 0.0, time_sum_sq = 0.0;
    double result_sum = 0.0;
    for (int i = 0; i < NUM_TIMING_RUNS; i++) {
        time_sum += times[i];
        time_sum_sq += times[i] * times[i];
        result_sum += results[i];
    }
    
    res.time_mean = time_sum / NUM_TIMING_RUNS;
    double time_var = (time_sum_sq / NUM_TIMING_RUNS) - (res.time_mean * res.time_mean);
    res.time_std = sqrt(fmax(0.0, time_var));
    
    res.result = result_sum / NUM_TIMING_RUNS;
    res.error = fabs(res.result - problem->exact);
    res.rel_error = (problem->exact != 0.0) ? res.error / fabs(problem->exact) : res.error;
    
    // FOM calculations
    if (res.error > 1e-15 && res.n_eval > 0) {
        res.fom_eval = 1.0 / (res.error * res.error * res.n_eval);
        res.fom_time = 1.0 / (res.error * res.error * res.time_mean);
    } else {
        res.fom_eval = (res.error < 1e-15) ? 1e20 : 0.0;
        res.fom_time = (res.error < 1e-15) ? 1e20 : 0.0;
    }
    
    return res;
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================

int main(int argc, char** argv) {
    seed_rng(12345);
    
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║   ADVANCED NUMERICAL INTEGRATION BENCHMARK SUITE v2.0          ║\n");
    printf("║   Comprehensive Figure of Merit Analysis                       ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    // Define test problems
    TestProblem problems[] = {
        {"Exponential", f_exp, 0.0, 1.0, exact_exp(0, 1), "Easy"},
        {"Polynomial_x7", f_poly7, 0.0, 1.0, exact_poly7(0, 1), "Easy"},
        {"Sin_10x", f_sin10, 0.0, 2*M_PI, exact_sin10(0, 2*M_PI), "Medium"},
        {"Gaussian", f_gaussian, -3.0, 3.0, exact_gaussian(), "Easy"},
        {"Runge_Function", f_runge, 0.0, 1.0, exact_runge(), "Hard"},
        {"Weak_Singular", f_weak_sing, 0.0, 1.0, exact_weak_sing(), "Hard"},
        {"Cos_50pi", f_cos50pi, 0.0, 1.0, exact_cos50pi(), "Very_Hard"},
        {"Sine", f_sin, 0.0, M_PI, exact_sin_0_pi(), "Easy"},
        {"Sin_times_Cos", f_sin_cos, 0.0, M_PI, exact_sin_cos(), "Easy"},
        {"Rational_1over1+x2", f_rational, 0.0, 1.0, exact_rational(), "Medium"}
    };
    int n_problems = sizeof(problems) / sizeof(problems[0]);
    
    // Define methods
    Method1D methods[] = {
        {"Midpoint_n50", midpoint_rule, 50, "Newton-Cotes"},
        {"Midpoint_n200", midpoint_rule, 200, "Newton-Cotes"},
        {"Trapezium_n50", trapezium_rule, 50, "Newton-Cotes"},
        {"Trapezium_n200", trapezium_rule, 200, "Newton-Cotes"},
        {"Simpson_n50", simpson_rule, 50, "Newton-Cotes"},
        {"Simpson_n200", simpson_rule, 200, "Newton-Cotes"},
        {"Simpson38_n51", simpson_38_rule, 51, "Newton-Cotes"},
        {"Boole_n100", boole_rule, 100, "Newton-Cotes"},
        {"Gauss2_n25", gauss_legendre_2, 25, "Gaussian"},
        {"Gauss3_n25", gauss_legendre_3, 25, "Gaussian"},
        {"Gauss5_n25", gauss_legendre_5, 25, "Gaussian"},
        {"Gauss7_n25", gauss_legendre_7, 25, "Gaussian"},
        {"Gauss10_n25", gauss_legendre_10, 25, "Gaussian"},
        {"Gauss5_n50", gauss_legendre_5, 50, "Gaussian"},
        {"Romberg_k8", romberg_integration, 8, "Extrapolation"},
        {"ClenshawCurtis_n50", clenshaw_curtis, 50, "Chebyshev"},
        {"TanhSinh_lv5", tanh_sinh, 5, "DoubleExp"},
        {"MonteCarlo_1e4", monte_carlo_uniform, 10000, "Stochastic"},
        {"QuasiMC_1e4", monte_carlo_quasi, 10000, "Quasi-Random"}
    };
    int n_methods = sizeof(methods) / sizeof(methods[0]);
    
    // Allocate results matrix
    BenchmarkResult** results = malloc(n_problems * sizeof(BenchmarkResult*));
    for (int i = 0; i < n_problems; i++) {
        results[i] = malloc(n_methods * sizeof(BenchmarkResult));
    }
    
    // Run benchmarks
    printf("Running comprehensive benchmarks...\n");
    printf("(Averaging over %d runs per method)\n\n", NUM_TIMING_RUNS);
    
    int total_tests = n_problems * n_methods;
    int completed = 0;
    
    for (int p = 0; p < n_problems; p++) {
        printf("\n[%d/%d] Problem: %s (Difficulty: %s)\n", 
               p+1, n_problems, problems[p].name, problems[p].difficulty);
        printf("  Domain: [%.2f, %.2f], Exact: %.10e\n", 
               problems[p].a, problems[p].b, problems[p].exact);
        
        for (int m = 0; m < n_methods; m++) {
            results[p][m] = benchmark_method_1d(&methods[m], &problems[p]);
            completed++;
            
            // Progress indicator
            if (completed % 10 == 0 || completed == total_tests) {
                printf("  Progress: %d/%d (%.1f%%)\r", 
                       completed, total_tests, 100.0 * completed / total_tests);
                fflush(stdout);
            }
        }
    }
    
    printf("\n\n✓ Benchmarking complete!\n\n");
    
    // Export results
    export_to_csv("integration_results.csv", problems, n_problems, methods, n_methods, results);
    
    // Convergence analysis
    printf("\nPerforming convergence rate analysis...\n");
    FILE* conv_file = fopen("convergence_analysis.csv", "w");
    fprintf(conv_file, "Method,n10,n20,n40,n80,n160,n320,n640,n1280\n");
    
    double (*conv_methods[])(double (*)(double), double, double, int) = {
        midpoint_rule, trapezium_rule, simpson_rule, 
        gauss_legendre_3, gauss_legendre_5, clenshaw_curtis
    };
    const char* conv_names[] = {
        "Midpoint", "Trapezium", "Simpson", "Gauss3", "Gauss5", "ClenshawCurtis"
    };
    
    for (int i = 0; i < 6; i++) {
        analyze_convergence(f_exp, 0.0, 1.0, exact_exp(0, 1), 
                          conv_methods[i], conv_names[i], conv_file);
    }
    fclose(conv_file);
    printf("✓ Convergence analysis exported to convergence_analysis.csv\n");
    
    // Summary statistics
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║ SUMMARY: TOP 5 METHODS BY OVERALL FOM (averaged across all)   ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    
    double method_avg_fom[n_methods];
    for (int m = 0; m < n_methods; m++) {
        double sum = 0.0;
        int count = 0;
        for (int p = 0; p < n_problems; p++) {
            if (results[p][m].fom_eval > 0 && results[p][m].fom_eval < 1e15) {
                sum += log10(results[p][m].fom_eval);
                count++;
            }
        }
        method_avg_fom[m] = (count > 0) ? sum / count : -1e10;
    }
    
    // Sort by FOM
    int indices[n_methods];
    for (int i = 0; i < n_methods; i++) indices[i] = i;
    for (int i = 0; i < n_methods-1; i++) {
        for (int j = i+1; j < n_methods; j++) {
            if (method_avg_fom[indices[j]] > method_avg_fom[indices[i]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }
    
    printf("\n%-25s %15s %15s\n", "Method", "Avg_log10(FOM)", "Category");
    printf("─────────────────────────────────────────────────────────────────\n");
    for (int i = 0; i < 5 && i < n_methods; i++) {
        int idx = indices[i];
        printf("%-25s %15.2f %15s\n", 
               methods[idx].name, method_avg_fom[idx], methods[idx].category);
    }
    
    printf("\n✓ Analysis complete! Check CSV files for detailed results.\n");
    printf("  Run the Python visualization script to generate plots.\n\n");
    
    // Cleanup
    for (int i = 0; i < n_problems; i++) {
        free(results[i]);
    }
    free(results);
    
    return 0;
}
