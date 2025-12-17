# Makefile for Advanced Numerical Integration Benchmark Suite
# Author: Integration Benchmark Project
# Date: 2024

# Compiler and flags
CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra -std=c11
LDFLAGS = -lm -lrt
DEBUG_FLAGS = -g -O0 -DDEBUG

# Profiling flags
PROF_FLAGS = -pg
PERF_FLAGS = -fno-omit-frame-pointer

# Target executable
TARGET = integration_bench
DEBUG_TARGET = integration_bench_debug
PROF_TARGET = integration_bench_prof

# Source files
SOURCES = integration_benchmark.c
HEADERS =

# Python visualization script
PYTHON = python3
VIZ_SCRIPT = visualize_results.py

# Output directories
PLOT_DIR = plots
DATA_DIR = data

# Default target
.PHONY: all
all: $(TARGET)

# Build optimized version
$(TARGET): $(SOURCES) $(HEADERS)
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║ Building Optimized Integration Benchmark Suite                ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)
	@echo "✓ Build complete: $(TARGET)"
	@echo ""

# Build debug version
.PHONY: debug
debug: $(DEBUG_TARGET)

$(DEBUG_TARGET): $(SOURCES) $(HEADERS)
	@echo "Building debug version..."
	$(CC) $(DEBUG_FLAGS) $(SOURCES) -o $(DEBUG_TARGET) $(LDFLAGS)
	@echo "✓ Debug build complete: $(DEBUG_TARGET)"

# Build with profiling support
.PHONY: profile
profile: $(PROF_TARGET)

$(PROF_TARGET): $(SOURCES) $(HEADERS)
	@echo "Building with profiling support..."
	$(CC) $(CFLAGS) $(PROF_FLAGS) $(SOURCES) -o $(PROF_TARGET) $(LDFLAGS)
	@echo "✓ Profile build complete: $(PROF_TARGET)"
	@echo "Run './$(PROF_TARGET)' then 'gprof $(PROF_TARGET) gmon.out > analysis.txt'"

# Build for perf analysis
.PHONY: perf
perf:
	@echo "Building for perf analysis..."
	$(CC) $(CFLAGS) $(PERF_FLAGS) $(SOURCES) -o $(TARGET)_perf $(LDFLAGS)
	@echo "✓ Perf build complete"
	@echo "Run: 'perf record -g ./$(TARGET)_perf && perf report'"

# Run benchmark
.PHONY: run
run: $(TARGET)
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║ Running Integration Benchmark Suite                           ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@mkdir -p $(DATA_DIR)
	./$(TARGET)
	@echo ""
	@echo "✓ Benchmark complete! Results saved to:"
	@echo "  - integration_results.csv"
	@echo "  - convergence_analysis.csv"

# Run and visualize
.PHONY: full
full: clean all run visualize

# Generate visualizations
.PHONY: visualize
visualize:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║ Generating Visualizations                                      ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@mkdir -p $(PLOT_DIR)
	$(PYTHON) $(VIZ_SCRIPT)
	@echo ""

# Performance profiling with perf
.PHONY: perf-stat
perf-stat: $(TARGET)
	@echo "Running performance analysis with perf..."
	perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses ./$(TARGET)

# Cache analysis
.PHONY: cache-analysis
cache-analysis: $(TARGET)
	@echo "Analyzing cache performance..."
	valgrind --tool=cachegrind ./$(TARGET)
	@echo "View results with: cg_annotate cachegrind.out.<pid>"

# Memory leak check
.PHONY: memcheck
memcheck: $(DEBUG_TARGET)
	@echo "Checking for memory leaks..."
	valgrind --leak-check=full --show-leak-kinds=all ./$(DEBUG_TARGET)

# Quick test (fewer runs)
.PHONY: test
test:
	@echo "Running quick test..."
	$(CC) $(CFLAGS) -DNUM_RUNS=3 $(SOURCES) -o $(TARGET)_test $(LDFLAGS)
	./$(TARGET)_test
	@rm -f $(TARGET)_test

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TARGET) $(DEBUG_TARGET) $(PROF_TARGET) $(TARGET)_perf $(TARGET)_test
	rm -f gmon.out analysis.txt
	rm -f cachegrind.out.*
	rm -f *.o
	@echo "✓ Clean complete"

# Clean all generated files
.PHONY: cleanall
cleanall: clean
	@echo "Cleaning all generated files..."
	rm -f *.csv
	rm -rf $(PLOT_DIR)
	rm -rf $(DATA_DIR)
	@echo "✓ All generated files removed"

# Display help
.PHONY: help
help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║ Integration Benchmark Suite - Makefile Help                   ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Available targets:"
	@echo "  make              - Build optimized version"
	@echo "  make run          - Build and run benchmark"
	@echo "  make full         - Clean, build, run, and visualize (complete workflow)"
	@echo "  make visualize    - Generate plots from existing CSV data"
	@echo ""
	@echo "Build variants:"
	@echo "  make debug        - Build with debug symbols"
	@echo "  make profile      - Build with profiling support (gprof)"
	@echo "  make perf         - Build for perf analysis"
	@echo ""
	@echo "Performance analysis:"
	@echo "  make perf-stat    - Run with perf stat (requires Linux perf)"
	@echo "  make cache-analysis - Analyze cache performance (requires valgrind)"
	@echo "  make memcheck     - Check for memory leaks (requires valgrind)"
	@echo ""
	@echo "Utilities:"
	@echo "  make test         - Quick test run (fewer iterations)"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make cleanall     - Remove all generated files"
	@echo "  make help         - Display this help message"
	@echo ""
	@echo "Example workflows:"
	@echo "  1. Full analysis: make full"
	@echo "  2. Re-visualize:  make visualize"
	@echo "  3. Debug run:     make debug && ./integration_bench_debug"
	@echo "  4. Profile:       make profile && ./integration_bench_prof && gprof ..."
	@echo ""

# Phony target to avoid conflicts with files
.PHONY: all clean cleanall run visualize help test debug profile perf perf-stat \
        cache-analysis memcheck full
