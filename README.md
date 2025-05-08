# Monte Carlo Option Pricing Engine

A high-performance Monte Carlo simulation engine for pricing financial options with advanced variance reduction techniques and performance benchmarking.

## Features

- **High-performance C++ core engine** with multi-threading support
- **Multiple option types** - European Call, Put, and Digital options
- **Advanced variance reduction techniques**:
  - Antithetic Variates
  - Control Variates
  - Combined approach
- **Comprehensive benchmarking and analysis**:
  - Performance metrics
  - Convergence analysis
  - Efficiency comparison across techniques
- **Interactive Streamlit dashboard** for real-time option pricing
- **Integration with financial market data** via Yahoo Finance API
- **Visualization tools** for analyzing convergence and performance

## Installation

### Prerequisites

- C++ compiler with C++11 support
- Python 3.7+
- pip package manager

### Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Compilation

Compile the C++ engine:

```bash
g++ -o monte_carlo_option monte_carlo_option.cpp -std=c++11 -pthread -O3
```

## Usage

### Basic Usage

Price a call option with 100,000 simulations:

```bash
./monte_carlo_option 100000 call --K=100
```

### With Variance Reduction

Apply variance reduction techniques:

```bash
./monte_carlo_option 100000 call --K=100 --vr=antithetic
./monte_carlo_option 100000 call --K=100 --vr=control
./monte_carlo_option 100000 call --K=100 --vr=both
```

### Running Benchmarks

Generate performance benchmark reports:

```bash
./monte_carlo_option 100000 --benchmark call --K=100
python visualize_benchmarks.py --run --output=benchmark_report.pdf
```

### Interactive Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

## Performance Optimization

The Monte Carlo engine includes multiple performance optimizations:

1. **Multi-threading** - Parallelizes simulation across CPU cores
2. **Variance Reduction Techniques**:
   - **Antithetic Variates** - Generates negatively correlated paths to reduce variance
   - **Control Variates** - Uses analytical Black-Scholes as control for variance reduction
   - **Combined Approach** - Leverages both techniques simultaneously

### Benchmarking Results

Benchmark results demonstrate significant improvements from variance reduction:

| Technique | Error Reduction | Performance Impact |
|-----------|----------------|-------------------|
| Antithetic | ~50% | +5% execution time |
| Control Variate | ~70% | +10% execution time |
| Combined | ~80% | +15% execution time |

*Actual results will vary based on option parameters and hardware*

## Project Structure

- `monte_carlo_option.cpp` - C++ implementation of the Monte Carlo engine
- `dashboard.py` - Streamlit dashboard for interactive option pricing
- `visualize_benchmarks.py` - Benchmark visualization and analysis tool
- `get_stock_data.py` - Utility for fetching market data
- `visualize_convergence.py` - Tool for analyzing convergence properties

## Implementation Details

### Monte Carlo Simulation

The implementation follows these steps:
1. Generate random paths for the underlying asset
2. Calculate option payoffs for each path
3. Average payoffs and discount to present value
4. Apply variance reduction techniques as specified

### Black-Scholes Implementation

For European options, the implementation includes the Black-Scholes analytical solution for:
- Price comparison and validation
- Control variate basis

### Variance Reduction Techniques

#### Antithetic Variates
For each random normal Z, we use both Z and -Z to generate two negatively correlated paths, reducing the variance of the estimator.

#### Control Variates
We use the analytical Black-Scholes price as a control variate to adjust the Monte Carlo estimator, significantly reducing variance.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Black-Scholes formula implementation based on standard financial mathematics
- Variance reduction techniques inspired by academic literature on financial derivatives pricing
- Dashboard design leverages Streamlit's interactive capabilities 