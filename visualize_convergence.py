import subprocess
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless environments
import matplotlib.pyplot as plt

simulation_sizes = [1000, 5000, 10000, 50000, 100000, 200000, 500000, 1000000]

prices = []
lowers = []
uppers = []

for N in simulation_sizes:
    # Run the simulation script and capture the output
    result = subprocess.run(['./monte_carlo_option', str(N)], capture_output=True, text=True)
    
    for line in result.stdout.splitlines():
        if "," in line:
            parts = line.strip().split(",")
            price = float(parts[1])
            lower = float(parts[2])
            upper = float(parts[3])
            prices.append(price)
            lowers.append(lower)
            uppers.append(upper)
            print(f"N={N}: Price={price:.4f}, 95% CI=({lower:.4f}, {upper:.4f})")

plt.figure(figsize=(10, 6))
plt.plot(simulation_sizes, prices, marker='o', label='Estimated Price')
plt.plot(simulation_sizes, prices, marker='o', label='Estimated Price')
plt.fill_between(simulation_sizes, lowers, uppers, color='lightblue', alpha=0.5, label='95% Confidence Interval')
plt.axhline(y=10.45, color='red', linestyle='--', label='Black-Scholes Benchmark (≈10.45)')
plt.xlabel("Number of Simulations")
plt.ylabel("Estimated Option Price")
plt.title("Monte Carlo Convergence (C++ Results)")
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("cpp_convergence_plot.png")
