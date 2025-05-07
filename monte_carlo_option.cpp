#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <thread>
#include <cstdlib>

using namespace std;

double normal_cdf(double x) {
    return 0.5 * erfc(-x * sqrt(2)); // Cumulative distribution function for standard normal distribution
}

double black_scholes_call(double S0, double K, double r, double sigma,
                           double T) {
    double d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T)); // d1 term
    double d2 = d1 - sigma * sqrt(T); // d2 term
    return S0 * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2); // Call option price
}

enum class OptionType { Call, Put, Digital };

double calculate_payoff(double ST, double K, OptionType type){
    switch(type){
        case OptionType::Call:
            return max(ST - K, 0.0); // Call option payoff
        case OptionType::Put:
            return max(K - ST, 0.0); // Put option payoff
        case OptionType::Digital:
            return (ST > K) ? 1.0 : 0.0; // Digital option payoff
        default:
            return 0.0; // Default case
    }
}

void simulate_chunk(int start, int end, double S0, double K, double r,
                    double sigma, double T, OptionType type, double& payoff_sum, 
                    double & payoff_sq_sum){

    random_device rd; // Random number generator seed
    mt19937 gen(rd()); // Mersenne Twister generator
    normal_distribution<double> dist(0.0, 1.0); // Normal distribution

    double local_payoff_sum = 0.0; // Local sum of payoffs
    double local_payoff_sq_sum = 0.0; // Local sum of squared payoffs

    for (int i = start; i < end; ++i){
        double Z = dist(gen);
        double ST = S0 * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z); // Simulated stock price at maturity
        double payoff = calculate_payoff(ST, K, type); // Payoff of the option
        local_payoff_sum += payoff; // Accumulate payoffs
        local_payoff_sq_sum += payoff * payoff; // Accumulate squared payoffs
    }
    payoff_sum = local_payoff_sum;
    payoff_sq_sum = local_payoff_sq_sum; // Store local results in reference variables
}

pair<double, double> read_params(const string& filename){
    ifstream file(filename);
    if(!file){
        cerr<< "Error opening file: " << filename << endl;
        exit(1);
    }

    string line;
    getline(file, line);
    stringstream ss(line);
    string token;

    double S0, sigma;

    getline(ss, token, ',');
    S0 = stod(token); // Initial stock price
    getline(ss, token, ',');
    sigma = stod(token); // Volatility

    file.close();
    return {S0, sigma};
}

int main(int argc, char* argv[]){

    bool csv_output = false;
    if (argc >= 3 && string(argv[2]) == "--csv") {
        csv_output = true;
    }

    OptionType type = OptionType::Call; // Default option type

    if (argc < 2) {
        cerr << "Usage: ./monte_carlo_option <num_simulations>\n";
        return 1;
    }

    int N = atoi(argv[1]);  // Read number of simulations from CLI
    cout << "Monte Carlo Option Pricing\n" << endl;
    pair<double, double> params = read_params("params.txt");
    double S0 = params.first;
    double sigma = params.second; // Read parameters from file
    double K = 100.0; // Strike price
    double r = 0.05; // Risk-free interest rate
    double T = 1.0; // Time to maturity in years

    int num_threads = 4;
    int chunk_size = N / num_threads; // Size of each chunk

    vector<thread> threads;
    vector<double> payoff_sums(num_threads, 0.0); // Vector to store local sums of payoffs
    vector<double> payoff_sq_sums(num_threads, 0.0); // Vector to store local sums of squared payoffs

    for(int i = 0; i < num_threads; ++i){
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? N : start + chunk_size; // Handle last chunk
        threads.emplace_back(simulate_chunk, start, end, S0, K, r, sigma, T, type,
                             ref(payoff_sums[i]), ref(payoff_sq_sums[i]));
    }
    for(auto& t : threads){
        t.join(); // Wait for all threads to finish
    }

    double total_payoff_sum = 0.0; // Total sum of payoffs
    double total_payoff_sq_sum = 0.0; // Total sum of squared payoffs
    for (int i = 0; i < num_threads; ++i){
        total_payoff_sum += payoff_sums[i]; // Accumulate local sums
        total_payoff_sq_sum += payoff_sq_sums[i]; // Accumulate local squared sums
    }
    double mean_payoff = total_payoff_sum / N; // Mean payoff
    double std_dev = sqrt((total_payoff_sq_sum / N - mean_payoff * mean_payoff) / (N - 1)); // Standard deviation of payoffs
    double standard_error = std_dev / sqrt(N); // Standard error of the mean
    double z = 1.96; // Z-score for 95% confidence interval
    double lower_bound = exp(-r * T) * (mean_payoff - z * standard_error); // Lower bound of confidence interval
    double upper_bound = exp(-r * T) * (mean_payoff + z * standard_error); // Upper bound of confidence interval
    double discounted_price = exp(-r * T) * (total_payoff_sum / N); // Discounted average payoff

    if (csv_output) {
        cout << N << "," << discounted_price << "," << lower_bound << "," << upper_bound << endl;
    } else {
        if (type == OptionType::Call) {
            double analytical_call = black_scholes_call(S0, K, r, sigma, T);
            cout << "Analytical Black-Scholes Call Price: " << analytical_call << endl;
            cout << "Simulated Call Price (Monte Carlo): " << discounted_price << endl;
            cout << "Absolute Error: " << abs(discounted_price - analytical_call) << endl;
        }
        cout << "Estimated Option Price: " << discounted_price << endl;
        cout << "95% Confidence Interval: [" << lower_bound << ", " << upper_bound << "]" << endl;
    }
    return 0;
    
}

