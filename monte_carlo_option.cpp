#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

using namespace std;

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

int main(){
    cout << "Monte Carlo Option Pricing\n" << endl;
    pair<double, double> params = read_params("params.txt");
    double S0 = params.first;
    double sigma = params.second; // Read parameters from file
    double K = 100.0; // Strike price
    double r = 0.05; // Risk-free interest rate
    double T = 1.0; // Time to maturity in years
    int N = 100000; // Number of simulations

    random_device rd; // Random number generator seed
    mt19937 gen(rd()); // Mersenne Twister generator
    normal_distribution<double> dist(0.0, 1.0); // Normal distribution

    double payoff_sum = 0.0; // Sum of payoffs
    double payoff_sum_sq = 0.0; // Sum of squared payoffs

    for(int i = 0; i < N; ++i){
        double Z = dist(gen);
        double ST = S0 * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z); // Simulated stock price at maturity
        double payoff = max(ST - K, 0.0); // Payoff of the option
        payoff_sum += payoff; // Accumulate payoffs
        payoff_sum_sq += payoff * payoff; // Accumulate squared payoffs
    }


    double mean_payoff = payoff_sum / N; // Mean payoff
    double std_dev = sqrt((payoff_sum_sq / N - mean_payoff * mean_payoff) / (N - 1)); // Standard deviation of payoffs
    double standard_error = std_dev / sqrt(N); // Standard error of the mean
    double z = 1.96; // Z-score for 95% confidence interval
    double lower_bound = exp(-r * T) * (mean_payoff - z * standard_error); // Lower bound of confidence interval
    double upper_bound = exp(-r * T) * (mean_payoff + z * standard_error); // Upper bound of confidence interval
    double option_price = exp(-r * T) * (payoff_sum / N); // Discounted average payoff
    cout << "Option Price: " << option_price << endl; // Output the option price
    cout << "95% Confidence Interval: [" << lower_bound << ", " << upper_bound << "]" << endl; // Output the confidence interval
    cout << "Mean Payoff: " << mean_payoff << endl; // Output the mean payoff
    return 0;
}

