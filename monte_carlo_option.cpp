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
    return 0.5 * erfc(-x * sqrt(2));
}

double black_scholes_call(double S0, double K, double r, double sigma, double T) {
    double d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return S0 * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2);
}

enum class OptionType { Call, Put, Digital };

double calculate_payoff(double ST, double K, OptionType type){
    switch(type){
        case OptionType::Call: return max(ST - K, 0.0);
        case OptionType::Put: return max(K - ST, 0.0);
        case OptionType::Digital: return (ST > K) ? 1.0 : 0.0;
        default: return 0.0;
    }
}

void simulate_chunk(int start, int end, double S0, double K, double r,
                    double sigma, double T, OptionType type, double& payoff_sum, 
                    double & payoff_sq_sum){

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 1.0);

    double local_payoff_sum = 0.0;
    double local_payoff_sq_sum = 0.0;

    for (int i = start; i < end; ++i){
        double Z = dist(gen);
        double ST = S0 * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);
        double payoff = calculate_payoff(ST, K, type);
        local_payoff_sum += payoff;
        local_payoff_sq_sum += payoff * payoff;
    }
    payoff_sum = local_payoff_sum;
    payoff_sq_sum = local_payoff_sq_sum;
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
    S0 = stod(token);
    getline(ss, token, ',');
    sigma = stod(token);

    file.close();
    return {S0, sigma};
}

int main(int argc, char* argv[]){
    cerr << "[DEBUG] Entered main()\n";
    if (argc < 2) {
        cerr << "Usage: ./monte_carlo_option <num_simulations> [--csv] [call|put|digital]\n";
        return 1;
    }

    int N = atoi(argv[1]);
    bool csv_output = false;
    OptionType type = OptionType::Call;

    double K = 100.0;  // Will be overwritten if --K= is passed
for (int i = 2; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--csv") csv_output = true;
        else if (arg == "put") type = OptionType::Put;
        else if (arg == "digital") type = OptionType::Digital;
        else if (arg == "call") type = OptionType::Call;
        
    else if (arg.rfind("--K=", 0) == 0) {
        K = stod(arg.substr(4));
    }
        else {
            cerr << "Unknown argument: " << arg << endl;
            return 1;
        }
    }

    pair<double, double> params = read_params("params.txt");
    double S0 = params.first;
    double sigma = params.second;
    double r = 0.05;
    double T = 1.0;

    int num_threads = 4;
    int chunk_size = N / num_threads;

    vector<thread> threads;
    vector<double> payoff_sums(num_threads, 0.0);
    vector<double> payoff_sq_sums(num_threads, 0.0);

    for(int i = 0; i < num_threads; ++i){
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? N : start + chunk_size;
        threads.emplace_back(simulate_chunk, start, end, S0, K, r, sigma, T, type,
                             ref(payoff_sums[i]), ref(payoff_sq_sums[i]));
    }
    for(auto& t : threads){
        t.join();
    }

    double total_payoff_sum = 0.0;
    double total_payoff_sq_sum = 0.0;
    for (int i = 0; i < num_threads; ++i){
        total_payoff_sum += payoff_sums[i];
        total_payoff_sq_sum += payoff_sq_sums[i];
    }
    double mean_payoff = total_payoff_sum / N;
    double std_dev = sqrt((total_payoff_sq_sum / N - mean_payoff * mean_payoff) / (N - 1));
    double standard_error = std_dev / sqrt(N);
    double z = 1.96;
    double lower_bound = exp(-r * T) * (mean_payoff - z * standard_error);
    double upper_bound = exp(-r * T) * (mean_payoff + z * standard_error);
    double discounted_price = exp(-r * T) * (total_payoff_sum / N);

    if (csv_output) {
        cerr << "[DEBUG] CSV mode ON | OptionType: ";
        switch (type) {
            case OptionType::Call: cerr << "Call\n"; break;
            case OptionType::Put: cerr << "Put\n"; break;
            case OptionType::Digital: cerr << "Digital\n"; break;
        }
        cerr << "[DEBUG] Params => S0: " << S0 << ", K: " << K << ", sigma: " << sigma << ", r: " << r << ", T: " << T << ", N: " << N << endl;
        cout << N << "," << discounted_price << "," << lower_bound << "," << upper_bound << endl;
    } else {
        cout << "S0: " << S0 << ", K: " << K << ", sigma: " << sigma
             << ", r: " << r << ", T: " << T << ", N: " << N << endl;
        cout << "Option Type: ";
        switch (type) {
            case OptionType::Call: cout << "Call" << endl; break;
            case OptionType::Put: cout << "Put" << endl; break;
            case OptionType::Digital: cout << "Digital" << endl; break;
        }

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
