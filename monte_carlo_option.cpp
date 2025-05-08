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
#include <chrono>
#include <iomanip>

using namespace std;

// Variance reduction technique enum
enum class VarianceReduction { None, Antithetic, ControlVariate, Both };

double normal_cdf(double x) {
    return 0.5 * erfc(-x * sqrt(2));
}

double black_scholes_call(double S0, double K, double r, double sigma, double T) {
    double d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return S0 * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2);
}

double black_scholes_put(double S0, double K, double r, double sigma, double T) {
    double d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return K * exp(-r * T) * normal_cdf(-d2) - S0 * normal_cdf(-d1);
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
                    double& payoff_sq_sum, VarianceReduction vr){

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 1.0);

    double local_payoff_sum = 0.0;
    double local_payoff_sq_sum = 0.0;
    
    // For control variate technique
    double analytical_price = 0.0;
    if (vr == VarianceReduction::ControlVariate || vr == VarianceReduction::Both) {
        if (type == OptionType::Call) {
            analytical_price = black_scholes_call(S0, K, r, sigma, T);
        } else if (type == OptionType::Put) {
            analytical_price = black_scholes_put(S0, K, r, sigma, T);
        }
    }

    int effective_simulations = end - start;
    if (vr == VarianceReduction::Antithetic || vr == VarianceReduction::Both) {
        effective_simulations /= 2; // We'll do half the simulations with antithetic pairs
    }

    for (int i = 0; i < effective_simulations; ++i){
        double Z = dist(gen);
        
        // Standard path
        double ST = S0 * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);
        double payoff = calculate_payoff(ST, K, type);
        
        if (vr == VarianceReduction::Antithetic || vr == VarianceReduction::Both) {
            // Antithetic path (using -Z)
            double ST_anti = S0 * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * (-Z));
            double payoff_anti = calculate_payoff(ST_anti, K, type);
            
            // Average the two payoffs
            payoff = (payoff + payoff_anti) / 2.0;
        }
        
        if ((vr == VarianceReduction::ControlVariate || vr == VarianceReduction::Both) && 
            (type == OptionType::Call || type == OptionType::Put)) {
            // Only apply control variate for call or put (not digital)
            // Use Black-Scholes as control variate
            double control_payoff = 0.0;
            
            if (type == OptionType::Call) {
                control_payoff = max(ST - K, 0.0);
            } else if (type == OptionType::Put) {
                control_payoff = max(K - ST, 0.0);
            }
            
            // Apply control variate adjustment with beta = 1.0 (simplified)
            double adjusted_payoff = payoff - (control_payoff - analytical_price * exp(r * T));
            payoff = adjusted_payoff;
        }
        
        local_payoff_sum += payoff;
        local_payoff_sq_sum += payoff * payoff;
    }
    
    // Scale back to the original simulation count if using antithetic
    if (vr == VarianceReduction::Antithetic || vr == VarianceReduction::Both) {
        local_payoff_sum *= 2;
        local_payoff_sq_sum *= 2;
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

struct BenchmarkResult {
    double price;
    double lower_bound;
    double upper_bound;
    double execution_time_ms;
    double abs_error;
    double std_dev;
};

BenchmarkResult run_simulation(int N, double S0, double K, double r, double sigma, double T, 
                             OptionType type, VarianceReduction vr, int num_threads = 4) {
    auto start_time = chrono::high_resolution_clock::now();
    
    int chunk_size = N / num_threads;

    vector<thread> threads;
    vector<double> payoff_sums(num_threads, 0.0);
    vector<double> payoff_sq_sums(num_threads, 0.0);

    for(int i = 0; i < num_threads; ++i){
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? N : start + chunk_size;
        threads.emplace_back(simulate_chunk, start, end, S0, K, r, sigma, T, type,
                             ref(payoff_sums[i]), ref(payoff_sq_sums[i]), vr);
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
    
    auto end_time = chrono::high_resolution_clock::now();
    double execution_time_ms = chrono::duration<double, milli>(end_time - start_time).count();
    
    // Calculate absolute error if analytical solution is available
    double abs_error = 0.0;
    if (type == OptionType::Call) {
        double analytical_call = black_scholes_call(S0, K, r, sigma, T);
        abs_error = abs(discounted_price - analytical_call);
    } else if (type == OptionType::Put) {
        double analytical_put = black_scholes_put(S0, K, r, sigma, T);
        abs_error = abs(discounted_price - analytical_put);
    }
    
    return {
        discounted_price,
        lower_bound,
        upper_bound,
        execution_time_ms,
        abs_error,
        std_dev
    };
}

int main(int argc, char* argv[]){
    cerr << "[DEBUG] Entered main()\n";
    if (argc < 2) {
        cerr << "Usage: ./monte_carlo_option <num_simulations> [--csv] [call|put|digital] [--vr=none|antithetic|control|both]\n";
        return 1;
    }

    int N = atoi(argv[1]);
    bool csv_output = false;
    bool benchmark_mode = false;
    OptionType type = OptionType::Call;
    VarianceReduction vr = VarianceReduction::None;

    double K = 100.0;  // Will be overwritten if --K= is passed
    for (int i = 2; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--csv") csv_output = true;
        else if (arg == "--benchmark") benchmark_mode = true;
        else if (arg == "put") type = OptionType::Put;
        else if (arg == "digital") type = OptionType::Digital;
        else if (arg == "call") type = OptionType::Call;
        else if (arg.rfind("--K=", 0) == 0) {
            K = stod(arg.substr(4));
        }
        else if (arg.rfind("--vr=", 0) == 0) {
            string vr_str = arg.substr(5);
            if (vr_str == "antithetic") vr = VarianceReduction::Antithetic;
            else if (vr_str == "control") vr = VarianceReduction::ControlVariate;
            else if (vr_str == "both") vr = VarianceReduction::Both;
            else vr = VarianceReduction::None;
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
    
    string option_type_str;
    switch (type) {
        case OptionType::Call: option_type_str = "Call"; break;
        case OptionType::Put: option_type_str = "Put"; break;
        case OptionType::Digital: option_type_str = "Digital"; break;
    }
    
    string vr_str;
    switch (vr) {
        case VarianceReduction::None: vr_str = "None"; break;
        case VarianceReduction::Antithetic: vr_str = "Antithetic"; break;
        case VarianceReduction::ControlVariate: vr_str = "Control Variate"; break;
        case VarianceReduction::Both: vr_str = "Both (Antithetic + Control)"; break;
    }

    if (benchmark_mode) {
        // Quick simple benchmark mode - run one test for each variance reduction technique
        if (argc > 2 && std::string(argv[2]) == "--simple") {
            cout << "Running simple benchmark for " << option_type_str << " option:" << endl;
            cout << "S0: " << S0 << ", K: " << K << ", r: " << r << ", sigma: " << sigma << ", T: " << T << endl;
            cout << "----------------------------------------------------------------------------------------------------------" << endl;
            cout << setw(15) << "Simulations" << setw(20) << "Technique" << setw(15) << "Price" << setw(15) << "Std Dev" 
                 << setw(15) << "Abs Error" << setw(15) << "Time (ms)" << setw(20) << "Efficiency" << endl;
            cout << "----------------------------------------------------------------------------------------------------------" << endl;
            
            vector<VarianceReduction> vr_techniques = {
                VarianceReduction::None, 
                VarianceReduction::Antithetic, 
                VarianceReduction::ControlVariate,
                VarianceReduction::Both
            };
            
            for (auto technique : vr_techniques) {
                BenchmarkResult result = run_simulation(10000, S0, K, r, sigma, T, type, technique);
                
                string tech_name;
                switch (technique) {
                    case VarianceReduction::None: tech_name = "None"; break;
                    case VarianceReduction::Antithetic: tech_name = "Antithetic"; break;
                    case VarianceReduction::ControlVariate: tech_name = "Control Variate"; break;
                    case VarianceReduction::Both: tech_name = "Both"; break;
                }
                
                double efficiency = 0.0;
                if (result.abs_error > 0 && result.execution_time_ms > 0) {
                    efficiency = 1.0 / (result.abs_error * result.abs_error * result.execution_time_ms);
                }
                
                cout << setw(15) << 10000 << setw(20) << tech_name 
                     << setw(15) << fixed << setprecision(6) << result.price 
                     << setw(15) << result.std_dev
                     << setw(15) << result.abs_error
                     << setw(15) << result.execution_time_ms
                     << setw(20) << scientific << setprecision(4) << efficiency << endl;
            }
            
            return 0;
        }
    }

    // Run a single simulation with the specified parameters
    BenchmarkResult result = run_simulation(N, S0, K, r, sigma, T, type, vr);

    if (csv_output) {
        cerr << "[DEBUG] CSV mode ON | OptionType: " << option_type_str << " | VR: " << vr_str << endl;
        cerr << "[DEBUG] Params => S0: " << S0 << ", K: " << K << ", sigma: " << sigma << ", r: " << r << ", T: " << T << ", N: " << N << endl;
        cout << N << "," << result.price << "," << result.lower_bound << "," << result.upper_bound << "," << result.execution_time_ms << "," << vr_str << endl;
    } else {
        cout << "S0: " << S0 << ", K: " << K << ", sigma: " << sigma
             << ", r: " << r << ", T: " << T << ", N: " << N << endl;
        cout << "Option Type: " << option_type_str << endl;
        cout << "Variance Reduction: " << vr_str << endl;
        cout << "Execution Time: " << result.execution_time_ms << " ms" << endl;

        if (type == OptionType::Call) {
            double analytical_call = black_scholes_call(S0, K, r, sigma, T);
            cout << "Analytical Black-Scholes Call Price: " << analytical_call << endl;
            cout << "Simulated Call Price (Monte Carlo): " << result.price << endl;
            cout << "Absolute Error: " << result.abs_error << endl;
        } else if (type == OptionType::Put) {
            double analytical_put = black_scholes_put(S0, K, r, sigma, T);
            cout << "Analytical Black-Scholes Put Price: " << analytical_put << endl;
            cout << "Simulated Put Price (Monte Carlo): " << result.price << endl;
            cout << "Absolute Error: " << result.abs_error << endl;
        }
        cout << "Estimated Option Price: " << result.price << endl;
        cout << "95% Confidence Interval: [" << result.lower_bound << ", " << result.upper_bound << "]" << endl;
        cout << "Standard Deviation: " << result.std_dev << endl;
    }
    return 0;
}
