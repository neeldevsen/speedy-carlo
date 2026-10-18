#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <thread>
#include <random>

struct OptionDataBS
{
    double S {};
    double K {};
    double r {};
    double sigma {};
    double T {};  
    double k1 {};
    double k2 {};
    double k3 {}; 

    // this basically makes the equation S_T = S*exp(k2 + k3*Z)
    // the value of the option is in the form V = k1 * payoff
    
    explicit OptionDataBS(double pS,double pK,double pr,double psigma,double pT) :
    S{pS},K{pK},r{pr},sigma{psigma},T{pT},k1{std::exp(-pr*pT)},k2{(pr-0.5*psigma*psigma) * pT}, k3{psigma*std::sqrt(pT)} 
    {}
};

struct Options
{
    double call_payoffs {};
    double put_payoffs {};

    Options(double pcall, double pput) :
    call_payoffs{pcall}, put_payoffs{pput} {}

    Options(void) :
    call_payoffs{0.0}, put_payoffs{0.0 } {}
};

void worker(const int id, const OptionDataBS& data, const std::uint64_t N, const unsigned long long seed,  Options& payoffs)
{
    double local_payoff_call {};
    double local_payoff_put {};

    
    std::mt19937 mt (seed + static_cast<unsigned long long>(id));
    std::normal_distribution<double> W (0.0,1.0);

    for (std::uint64_t i {0}; i < N; ++i)
    {
        double Z {W(mt)};
        double S_T {data.S * std::exp(data.k2 + data.k3 * Z)};
        local_payoff_call += std::max(S_T - data.K, 0.0);
        local_payoff_put += std::max(data.K - S_T, 0.0); 
    }
    payoffs.call_payoffs = data.k1 * local_payoff_call;
    payoffs.put_payoffs = data.k1 * local_payoff_put;

}

void experiment(int N_IV, unsigned int P) // N_IV means number of independent variables
{
    OptionDataBS data {100, 100, 0.01, 0.3, 1};

    std::ofstream csv{"../data/MT_data.csv"};
    csv << std::setprecision(15);
    csv << "N,Time,Call,Put\n";
    std::random_device rd {};
    int N {30};
    int N_old {33};
    int N_actual {};
    for (int i{0}; i < N_IV; ++i)
    {
        N = std::floor(std::pow(static_cast<double>(10.0), static_cast<double>(1.0 + (7.0*i) / (N_IV - 1.0))));
        N_actual = (N / P) * P;
        if (N_old == N_actual or N_actual == 0) {continue;}

        std::vector<Options> payoffs (P);
        OptionDataBS data {100, 100, 0.01, 0.3, 1};
        Options payoff_final {0.0, 0.0};
        
        std::vector<std::thread> thr;
        unsigned long long seed {rd()};
        auto start {std::chrono::steady_clock::now()};
        for (int j {}; j < P; ++j)
        {
            thr.emplace_back(worker, j, data, N/P, seed, std::ref(payoffs[j]));
        }

        for (auto& t : thr)
        {
            t.join();
        }

        for (auto& payoff : payoffs)
        {
            payoff_final.call_payoffs += payoff.call_payoffs;
            payoff_final.put_payoffs += payoff.put_payoffs;
        }

        payoff_final.call_payoffs /= N_actual;
        payoff_final.put_payoffs /= N_actual;

        auto end {std::chrono::steady_clock::now()};
        auto difference = std::chrono::duration<double,std::milli>(end - start).count();

        csv << N_actual << ',' << difference << ',' << payoff_final.call_payoffs << ',' << payoff_final.put_payoffs << '\n';
       
        N_old = N_actual;
    }
}
 
int main()
{
    unsigned int P {std::thread::hardware_concurrency()};
    std::cout << "# Threads =" << P << '\n';

    experiment(1000, P);
    return 0;
}