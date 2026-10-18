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

void worker(const int id, const OptionDataBS& data, const std::uint64_t N, Options& payoffs)
{
    double local_payoff_call {};
    double local_payoff_put {};
    
    std::mt19937 mt (id);
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


int main()
{
    const std::uint64_t N {1000000000LL};
    unsigned int P {std::thread::hardware_concurrency()};
    std::cout << "# Threads =" << P << '\n';
    std::vector<Options> payoffs (P);
    OptionDataBS data {100, 100, 0.01, 0.3, 1};
    Options payoff_final {0.0, 0.0};
    
    std::vector<std::thread> thr;

    for (int i {}; i < P; ++i)
    {
        thr.emplace_back(worker, i, data, N/P, std::ref(payoffs[i]));
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

    payoff_final.call_payoffs /= N;
    payoff_final.put_payoffs /= N;

    std::cout << "Call price: " << payoff_final.call_payoffs << '\n';
    std::cout << "Put price: " << payoff_final.put_payoffs << '\n';

    return 0;
}