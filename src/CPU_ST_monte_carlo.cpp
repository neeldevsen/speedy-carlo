
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
    call_payoffs{0.0}, put_payoffs{0.0} {}
};

Options monteCarlo(const OptionDataBS& data, const std::uint64_t N)
{
    Options payoffs {};
    
    std::mt19937 mt (42);
    std::normal_distribution<double> W (0.0,1.0);

    for (std::uint64_t i {0}; i < N; ++i)
    {
        double Z {W(mt)};
        double S_T {data.S * std::exp(data.k2 + data.k3 * Z)};
        payoffs.call_payoffs += std::max(S_T - data.K, 0.0);
        payoffs.put_payoffs += std::max(data.K - S_T, 0.0); 
    }
    payoffs.call_payoffs = data.k1 * payoffs.call_payoffs / N;
    payoffs.put_payoffs = data.k1 * payoffs.put_payoffs / N;

    return payoffs;
}


int main()
{
    OptionDataBS data {100, 100, 0.01, 0.3, 1};
    const std::uint64_t N {static_cast<int>(1e8)};
    Options payoffs {monteCarlo(data, N)};

    std::cout << "Call price: " << payoffs.call_payoffs << '\n';
    std::cout << "Put price: " << payoffs.put_payoffs << '\n';

    return 0;
}