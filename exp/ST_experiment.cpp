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

Options monteCarlo(const OptionDataBS& data, const int N)
{
    Options payoffs {};
    
    std::random_device rd {};
    std::mt19937 mt (rd());
    std::normal_distribution<double> W (0.0,1.0);

    for (int i {0}; i < N; ++i)
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

void experiment(int N_IV) // N_IV means number of independent variables
{
    OptionDataBS data {100, 100, 0.01, 0.3, 1};

    std::ofstream csv{"../data/ST_data.csv"};
    csv << std::setprecision(15);
    csv << "N,Time,Call,Put\n";
    int N {10};
    int N_old {11};
    for (int i{0}; i < N_IV; ++i)
    {
        N = std::floor(std::pow(static_cast<double>(10.0), static_cast<double>(1.0 + (7.0*i) / (N_IV - 1.0))));
        if (N_old == N) {continue;}

        OptionDataBS data {100, 100, 0.01, 0.3, 1};
        auto start {std::chrono::steady_clock::now()};
        Options payoff_final {monteCarlo(data, N)};
        auto end {std::chrono::steady_clock::now()};
        auto difference = std::chrono::duration<double,std::milli>(end - start).count();
        csv << N << ',' << difference << ',' << payoff_final.call_payoffs << ',' << payoff_final.put_payoffs << '\n';
        N_old = N;
    }
}

int main()
{
    experiment(1000);
    return 0;
}
