#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
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
    thrust::device_vector<double> call_payoffs {};
    thrust::device_vector<double> put_payoffs {};

    explicit Options(int N) :
    call_payoffs(N, 0.0), put_payoffs(N, 0.0) {}

    struct Payoffs
    {
        double call_payoffs {};
        double put_payoffs {};
    };
    
    Payoffs avg(int N)
    {
        Payoffs payoff {};
        payoff.call_payoffs = thrust::reduce((*this).call_payoffs.begin(), (*this).call_payoffs.end(), 0.0) / static_cast<double>(N);
        payoff.put_payoffs = thrust::reduce((*this).put_payoffs.begin(), (*this).put_payoffs.end(), 0.0) / static_cast<double>(N);

        return payoff;
    }
};

__global__ void GPU_monte_carlo(const OptionDataBS* data, const int N, const unsigned long long seed, double* call_payoffs, double* put_payoffs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < N)
    {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, i, 0ULL, &state);
        double Z = curand_normal_double(&state);
        double S_T {(*data).S * std::exp((*data).k2 + (*data).k3 * Z)};
        *(call_payoffs + i) = (*data).k1 * fmax(S_T - (*data).K, 0.0);
        *(put_payoffs + i) = (*data).k1 * fmax((*data).K - S_T, 0.0);
    }
}

void experiment(int N_IV) // N_IV means number of independent variables
{
    std::random_device rd {};
    int threads = 256;
    
    OptionDataBS h_data {100, 100, 0.01, 0.3, 1};
    OptionDataBS* data = nullptr;
    
    cudaMalloc(reinterpret_cast<void**>(&data), sizeof(OptionDataBS));
    cudaMemcpy(data, &h_data, sizeof(OptionDataBS),cudaMemcpyHostToDevice);

    std::ofstream csv{"../data/CUDA_data.csv"};
    csv << std::setprecision(15);
    csv << "N,Time,Call,Put\n";
    int N {10};
    int N_old {11};
    for (int i{0}; i < N_IV; ++i)
    {
        N = std::floor(std::pow(static_cast<double>(10.0), static_cast<double>(1.0 + (7.0*i) / (N_IV - 1.0))));
        if (N_old == N) {continue;}
        Options payoff_vector(N);
        int blocks = (N + threads - 1) / threads;
        unsigned long long seed {rd()};
        auto start = std::chrono::steady_clock::now();
        GPU_monte_carlo<<<blocks, threads>>>(data, N, seed, thrust::raw_pointer_cast(payoff_vector.call_payoffs.data()), thrust::raw_pointer_cast(payoff_vector.put_payoffs.data()));
        Options::Payoffs payoffs {payoff_vector.avg(N)};
        cudaDeviceSynchronize();
        auto end = std::chrono::steady_clock::now();
        auto difference = std::chrono::duration<double,std::milli>(end - start).count();
        csv << N << ',' << difference << ',' << payoffs.call_payoffs << ',' << payoffs.put_payoffs << '\n';
        N_old = N;
    }
    cudaFree(data);
}

int main()
{
    experiment(1000);
    return 0;
} 