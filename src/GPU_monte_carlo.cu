#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <chrono>
#include <cmath>

struct OptionDataBS
{
    double S {};
    double K {};
    double r {};
    double sigma {};
    double T {};   
    
    explicit OptionDataBS(double pS,double pK,double pr,double psigma,double pT) :
    S{pS},K{pK},r{pr},sigma{psigma},T{pT} {} 
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

__global__ void GPU_monte_carlo(const OptionDataBS* data, const int N, double* call_payoffs, double* put_payoffs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        curandStatePhilox4_32_10_t state;
        curand_init(1234ULL, i, 0ULL, &state);
        double Z = curand_normal_double(&state);
        double St {(*data).S * std::exp(((*data).r - 0.5 * (*data).sigma * (*data).sigma) * (*data).T + (*data).sigma * std::sqrt((*data).T) * Z)};
        *(call_payoffs + i) = std::exp(-(*data).r * (*data).T) * fmax(St - (*data).K, 0.0);
        *(put_payoffs + i) = std::exp(-(*data).r * (*data).T) * fmax((*data).K - St, 0.0);
    }
}

int main()
{
    int N {static_cast<int>(1e8)};

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    OptionDataBS h_data {100, 100, 0.01, 0.3, 1};

    OptionDataBS* data = nullptr;

    Options payoff_vector(N);

    cudaMalloc(reinterpret_cast<void**>(&data), sizeof(OptionDataBS));

    cudaMemcpy(data, &h_data, sizeof(OptionDataBS),cudaMemcpyHostToDevice);
  
    GPU_monte_carlo<<<blocks, threads>>>(data, N, thrust::raw_pointer_cast(payoff_vector.call_payoffs.data()), thrust::raw_pointer_cast(payoff_vector.put_payoffs.data()));
    Options::Payoffs payoffs {payoff_vector.avg(N)};

    std::cout << "Call price: " << payoffs.call_payoffs << '\n';
    std::cout << "Put price: " << payoffs.put_payoffs << '\n';

    cudaFree(data);

    return 0;
} 