#include <iostream>
#include <chrono>

void add_vectors(const float* arr1, const float* arr2, float* result, int N)
{
    for (int i{}; i < N; ++i)
    {
        *(result + i) = *(arr1 + i) + *(arr2 + i);
    }
}

void generate_vector(float* result, int N)
{
    for (int i{}; i < N; ++i)
    {
       *(result + i) = i + 1;
    }
}


void print_array(const float* arr, int N)
{
    for (int i {}; i < N; ++i)
    {
        std::cout << *(arr + i) << '\n';
    }
}

int main()
{
    int N = 100000000;
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    d_A = new float[N]{};
    d_B = new float[N]{};
    d_C = new float[N]{};

    auto start = std::chrono::steady_clock::now();
    generate_vector(d_A, N);
    generate_vector(d_B, N);
    add_vectors(d_A, d_B, d_C, N);

    auto end = std::chrono::steady_clock::now();

    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Time elapsed: " << elapsed << '\n';

    delete[] d_A;
    delete[] d_B;
    delete[] d_C;

    return 0;
}