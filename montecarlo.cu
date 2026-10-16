__global__ void square(float* arr, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        arr[i] *= arr[i];
    }
}