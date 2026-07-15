// Small enough to inspect end to end: CUDA -> PTX -> cubin -> SASS.
extern "C" __global__ void smoke_axpy(
    const float* x,
    float* y,
    float alpha,
    unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        y[index] = fmaf(alpha, x[index], y[index]);
    }
}
