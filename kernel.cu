﻿#define _USE_MATH_DEFINES
// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "device_launch_parameters.h"

using namespace std;

// Complex data type
typedef float2 Complex;

static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void TwiddleMult3d(Complex*, Complex*, int*);

//declaration
cudaError_t fft_3d(int, int, int);
cudaError_t fft_3d_cufft(int, int, int);
cudaError_t test_3d_dec();
cudaError_t test_3d_dec_big();


cudaError_t cudaStatus = cudaSetDevice(0);

int main()
{
    int x = 32;
    int y = 64;
    int z = 1024;
    //test_3d_dec();
    test_3d_dec_big();
   // fft_3d(x, y, z);
   // fft_3d_cufft(x, y, z);

    return 0;
}

cudaError_t test_3d_dec() {
    int X = 16;
    int Y = 32;
    int Z = 128;
    printf("X = %d, Y = %d, Z = %d \n", X, Y, Z);
    int deg = (int)log2(Z);
    long long Z1 = (int)pow(2, deg / 2);
    long long Z2 = (int)pow(2, (deg + 1) / 2);

    long long  signalSize = X * Y * Z;

    // float2 2 * 4byte each element
    long long gpu_mem_size = X * Y * Z; // elemnts for 2GB
    long long gpu_mem_size_b = gpu_mem_size * sizeof(Complex); // bytes for 2GB 


    // Allocate host and device memory for the signal
    Complex* h_signal = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * signalSize));
    Complex* h_result = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * signalSize));
    Complex* d_signal;
    Complex* d_result;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal), gpu_mem_size_b));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_result), gpu_mem_size_b));


    srand(3);
    // Initialize signal
    for (unsigned int i = 0; i < signalSize; ++i) {
        h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
        h_signal[i].y = 0;
    }


    //allocate memory for twiddle

    Complex* h_twiddle = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * Z1 * Z2));
    Complex* d_twiddle;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_twiddle), sizeof(Complex) * Z1 * Z2));

    //compute twidlle factors
    // exp(2 * pi * (m - 1) * (j - 1) / n), m - ñòðîêà, j - ñòîëáåö
    for (int i = 0; i < Z2; i++) {
        for (int j = 0; j < Z1; j++) {
            h_twiddle[i * Z1 + j].x = real(polar(1.0, -2 * M_PI * i * j / Z));
            h_twiddle[i * Z1 + j].y = imag(polar(1.0, -2 * M_PI * i * j / Z));
            // cout << "(" << h_twiddle[i * Z1 + j].x << "; " << h_twiddle[i * Z1 + j].y << ") ";
        }
        //  cout << "\n";
    }



    int C = Z1;
    int W = gpu_mem_size / C;

    int tC = C;
    int tW = W;

    int* h_vars = (int*)malloc(3 * sizeof(int));
    h_vars[0] = C; h_vars[1] = W, h_vars[2] = X * Y;

    int* d_vars;
    checkCudaErrors(cudaMalloc(&d_vars, 3 * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_vars, h_vars, 3 * sizeof(int), cudaMemcpyHostToDevice));


    checkCudaErrors(cudaMemcpyAsync(d_signal, h_signal, gpu_mem_size_b, cudaMemcpyHostToDevice));

    //transfer twiddle

    checkCudaErrors(cudaMemcpy(d_twiddle, h_twiddle, sizeof(Complex) * Z1 * Z2,
        cudaMemcpyHostToDevice));
    //make plan for d(Z1, XYZ2, XY)
    cufftHandle plan_advZ1;

    int n[1] = { tC };
    int inembed[] = { C };
    int onembed[] = { W };
    int istride = tW;
    int idist = 1;
    int ostride = X * Y;
    int odist = 1;
    int batch = X * Y;
    // might need for loop (?) cos of possibility of output overlap
    // transpose by advanced layout
    checkCudaErrors(cufftPlanMany(&plan_advZ1, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));
    for (int k = 0; k < tW / (X * Y); k++) {
        checkCudaErrors(cufftExecC2C(plan_advZ1, reinterpret_cast<cufftComplex*>(d_signal + (k * X * Y)),
            reinterpret_cast<cufftComplex*>(d_result + (k * X * Y * tC)), CUFFT_FORWARD));
    }
    cudaDeviceSynchronize();


    dim3 threadsPerBlock(8, 8, 16);
    dim3 numBlocks((C + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (W / (X * Y) + threadsPerBlock.y - 1) / threadsPerBlock.y,
        ((X * Y) + threadsPerBlock.z - 1) / threadsPerBlock.z);
    TwiddleMult3d << <numBlocks, threadsPerBlock >> > (d_result, d_twiddle, d_vars);
    cudaDeviceSynchronize();

     //make plan for d(Y, X, X)

    cufftHandle plan_advY;

    n[0] = Y;
    inembed[0] = tC;
    onembed[0] = tW;
    istride = X;
    idist = 1;
    ostride = X;
    odist = 1;
    batch = X;

    int h_C = C;
    checkCudaErrors(cufftPlanMany(&plan_advY, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));
    cudaDeviceSynchronize();
    int tZ2 = tW / (X * Y);
    for (int k = 0; k < h_C * tZ2; k++) {
        checkCudaErrors(cufftExecC2C(plan_advY, d_result + (k * X * Y),
            d_signal + (k * X * Y), CUFFT_FORWARD));
    }
    cudaDeviceSynchronize();

    C = Z2;
    W = gpu_mem_size / C;
    tC = C;
    tW = W;

    cufftHandle plan_advZ2;

    n[0] = tC;
    inembed[0] = tC;
    onembed[0] = tW;
    istride = tW;
    idist = 1;
    ostride = tW;
    odist = 1;
    batch = tW;


    checkCudaErrors(cufftPlanMany(&plan_advZ2, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));
    checkCudaErrors(cufftExecC2C(plan_advZ2, reinterpret_cast<cufftComplex*>(d_signal),
        reinterpret_cast<cufftComplex*>(d_result), CUFFT_FORWARD));


    cufftHandle plan_advX;

    n[0] = X;
    inembed[0] = tC;
    onembed[0] = tW;
    istride = 1;
    idist = X;
    ostride = 1;
    odist = X;
    batch = tC * tW / X;

    cout << "Checking if output is right\n";
    checkCudaErrors(cufftPlanMany(&plan_advX, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));

    checkCudaErrors(cufftExecC2C(plan_advX, d_result,
        d_signal, CUFFT_FORWARD));

    checkCudaErrors(cudaMemcpy(h_result, d_signal, gpu_mem_size_b,
        cudaMemcpyDeviceToHost));

    cudaMemcpy(d_signal, h_signal, X * Y * Z * sizeof(Complex), cudaMemcpyHostToDevice);
    cufftHandle plan;
    cufftPlan3d(&plan, Z, Y, X, CUFFT_C2C);
    cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);

    cudaMemcpy(h_signal, d_signal, Z * Y * X * sizeof(Complex),
        cudaMemcpyDeviceToHost);

    
    float delta = 0;
    for (int i = 0; i < X * Y * Z; i++) {
        //cout << h_result[i].x << " ";
        if (abs(h_result[i].x - h_signal[i].x) > .01) {
            cout << h_result[i].x << " " << h_signal[i].x << " x " << i << "\n";
        }
        if (abs(h_result[i].y - h_signal[i].y) > .01) {
            cout << h_result[i].y << " " << h_signal[i].y << " y " << i << "\n";
        }
        delta = max(abs(h_result[i].x - h_signal[i].x), delta);
        delta = max(abs(h_result[i].y - h_signal[i].y), delta);
    }
    cout << "Max delta: " << delta << "\n";
    cout << "Done\n";
    return cudaStatus;
}
cudaError_t fft_3d(int X,int Y, int Z) {

    printf("X = %d, Y = %d, Z = %d \n", X, Y, Z);

    int deg = (int)log2(Z);

    int Z1 = (int)pow(2, deg / 2);
    int Z2 = (int)pow(2, (deg + 1) / 2);
    long long signalSize = X * Y * Z;

    // float2 2 * 4byte each element
    int gpu_mem_size = 1048576; // 
    long long gpu_mem_size_b = gpu_mem_size * sizeof(Complex); // bytes


    // Allocate host and device memory for the signal
    /*
    Complex* h_signal = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * signalSize));
    Complex* h_result = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * signalSize));
    Complex* buffer = reinterpret_cast<Complex*>(malloc(gpu_mem_size_b));
    */
    
    Complex* h_signal;
    checkCudaErrors(cudaMallocHost((void**)&h_signal, (long long)sizeof(Complex) * signalSize));
    cout << "h_signal pinned: " << (long long)sizeof(Complex) * signalSize << "\n";
    Complex* h_result;
    checkCudaErrors(cudaMallocHost((void**)&h_result, (long long)sizeof(Complex) * signalSize));
    Complex* buffer;
    checkCudaErrors(cudaMallocHost((void**)&buffer, gpu_mem_size_b));
    
    Complex* d_signal;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal), gpu_mem_size_b));   
    Complex* d_result;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_result), gpu_mem_size_b));

    
    // Initialize signal
    srand(3);
    for (unsigned int i = 0; i < signalSize; ++i) {
        h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
        h_signal[i].y = 0;
    }
    
  
    //allocate memory for twiddle

   // Complex* h_twiddle = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * Z1 * Z2));
    Complex* h_twiddle;
    checkCudaErrors(cudaMallocHost((void**)&h_twiddle, sizeof(Complex) * Z1 * Z2));


    //compute twidlle factors
    // exp(2 * pi * (m - 1) * (j - 1) / n), m - строка, j - столбец
    for (int i = 0; i < Z2; i++) {
        for (int j = 0; j < Z1; j++) {
            h_twiddle[i * Z1 + j].x = (float)real(polar(1.0, -2 * M_PI * i * j / Z));
            h_twiddle[i * Z1 + j].y = (float)imag(polar(1.0, -2 * M_PI * i * j / Z));
        }
    }
    //streams might need later
    const int sNum = 8;
    cudaStream_t stream[sNum];
    for (int i = 0; i < sNum; ++i)
        checkCudaErrors(cudaStreamCreate(&stream[i]));


    cout <<"Z1 = "<< Z1 << ", Z2 = " << Z2 << "\n";

    int C = Z1;
    int W = gpu_mem_size / C;
    int tZ2 = W / (X * Y);

    
    Complex* d_twiddle;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_twiddle), C * W / (X * Y) * sizeof(Complex)));

    // variables for kernel {W, C, X*Y}
    int* h_vars = (int*)malloc(3 * sizeof(int));
    h_vars[0] = C; h_vars[1] = W, h_vars[2] = X * Y;

    int* d_vars;
    checkCudaErrors(cudaMalloc(&d_vars, 3 * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_vars, h_vars, 3 * sizeof(int), cudaMemcpyHostToDevice));


    // X * Y fits into memory
    int it_s = X * Y * Z2 / W;
   // cout << it_s << "\n";

    float trans = 0.0, twiddle = 0.0, fft = 0.0;

    //make plan for d(Z1, XYZ2, XY)
    cufftHandle plan_advZ1;

    int n[1] = { C };
    int inembed[] = { C };
    int onembed[] = { W };
    int istride = W;
    int idist = 1;
    int ostride = X * Y;
    int odist = 1;
    int batch = X * Y;

    // transpose by advanced layout
    checkCudaErrors(cufftPlanMany(&plan_advZ1, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));

    //make plan for d(Y, X, X)
    cufftHandle plan_advY;

    n[0] = Y;
    inembed[0] = C;
    onembed[0] = W;
    istride = X;
    idist = 1;
    ostride = X;
    odist = 1;
    batch = X;

    checkCudaErrors(cufftPlanMany(&plan_advY, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));

    dim3 threadsPerBlock(8, 8, 16);
    dim3 numBlocks((C + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (tZ2 + threadsPerBlock.y - 1) / threadsPerBlock.y,
        ((X * Y) + threadsPerBlock.z - 1) / threadsPerBlock.z);

    int bufN = sNum;
    int bufIt = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < it_s; i++) {

        bufIt = 0;
        for (unsigned int j = 0; j < C; j++) {
            if (j == C * (bufIt + 1) / bufN) {
                checkCudaErrors(cudaMemcpyAsync(d_signal + C * W * bufIt / bufN, buffer + C * W * bufIt / bufN, C * W / bufN * sizeof(Complex), cudaMemcpyHostToDevice, stream[bufIt]));
                bufIt++;
            }
            memcpy(buffer + (j * W), h_signal + (i * W) + (j * X * Y * Z2), W * sizeof(Complex));
            
        }
        checkCudaErrors(cudaMemcpyAsync(d_signal +  C * W * (bufN - 1) / bufN, buffer + C * W * (bufN - 1) / bufN, C * W / bufN  * sizeof(Complex), cudaMemcpyHostToDevice, stream[bufIt]));
        // checkCudaErrors(cudaMemcpyAsync(d_signal, buffer, C * W * sizeof(Complex), cudaMemcpyHostToDevice));
        

        //checkCudaErrors(cudaMemcpy(d_signal, buffer, gpu_mem_size_b, 
        //                           cudaMemcpyHostToDevice));

        //transfer twiddle
        // W / (X*Y) is Z2 / const
        
        checkCudaErrors(cudaMemcpyAsync(d_twiddle, h_twiddle + (i * C * tZ2), C * tZ2 * sizeof(Complex),
            cudaMemcpyHostToDevice));


        for (int k = 0; k < W / (X * Y); k++) {
            checkCudaErrors(cufftExecC2C(plan_advZ1, reinterpret_cast<cufftComplex*>(d_signal + (k * X * Y)),
                reinterpret_cast<cufftComplex*>(d_result + (k * X * Y * C)), CUFFT_FORWARD));
        }
        checkCudaErrors(cudaDeviceSynchronize());
      

        TwiddleMult3d << <numBlocks, threadsPerBlock >> > (d_result, d_twiddle, d_vars);
        checkCudaErrors(cudaDeviceSynchronize());
      

       // checkCudaErrors(cufftSetStream(plan_advY, stream[i % sNum]));
        for (int k = 0; k < C * tZ2; k++) {
            checkCudaErrors(cufftExecC2C(plan_advY, d_result + (k * X * Y),
                d_signal + (k * X * Y), CUFFT_FORWARD));
        }
        checkCudaErrors(cudaDeviceSynchronize());
       

        //transport to host
        checkCudaErrors(cudaMemcpyAsync(h_result + (i * C * W), d_signal, gpu_mem_size_b,
            cudaMemcpyDeviceToHost));

    }
    cufftDestroy(plan_advZ1);
    cufftDestroy(plan_advY);

    C = Z2;
    W = gpu_mem_size / C;
    it_s = X * Y * Z1 / W;

    //make plan for d(Z2, XYZ1, XYZ1)
    cufftHandle plan_advZ2;

    n[0] = C;
    istride = W;
    idist = 1;
    ostride = W;
    odist = 1;
    batch = W;

    checkCudaErrors(cufftPlanMany(&plan_advZ2, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));

    //make plan for d(X, 1, 1)
    cufftHandle plan_advX;

    n[0] = X;
    istride = 1;
    idist = X;
    ostride = 1;
    odist = X;
    batch = C * W / X;


    checkCudaErrors(cufftPlanMany(&plan_advX, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));

   
    for (int i = 0; i < it_s; i++) {
        bufIt = 0;
        for (unsigned int j = 0; j < C; j++) {
            if (j == C * (bufIt + 1) / bufN) {
                checkCudaErrors(cudaMemcpyAsync(d_signal + C * W * bufIt / bufN, buffer + C * W * bufIt / bufN, C * W / bufN * sizeof(Complex), cudaMemcpyHostToDevice));
                bufIt++;
            }
            memcpy(buffer + (j * W), h_result + (i * W) + (j * X * Y * Z1), W * sizeof(Complex));
        }
        checkCudaErrors(cudaMemcpyAsync(d_signal + C * W * (bufN - 1) / bufN, buffer + C * W * (bufN - 1) / bufN, C* W / bufN * sizeof(Complex), cudaMemcpyHostToDevice));

      


        checkCudaErrors(cufftExecC2C(plan_advZ2, reinterpret_cast<cufftComplex*>(d_signal),
            reinterpret_cast<cufftComplex*>(d_result), CUFFT_FORWARD));

       // checkCudaErrors(cufftSetStream(plan_advX, stream[i % sNum]));
        checkCudaErrors(cufftExecC2C(plan_advX, d_result,
             d_signal, CUFFT_FORWARD));
      

        for (int k = 0; k < bufN; k++) {
            checkCudaErrors(cudaMemcpyAsync(buffer + C * W * k / bufN, d_signal + C * W * k / bufN, C * W / bufN * sizeof(Complex),
                cudaMemcpyDeviceToHost, stream[k]));
        }
        for (int k = 0; k < bufN; k++) {
            checkCudaErrors(cudaStreamSynchronize(stream[k]));
            for (unsigned int j = k * C / bufN; j < C * (k + 1) / bufN; j++) {
                memcpy(h_signal + (i * W) + (j * X * Y * Z1), buffer + (j * W), W * sizeof(Complex));
            }
        }
       
    }

    cufftDestroy(plan_advZ2);
    cufftDestroy(plan_advX);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cout << "3d dec time, ms = " << milliseconds<<"\n";
    cout << "transport time  = " << trans<<"\n";
    cout << "twiddle time  = " << twiddle<<"\n";
    cout << "fft's time  = " << fft<<"\n";

    cudaFreeHost(h_signal);
    cudaFreeHost(h_result);
    cudaFreeHost(h_twiddle);
    
    cudaFree(d_result);
    cudaFree(d_signal);
    cudaFree(d_twiddle);

    return cudaStatus;
}

cudaError_t test_3d_dec_big() {
    int X = 32;
    int Y = 64;
    int Z = 1024;
    printf("X = %d, Y = %d, Z = %d \n", X, Y, Z);

    int deg = (int)log2(Z);

    int Z1 = (int)pow(2, deg / 2);
    int Z2 = (int)pow(2, (deg + 1) / 2);
    long long signalSize = X * Y * Z;

    // float2 2 * 4byte each element
    int gpu_mem_size = 1048576; // 
    long long gpu_mem_size_b = gpu_mem_size * sizeof(Complex); // bytes


    // Allocate host and device memory for the signal
    /*
    Complex* h_signal = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * signalSize));
    Complex* h_result = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * signalSize));
    Complex* buffer = reinterpret_cast<Complex*>(malloc(gpu_mem_size_b));
    */

    Complex* h_signal;
    checkCudaErrors(cudaMallocHost((void**)&h_signal, (long long)sizeof(Complex) * signalSize));
    cout << "h_signal pinned: " << (long long)sizeof(Complex) * signalSize << "\n";
    Complex* h_result;
    checkCudaErrors(cudaMallocHost((void**)&h_result, (long long)sizeof(Complex) * signalSize));
    Complex* buffer;
    checkCudaErrors(cudaMallocHost((void**)&buffer, gpu_mem_size_b));

    Complex* d_signal;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal), gpu_mem_size_b));
    Complex* d_result;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_result), gpu_mem_size_b));


    // Initialize signal
    srand(3);
    for (unsigned int i = 0; i < signalSize; ++i) {
        h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
        h_signal[i].y = 0;
    }


    //allocate memory for twiddle

   // Complex* h_twiddle = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * Z1 * Z2));
    Complex* h_twiddle;
    checkCudaErrors(cudaMallocHost((void**)&h_twiddle, sizeof(Complex) * Z1 * Z2));


    //compute twidlle factors
    // exp(2 * pi * (m - 1) * (j - 1) / n), m - строка, j - столбец
    for (int i = 0; i < Z2; i++) {
        for (int j = 0; j < Z1; j++) {
            h_twiddle[i * Z1 + j].x = (float)real(polar(1.0, -2 * M_PI * i * j / Z));
            h_twiddle[i * Z1 + j].y = (float)imag(polar(1.0, -2 * M_PI * i * j / Z));
        }
    }
    //streams might need later
    const int sNum = 4;
    cudaStream_t stream[sNum];
    for (int i = 0; i < sNum; ++i)
        checkCudaErrors(cudaStreamCreate(&stream[i]));


    cout << "Z1 = " << Z1 << ", Z2 = " << Z2 << "\n";

    int C = Z1;
    int W = gpu_mem_size / C;
    int tZ2 = W / (X * Y);


    Complex* d_twiddle;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_twiddle), C * W / (X * Y) * sizeof(Complex)));

    // variables for kernel {W, C, X*Y}
    int* h_vars = (int*)malloc(3 * sizeof(int));
    h_vars[0] = C; h_vars[1] = W, h_vars[2] = X * Y;

    int* d_vars;
    checkCudaErrors(cudaMalloc(&d_vars, 3 * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_vars, h_vars, 3 * sizeof(int), cudaMemcpyHostToDevice));


    // X * Y fits into memory
    int it_s = X * Y * Z2 / W;
    cout << it_s << "\n";

    float trans = 0.0, twiddle = 0.0, fft = 0.0;

    //make plan for d(Z1, XYZ2, XY)
    cufftHandle plan_advZ1;

    int n[1] = { C };
    int inembed[] = { C };
    int onembed[] = { W };
    int istride = W;
    int idist = 1;
    int ostride = X * Y;
    int odist = 1;
    int batch = X * Y;

    // transpose by advanced layout
    checkCudaErrors(cufftPlanMany(&plan_advZ1, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));

    //make plan for d(Y, X, X)
    cufftHandle plan_advY;

    n[0] = Y;
    inembed[0] = C;
    onembed[0] = W;
    istride = X;
    idist = 1;
    ostride = X;
    odist = 1;
    batch = X;

    checkCudaErrors(cufftPlanMany(&plan_advY, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));

    dim3 threadsPerBlock(8, 8, 16);
    dim3 numBlocks((C + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (tZ2 + threadsPerBlock.y - 1) / threadsPerBlock.y,
        ((X * Y) + threadsPerBlock.z - 1) / threadsPerBlock.z);

    int bufN = sNum;
    int bufIt = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < it_s; i++) {

        bufIt = 0;
        for (unsigned int j = 0; j < C; j++) {
            
            if (j == C * (bufIt + 1) / bufN) {
                
                checkCudaErrors(cudaMemcpyAsync(d_signal + C * W * bufIt / bufN, buffer + C * W * bufIt / bufN, C * W / bufN * sizeof(Complex), cudaMemcpyHostToDevice, stream[bufIt]));
                bufIt++;
            }
            memcpy(buffer + (j * W), h_signal + (i * W) + (j * X * Y * Z2), W * sizeof(Complex));

        }
        
        checkCudaErrors(cudaMemcpyAsync(d_signal + C * W * (bufN - 1) / bufN, buffer + C * W * (bufN - 1) / bufN, C * W / bufN * sizeof(Complex), cudaMemcpyHostToDevice, stream[bufIt]));
        // checkCudaErrors(cudaMemcpyAsync(d_signal, buffer, C * W * sizeof(Complex), cudaMemcpyHostToDevice));


        //checkCudaErrors(cudaMemcpy(d_signal, buffer, gpu_mem_size_b, 
        //                           cudaMemcpyHostToDevice));

        //transfer twiddle
        // W / (X*Y) is Z2 / const

        checkCudaErrors(cudaMemcpyAsync(d_twiddle, h_twiddle + (i * C * tZ2), C * tZ2 * sizeof(Complex),
            cudaMemcpyHostToDevice));


        for (int k = 0; k < W / (X * Y); k++) {
            checkCudaErrors(cufftExecC2C(plan_advZ1, reinterpret_cast<cufftComplex*>(d_signal + (k * X * Y)),
                reinterpret_cast<cufftComplex*>(d_result + (k * X * Y * C)), CUFFT_FORWARD));
        }
        checkCudaErrors(cudaDeviceSynchronize());


        TwiddleMult3d << <numBlocks, threadsPerBlock >> > (d_result, d_twiddle, d_vars);
        checkCudaErrors(cudaDeviceSynchronize());


        // checkCudaErrors(cufftSetStream(plan_advY, stream[i % sNum]));
        for (int k = 0; k < C * tZ2; k++) {
            checkCudaErrors(cufftExecC2C(plan_advY, d_result + (k * X * Y),
                d_signal + (k * X * Y), CUFFT_FORWARD));
        }
        checkCudaErrors(cudaDeviceSynchronize());


        //transport to host
        checkCudaErrors(cudaMemcpyAsync(h_result + (i * C * W), d_signal, gpu_mem_size_b,
            cudaMemcpyDeviceToHost));

    }
    cufftDestroy(plan_advZ1);
    cufftDestroy(plan_advY);

    C = Z2;
    W = gpu_mem_size / C;
    it_s = X * Y * Z1 / W;

    //make plan for d(Z2, XYZ1, XYZ1)
    cufftHandle plan_advZ2;

    n[0] = C;
    istride = W;
    idist = 1;
    ostride = W;
    odist = 1;
    batch = W;

    checkCudaErrors(cufftPlanMany(&plan_advZ2, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));

    //make plan for d(X, 1, 1)
    cufftHandle plan_advX;

    n[0] = X;
    istride = 1;
    idist = X;
    ostride = 1;
    odist = X;
    batch = C * W / X;


    checkCudaErrors(cufftPlanMany(&plan_advX, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));

    
    for (int i = 0; i < it_s; i++) {
        bufIt = 0;
        for (unsigned int j = 0; j < C; j++) {
            if (j == C * (bufIt + 1) / bufN) {
                
                checkCudaErrors(cudaMemcpyAsync(d_signal + C * W * bufIt / bufN, buffer + C * W * bufIt / bufN, C * W / bufN * sizeof(Complex), cudaMemcpyHostToDevice));
                bufIt++;
            }
            memcpy(buffer + (j * W), h_result + (i * W) + (j * X * Y * Z1), W * sizeof(Complex));
        }
        
        checkCudaErrors(cudaMemcpyAsync(d_signal + C * W * (bufN - 1) / bufN, buffer + C * W * (bufN - 1) / bufN, C * W / bufN * sizeof(Complex), cudaMemcpyHostToDevice));




        checkCudaErrors(cufftExecC2C(plan_advZ2, reinterpret_cast<cufftComplex*>(d_signal),
            reinterpret_cast<cufftComplex*>(d_result), CUFFT_FORWARD));

        // checkCudaErrors(cufftSetStream(plan_advX, stream[i % sNum]));
        checkCudaErrors(cufftExecC2C(plan_advX, d_result,
            d_signal, CUFFT_FORWARD));


        for (int k = 0; k < bufN; k++) {
            
            checkCudaErrors(cudaMemcpyAsync(buffer + C * W * k / bufN, d_signal + C * W * k / bufN, C * W / bufN * sizeof(Complex),
                cudaMemcpyDeviceToHost, stream[k]));
        }
        for (int k = 0; k < bufN; k++) {
            checkCudaErrors(cudaStreamSynchronize(stream[k]));
            
            for (unsigned int j = k * C / bufN; j < C * (k + 1) / bufN; j++) {
                memcpy(h_result + (i * W) + (j * X * Y * Z1), buffer + (j * W), W * sizeof(Complex));
            }
        }

    }

    cufftDestroy(plan_advZ2);
    cufftDestroy(plan_advX);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cout << "3d dec time, ms = " << milliseconds << "\n";

    cudaFree(d_result);
    cudaFree(d_signal);
    cudaFree(d_twiddle);

    
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal), sizeof(Complex)* signalSize));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    checkCudaErrors(cudaMemcpy(d_signal, h_signal, X* Y* Z * sizeof(Complex), cudaMemcpyHostToDevice));
    cufftHandle plan;

    checkCudaErrors(cufftPlan3d(&plan, Z, Y, X, CUFFT_C2C));
    checkCudaErrors(cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_signal, d_signal, Z* Y* X * sizeof(Complex),
        cudaMemcpyDeviceToHost));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "cufft time, ms = " << milliseconds << "\n";

    for (int i = 0; i < X * Y * Z; i++) {
        //cout << h_result[i].x << " ";
        if (abs(h_result[i].x - h_signal[i].x) > 1) {
            cout << h_result[i].x << " " << h_signal[i].x << " x " << i << "\n";
        }
        if (abs(h_result[i].y - h_signal[i].y) > 1) {
            cout << h_result[i].y << " " << h_signal[i].y << " y " << i << "\n";
        }
    }

    cudaFree(d_signal);

    cudaFreeHost(h_signal);
    cudaFreeHost(h_result);
    cudaFreeHost(h_twiddle);



    return cudaStatus;
}
cudaError_t fft_3d_cufft(int X, int Y, int Z) {

    int signalSize = X * Y * Z;

    Complex* h_signal;
    checkCudaErrors(cudaMallocHost((void**)&h_signal, sizeof(Complex) * signalSize));
    Complex* d_signal;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal), sizeof(Complex) * signalSize));

    srand(3);
    for (unsigned int i = 0; i < signalSize; ++i) {
        h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
        h_signal[i].y = 0;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    checkCudaErrors(cudaMemcpy(d_signal, h_signal, X * Y * Z * sizeof(Complex), cudaMemcpyHostToDevice));

    cufftHandle plan;

    checkCudaErrors(cufftPlan3d(&plan, Z, Y, X, CUFFT_C2C));
    checkCudaErrors(cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_signal, d_signal, Z * Y * X * sizeof(Complex),
        cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "cufft time, ms = " << milliseconds << "\n";

    cudaFreeHost(h_signal);
    cudaFree(d_signal);
}


////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex pointwise multiplication
static __global__ void TwiddleMult(Complex* X, Complex* twiddle, int* d_vars) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int W = d_vars[1];
    int C = d_vars[0];
    if (i < C && j < W)
        X[j*C + i] = ComplexMul(X[j * C + i], twiddle[i*W + j]);
}
static __global__ void TwiddleMult3d(Complex* X, Complex* twiddle, int* d_vars) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int xy = d_vars[2];
    int W = d_vars[1];
    int C = d_vars[0];

   // int Z2 = (k + xy * (i + C * j)) / (xy * C);

    if (i < C && j < W / xy && k < xy) {
        X[k + xy * (i + C * j)] = ComplexMul(X[k + xy * (i + C * j)], twiddle[i + j * C]);
        //X[k + xy * (i + C * j)].x = xy;
        //X[k + xy * (i + C * j)].y = k;
    }
}