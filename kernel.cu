#define _USE_MATH_DEFINES
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

static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void TwiddleMult(Complex*, Complex*);
static __global__ void TwiddleMult(Complex*, Complex*, int *);
__managed__ int C;
__managed__ int W;




//declaration
cudaError_t fft_1d(long long);
cudaError_t fft_3d(long long, long long, long long);
cudaError_t test();
cudaError_t test_3d_naive();
cudaError_t test_3d_dec();

cudaError_t cudaStatus = cudaSetDevice(0);

int main()
{

    test_3d_dec();

    return 0;
}

// 
cudaError_t test()
{
    int signalSize = 32;
    Complex* h_signal = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * signalSize));
    Complex* h_result = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * signalSize));
    Complex* h_twiddle = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * signalSize));

    Complex* d_signal;
    Complex* d_twiddle;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal), sizeof(Complex) * signalSize));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_twiddle), sizeof(Complex) * signalSize));

    srand(2);
    for (unsigned int i = 0; i < signalSize; ++i) {
        h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
        h_signal[i].y = 0;
    }

    cout << "signal\n";
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            cout <<"(" <<h_signal[i * 4 + j].x << ", "<<h_signal[i*4 + j].y<<") ";
        }
        cout << "\n";
    }
    cout << "\n";
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            cout << h_signal[i * 4 + j].x<<"\n";
        }
        
    }
    cout << "\n";

    int X1 = 8, X2 = 4;
    
    for (int i = 0; i < X1  ; i++) {
        for (int j = 0; j < X2; j++) {
            h_twiddle[i * X2 + j].x = real(polar(1.0, -2 * M_PI * i * j / signalSize));
            h_twiddle[i * X2 + j].y = imag(polar(1.0, -2 * M_PI * i * j / signalSize));
        }
    }
    cout << "\n";
    cout << "twiddle\n";
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            cout << "(" << h_twiddle[i * 4 + j].x << ", " << h_twiddle[i * 4 + j].y << ") ";
        }
        cout << "\n";
    }

    checkCudaErrors(cudaMemcpy(d_twiddle, h_twiddle, sizeof(Complex) * signalSize,
        cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_signal, h_signal, sizeof(Complex) * signalSize, cudaMemcpyHostToDevice));

    C = 8;
    W = 4;

    cufftHandle plan_adv2;
    

    int n[1] = { C };
    int inembed[] = { C };
    int onembed[] = { W };
    int istride = W;
    int idist = 1;
    int ostride = 1;
    int odist = C;
    int batch = W;

    checkCudaErrors(cufftPlanMany(&plan_adv2, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));
    checkCudaErrors(cufftExecC2C(plan_adv2, reinterpret_cast<cufftComplex*>(d_signal),
        reinterpret_cast<cufftComplex*>(d_signal), CUFFT_FORWARD));
    checkCudaErrors(cudaMemcpy(h_result, d_signal, sizeof(Complex) * signalSize,
        cudaMemcpyDeviceToHost));
    
    cout << "\n";
    cout << "first round fft\n";
    for (int i = 0; i < X2; i++) {
        for (int j = 0; j < X1; j++) {
            cout << "(" << h_result [i * X1 + j] .x << ", " << h_result[i * X1 + j].y << ") ";
        }
        cout << "\n";
    }

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((C + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (W + threadsPerBlock.y - 1) / threadsPerBlock.y);
    TwiddleMult <<<numBlocks, threadsPerBlock >>> (d_signal, d_twiddle);
    checkCudaErrors(cudaMemcpy(h_result, d_signal, sizeof(Complex) * signalSize,
        cudaMemcpyDeviceToHost));
   
    cout << "\n";
    cout << "mult by twiddle\n";
    for (int i = 0; i < X2; i++) {
        for (int j = 0; j < X1; j++) {
            cout << "(" << h_result[i * X1 + j].x << ", " << h_result[i * X1 + j].y << ") ";
        }
        cout << "\n";
    }


    cufftHandle plan_adv;

    n[0] = { W };
    inembed[0] = { C };
    onembed[0] = { C };
    istride = C;
    idist = 1;
    ostride = C;
    odist = 1;
    batch = C;
    checkCudaErrors(cufftPlanMany(&plan_adv, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));
    checkCudaErrors(cufftExecC2C(plan_adv, reinterpret_cast<cufftComplex*>(d_signal),
        reinterpret_cast<cufftComplex*>(d_signal), CUFFT_FORWARD));
    checkCudaErrors(cudaMemcpy(h_result, d_signal, sizeof(Complex) * signalSize,
        cudaMemcpyDeviceToHost));
    
    cout << "\n";
    cout << "result\n";
    for (int i = 0; i < X1 * X2; i++) {
        
        cout << "(" << h_result[i].x<< ", " << h_result[i].y << ") \n";
        
    }
    cudaFree(d_signal);
    cudaFree(d_twiddle);
    free(h_signal);
    free(h_result);
    free(h_twiddle);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    

Error:

    return cudaStatus;
}
cudaError_t test_3d_naive() {

    int signalSizeX = 8;
    int signalSizeY = 2;
    int signalSizeZ = 4;

    // Allocate host memory for the signal
    Complex* h_signal = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * signalSizeY * signalSizeX * signalSizeZ));
    Complex* h_result = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * signalSizeY * signalSizeX * signalSizeZ));
    srand(3);
    // Initialize the memory for the signal
    for (unsigned int i = 0; i < signalSizeY * signalSizeX * signalSizeZ; ++i) {
        h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
        h_signal[i].y = 0;
    }

    for (int z = 0; z < signalSizeZ; z++) {
        for (int y = 0; y < signalSizeY; y++) {
            for (int x = 0; x < signalSizeX; x++) {

                cout << "(" << h_signal[x + signalSizeX * (y + signalSizeY * z)].x << "; " << h_signal[x + signalSizeX * (y + signalSizeY * z)].y << ") ";
            }
            cout << "\n";

        }
        cout << "\n\n";
    }

    
    long long gpu_mem_size = signalSizeY * signalSizeX * signalSizeZ; 
    long long gpu_mem_size_b = signalSizeY * signalSizeX * signalSizeZ * sizeof(Complex); 

    Complex* d_signal;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal), gpu_mem_size_b));


    //assume that X * Y can fit in gpu
    int tC = signalSizeZ;
    int tW = signalSizeY * signalSizeX;
    Complex* buffer = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * tC * tW));

    checkCudaErrors(cudaMemcpy(d_signal, h_signal, tC * tW * sizeof(Complex), cudaMemcpyHostToDevice));


    //make plan for z fft
    cufftHandle plan_advZ;

    int n[1] = { tC };
    int inembed[] = { tW };
    int onembed[] = { tW };
    int istride = tW;
    int idist = 1;
    int ostride = tW;
    int odist = 1;
    int batch = tW;


    checkCudaErrors(cufftPlanMany(&plan_advZ, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));
    checkCudaErrors(cufftExecC2C(plan_advZ, reinterpret_cast<cufftComplex*>(d_signal),
        reinterpret_cast<cufftComplex*>(d_signal), CUFFT_FORWARD));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_result, d_signal, signalSizeX * signalSizeZ * signalSizeY * sizeof(Complex),
        cudaMemcpyDeviceToHost));


    cout << "z fft\n";
    for (int z = 0; z < signalSizeZ; z++) {
        for (int y = 0; y < signalSizeY; y++) {
            for (int x = 0; x < signalSizeX; x++) {

                cout << "(" << h_result[x + signalSizeX * (y + signalSizeY * z)].x << "; " << h_result[x + signalSizeX * (y + signalSizeY * z)].y << ") ";
            }
            cout << "\n";

        }
        cout << "\n\n\n";
    }

    cufftHandle plan_advX;

    // X fft
    n[0] = (int)signalSizeX;
    inembed[0] = tW;
    onembed[0] = tW;
    istride = 1;
    idist = (int)signalSizeX;
    ostride = 1;
    odist = (int)signalSizeX;
    batch = signalSizeY * signalSizeZ;

    checkCudaErrors(cufftPlanMany(&plan_advX, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));
    checkCudaErrors(cufftExecC2C(plan_advX, reinterpret_cast<cufftComplex*>(d_signal),
        reinterpret_cast<cufftComplex*>(d_signal), CUFFT_FORWARD));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_result, d_signal, signalSizeX * signalSizeZ * signalSizeY * sizeof(Complex),
        cudaMemcpyDeviceToHost));

    cout << "x fft\n";
    for (int z = 0; z < signalSizeZ; z++) {
        for (int y = 0; y < signalSizeY; y++) {
            for (int x = 0; x < signalSizeX; x++) {

                cout << "(" << h_result[x + signalSizeX * (y + signalSizeY * z)].x << "; " << h_result[x + signalSizeX * (y + signalSizeY * z)].y << ") ";
            }
            cout << "\n";

        }
        cout << "\n\n";
    }


    
    // fft size y
    tC = signalSizeY;
    tW = gpu_mem_size / tC;

    cufftHandle plan_advY;
    n[0] = signalSizeY;
    
    inembed[0] = tW ;
    onembed[0] = tW;
    istride = signalSizeX;
    idist = 1;
    ostride = signalSizeX;
    odist = 1;
    batch = signalSizeX;

    checkCudaErrors(cufftPlanMany(&plan_advY, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));
    for (int k = 0; k < signalSizeZ; k++) {

        checkCudaErrors(cufftExecC2C(plan_advY, d_signal + (k * signalSizeX * signalSizeY),
            d_signal + (k * signalSizeX * signalSizeY), CUFFT_FORWARD));
    }

    //transport to host

    checkCudaErrors(cudaMemcpy(h_result, d_signal, signalSizeX * signalSizeZ * signalSizeY * sizeof(Complex),
        cudaMemcpyDeviceToHost));


    cout << "result\n";
    for (int z = 0; z < signalSizeZ; z++) {
        for (int y = 0; y < signalSizeY; y++) {
            for (int x = 0; x < signalSizeX; x++) {

                cout << "(" << h_result[x + signalSizeX * (y + signalSizeY * z)].x << "; " << h_result[x + signalSizeX * (y + signalSizeY * z)].y << ") ";
            }
            cout << "\n";

        }
        cout << "\n\n";
    }


Error:

    return cudaStatus;

}

cudaError_t test_3d_dec() {
    long long X = 4;
    long long Y = 2;
    long long Z = 8;

    long long Z1 = (int)(sqrt(Z));
    long long Z2 = Z / Z1;

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
    cout << "signal\n";
    for (int z = 0; z < Z; z++) {
        for (int y = 0; y < Y; y++) {
            for (int x = 0; x < X; x++) {

                cout << "(" << h_signal[x + X * (y + Y * z)].x << "; " << h_signal[x + X * (y + Y * z)].y << ") ";
            }
            cout << "\n";

        }
        cout << "\n";
    }

    cout << "(z1,z2,xy)\n";
    for (int z = 0; z < Z1; z++) {
        for (int y = 0; y < Z2; y++) {
            for (int x = 0; x < X * Y; x++) {

                cout << "(" << h_signal[x + X * Y * (y + Z2 * z)].x << "; " << h_signal[x + X * Y * (y + Z2 * z)].y << ") ";
            }
            cout << "\n";

        }
        cout << "\n";
    }
    //allocate memory for twiddle

    Complex* h_twiddle = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * Z1 * Z2));
    Complex* d_twiddle;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_twiddle), gpu_mem_size));

    //compute twidlle factors
    // exp(2 * pi * (m - 1) * (j - 1) / n), m - строка, j - столбец
    for (int i = 0; i < Z1; i++) {
        for (int j = 0; j < Z2; j++) {
            h_twiddle[i * Z2 + j].x = real(polar(1.0, -2 * M_PI * i * j / signalSize));
            h_twiddle[i * Z2 + j].y = imag(polar(1.0, -2 * M_PI * i * j / signalSize));
            cout << "(" << h_twiddle[i * Z2 + j].x << "; " << h_twiddle[i * Z2 + j].y << ") ";
        }
        cout << "\n";
    }



    C = Z1;
    W = gpu_mem_size / C;

    int tC = C;
    int tW = W;
    


    checkCudaErrors(cudaMemcpyAsync(d_signal, h_signal, gpu_mem_size_b, cudaMemcpyHostToDevice));

    //transfer twiddle

    checkCudaErrors(cudaMemcpy(d_twiddle, h_twiddle, sizeof(Complex) * Z1 * Z2,
        cudaMemcpyHostToDevice));
    //make plan for d(Z1, XYZ2, XY)
    cufftHandle plan_advZ1;

    int n[1] = { C };
    int inembed[] = { C };
    int onembed[] = { W };
    int istride = W;
    int idist = 1;
    int ostride = X * Y ;
    int odist = 1;
    int batch = X * Y;
    // might need for loop (?) cos of possibility of output overlap
    // transpose by advanced layout
    checkCudaErrors(cufftPlanMany(&plan_advZ1, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));
    for (int k = 0; k < Z2; k++) {
        checkCudaErrors(cufftExecC2C(plan_advZ1, reinterpret_cast<cufftComplex*>(d_signal + (k * X * Y)),
            reinterpret_cast<cufftComplex*>(d_result + (k * X * Y * Z1)), CUFFT_FORWARD));
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(h_signal, d_result, X * Y * Z * sizeof(Complex), cudaMemcpyDeviceToHost));
    // have to mult by twiddle factor
    int * XY;
    cudaMallocManaged(&XY, sizeof(int));
    *XY = (int)X * Y;
    /*
    cout << "z1 fft bf twiddle (z2,z1,xy)\n";
    for (int z = 0; z < Z2; z++) {
        for (int y = 0; y < Z1; y++) {
            for (int x = 0; x < X * Y; x++) {

                cout << "(" << h_signal[x + X * Y * (y + Z1 * z)].x << "; " << h_signal[x + X * Y * (y + Z1 * z)].y << ") ";
            }
            cout << "\n";

        }
        cout << "\n";
    }
    */
    dim3 threadsPerBlock(16, 16, 4);
    dim3 numBlocks((C + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (W/(X * Y) + threadsPerBlock.y - 1) / threadsPerBlock.y, ((X * Y) + threadsPerBlock.z - 1) / threadsPerBlock.z);
    TwiddleMult << <numBlocks, threadsPerBlock >> > (d_result, d_twiddle, XY);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(h_signal, d_result, X * Y * Z * sizeof(Complex), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    cout << "z1 fft (z1,z2,xy)\n";
    for (int z = 0; z < Z2; z++) {
        for (int y = 0; y < Z1; y++) {
            for (int x = 0; x < X * Y; x++) {

                cout << "(" << h_signal[x + X * Y * (y + Z1 * z)].x << "; " << h_signal[x + X * Y * (y + Z1 * z)].y << ") ";
            }
            cout << "\n";

        }
        cout << "\n";
    }
    cout << "xyz\n";
    for (int z = 0; z < Z; z++) {
        for (int y = 0; y < Y; y++) {
            for (int x = 0; x < X; x++) {

                cout << "(" << h_signal[x + X * (y + Y * z)].x << "; " << h_signal[x + X * (y + Y * z)].y << ") ";
            }
            cout << "\n";

        }
        cout << "\n";
    }
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

    int h_C = C;
    checkCudaErrors(cufftPlanMany(&plan_advY, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));
    cudaDeviceSynchronize();
    int tZ2 = W / (X * Y);
    for (int k = 0; k < h_C * tZ2; k++) {
        checkCudaErrors(cufftExecC2C(plan_advY, d_signal + (k * X * Y),
            d_signal + (k * X * Y), CUFFT_FORWARD));
    }
    cudaDeviceSynchronize();
    //transport to host
    checkCudaErrors(cudaMemcpy(h_signal, d_signal, gpu_mem_size_b,
        cudaMemcpyDeviceToHost));
    cout << "y fft xyz\n";
    for (int z = 0; z < Z; z++) {
        for (int y = 0; y < Y; y++) {
            for (int x = 0; x < X; x++) {

                cout << "(" << h_signal[x + X * (y + Y * z)].x << "; " << h_signal[x + X * (y + Y * z)].y << ") ";
            }
            cout << "\n";

        }
        cout << "\n";
    }
    
    C = Z2;
    W = gpu_mem_size / C;

    checkCudaErrors(cudaMemcpyAsync(d_signal, h_signal, gpu_mem_size_b, cudaMemcpyHostToDevice));
        

    //make plan for d(Z2, XYZ1, XYZ1)
    cufftHandle plan_advZ2;

    n[0] =  C;
    inembed[0] =  C ;
    onembed[0] =  W ;
    istride = W;
    idist = 1;
    ostride = W;
    odist = 1;
    batch = W;


    checkCudaErrors(cufftPlanMany(&plan_advZ2, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));
    checkCudaErrors(cufftExecC2C(plan_advZ2, reinterpret_cast<cufftComplex*>(d_signal),
        reinterpret_cast<cufftComplex*>(d_signal), CUFFT_FORWARD));
    checkCudaErrors(cudaMemcpy(h_signal, d_signal, gpu_mem_size_b, cudaMemcpyDeviceToHost));
    cout << "z2 fft (z2,z1,xy)\n";
    for (int z = 0; z < Z2; z++) {
        for (int y = 0; y < Z1; y++) {
            for (int x = 0; x < X * Y; x++) {

                cout << "(" << h_signal[x + X * Y * (y + Z1 * z)].x << "; " << h_signal[x + X * Y * (y + Z1 * z)].y << ") ";
            }
            cout << "\n";

        }
        cout << "\n";
    }
    cout << "y fft xyz\n";
    for (int z = 0; z < Z; z++) {
        for (int y = 0; y < Y; y++) {
            for (int x = 0; x < X; x++) {

                cout << "(" << h_signal[x + X * (y + Y * z)].x << "; " << h_signal[x + X * (y + Y * z)].y << ") ";
            }
            cout << "\n";

        }
        cout << "\n";
    }

    //make plan for d(X, 1, 1)

    cufftHandle plan_advX;

    n[0] = X;
    inembed[0] = C;
    onembed[0] = W;
    istride = 1;
    idist = X;
    ostride = 1;
    odist = X;
    batch = C * W / X;


    checkCudaErrors(cufftPlanMany(&plan_advX, 1, n, inembed, istride, idist,
        onembed, ostride, odist, CUFFT_C2C, batch));

    checkCudaErrors(cufftExecC2C(plan_advX, d_signal,
        d_signal, CUFFT_FORWARD));

    checkCudaErrors(cudaMemcpy(h_result, d_signal, gpu_mem_size_b,
        cudaMemcpyDeviceToHost));
    cout << "y fft xyz\n";
    for (int z = 0; z < Z; z++) {
        for (int y = 0; y < Y; y++) {
            for (int x = 0; x < X; x++) {

                cout << "(" << h_result[x + X * (y + Y * z)].x << "; " << h_result[x + X * (y + Y * z)].y << ") ";
            }
            cout << "\n";

        }
        cout << "\n";
    }

    return cudaStatus;
}

// signal size is power of 2
// 2^34 ?
// two round 1d fft algo
cudaError_t fft_1d(long long signalSize) {



    int X1 = (int)sqrt(signalSize);
    int X2 = signalSize / X1;
    // float2 2 * 4byte each element
    long long gpu_mem_size = 268435456; // bytes for 2GB
    long long gpu_mem_size_b = 268435456 * sizeof(Complex); // bytes for 2GB 

    // Allocate host and device memory for the signal
    Complex* h_signal = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * signalSize));
    Complex* h_result = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * signalSize));
    Complex* d_signal;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal), gpu_mem_size));

    // Initialize signal
    for (unsigned int i = 0; i < signalSize; ++i) {
        h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
        h_signal[i].y = 0;
    }


    //memory for twiddle
    Complex* h_twiddle = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * X1 * X2));
    Complex* d_twiddle;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_twiddle), gpu_mem_size));
    //compute twidlle factors
    // exp(2 * pi * (m - 1) * (j - 1) / n), m - строка, j - столбец
    for (int i = 0; i < X1; i++) {
        for (int j = 0; j < X2; j++) {
            h_twiddle[i * X2 + j].x = real(polar(1.0, -2 * M_PI * i * j / signalSize));
            h_twiddle[i * X2 + j].y = imag(polar(1.0, -2 * M_PI * i * j / signalSize));
        }
    }


    C = X1;
    W = gpu_mem_size / C;

    // do i really need buffer ?
    Complex* buffer = reinterpret_cast<Complex*>(malloc(gpu_mem_size_b));


    int it_s = X2 / W;
    for (unsigned int i = 0; i < it_s; i += 1) {
        //transfer data to gpu
        for (unsigned int j = 0; j < C; j++) {
            memcpy(buffer + (j * W), h_signal + (i * W) + (j * X2), W * sizeof(Complex));
            checkCudaErrors(cudaMemcpyAsync(d_signal + (j * W), buffer + (j * W), W * sizeof(Complex), cudaMemcpyHostToDevice));
        }

        //checkCudaErrors(cudaMemcpy(d_signal, buffer, gpu_mem_size_b, 
        //                           cudaMemcpyHostToDevice));

        //transfer twiddle
        for (unsigned int j = 0; j < C; j++) {
            memcpy(buffer + (j * W), h_twiddle + (i * W) + (j * X2), W * sizeof(Complex));
        }
        checkCudaErrors(cudaMemcpy(d_twiddle, buffer, gpu_mem_size_b,
            cudaMemcpyHostToDevice));
        //make plan
        cufftHandle plan_adv;

        int n[1] = { C };
        int inembed[] = { C };
        int onembed[] = { W };
        int istride = W;
        int idist = 1;
        int ostride = 1;
        int odist = C;
        int batch = W;

        //transposing layout
        checkCudaErrors(cufftPlanMany(&plan_adv, 1, n, inembed, istride, idist,
                                        onembed, ostride, odist, CUFFT_C2C,  batch));
        checkCudaErrors(cufftExecC2C(plan_adv, reinterpret_cast<cufftComplex*>(d_signal), 
                                     reinterpret_cast<cufftComplex*>(d_signal), CUFFT_FORWARD));
        // twiddle factor multiplication
        dim3 threadsPerBlock(32, 32);
        dim3 numBlocks((C + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (W + threadsPerBlock.y - 1) / threadsPerBlock.y);
        TwiddleMult<<<numBlocks, threadsPerBlock>>>(d_signal, d_twiddle);
        //transport to host

        checkCudaErrors(cudaMemcpy(d_signal, buffer, gpu_mem_size_b,
                cudaMemcpyDeviceToHost));
        
        memcpy(h_result + (i * C * W), buffer, W * C * sizeof(Complex));
        
    }

    // fft size x2
    C = X2;
    W = gpu_mem_size / C;

    it_s = X1 / W;
    for (unsigned int i = 0; i < it_s; i += 1) {
        //transfer data to gpu
        for (unsigned int j = 0; j < C; j++) {
            memcpy(buffer + (j * W), h_result + (i * W) + (j * X2), W * sizeof(Complex));
            checkCudaErrors(cudaMemcpyAsync(d_signal + (j * W), buffer + (j * W), W * sizeof(Complex), cudaMemcpyHostToDevice));
        }

        //checkCudaErrors(cudaMemcpy(d_signal, buffer, gpu_mem_size_b, 
        //                           cudaMemcpyHostToDevice));

        //make plan
        cufftHandle plan_adv;
        int n[] = { W };
        int inembed[] = { C };
        int onembed[] = { C };
        int istride = C;
        int idist = 1;
        int ostride = C;
        int odist = 1;
        int batch = C;


        checkCudaErrors(cufftPlanMany(&plan_adv, 1, n, inembed, istride, idist,
            onembed, ostride, odist, CUFFT_C2C, batch));
        checkCudaErrors(cufftExecC2C(plan_adv, reinterpret_cast<cufftComplex*>(d_signal),
            reinterpret_cast<cufftComplex*>(d_signal), CUFFT_FORWARD));

        //transport to host

        checkCudaErrors(cudaMemcpy(d_signal, buffer, gpu_mem_size_b,
            cudaMemcpyDeviceToHost));

        memcpy(h_result + (i * C * W), buffer, W * C);

    }

    
    return cudaStatus;
}

cudaError_t fft_naive_3d(long long signalSizeY, long long signalSizeX, long long signalSizeZ){

    // Allocate host memory for the signal
    Complex* h_signal = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * signalSizeY * signalSizeX * signalSizeZ));
    Complex* h_result = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * signalSizeY * signalSizeX * signalSizeZ));

    // Initialize the memory for the signal
    for (unsigned int i = 0; i < signalSizeY * signalSizeX * signalSizeZ; ++i) {
        h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
        h_signal[i].y = 0;
    }

    // how much am i able to use (?)
    // float2 2 * 4byte each element
    long long gpu_mem_size = 268435456; // elements for 2 gb
    long long gpu_mem_size_b = 268435456 * sizeof(Complex); // bytes for 2GB 

    Complex* d_signal;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal), gpu_mem_size));

    if (signalSizeX * signalSizeY < gpu_mem_size) {
        fprintf(stderr, "Cannot use that algorithm, consider using Z - decomposition\n");
        goto Error;
    }
    //assume that X * Y can fit in gpu
    C = signalSizeZ;
    W = gpu_mem_size / C;
    Complex* buffer = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * C * W));


    int it_s = signalSizeY * signalSizeX / W;
    for (int i = 0; i < it_s; i += 1) {
        //transfer data to gpu
        for (unsigned int j = 0; j < C; j++) {
            memcpy(buffer + (j * W), h_signal + (i * W) + (j * signalSizeX * signalSizeY), W);
            checkCudaErrors(cudaMemcpyAsync(d_signal + (j * W), buffer + (j * W), W, cudaMemcpyHostToDevice));
        }

        //checkCudaErrors(cudaMemcpy(d_signal, buffer, gpu_mem_size_b, 
        //                           cudaMemcpyHostToDevice));

        //transfer twiddle


        //make plan for z fft
        cufftHandle plan_advZ;

        int n[1] = { C };
        int inembed[] = { W };
        int onembed[] = { W };
        int istride = W;
        int idist = 1;
        int ostride = W;
        int odist = 1;
        int batch = W;

        //checkCudaErrors(cufftCreate(&plan_adv));
        //it is transpose
        checkCudaErrors(cufftPlanMany(&plan_advZ, 1, n, inembed, istride, idist,
            onembed, ostride, odist, CUFFT_C2C, batch));
        checkCudaErrors(cufftExecC2C(plan_advZ, reinterpret_cast<cufftComplex*>(d_signal),
            reinterpret_cast<cufftComplex*>(d_signal), CUFFT_FORWARD));
        cufftHandle plan_advX;

        // X fft
        n[0] = { (int)signalSizeX };
        inembed[0] = { W };
        onembed[0] = { W };
        istride = 1;
        idist = (int)signalSizeX;
        ostride = 1;
        odist = (int)signalSizeX;
        batch = C * W / signalSizeX;
        //transport to host


        checkCudaErrors(cufftPlanMany(&plan_advX, 1, n, inembed, istride, idist,
            onembed, ostride, odist, CUFFT_C2C, batch));
        checkCudaErrors(cufftExecC2C(plan_advX, reinterpret_cast<cufftComplex*>(d_signal),
            reinterpret_cast<cufftComplex*>(d_signal), CUFFT_FORWARD));
        checkCudaErrors(cudaMemcpy(buffer, d_signal, gpu_mem_size_b,
            cudaMemcpyDeviceToHost));
        memcpy(h_result + (i * C * W), buffer, C * W * sizeof(Complex));

    }
    // fft size x2
    C = signalSizeY;
    W = gpu_mem_size / C;



    it_s = signalSizeX * signalSizeZ / W;
    for (unsigned int i = 0; i < it_s; i += 1) {
        //transfer data to gpu
        
         memcpy(buffer, h_result + (i * W * C), C * W * sizeof(Complex));
         checkCudaErrors(cudaMemcpyAsync(d_signal, buffer, W * C, cudaMemcpyHostToDevice));
        

        //checkCudaErrors(cudaMemcpy(d_signal, buffer, gpu_mem_size_b, 
        //                           cudaMemcpyHostToDevice));

        //make plan
        cufftHandle plan_advY;
        int n[] = { C };
        int inembed[] = { W };
        int onembed[] = { W };
        int istride = signalSizeX;
        int idist = 1;
        int ostride = signalSizeX;
        int odist = 1;
        int batch = signalSizeX;

        //checkCudaErrors(cufftCreate(&plan_adv));
        //does it transpose? (stride, dist)
        checkCudaErrors(cufftPlanMany(&plan_advY, 1, n, inembed, istride, idist,
            onembed, ostride, odist, CUFFT_C2C, batch));

        for (int k = 0; k < W / signalSizeX; k++) {
            checkCudaErrors(cufftExecC2C(plan_advY, d_signal + (k * signalSizeX * C),
                d_signal + (k * signalSizeX * C), CUFFT_FORWARD));
        }

        //transport to host

        checkCudaErrors(cudaMemcpy(buffer, d_signal, gpu_mem_size_b,
            cudaMemcpyDeviceToHost));

        memcpy(h_result + (i * C * W), buffer, C * W * sizeof(Complex));

    }

    Error:

    return cudaStatus;
}

cudaError_t fft_3d(long long signalSizeX,long long signalSizeY, long long  signalSizeZ) {

    long long X = signalSizeX;
    long long Y = signalSizeY;
    long long Z = signalSizeZ;

    long long Z1 = (int)(sqrt(Z));
    long long Z2 = Z / Z1;

    long long  signalSize = X * Y * Z;

    // float2 2 * 4byte each element
    long long gpu_mem_size = 268435456; // elemnts for 2GB
    long long gpu_mem_size_b = 268435456 * sizeof(Complex); // bytes for 2GB 


    // Allocate host and device memory for the signal
    Complex* h_signal = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * signalSize));
    Complex* h_result = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * signalSize));
    Complex* d_signal;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_signal), gpu_mem_size));


    srand(3);
    // Initialize signal
    for (unsigned int i = 0; i < signalSize; ++i) {
        h_signal[i].x = rand() / static_cast<float>(RAND_MAX);
        h_signal[i].y = 0;
    }


    //allocate memory for twiddle

    Complex* h_twiddle = reinterpret_cast<Complex*>(malloc(sizeof(Complex) * Z1 * Z2));
    Complex* d_twiddle;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_twiddle), gpu_mem_size));

    //compute twidlle factors
    // exp(2 * pi * (m - 1) * (j - 1) / n), m - строка, j - столбец
    for (int i = 0; i < Z1; i++) {
        for (int j = 0; j < Z2; j++) {
            h_twiddle[i * Z2 + j].x = real(polar(1.0, -2 * M_PI * i * j / signalSize));
            h_twiddle[i * Z2 + j].y = imag(polar(1.0, -2 * M_PI * i * j / signalSize));
        }
    }
    


    C = Z1;
    W = gpu_mem_size / C;


    Complex* buffer = reinterpret_cast<Complex*>(gpu_mem_size_b);


    // X * Y fits into memory

    int it_s = X * Y * Z2 / W;
    for (unsigned int i = 0; i < it_s; i += 1) {
        //transfer data to gpu
        for (unsigned int j = 0; j < C; j++) {
            memcpy(buffer + (j * W), h_signal + (i * W) + (j * X * Y * Z2), W * sizeof(Complex));
            checkCudaErrors(cudaMemcpyAsync(d_signal + (j * W), buffer + (j * W), W * sizeof(Complex), cudaMemcpyHostToDevice));
        }

        //checkCudaErrors(cudaMemcpy(d_signal, buffer, gpu_mem_size_b, 
        //                           cudaMemcpyHostToDevice));

        //transfer twiddle
        
        memcpy(buffer, h_twiddle + (i * W/(X * Y)) , C * W / (X * Y) * sizeof(Complex));
        
        checkCudaErrors(cudaMemcpy(d_twiddle, buffer, gpu_mem_size_b,
            cudaMemcpyHostToDevice));
        //make plan for d(Z1, XYZ2, XY)
        cufftHandle plan_advZ1;

        int n[1] = { C };
        int inembed[] = { C };
        int onembed[] = { W };
        int istride = W;
        int idist = 1;
        int ostride = X * Y;
        int odist = 1;
        int batch = W;
        // might need for loop (?) cos of possibility of output overlap
        // transpose by advanced layout
        checkCudaErrors(cufftPlanMany(&plan_advZ1, 1, n, inembed, istride, idist,
            onembed, ostride, odist, CUFFT_C2C, batch));
        checkCudaErrors(cufftExecC2C(plan_advZ1, reinterpret_cast<cufftComplex*>(d_signal),
            reinterpret_cast<cufftComplex*>(d_signal), CUFFT_FORWARD));
        // have to mult by twiddle factor
        int h_XY = X * Y;
        /*
        checkCudaErrors(cudaMemset(&XY, 0, sizeof(int)));
        checkCudaErrors(cudaMemcpyFromSymbol(&h_XY, &XY, sizeof(int)));

        dim3 threadsPerBlock(32, 32);
        dim3 numBlocks((C + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (W + threadsPerBlock.y - 1) / threadsPerBlock.y);
        //TwiddleMult <<<numBlocks, threadsPerBlock >>> (d_signal, d_twiddle, XY);
        cudaDeviceSynchronize();
        */

        //make plan for d(Y, X, X)

        cufftHandle plan_advY;

        n[0] = Y ;
        inembed[0] = C;
        onembed[0] = W;
        istride = X;
        idist = 1;
        ostride = X;
        odist = 1;
        batch = X;

        
        checkCudaErrors(cufftPlanMany(&plan_advY, 1, n, inembed, istride, idist,
            onembed, ostride, odist, CUFFT_C2C, batch));
        int tZ2 = W / (X * Y);
        for (int k = 0; k < C * tZ2; k++) {
            checkCudaErrors(cufftExecC2C(plan_advY, d_signal + (k * X * Y),
                d_signal + (k * X * Y), CUFFT_FORWARD));
        }

        //transport to host
        checkCudaErrors(cudaMemcpy(buffer, d_signal, gpu_mem_size_b,
            cudaMemcpyDeviceToHost));
        for (unsigned int j = 0; j < C; j++) {
            memcpy(h_signal + (i * W) + (j * X * Y * Z2), buffer, W * sizeof(Complex));
        }
    }
    C = Z2;
    W = gpu_mem_size / C;

    it_s = X * Y * Z1 / W;
    for (unsigned int i = 0; i < it_s; i += 1) {
        //transfer data to gpu
        for (unsigned int j = 0; j < C; j++) {
            memcpy(buffer + (j * W), h_signal + (i * W) + (j * X * Y * Z1), W * sizeof(Complex));
            checkCudaErrors(cudaMemcpyAsync(d_signal + (j * W), buffer + (j * W), W * sizeof(Complex), cudaMemcpyHostToDevice));
        }


        
        //make plan for d(Z2, XYZ1, XYZ1)
        cufftHandle plan_advZ1;

        int n[1] = { C };
        int inembed[] = { C };
        int onembed[] = { W };
        int istride = W;
        int idist = 1;
        int ostride = W;
        int odist = 1;
        int batch = W;

       
        checkCudaErrors(cufftPlanMany(&plan_advZ1, 1, n, inembed, istride, idist,
            onembed, ostride, odist, CUFFT_C2C, batch));
        checkCudaErrors(cufftExecC2C(plan_advZ1, reinterpret_cast<cufftComplex*>(d_signal),
            reinterpret_cast<cufftComplex*>(d_signal), CUFFT_FORWARD));
     

        
        //make plan for d(X, 1, 1)

        cufftHandle plan_advY;

        n[0] = X;
        inembed[0] = C;
        onembed[0] = W;
        istride = 1;
        idist = X;
        ostride = 1;
        odist = X;
        batch = C * W / X;


        checkCudaErrors(cufftPlanMany(&plan_advY, 1, n, inembed, istride, idist,
            onembed, ostride, odist, CUFFT_C2C, batch));

        checkCudaErrors(cufftExecC2C(plan_advY, d_signal,
             d_signal, CUFFT_FORWARD));
        
        checkCudaErrors(cudaMemcpy(buffer, d_signal, gpu_mem_size_b,
            cudaMemcpyDeviceToHost));
        
        memcpy(h_result + (i * W) , buffer, gpu_mem_size_b);
        
    }


    return cudaStatus;
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s) {
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex pointwise multiplication
static __global__ void TwiddleMult(Complex* X, Complex* twiddle) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < C && j < W)
        X[j*C + i] = ComplexMul(X[j * C + i], twiddle[i*W + j]);
}
static __global__ void TwiddleMult(Complex* X, Complex* twiddle, int * XY) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int xy = *XY;
    if (i < C && j < W/xy && k < xy)
        X[k + xy * (i + C * j)] = ComplexMul(X[k + xy * (i + C * j)], twiddle[i * C + j]);
        //X[k + xy * (i + C * j)].x = i;
        //X[k + xy * (i + C * j)].y = j;
}