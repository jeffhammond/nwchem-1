#include <stdio.h>
#include <stdlib.h>

#if defined(CUBLAS)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#else
#error No GPU BLAS!
#endif

static inline
void * gpualloc(size_t n)
{
    double * x;
    cudaError_t cuda_error = cudaMalloc((void**)&x, n*sizeof(*x));
    if (cuda_error != cudaSuccess) {
        printf("cudaMalloc epäonnistui (%d)\n", cuda_error);
    }
    return x;
}

static inline
void gpufree(void * x)
{
    cudaError_t cuda_error = cudaFree((void*)x);
    if (cuda_error != cudaSuccess) {
        printf("cudaFree epäonnistui (%d)\n", cuda_error);
    }
}

void ccsd_trpdrv_omp_cbody_(double * restrict f1n, double * restrict f1t,
                            double * restrict f2n, double * restrict f2t,
                            double * restrict f3n, double * restrict f3t,
                            double * restrict f4n, double * restrict f4t,
                            double * restrict eorb,
                            int    * restrict ncor_, int * restrict nocc_, int * restrict nvir_,
                            double * restrict emp4_, double * restrict emp5_,
                            int    * restrict a_, int * restrict i_, int * restrict j_, int * restrict k_, int * restrict klo_,
                            double * restrict tij, double * restrict tkj, double * restrict tia, double * restrict tka,
                            double * restrict xia, double * restrict xka, double * restrict jia, double * restrict jka,
                            double * restrict kia, double * restrict kka, double * restrict jij, double * restrict jkj,
                            double * restrict kij, double * restrict kkj,
                            double * restrict dintc1, double * restrict dintx1, double * restrict t1v1,
                            double * restrict dintc2, double * restrict dintx2, double * restrict t1v2)
{
    double emp4 = *emp4_;
    double emp5 = *emp5_;

    double emp4i = 0.0;
    double emp5i = 0.0;
    double emp4k = 0.0;
    double emp5k = 0.0;

    const int ncor = *ncor_;
    const int nocc = *nocc_;
    const int nvir = *nvir_;
    const int nbf  = ncor + nocc + nvir;

    /* convert from Fortran to C offset convention... */
    const int a   = *a_ - 1;
    const int i   = *i_ - 1;
    const int j   = *j_ - 1;
    const int k   = *k_ - 1;
    const int klo = *klo_ - 1;

    const int lnov = nocc * nvir;
    const int lnvv = nvir * nvir;

    const cblas_int nv = nvir;
    const cblas_int no = nocc;

    const double eaijk = eorb[a] - (eorb[ncor+i] + eorb[ncor+j] + eorb[ncor+k]);

    cudaError_t cuda_error;
    cublasStatus_t cublas_error;
    cublasHandle_t cublas_handle;

    cublas_error = cublasCreate(&cublas_handle);
    if (cublas_error != CUBLAS_STATUS_SUCCESS) {
        printf("cublasCreate epäonnistui (%d)\n", cublas_error);
    }

    double * f1n = gpualloc(nvir*nvir);
    double * f2n = gpualloc(nvir*nvir);
    double * f3n = gpualloc(nvir*nvir);
    double * f4n = gpualloc(nvir*nvir);
    double * f1t = gpualloc(nvir*nvir);
    double * f2t = gpualloc(nvir*nvir);
    double * f3t = gpualloc(nvir*nvir);
    double * f4t = gpualloc(nvir*nvir);

    double * eorb = gpualloc(nbf);

    // Tij    lnvv
    double * tij = gpualloc(lnvv);
    // Tkj    kchunk*lnvv
    double * tkj = gpualloc(lnvv*kchunk);
    // Tia    lnov*nocc
    double * tia = gpualloc(lnov*nocc);
    // Tka    kchunk*lnov
    double * tka = gpualloc(lnov*kchunk);
    // Xia    lnov*nocc
    double * xia = gpualloc(lnov*nocc);

    // Xka    kchunk*lnov
    double * xka = gpualloc(lnov*kchunk);
    // Jia    lnvv
    double * jia = gpualloc(lnvv);
    // Jka    kchunk*lnvv
    double * jka = gpualloc(lnvv*kchunk);
    // Kia    lnvv
    double * kia = gpualloc(lnvv);
    // Kka    kchunk*lnvv
    double * kka = gpualloc(lnvv*kchunk);
    // Jij    lnov*nocc
    double * jij = gpualloc(lnov*nocc);
    // Jkj    kchunk*lnov
    double * jkj = gpualloc(lnov*kchunk);
    // Kij    lnov*nocc
    double * kij = gpualloc(lnov*nocc);
    // Kkj    kchunk*lnov
    double * kkj = gpualloc(lnov*kchunk);

    // Dja    lnov
    // Djka   nvir*kchunk
    // Djia   nvir*nocc
    double * dintc1 = gpualloc();
    double * dintx1 = gpualloc();
    double * dintc2 = gpualloc();
    double * dintx2 = gpualloc();

    double * t1v1 = gpualloc();
    double * t1v2 = gpualloc();



    #pragma omp target data \
                       map(to: f1n[0:nvir*nvir], f1t[0:nvir*nvir], \
                               f2n[0:nvir*nvir], f2t[0:nvir*nvir], \
                               f3n[0:nvir*nvir], f3t[0:nvir*nvir], \
                               f4n[0:nvir*nvir], f4t[0:nvir*nvir] ) \
                       map(to: dintc1[0:nvir], dintc2[0:nvir], \
                               dintx1[0:nvir], dintx2[0:nvir], \
                               t1v1[0:nvir],   t1v2[0:nvir] ) \
                       map(to: eorb[0:ncor+nocc+nvir] ) \
                       map(to: nv, no) \
                       map(to: ncor, nocc, nvir, eaijk) \
                       map(tofrom: emp5i, emp4i, emp5k, emp4k) \
                       device(device_num)
    {
        {
            #pragma omp target variant dispatch device(device_num) use_device_ptr(jia,tkj,f1n)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, nv, nv, nv, 1.0, jia, nv, &tkj[(k-klo)*lnvv], nv, 0.0, f1n, nv);
            #pragma omp target variant dispatch device(device_num) use_device_ptr(tia,kkj,f1n)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, no, -1.0, tia, nv, &kkj[(k-klo)*lnov], no, 1.0, f1n, nv);

            #pragma omp target variant dispatch device(device_num) use_device_ptr(kia,tkj,f2n)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, nv, nv, nv, 1.0, kia, nv, &tkj[(k-klo)*lnvv], nv, 0.0, f2n, nv);
            #pragma omp target variant dispatch device(device_num) use_device_ptr(xia,kkj,f2n)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, no, -1.0, xia, nv, &kkj[(k-klo)*lnov], no, 1.0, f2n, nv);

            #pragma omp target variant dispatch device(device_num) use_device_ptr(jia,tkj,f3n)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, nv, 1.0, jia, nv, &tkj[(k-klo)*lnvv], nv, 0.0, f3n, nv);
            #pragma omp target variant dispatch device(device_num) use_device_ptr(tia,jkj,f3n)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, no, -1.0, tia, nv, &jkj[(k-klo)*lnov], no, 1.0, f3n, nv);

            #pragma omp target variant dispatch device(device_num) use_device_ptr(kia,tkj,f4n)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, nv, 1.0, kia, nv, &tkj[(k-klo)*lnvv], nv, 0.0, f4n, nv);
            #pragma omp target variant dispatch device(device_num) use_device_ptr(xia,jkj,f4n)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, no, -1.0, xia, nv, &jkj[(k-klo)*lnov], no, 1.0, f4n, nv);

            #pragma omp target variant dispatch device(device_num) use_device_ptr(jka,tij,f1t)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, nv, nv, nv, 1.0, &jka[(k-klo)*lnvv], nv, tij, nv, 0.0, f1t, nv);
            #pragma omp target variant dispatch device(device_num) use_device_ptr(tka,kij,f1t)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, no, -1.0, &tka[(k-klo)*lnov], nv, kij, no, 1.0, f1t, nv);

            #pragma omp target variant dispatch device(device_num) use_device_ptr(kka,tij,f2t)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, nv, nv, nv, 1.0, &kka[(k-klo)*lnvv], nv, tij, nv, 0.0, f2t, nv);
            #pragma omp target variant dispatch device(device_num) use_device_ptr(xka,kij,f2t)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, no, -1.0, &xka[(k-klo)*lnov], nv, kij, no, 1.0, f2t, nv);

            #pragma omp target variant dispatch device(device_num) use_device_ptr(jka,tij,f3t)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, nv, 1.0, &jka[(k-klo)*lnvv], nv, tij, nv, 0.0, f3t, nv);
            #pragma omp target variant dispatch device(device_num) use_device_ptr(tka,jij,f3t)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, no, -1.0, &tka[(k-klo)*lnov], nv, jij, no, 1.0, f3t, nv);

            #pragma omp target variant dispatch device(device_num) use_device_ptr(kka,tij,f4t)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, nv, 1.0, &kka[(k-klo)*lnvv], nv, tij, nv, 0.0, f4t, nv);
            #pragma omp target variant dispatch device(device_num) use_device_ptr(xka,jij,f4t)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nv, nv, no, -1.0, &xka[(k-klo)*lnov], nv, jij, no, 1.0, f4t, nv);
        }

        #pragma omp target teams distribute parallel for collapse(2) \
                                 reduction(+:emp5i,emp4i,emp5k,emp4k) \
                                 device(device_num)
        for (int b = 0; b < nvir; ++b)
          for (int c = 0; c < nvir; ++c) {

                const double denom = -1.0 / (eorb[ncor+nocc+b] + eorb[ncor+nocc+c] + eaijk);

                // nvir < 10000 so this should never overflow
                const int bc = b+c*nvir;
                const int cb = c+b*nvir;

                const double f1nbc = f1n[bc];
                const double f1tbc = f1t[bc];
                const double f1ncb = f1n[cb];
                const double f1tcb = f1t[cb];

                const double f2nbc = f2n[bc];
                const double f2tbc = f2t[bc];
                const double f2ncb = f2n[cb];
                const double f2tcb = f2t[cb];

                const double f3nbc = f3n[bc];
                const double f3tbc = f3t[bc];
                const double f3ncb = f3n[cb];
                const double f3tcb = f3t[cb];

                const double f4nbc = f4n[bc];
                const double f4tbc = f4t[bc];
                const double f4ncb = f4n[cb];
                const double f4tcb = f4t[cb];

                emp4i += denom * (f1tbc+f1ncb+f2tcb+f3nbc+f4ncb) * (f1tbc-f2tbc*2-f3tbc*2+f4tbc)
                       - denom * (f1nbc+f1tcb+f2ncb+f3ncb) * (f1tbc*2-f2tbc-f3tbc+f4tbc*2)
                       + denom * 3 * (f1nbc*(f1nbc+f3ncb+f4tcb*2) +f2nbc*f2tcb+f3nbc*f4tbc);
                emp4k += denom * (f1nbc+f1tcb+f2ncb+f3tbc+f4tcb) * (f1nbc-f2nbc*2-f3nbc*2+f4nbc)
                       - denom * (f1tbc+f1ncb+f2tcb+f3tcb) * (f1nbc*2-f2nbc-f3nbc+f4nbc*2)
                       + denom * 3 * (f1tbc*(f1tbc+f3tcb+f4ncb*2) +f2tbc*f2ncb+f3tbc*f4nbc);

                const double t1v1b = t1v1[b];
                const double t1v2b = t1v2[b];

                const double dintx1c = dintx1[c];
                const double dintx2c = dintx2[c];
                const double dintc1c = dintc1[c];
                const double dintc2c = dintc2[c];

                emp5i += denom * t1v1b * dintx1c * (f1tbc+f2nbc+f4ncb-(f3tbc+f4nbc+f2ncb+f1nbc+f2tbc+f3ncb)*2
                                                    +(f3nbc+f4tbc+f1ncb)*4)
                       + denom * t1v1b * dintc1c * (f1nbc+f4nbc+f1tcb -(f2nbc+f3nbc+f2tcb)*2);
                emp5k += denom * t1v2b * dintx2c * (f1nbc+f2tbc+f4tcb -(f3nbc+f4tbc+f2tcb +f1tbc+f2nbc+f3tcb)*2
                                                    +(f3tbc+f4nbc+f1tcb)*4)
                       + denom * t1v2b * dintc2c * (f1tbc+f4tbc+f1ncb -(f2tbc+f3tbc+f2ncb)*2);
        }
    }

    emp4 += emp4i;
    emp5 += emp5i;

    if (*i_ != *k_) {
        emp4 += emp4k;
        emp5 += emp5k;
    }

    *emp4_ = emp4;
    *emp5_ = emp5;

    return;
}

