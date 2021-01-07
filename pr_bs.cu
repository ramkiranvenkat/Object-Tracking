#include <stdint.h> /* for uint64 definition */
#include <time.h> /* for clock_gettime() */
#include <stdio.h>
#include <stdlib.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define BILLION 1000000000L
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
// parallel resampling with binary search
__global__ void par_resampling_bs(double *cdfd, int *new_idxd, int N, double u0) {
	int t = blockIdx.x * blockDim.x + threadIdx.x;
	double u = ((double)t + u0)/(double)N;
	int m, l = 0, r = N - 1;
	while(1) {
		m = (l + r) / 2;
		if (cdfd[m] < u && cdfd[m+1] >= u)		break;
		else if (cdfd[m] < u && cdfd[m+1] < u) {
			l = m + 1;
		} else if (cdfd[m] > u && cdfd[m+1] > u) {
			r = m;
		}
		if (r <= l) {
			--m;
			break;
		}
	}
	new_idxd[t] = m+1;
}
void seq_resampling(double *cdf, int *new_idx, int N, double u0) {
	double u = u0 / (double)N;
	int icdf = 0;
	for (int i = 0; i < N; ++i) {
		while(cdf[icdf] < u)    ++icdf;
		new_idx[i] = icdf;
		u += (1.0 / N);
	}
}
void printVals(int *new_idx_ser, int *new_idx_par, int N) {
	// for (int i =0; i < N; ++i)		printf("%6d", new_idx_ser[i]);
	// printf("\n\n");
	// for (int i =0; i < N; ++i)		printf("%6d", new_idx_par[i]);
	// printf("\n\n");
	FILE *fNum;
// printing ser
  char fn[50] = "";
  sprintf(fn, "outser.csv");
  fNum = fopen(fn, "w");
  if (fNum == NULL) {
    printf("Error with fNum file.\n");
  }
  for (unsigned int i = 0; i < N; ++i) {
    fprintf(fNum, "%6d\n", new_idx_ser[i]);
  }
  fclose(fNum);
// printing par
  sprintf(fn, "outpar.csv");
  fNum = fopen(fn, "w");
  if (fNum == NULL) {
    printf("Error with fNum file.\n");
  }
  for (unsigned int i = 0; i < N; ++i) {
    fprintf(fNum, "%6d\n", new_idx_par[i]);
  }
  fclose(fNum);
}
int check_correctness(int *new_idx_ser, int *new_idx_par, int N) {
	for (int i = 0; i < N; ++i) {
		if (new_idx_ser[i] != new_idx_par[i])		return 0;
	}
	return 1;
}
void printcdfandu(double *cdf, double u0, int N) {
	FILE *fNum;
// printing cdf
  char fn[50] = "";
  sprintf(fn, "cdf.csv");
  fNum = fopen(fn, "w");
  if (fNum == NULL) {
    printf("Error with fNum file.\n");
  }
  for (unsigned int i = 0; i < N; ++i) {
    fprintf(fNum, "%f\n", cdf[i]);
  }
  fclose(fNum);
// printing u vals
  sprintf(fn, "uvals.csv");
  fNum = fopen(fn, "w");
  if (fNum == NULL) {
    printf("Error with fNum file.\n");
  }
  for (unsigned int i = 0; i < N; ++i) {
    fprintf(fNum, "%f\n", ((double)i + u0)/(double)N);
  }
  fclose(fNum);
}

int main() {
	srand(5489ULL);
	uint64_t diff;
  struct timespec start, end;
	FILE *fp;
	char fn[50] = "";
  sprintf(fn, "res_bs.tsv");
  fp = fopen(fn, "w");
  if (fp == NULL) {
    printf("Error with fp file.\n");
  }
	fprintf(fp, "N\tserial time\tparallel time\n");
	for (long long int N = 1024; N <= 1024*1024; N *= 2) {
		double *w = (double *)malloc(sizeof(double) * N);
		// double w[] = {0.1, 0.05, 0.05, 0.3, 0.1, 0.4};
		double totalw = 0.0;
		for (int i = 0; i < N; ++i) {
			w[i] = (double)rand() / RAND_MAX;
			totalw += w[i];
		}
		double *cdf = (double *)malloc(sizeof(double) * N);
		cdf[0] = w[0] / totalw;
		for (int i = 1; i < N; ++i) {
			cdf[i] = cdf[i - 1] + (w[i] / totalw);
		}
		int *new_idx_ser, *new_idx_par;
		new_idx_ser = (int *)malloc(sizeof(int) * N);
		new_idx_par = (int *)malloc(sizeof(int) * N);
		double u0 = (double)rand() / RAND_MAX;

	/////////////////////////////////////////////////////////////////////////////////////
		
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start); /* mark start time */
		seq_resampling(cdf, new_idx_ser, N, u0);
		// printf("Completed serial.\n");
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end); /* mark the end time */
		diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
		fprintf(fp, "%d\t%-7.3f", N, (double)diff / BILLION * 1000);  // ms
		// printf("s: %-7.3f\n", (double)diff / BILLION * 1000);  // ms

	/////////////////////////////////////////////////////////////////////////////////////
		
		gpuErrchk( cudaSetDevice(0) );
		double *cdfd;
		int *new_idxd;
		gpuErrchk( cudaMalloc((void**)&cdfd    , N * sizeof(double)) );
		gpuErrchk( cudaMalloc((void**)&new_idxd, N * sizeof(int)) );
		cudaMemcpy(cdfd, cdf, N * sizeof(double), cudaMemcpyHostToDevice);
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start); /* mark start time */
		par_resampling_bs<<<(N+1023)/1024, 1024>>>(cdfd, new_idxd, N, u0);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end); /* mark the end time */
		diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
		fprintf(fp, "\t%-7.3f\n", (double)diff / BILLION * 1000);  // ms
		// printf("p: %-7.3f\n", (double)diff / BILLION * 1000);  // ms
		
		gpuErrchk( cudaMemcpy(new_idx_par, new_idxd, N * sizeof(int), cudaMemcpyDeviceToHost) );
		cudaFree(new_idxd);  cudaFree(cdfd);
		// printf("Completed parallel with bs.\n");

	/////////////////////////////////////////////////////////////////////////////////////
		// printVals(new_idx_ser, new_idx_par, N);
		// printcdfandu(cdf, u0, N);
		int ans = check_correctness(new_idx_ser, new_idx_par, N);
		printf("N : %10ld   correctness: %d (0: wrong, 1: correct)\n", N, ans);
	}
	fclose(fp);
	return 0;	
}