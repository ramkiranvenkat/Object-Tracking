#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <random>
#include <algorithm>
#include <cmath>
#include<time.h>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>

#define SPEED 3
#define trdMax 1024
#define NOEPT  3
using namespace std;
using namespace cv;

ofstream f1("parallelTiming");

//g++ ppp.cpp -o vc `pkg-config --cflags --libs opencv4`

// pending
// free memory
// histogram parallelization

__global__ void setup_kernel(curandState* state) 
{
  	int idx = blockIdx.x*trdMax + threadIdx.x;
  	curand_init(idx, idx, 0, &state[idx]);  // 1st input seed
}

__global__ void cudaColorHistInit(int* histogram)
{
	int i = threadIdx.x;
	histogram[i] = 0;
}

__global__ void cudaComputeColorHist(unsigned char* cudaImgArray,int* cudaColorHist,int NOEB,int NBINS)
{
	int idx = blockIdx.x*trdMax + threadIdx.x;
	int i,iaIdx = idx*NOEPT; 
	// we are interested in idx*NOEPT+i
	for (i=0;i<NOEPT;i++)
	{	
		printf("%d : %d %d\n",idx,((iaIdx+i)%3)*NBINS + ((int)cudaImgArray[iaIdx+i])/NOEB,((int)cudaImgArray[iaIdx+i]));
		atomicAdd(&cudaColorHist[((iaIdx+i)%3)*NBINS + ((int)cudaImgArray[iaIdx+i])/NOEB],1);
	}
}

__global__ void  cudaForwardModel(int* particles,float* Q,curandState* rState,int NOP)
{
	int idx = blockIdx.x*trdMax + threadIdx.x;
	
	particles[0+idx*4] = particles[0+idx*4] + (int)((float)particles[2+idx*4]*0.0625) + (int)(Q[0]*(float)curand_normal(&rState[idx]));
	particles[1+idx*4] = particles[1+idx*4] + (int)((float)particles[3+idx*4]*0.0625) + (int)(Q[1]*(float)curand_normal(&rState[idx]));
	particles[2+idx*4] = particles[2+idx*4] + (int)(Q[2]*(float)curand_normal(&rState[idx]));
	particles[3+idx*4] = particles[3+idx*4] + (int)(Q[3]*(float)curand_normal(&rState[idx]));
}

__global__ void cudaMeasurement(unsigned char* imgA,float* hist,int* particles,float* w,float* wm,int irows,int icols,int width,int height,int NBINS,int step)
{
	int idx = blockIdx.x*trdMax + threadIdx.x;
	int N = 3*NBINS;
	float lhist[30];
	w[idx] = 0;
	if (!((particles[0+idx*4] >= icols) || (particles[0+idx*4] < 0) || (particles[1+idx*4] >= irows) || (particles[1+idx*4] < 0))) 
	{
		int sx = (((particles[0+idx*4]+width )<icols)?(particles[0+idx*4]+width ):icols-1) - particles[0+idx*4];
		int sy = (((particles[1+idx*4]+height)<irows)?(particles[1+idx*4]+height):irows-1) - particles[1+idx*4];

		if (!((!sy) || (!sx)))
		{
			// color histogram
			int NOEB = 255/NBINS + 1;
			float sum = 0;
			for (int i=0;i<N;i++) lhist[i] = 0;
			int initPos = (particles[1+idx*4]*step+3*particles[0+idx*4]);
			for (int i=0,imax=sx*sy;i<imax;i++)
			{
				int r = i/sx;
				int c = i%sx;
				for (int k=0;k<3;k++) 
				{
						lhist[k*NBINS + ((int)imgA[initPos + r*step+c*3+k])/NOEB]+=1.0;
						sum+=1.0;
				}
			}
			for (int i=0;i<N;i++) w[idx] += sqrt(hist[i]*lhist[i]/sum);
		}
	}
	wm[idx] = w[idx];
	__syncthreads();
}

__global__ void vecSum(float* a,float* b)
{
        int tindex = threadIdx.x;
	int bindex = blockIdx.x;
	int blkOff = blockIdx.x*trdMax;
        a[tindex+blkOff] = a[tindex+blkOff]+b[bindex];
        __syncthreads();
	if ((tindex + bindex) == 0) a[tindex] = 1.0;
}

__global__ void vecDiv(float* cudaCorr,float* sum)
{
	int idx = blockIdx.x*trdMax+threadIdx.x;
	cudaCorr[idx] = cudaCorr[idx]/sum[0];
}
__global__ void presumBlock(float* a,float* b,int N)
{
        int idx = threadIdx.x;
	int blkOff = blockIdx.x*trdMax;
	int offset = 1;
	for (int i = N>>1; i > 0; i >>= 1) // build sum in place up the tree     
	{         
		__syncthreads();
		if (idx < i)            
		{ 
			int ai = offset*(2*idx+1)-1; 
			int bi = offset*(2*idx+2)-1;             
			a[bi + blkOff] += a[ai + blkOff];           
		}         
		offset *= 2;     
	}
	if (idx == 0) 
	{ 
		b[blockIdx.x] = a[N - 1 + blkOff];
		a[N - 1 + blkOff] = 0; 
	}
	
	for (int i = 1; i < N; i*= 2) // traverse down tree & build scan     
	{         
		offset >>= 1;         
		__syncthreads(); 
		if (idx < i)         
		{ 
			int ai = offset*(2*idx+1)-1; 
			int bi = offset*(2*idx+2)-1; 
			float t   = a[ai + blkOff];             
			a[ai + blkOff]  = a[bi + blkOff];             
			a[bi+ blkOff] += t;         
		}     
	}
}

__global__ void presum(float* a,float* b,int N)
{
        int idx = threadIdx.x;
	int offset = 1;
	for (int i = N>>1; i > 0; i >>= 1) // build sum in place up the tree     
	{         
		__syncthreads();
		if (idx < i)            
		{ 
			int ai = offset*(2*idx+1)-1; 
			int bi = offset*(2*idx+2)-1;             
			a[bi] += a[ai];           
		}         
		offset *= 2;     
	}
	if (idx == 0) 
	{
		b[0] = a[N-1];
		a[N - 1 ] = 0; 
	}
	
	for (int i = 1; i < N; i*= 2) // traverse down tree & build scan     
	{         
		offset >>= 1;         
		__syncthreads(); 
		if (idx < i)         
		{ 
			int ai = offset*(2*idx+1)-1; 
			int bi = offset*(2*idx+2)-1; 
			float t   = a[ai];             
			a[ai]  = a[bi];             
			a[bi] += t;         
		}     
	}
}

__global__ void cudaResample(int *p,float* cdfp,float u0,int N)
{
	int t = blockIdx.x * blockDim.x + threadIdx.x;
  	int idx = t;
  	float u = ((float)t + u0) / (float)N;
  	char m = 1;
  	int el = 0;
	
	while (m) 
	{
		if (t >= (N - el))	m = 0;
		else m = (cdfp[(t + el+1)%N] < u);
		if (m) idx++;
		el++;
	}	
	el = 1;
	while(!m) {
		if (t <= el)	m = 1;
		else m = (cdfp[(t - el+1)%N] < u);
		if (!m) idx--;
		el++;
	}
  	float pn[4];
	for (int i=0;i<4;i++) pn[i] = p[4*idx + i];
  	__syncthreads();
  	for (int i=0;i<4;i++) p[4*t+i] = pn[i];
}

__global__ void bsc(float* p,float* r,int N)
{
	int idx = threadIdx.x;	
	r[0] = 0;
	__syncthreads();
	atomicAdd(r,sqrt(p[N+idx]*p[idx]));
}

__global__ void cudaImagePrint(unsigned char* imgA,int step,int r,int c)
{
	int idx = threadIdx.x;
	int i = idx/c;
	int j = idx%c;
	printf("%d\t: %d %d %d\n",idx,imgA[i*step+j*3+0],imgA[i*step+j*3+1],imgA[i*step+j*3+2]);
}

__global__ void cudaFindMaxItr(float *av,int *aidx,float *bv,int *bidx,int N)
{
        int idx = threadIdx.x;
	int blkOff = blockIdx.x*trdMax;
	int offset = 1;
	aidx[idx] = idx;
	__syncthreads();
	for (int i = N>>1; i > 0; i >>= 1) // build sum in place up the tree     
	{         
		__syncthreads();
		if (idx < i)            
		{ 
			int ai = offset*(2*idx+1)-1; 
			int bi = offset*(2*idx+2)-1; 
			aidx[bi + blkOff] = ((av[bi + blkOff] > av[ai + blkOff])?aidx[bi + blkOff]:aidx[ai + blkOff]);              
			av[bi + blkOff] = ((av[bi + blkOff] > av[ai + blkOff])?av[bi + blkOff]:av[ai + blkOff]);  
			        
		}         
		offset *= 2;     
	}
	__syncthreads();
	if (idx == 0) 
	{
		bv[blockIdx.x] = av[N - 1 + blkOff];
		bidx[blockIdx.x] = aidx[N - 1 + blkOff];
	}
}

/// cuda functions above

float median(float inp[], int NOP)
{
	float *another = new float[NOP];
  	for (int i = 0; i < NOP; i++) another[i] = inp[i];
  	sort(another, another + NOP);
  	float out = another[NOP / 2];
  	delete another;
  	return out;
}

int argmax(float inp[], int NOP)
{
  	int ind = 0;
  	float maxval = 0.0;
  	for (int i = 0; i < NOP; ++i)
  	{
    		if (inp[i] > maxval)
    		{
      			maxval = inp[i];
      			ind = i;
    		}
  	}
  	return ind;
}

void computeColorHist(int NBINS,Mat inpImg,float* colorhist,int placeHolder)
{
	int NOEB = 255/NBINS + 1;
	Vec3b pixel;
	placeHolder = placeHolder*3;

	float sum = 0;
	Size s = inpImg.size();

	for (int i=0;i<3;i++)
	{
		for (int j=0;j<NBINS;j++) colorhist[(i+placeHolder)*NBINS + j] = 0;
	}

	for (int i=0,imax=s.height;i<imax;i++)
	{
		for (int j=0,jmax=s.width;j<jmax;j++)
		{
			pixel =  inpImg.at<Vec3b>(i,j);
			for (int k=0;k<3;k++) 
			{
				colorhist[(k+placeHolder)*NBINS + ((int)pixel[k])/NOEB]+=1.0;
				sum+=1.0;
			}
		}		
	}
	
	for (int i=0;i<3;i++)
	{
		for (int j=0;j<NBINS;j++) 
		{
		colorhist[(i+placeHolder)*NBINS + j] = colorhist[(i+placeHolder)*NBINS + j]/sum;
		/*
		if (isnan(colorhist[i+placeHolder*NBINS+j]))
		{
			cout << "Bug variable" << " " << s.height << " " << s.width <<  endl;
			cout << sum << endl;
			exit(0);
		}
		*/

		}
	}
}

float BhattacharyyaSimilarityCoefficient(int N,float *hist)
{
	float ret = 0;
	for (int i=0;i<3*N;i++) ret += sqrt(hist[i]*hist[i+3*N]);
	return ret;
}


void particleImage(Mat img,int* p,int NOP,Scalar color,int xoffset,int yoffset)
{
	for (int i = 0; i < NOP; i++)
      	{
          	Point plocal(p[0+i*4]+xoffset, p[1+i*4]+yoffset);
          	drawMarker(img, plocal, color, MARKER_CROSS, 1, 8);
      	}
	imshow( "Particle Image", img );
}

int main()
{
	// Pseudo Random Number Generator
  	default_random_engine generator;
  	normal_distribution<float> distribution(0.0, 1.0);

  	// Create a VideoCapture object and open the input file
  	// If the input is the web camera, pass 0 instead of the video file name
  	VideoCapture cap("ball.mp4");
  	// Check if camera opened successfully
  	if (!cap.isOpened())
  	{
    		cout << "Error opening video stream or file" << endl;
    		return -1;
  	}
  	// object selection
  	float *P,*cudaP; P = new float[4];
	P[0] = 5;
	P[1] = 5;
	P[2] = 5;
	P[3] = 5;
	cudaMalloc((void**)&cudaP, 4*sizeof(float));
	cudaMemcpy(cudaP, P, 4*sizeof(float), cudaMemcpyHostToDevice);
  	float *Q,*cudaQ; Q = new float[4];
	Q[0] = 3;
	Q[1] = 3;
	Q[2] = 3;
	Q[3] = 3;
	cudaMalloc((void**)&cudaQ, 4*sizeof(float));
	cudaMemcpy(cudaQ, Q, 4*sizeof(float), cudaMemcpyHostToDevice);

	
  
  	int selectionBit = 0;
  	int NBINS  = 10;
  	float* colorhist  = new float[6*NBINS];
	float* cudaColorHist;
  	Rect r,rp;
  	Mat frame, imCrop;
  	int *particles,*nparticles, NOP = 65536; // number of particles
	int *cudaParticles;
  	particles = new int [4*NOP];
	nparticles = new int [4*NOP];

	float bscvl;
  	float *corr = new float[NOP]; // Correlation
	float *cudaCorr,*bCorr;
	cudaMalloc((void**)&cudaCorr,NOP*sizeof(float));
	cudaMalloc((void**)&bCorr,NOP/trdMax*sizeof(float));

  	Size imgS;
  	int selItr = 0;
	int imgstep;
	int nchannels;
	int NOE;
	float* bscv;
	float* cudaSum;

	float* cudaCorrMax;
	float* cudaCorrMaxVal;
	int*   cudaCorrMaxItr;
	int*   cudaMaxItr;
	int*   bCorrMax;
	cudaMalloc((void**)&cudaCorrMax,NOP*sizeof(float));
	cudaMalloc((void**)&cudaCorrMaxItr,NOP*sizeof(int));
	cudaMalloc((void**)&cudaMaxItr,sizeof(int));
	cudaMalloc((void**)&cudaCorrMaxVal,sizeof(float));
	cudaMalloc((void**)&bCorrMax,NOP/trdMax*sizeof(int));
	
	cout << (int)ceil(NOP/trdMax) << " " <<  trdMax << endl;
	curandState* randStates;
	unsigned char* cudaImgArray;
	cudaMalloc(&randStates, NOP * sizeof(curandState));
	cudaMalloc((void**)&cudaColorHist, 6*NBINS*sizeof(float));
	cudaMalloc((void**)&cudaParticles, 4*NOP*sizeof(int));
	cudaMalloc((void**)&bscv,sizeof(float));
	cudaMalloc((void**)&cudaSum,sizeof(float));
	//exit(0);
	setup_kernel<<<(int)ceil((float)NOP/trdMax), trdMax>>>(randStates);
	struct timespec start, finish,startf,finishf;
	
	cudaDeviceSynchronize();
  	while (1)
  	{
		selItr++;
    		cap >> frame;
    		if (frame.empty())
      		break;

    		// cvtColor(frame, grey, COLOR_BGR2GRAY);
    		imshow( "Frame", frame );
    		if (selItr < 0) continue;
	
    		if (!selectionBit)
    		{
      			// Region of Interest selection
      			selectionBit = 1;
      			r = selectROI(frame);
			Mat imcrop = frame(r);
	  		imshow("Selected Dist", imcrop);
      			// cout << r << endl;
//cout << r.x << " " << r.y << " " << r.height << " " << r.width << " " << r.x + r.width / 2 << " " << r.y + r.height / 2 << endl;
//cout << imcrop.step[0] << " " << imcrop.step[1] << " " << imcrop.step[2] << endl;
			/*
			Vec3b pixel =  imcrop.at<Vec3b>(0,0);
			cout << pixel << ", "; 
			for  (int i=0;i<3;i++)cout << (int)pixel[i] << " ";
			cout << endl;*/

			//unsigned char*  imarray = (unsigned char*)imcrop.data;	
			
			unsigned char*  imarray = imcrop.ptr<unsigned char>(0);
			imgstep = imcrop.step[0];
			nchannels = imcrop.channels();
			NOE = imcrop.rows*imcrop.cols*nchannels;

			cudaMalloc((void**)&cudaImgArray,frame.rows*frame.step[0]*sizeof(unsigned char));
			//cudaMemcpy(cudaImgArray, imarray,imcrop.rows*imgstep*sizeof(unsigned char), cudaMemcpyHostToDevice);
			//cudaImagePrint<<<1,imcrop.rows*imcrop.cols>>>(cudaImgArray,imgstep,imcrop.rows,imcrop.cols);
			

			//cout << NOE << " " << (int)ceil((float)NOE/trdMax/NOEPT) << endl;exit(0);
			/*
			for(int i=0; i<imcrop.rows; ++i)
			{
			    	for(int j=0; j<imcrop.cols; ++j)
				{
					cout << i << " " << j << " ";
					for(int ch=0; ch<nchannels; ++ch)
					{
				    		int val = imarray[(imgstep*i) + (nchannels*j) + ch];
						cout << val << " ";
					}
					cout << " | ";
					Vec3b pixel =  imcrop.at<Vec3b>(i,j);
					cout << (int)pixel[0] << " " << (int)pixel[1] << " " << (int)pixel[2]; 
					cout << endl;
			    	}
				cout << "---------------------------------------" << endl;
			}
			
			for (int i=0;i<imcrop.rows*imcrop.cols;i++)
			{
				int r = i/imcrop.cols;
				int c = i%imcrop.cols;
				cout << r << " " << c << " ";
				for(int ch=0; ch<nchannels; ++ch)
				{
					int val = imarray[(imgstep*r) + (nchannels*c) + ch];
					cout << val << " ";
				}
				cout << endl;
			}*/
			//cout << imcrop.step << " " << imcrop.channels() << endl;exit(0);
			
			computeColorHist(NBINS,imcrop,colorhist,0); // 0 past 1 present
			cudaMemcpy(cudaColorHist, colorhist, 3*NBINS*sizeof(float), cudaMemcpyHostToDevice);
			/*
			for (int i=0;i<3*NBINS;i++) cout << colorhist[i] << " ";
			cout << endl;
			// making colorHistogramZero
			cudaColorHistInit<<<1,3*NBINS>>>(&cudaColorHist[3*0*NBINS]);
			cudaMemcpy(cudaImgArray, imarray, NOE*sizeof(unsigned char), cudaMemcpyHostToDevice);
			int NOEB = 255/NBINS + 1;
			cudaComputeColorHist<<<(int)ceil((float)NOE/trdMax/NOEPT),trdMax>>>(cudaImgArray,&cudaColorHist[3*0*NBINS],NOEB,NBINS);
			cudaMemcpy(&colorhist[3*NBINS], cudaColorHist, 3*NBINS*sizeof(int), cudaMemcpyDeviceToHost);
			for (int i=0;i<3*NBINS;i++) cout << colorhist[i+3*NBINS] << " ";
			cout << endl;
			exit(0);
			/*
			for (int i=0;i<3;i++)
			{
				for (int j=0;j<NBINS;j++) cout << colorhist[i*NBINS+j] << " ";
				cout << endl;
			}*/
		
			int randomVariable;
		    	imgS = frame.size();
		    	int irows = imgS.height;
		    	int icols = imgS.width;

		  	for (int i = 0; i < NOP; i++)
		  	{
				
		    		while (1)
		    		{
		      			randomVariable = (int)(P[0]*distribution(generator));
		      			if (randomVariable + r.x < icols && randomVariable + r.x > 0) // width axis
					break;
		    		}
		    		particles[0+i*4] = randomVariable + r.x; // x pos  
		    		while (1)
		    		{
		      			randomVariable = (int)(P[1]*distribution(generator));
		      			if (randomVariable + r.y < irows && randomVariable + r.y > 0) // height axis
					break;
		    		}
		    		particles[1+i*4] = randomVariable + r.y; // y pos - top left corner
				particles[2+i*4] = (int)(P[2]*distribution(generator)); // x - vel
				particles[3+i*4] = (int)(P[3]*distribution(generator)); // y - vel
		   	}
			cudaMemcpy(cudaParticles, particles, 4*NOP*sizeof(int), cudaMemcpyHostToDevice);
		   	//Scalar color(0,0,200);
		   	//particleImage(frame,particles,NOP,color,0*r.width/2,0*r.height/2);
		}
	
    		else
    		{
      			// create new particles
			double time_seq = 0;
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &startf);
		  	imgS = frame.size();
		  	int irows = imgS.height;
		  	int icols = imgS.width;

		  	int maxItr;
			cudaMemcpy(cudaImgArray, frame.ptr<unsigned char>(0),frame.rows*frame.step[0]*sizeof(unsigned char), cudaMemcpyHostToDevice);
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &finish); 
			{
			    	long seconds = finish.tv_sec - start.tv_sec; 
			    	long ns = finish.tv_nsec - start.tv_nsec;
			    	time_seq = (double)seconds + (double)ns/(double)1000000000;
				f1 << time_seq << " ";
			}

			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);			
			cudaForwardModel<<<(int)ceil((float)NOP/trdMax), trdMax>>>(cudaParticles,cudaQ,randStates,NOP);
			

cudaMeasurement<<<(int)ceil((float)NOP/trdMax), trdMax>>>	(cudaImgArray,cudaColorHist,cudaParticles,cudaCorr,cudaCorrMax,
							   	 irows,icols,r.width,r.height,NBINS,frame.step[0]);
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &finish); 
			{
			    	long seconds = finish.tv_sec - start.tv_sec; 
			    	long ns = finish.tv_nsec - start.tv_nsec;
			    	time_seq = (double)seconds + (double)ns/(double)1000000000;
				f1 << time_seq << " ";
			}

			cudaFindMaxItr<<<(int)ceil((float)NOP/trdMax), trdMax>>>(cudaCorrMax,cudaCorrMaxItr,bCorr,bCorrMax,trdMax);
			int bclcl[NOP/trdMax];
			cudaMemcpy(bclcl,bCorrMax,NOP/trdMax*sizeof(int), cudaMemcpyDeviceToHost);
			//for (int i=0;i<NOP/trdMax;i++) cout << " " << bclcl[i];
			//cout << endl;
			//cudaFindMaxItr<<<1,NOP/trdMax>>>(bCorr,bCorrMax,cudaCorrMaxVal,cudaCorrMaxItr,NOP/trdMax);
			
			maxItr = bclcl[0];
			
			int maxx,maxy;
			cudaMemcpy(&maxx,&cudaParticles[0+maxItr*4],sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&maxy,&cudaParticles[1+maxItr*4],sizeof(int), cudaMemcpyDeviceToHost);
	  		
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
			presumBlock<<<NOP/trdMax,trdMax>>>(cudaCorr,bCorr,trdMax);
			presum<<<1,NOP/trdMax>>>(bCorr,cudaSum,NOP/trdMax);
			vecSum<<<NOP/trdMax,trdMax>>>(cudaCorr,bCorr);
			vecDiv<<<NOP/trdMax,trdMax>>>(cudaCorr,cudaSum);
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &finish); 
			{
			    	long seconds = finish.tv_sec - start.tv_sec; 
			    	long ns = finish.tv_nsec - start.tv_nsec;
			    	time_seq = (double)seconds + (double)ns/(double)1000000000;
				f1 << time_seq << " ";
			}

	  		float rpt = 1.0 /(float)NOP * ((float)rand() / RAND_MAX);
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
			cudaResample<<<NOP/trdMax,trdMax>>>(cudaParticles,cudaCorr,rpt,NOP);
			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &finish); 
			{
			    	long seconds = finish.tv_sec - start.tv_sec; 
			    	long ns = finish.tv_nsec - start.tv_nsec;
			    	time_seq = (double)seconds + (double)ns/(double)1000000000;
				f1 << time_seq << " ";
			}

			clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &finishf); 
			{
			    	long seconds = finishf.tv_sec - startf.tv_sec; 
			    	long ns = finishf.tv_nsec - startf.tv_nsec;
			    	time_seq = (double)seconds + (double)ns/(double)1000000000;
				f1 << time_seq << " ";
			}
			f1 << endl;
			Point p1((int)maxx,(int)maxy);
			Point p2((int)maxx+r.width,(int)maxy+r.height);

			Scalar clr(0,255,255);
			rectangle(frame,p1,p2,clr,2);
			imshow( "Frame", frame );

    		}
		cudaMemcpy(particles,cudaParticles,4*NOP*sizeof(int), cudaMemcpyDeviceToHost);
		Scalar color(0,200,200);
		particleImage(frame,particles,NOP,color,0*r.width/2,0*r.height/2);
    		// Press  ESC on keyboard to exit
    		char c = (char)waitKey(200);
    		if (c == 27) break;
  	}
  	// When everything done, release the video capture object
  	cap.release();

	cudaFree(cudaQ);
	cudaFree(cudaP);
	
  	// Closes all the frames
  	destroyAllWindows();
  	return 0;
}
