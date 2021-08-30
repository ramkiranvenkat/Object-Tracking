#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#define SPEED 3
using namespace std;
using namespace cv;
ofstream f1("serialTiming");
//g++ pps.cpp -o vc `pkg-config --cflags --libs opencv4`

double median(double inp[], int NOP)
{
  double *another = new double[NOP];
  for (int i = 0; i < NOP; i++)
    another[i] = inp[i];
  sort(another, another + NOP);
  double out = another[NOP / 2];
  delete another;
  return out;
}

int argmax(double inp[], int NOP)
{
  int ind = 0;
  double maxval = 0.0;
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

void computeColorHist(int NBINS,Mat inpImg,double** colorhist,int placeHolder)
{
	int NOEB = 255/NBINS + 1;
	Vec3b pixel;
	placeHolder = placeHolder*3;

	double sum = 0;
	Size s = inpImg.size();

	for (int i=0;i<3;i++)
	{
		for (int j=0;j<NBINS;j++) colorhist[i+placeHolder][j] = 0;
	}

	for (int i=0,imax=s.height;i<imax;i++)
	{
		for (int j=0,jmax=s.width;j<jmax;j++)
		{
			pixel =  inpImg.at<Vec3b>(i,j);
			for (int k=0;k<3;k++) 
			{
				colorhist[k+placeHolder][((int)pixel[k])/NOEB]+=1.0;
				sum+=1.0;
			}
		}		
	}
	
	for (int i=0;i<3;i++)
	{
		for (int j=0;j<NBINS;j++) 
		{
		colorhist[i+placeHolder][j] = colorhist[i+placeHolder][j]/sum;
		/*
		if (isnan(colorhist[i+placeHolder][j]))
		{
			cout << "Bug variable" << " " << s.height << " " << s.width <<  endl;
			cout << sum << endl;
			exit(0);
		}
		*/
		}
	}
}

double BhattacharyyaSimilarityCoefficient(int N,double **hist)
{
	double ret = 0;
	for (int i=0;i<3;i++)
	{
		for (int j=0;j<N;j++) 
		{
			/*
			if (isnan(sqrt(hist[i][j]*hist[i+3][j])))
			{
				cout << "BSC" << endl;
				cout << hist[i][j] << " " << hist[i+3][j] << endl;
				exit(0);
			}
			*/
			ret += sqrt(hist[i][j]*hist[i+3][j]);
		}
	}
	return ret;
}

int  forwardModel(int p,int v)
{
	return p+(int)((double)v*0.0625); // 0.0625 = 1/16 frame rate
}

void particleImage(Mat img,int** p,int NOP,Scalar color,int xoffset,int yoffset)
{
	for (int i = 0; i < NOP; i++)
      {
          Point plocal(p[0][i]+xoffset, p[1][i]+yoffset);
          drawMarker(img, plocal, color, MARKER_CROSS, 1, 8);
      }
	imshow( "Particle Image", img );
}

int main()
{
  // Pseudo Random Number Generator
  default_random_engine generator;
  normal_distribution<double> distribution(0.0, 1.0);

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
  double P[] = {5,5,5,5};
  double Q[] = {3,3,3,3};
  
  int selectionBit = 0;
  int NBINS  = 10;
  double** colorhist  = new double*[6];
  for (int i=0;i<6;i++) colorhist[i] = new double[NBINS];
  Rect r,rp;
  Mat frame, imCrop;
  int **particles,**nparticles, NOP = 65536; // number of particles
  particles = new int *[4];

  particles[0] = new int[NOP];    // X position
  particles[1] = new int[NOP];    // Y position
  particles[2] = new int[NOP];    // X velocity
  particles[3] = new int[NOP];    // Y velocity
  nparticles = new int *[4];

  nparticles[0] = new int[NOP];    // X position
  nparticles[1] = new int[NOP];    // Y position
  nparticles[2] = new int[NOP];    // X velocity
  nparticles[3] = new int[NOP];    // Y velocity

  double *corr = new double[NOP]; // Correlation
  Size imgS;
  int selItr = 0;
  struct timespec start, finish,startf,finishf;
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
        cout << r.x << " " << r.y << " " << r.height << " " << r.width << " " << r.x + r.width / 2 << " " << r.y + r.height / 2 << endl;
		/*
		Vec3b pixel =  imcrop.at<Vec3b>(0,0);
		cout << pixel << ", "; 
		for  (int i=0;i<3;i++)cout << (int)pixel[i] << " ";
		cout << endl;*/

		computeColorHist(NBINS,imcrop,colorhist,0); // 0 past 1 present
		/*
		for (int i=0;i<3;i++)
		{
			for (int j=0;j<NBINS;j++) cout << colorhist[i][j] << " ";
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
		    particles[0][i] = randomVariable + r.x; // x pos  
		    while (1)
		    {
		      randomVariable = (int)(P[1]*distribution(generator));
		      if (randomVariable + r.y < irows && randomVariable + r.y > 0) // height axis
		        break;
		    }
		    particles[1][i] = randomVariable + r.y; // y pos - top left corner

			particles[2][i] = (int)(P[2]*distribution(generator)); // x - vel
			particles[3][i] = (int)(P[3]*distribution(generator)); // y - vel
		   }

		   Scalar color(0,0,200);
		   particleImage(frame,particles,NOP,color,0*r.width/2,0*r.height/2);
	}
	
    else
    {
	double time_seq = 0;
      // create new particles
     clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &startf);
	f1 << 0 << " ";
      int randomVariable;
	  imgS = frame.size();
	  int irows = imgS.height;
	  int icols = imgS.width;
	  double sum = 0,maxw = 0;
	  int maxItr;
      for (int i = 0; i < NOP; i++)
      {

		// forward propagation
		particles[0][i] = forwardModel(particles[0][i],particles[2][i]) + (int)(Q[0]*distribution(generator));
		particles[1][i] = forwardModel(particles[1][i],particles[3][i]) + (int)(Q[1]*distribution(generator));

		particles[2][i] = particles[2][i] + (int)(Q[2]*distribution(generator));
		particles[3][i] = particles[3][i] + (int)(Q[3]*distribution(generator));

		if ((particles[0][i] >= icols) || (particles[0][i] < 0) || (particles[1][i] >= irows) || (particles[1][i] < 0)) 
		{
			corr[i] = 0.0;
			continue;
		}

		int sx = (((particles[0][i]+r.width)<icols)?(particles[0][i]+r.width):icols-1) - particles[0][i];
		int sy = (((particles[1][i]+r.height)<irows)?(particles[1][i]+r.height):irows-1) - particles[1][i];

		rp = Rect(particles[0][i], particles[1][i],sx,sy);		
		if ((!sy) || (!sx)) 
		{
			corr[i] = 0;
			continue;
		}
		computeColorHist(NBINS,frame(rp),colorhist,1); // measurement model
	
		corr[i] = BhattacharyyaSimilarityCoefficient(NBINS,colorhist);
		if (corr[i] > maxw) 
		{
			maxw = corr[i];
			maxItr = i;
		}
		//cout << corr[i] << " " << sum << endl;
		sum+= corr[i];
		if  (isnan(sum)) 
			{
	cout  << i << endl;exit(0);
}
      }
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &finish); 
			{
			    	long seconds = finish.tv_sec - start.tv_sec; 
			    	long ns = finish.tv_nsec - start.tv_nsec;
			    	time_seq = (double)seconds + (double)ns/(double)1000000000;
				f1 << time_seq << " ";
			}
      int maxx = particles[0][maxItr];
      int maxy = particles[1][maxItr];
	  // resample corr - weight, particles
	  ////////////////////////////////////////////////////////////////
	  double rpt = 1.0 /(double)NOP * ((double)rand() / RAND_MAX);
 clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
      double cdf[NOP];
      cdf[0] = corr[0] / sum;
      for (int i = 1; i < NOP; ++i) cdf[i] = cdf[i - 1] + (corr[i] / sum);
clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &finish); 
			{
			    	long seconds = finish.tv_sec - start.tv_sec; 
			    	long ns = finish.tv_nsec - start.tv_nsec;
			    	time_seq = (double)seconds + (double)ns/(double)1000000000;
				f1 << time_seq << " ";
			}
      int icdf = 0;
      // cout<<cdf[NOP - 1]<<"\n"; // should be 1
 clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
      for (int i = 0; i < NOP; ++i)
      {
        if (rpt < cdf[icdf])
        {
          // assign new pos for particle i
          nparticles[0][i] = particles[0][icdf];
	      nparticles[1][i] = particles[1][icdf];
		  nparticles[2][i] = particles[2][icdf];
	      nparticles[3][i] = particles[3][icdf];        
        }
        else
        {
          for (int j = icdf; j < NOP; ++j)
          {
            if (rpt < cdf[j]) break;
            ++icdf;
          }
          // assign new pos for particle i
          nparticles[0][i] = particles[0][icdf];
	  nparticles[1][i] = particles[1][icdf];
	  nparticles[2][i] = particles[2][icdf];
	  nparticles[3][i] = particles[3][icdf]; 

        }
        rpt += (1.0 / (double)NOP);
      }

	  double avgx=0,avgy=0;
	  for (int i = 1; i < NOP; ++i)
	  {
		  particles[0][i] = nparticles[0][i];
	      	  particles[1][i] = nparticles[1][i];
		  particles[2][i] = nparticles[2][i];
	          particles[3][i] = nparticles[3][i]; 
		  avgx += (double)particles[0][i]/(double)NOP;
		  avgy += (double)particles[1][i]/(double)NOP;
	  }
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
	//Point p1((int)avgx,(int)avgy);
	//Point p2((int)avgx+r.width,(int)avgy+r.height);

	Point p1((int)maxx,(int)maxy);
	Point p2((int)maxx+r.width,(int)maxy+r.height);

	Scalar clr(0,255,255);
	rectangle(frame,p1,p2,clr,2);
	imshow( "Frame", frame );

    }
	
	Scalar color(0,200,200);
	particleImage(frame,particles,NOP,color,0*r.width/2,0*r.height/2);
    // Press  ESC on keyboard to exit
    char c = (char)waitKey(200);
    if (c == 27)
      break;
  }
  // When everything done, release the video capture object
  cap.release();
  // Closes all the frames
  destroyAllWindows();
  return 0;
}
