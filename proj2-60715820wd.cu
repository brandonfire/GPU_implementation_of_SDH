/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the rc machines
   ==================================================================
*/
/*
* CUDA version by chengbin hu U#60715820
* Date 06/20/2015
*
*/



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
//#include <cuda.h>
#include <cuda_runtime.h>

#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;


bucket * histogram;		/* list of all buckets in the histogram   */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */

/* Thesea are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;


/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}

/*device function to caculate distance*/
__device__
double d_p2p_distance(double x1, double x2, double y1, double y2, double z1, double z2) {
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/* chengbin: kenrel function to caculate histogram*/


__global__
void D_PDH_baseline(double * x,double *y, double * z, bucket * hist, long long PDH_acount, double PDH_w, int n_buckets){
	/*shared tiling input use 1024 as 1024 is the largest threadnumber*/
	__shared__ double ix1[256];
	__shared__ double iy1[256];
	__shared__ double iz1[256];
	__shared__ double ix2[256];
	__shared__ double iy2[256];
	__shared__ double iz2[256];

	/*shared ouput*/
	extern __shared__ unsigned long long p_hist[];
	int i, j, d_pos,ti,k;
	//input[threadIdx.x]=atomlist[];
	int gd =gridDim.x;
	int bd = blockDim.x;
	int bdx = blockIdx.x;
	ti = threadIdx.x;
	i = bdx * bd + ti;
	
	
	for(j=ti;j<n_buckets;j+=bd)p_hist[j]=0;//iniatilize the ouput histogram
	//copy the anchor tile data to ix1,iy1,iz1 according to i
	if(i<PDH_acount){	
	ix1[ti] = x[i];
	iy1[ti] = y[i];
	iz1[ti] = z[i];}
	//ix2[ti] = x[i];
	//iy2[ti] = y[i];
	//iz2[ti] = z[i];
	__syncthreads();
	//calcute the points within one block.
	double dist;
	
	int lastblock = gd-1;
	int lastblocklength = PDH_acount - bd*(gd-1);
	if(bdx<lastblock)
	{
		
		for(j=ti+1; j<bd;j++)
		{
			dist = d_p2p_distance(ix1[ti],ix1[j],iy1[ti],iy1[j],iz1[ti],iz1[j]);
			d_pos = (int) (dist / PDH_w);
			atomicAdd(&(p_hist[d_pos]),1);
		}
		

		
	__syncthreads();
	} else 
	{
		
		if(i<PDH_acount)
		{

			
			
			for(j=ti+1; j<lastblocklength;j++)
			{
				dist = d_p2p_distance(ix1[ti],ix1[j],iy1[ti],iy1[j],iz1[ti],iz1[j]);
				d_pos = (int) (dist / PDH_w);
				atomicAdd(&(p_hist[d_pos]),1);
			}
		}
		__syncthreads();
	}
	__syncthreads();
	
	//calcute the points between blocks.
	int cycle = ceil(gd/2.0);//becareful the last block	
	for(k=1;k<cycle;k++)//caculate points between blocks
		{
			j = (bdx+k)%gd;
			if(j<lastblock) // j is not the last block
			{
				ix2[ti] = x[j* bd + ti];
				iy2[ti] = y[j* bd + ti];
				iz2[ti] = z[j* bd + ti];
				__syncthreads();
				if(i<PDH_acount)
				{
					for(int m = 0; m<bd; m++)
					{
						dist = d_p2p_distance(ix1[ti],ix2[m],iy1[ti],iy2[m],iz1[ti],iz2[m]);
						d_pos = (int) (dist / PDH_w);
						atomicAdd(&(p_hist[d_pos]),1);
					}
				}
				__syncthreads();
			} else //J is the last block
			{
				
				if(ti<lastblocklength)
				{
					ix2[ti] = x[j* bd + ti];
					iy2[ti] = y[j* bd + ti];
					iz2[ti] = z[j* bd + ti];
				}
				__syncthreads();
				if(i<PDH_acount)
				{
					for(int m = 0; m<lastblocklength; m++)
					{
						dist = d_p2p_distance(ix1[ti],ix2[m],iy1[ti],iy2[m],iz1[ti],iz2[m]);
						d_pos = (int) (dist / PDH_w);
						atomicAdd(&(p_hist[d_pos]),1);
					}
				}
				__syncthreads();
			}
	
		}//last half cycle for gridDim.x%2==0
		if(gd%2==0)
		{
			
			if(bdx<gd/2)
			{
				j = (bdx+cycle)%gd;
				if(j<lastblock) // j is not the last block
				{
					ix2[ti] = x[j* bd + ti];
					iy2[ti] = y[j* bd + ti];
					iz2[ti] = z[j* bd + ti];
					__syncthreads();
					if(i<PDH_acount)
					{
						for(int m = 0; m<bd; m++)
						{
							dist = d_p2p_distance(ix1[ti],ix2[m],iy1[ti],iy2[m],iz1[ti],iz2[m]);
							d_pos = (int) (dist / PDH_w);
							atomicAdd(&(p_hist[d_pos]),1);
						}
					}
					__syncthreads();
				} else //J is the last block
				{
					
					if(ti<lastblocklength)
					{
						ix2[ti] = x[j* bd + ti];
						iy2[ti] = y[j* bd + ti];
						iz2[ti] = z[j* bd + ti];
						
					}
					__syncthreads();
					if(i<PDH_acount)
					{
						for(int m = 0; m<lastblocklength; m++)
						{
							dist = d_p2p_distance(ix1[ti],ix2[m],iy1[ti],iy2[m],iz1[ti],iz2[m]);
							d_pos = (int) (dist / PDH_w);
							atomicAdd(&(p_hist[d_pos]),1);
						}
					
					}
					__syncthreads();
				}



			}

		}

	__syncthreads();
	for(j=ti;j<n_buckets;j+=bd)atomicAdd(&(hist[j].d_cnt),p_hist[j]);
	


}

__global__
void D_initialize(bucket * h, int n_buckets){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n_buckets) h[i].d_cnt=0;

}


/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


/* 
	print the counts in all buckets of the histogram 
	chengbin: changed to print specific histogram from argument
*/
void output_histogram(bucket * histogram1, int ident){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram1[i].d_cnt);
		total_cnt += histogram1[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
	if( ident == 1){
		total_cnt = 0;
		printf("\n difference between CPU and GPU \n");
		long long d;
		
        	for(i=0; i< num_buckets; i++) {
                if(i%5 == 0) /* we print 5 buckets in a row */
                        printf("\n%02d: ", i);
		d = histogram[i].d_cnt- histogram1[i].d_cnt;
                printf("%15lld ", d);
		total_cnt += d;
                /* we also want to make sure the total distance count is correct */
                if(i == num_buckets - 1)
                        printf("\n T:%lld \n", total_cnt);
                else printf("| ");
            }


	}


}




int main(int argc, char **argv)
{
	int block_size; 
	if(argc != 4) {
        printf("ERROR please input 3 arguments: %s {#of_samples} {bucket_width} {block_size} \n",argv[0]);
        exit(1);
    	}


	int i;
	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	block_size = atoi(argv[3]);
	if(block_size > 256) {
        printf("TOO BIG BLOCK SIZE ERROR. Due to size limitation of shared memory please use a blocksize <=256\n");
        exit(1);
    	}
//printf("args are %d and %f\n", PDH_acnt, PDH_res);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	
	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);
	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	double * h_x, * h_y, * h_z;//seperate host input array
	double * d_x,* d_y,* d_z;//seperate device input array		
	h_x = (double *)malloc(sizeof(double)*PDH_acnt);
	h_y = (double *)malloc(sizeof(double)*PDH_acnt);
	h_z = (double *)malloc(sizeof(double)*PDH_acnt);
	cudaMalloc((void**)&d_x, sizeof(double)*PDH_acnt);
	cudaMalloc((void**)&d_y, sizeof(double)*PDH_acnt);	
	cudaMalloc((void**)&d_z, sizeof(double)*PDH_acnt);
	/*move input array to seperate array*/
	for(i = 0;  i < PDH_acnt; i++) {
		h_x[i] = atom_list[i].x_pos;
		h_y[i] = atom_list[i].y_pos;
		h_z[i] = atom_list[i].z_pos;
	}

	/* start counting time */
	gettimeofday(&startTime, &Idunno);
	
	/* call CPU single thread version to compute the histogram */
	PDH_baseline();
	
	/* check the total running time */ 
	report_running_time();
	
	/* print out the histogram */
	output_histogram(histogram, 0);
	/*Chengbin: locate GPU memory for input data array*/
	//atom * d_atom_list;
	//cudaMalloc((void**)&d_atom_list, sizeof(atom)*PDH_acnt);
	
	/*Chengbin: locate GPU memory for output data array*/
	bucket * d_histogram;
	cudaMalloc((void**)&d_histogram, sizeof(bucket)*num_buckets);
	D_initialize<<<(int)ceil(num_buckets/256.0),256>>>(d_histogram,num_buckets);
	//const int inivalue = 0;
	//cudaMemset(d_histogram,inivalue,sizeof(bucket)*num_buckets);
	/*chengbin: locate GPU results histogrm*/
	bucket * cuda_histogram;
	cuda_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	/*chengbin: defince grid and block parameter*/
	dim3 dimGrid((int)ceil(PDH_acnt/(float)block_size),1,1);
	dim3 dimBlock(block_size,1,1);



	//kernel function to take input to generate histogram
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord( start, 0 );
	//cudaMemcpy(d_atom_list,atom_list,sizeof(atom)*PDH_acnt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x,h_x,sizeof(double)*PDH_acnt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y,h_y,sizeof(double)*PDH_acnt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z,h_z,sizeof(double)*PDH_acnt, cudaMemcpyHostToDevice);
	D_PDH_baseline<<<dimGrid,dimBlock,sizeof(bucket)*num_buckets>>>(d_x, d_y, d_z, d_histogram, PDH_acnt, PDH_res, num_buckets);	
	/*copy device result back to cuda result*/
	cudaMemcpy(cuda_histogram,d_histogram,sizeof(bucket)*num_buckets, cudaMemcpyDeviceToHost);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop );
	printf( "******** Total Running Time of Kernel = %f sec ******* \n", elapsedTime/1000 );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );
	
	output_histogram(cuda_histogram, 1);
	cudaFree(d_histogram);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);
	free(histogram);
	free(atom_list);
	free(h_x);
	free(h_y);
	free(h_z);	
	free(cuda_histogram);
	return 0;
}


