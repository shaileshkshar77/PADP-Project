#include <stdio.h>
#include <time.h>

#define N 64000
#define TPB 32
#define K 30
#define MAX_ITER 100

__device__ float distance(float x1, float x2)
{
	return sqrt((x2-x1)*(x2-x1));
}

__global__ void kMeansClusterAssignment(float *d_datapoints, int *d_clust_assn, float *d_centroids)
{
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx >= N) return;

	float min_dist = INFINITY;
	int closest_centroid = 0;

	for(int c = 0; c<K;++c)
	{
		float dist = distance(d_datapoints[idx],d_centroids[c]);

		if(dist < min_dist)
		{
			min_dist = dist;
			closest_centroid=c;
		}
	}

	d_clust_assn[idx]=closest_centroid;
}


__global__ void kMeansCentroidUpdate(float *d_datapoints, int *d_clust_assn, float *d_centroids, int *d_clust_sizes)
{

	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx >= N) return;

	const int s_idx = threadIdx.x;

	__shared__ float s_datapoints[TPB];
	s_datapoints[s_idx]= d_datapoints[idx];

	__shared__ int s_clust_assn[TPB];
	s_clust_assn[s_idx] = d_clust_assn[idx];

	__syncthreads();

	if(s_idx==0)
	{
		float b_clust_datapoint_sums[K]={0};
		int b_clust_sizes[K]={0};

		for(int j=0; j< blockDim.x; ++j)
		{
			int clust_id = s_clust_assn[j];
			b_clust_datapoint_sums[clust_id]+=s_datapoints[j];
			b_clust_sizes[clust_id]+=1;
		}

		for(int z=0; z < K; ++z)
		{
			atomicAdd(&d_centroids[z],b_clust_datapoint_sums[z]);
			atomicAdd(&d_clust_sizes[z],b_clust_sizes[z]);
		}
	}

	__syncthreads();

	if(idx < K){
		d_centroids[idx] = d_centroids[idx]/d_clust_sizes[idx]; 
	}

}


int main()
{

	//allocate memory on the device for the data points
	float *d_datapoints=0;
	//allocate memory on the device for the cluster assignments
	int *d_clust_assn = 0;
	//allocate memory on the device for the cluster centroids
	float *d_centroids = 0;
	//allocate memory on the device for the cluster sizes
	int *d_clust_sizes=0;

	float *yd_datapoints=0;
	int *yd_clust_assn = 0;
	float *yd_centroids = 0;
	int *yd_clust_sizes=0;

	clock_t start,end;

	cudaMalloc(&d_datapoints, N*sizeof(float));
	cudaMalloc(&d_clust_assn,N*sizeof(int));
	cudaMalloc(&d_centroids,K*sizeof(float));
	cudaMalloc(&d_clust_sizes,K*sizeof(float));

	cudaMalloc(&yd_datapoints, N*sizeof(float));
	cudaMalloc(&yd_clust_assn,N*sizeof(int));
	cudaMalloc(&yd_centroids,K*sizeof(float));
	cudaMalloc(&yd_clust_sizes,K*sizeof(float));

	float *h_centroids = (float*)malloc(K*sizeof(float));
	float *y_centroids = (float*)malloc(K*sizeof(float));
	float *h_datapoints = (float*)malloc(N*sizeof(float));
	float *y_datapoints = (float*)malloc(N*sizeof(float));
	int *h_clust_sizes = (int*)malloc(K*sizeof(int));
	int *y_clust_sizes = (int*)malloc(K*sizeof(int));

	srand(time(0));

	for(int c=0;c<K;++c)
	{
		h_centroids[c]=(float) rand() / (double)RAND_MAX;
		y_centroids[c]=(float) rand() / (double)RAND_MAX;
		printf("%f  %f\n", h_centroids[c],y_centroids[c]);
		h_clust_sizes[c]=0;
		y_clust_sizes[c]=0;
	}

	for(int d = 0; d < N; ++d)
	{
		h_datapoints[d] = (float) rand() / (double)RAND_MAX;
		y_datapoints[d] = (float) rand() / (double)RAND_MAX;
	}
	start=clock();

	cudaMemcpy(d_centroids,h_centroids,K*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_datapoints,h_datapoints,N*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_clust_sizes,h_clust_sizes,K*sizeof(int),cudaMemcpyHostToDevice);

	cudaMemcpy(yd_centroids,y_centroids,K*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(yd_datapoints,y_datapoints,N*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(yd_clust_sizes,y_clust_sizes,K*sizeof(int),cudaMemcpyHostToDevice);

	int cur_iter = 1;

	while(cur_iter < MAX_ITER)
	{
		//call cluster assignment kernel
		kMeansClusterAssignment<<<(N+TPB-1)/TPB,TPB>>>(d_datapoints,d_clust_assn,d_centroids);
		kMeansClusterAssignment<<<(N+TPB-1)/TPB,TPB>>>(yd_datapoints,yd_clust_assn,d_centroids);

		//copy new centroids back to host 
		cudaMemcpy(h_centroids,d_centroids,K*sizeof(float),cudaMemcpyDeviceToHost);
		cudaMemcpy(y_centroids,yd_centroids,K*sizeof(float),cudaMemcpyDeviceToHost);

		//reset centroids and cluster sizes (will be updated in the next kernel)
		cudaMemset(d_centroids,0.0,K*sizeof(float));
		cudaMemset(d_clust_sizes,0,K*sizeof(int));
		cudaMemset(yd_centroids,0.0,K*sizeof(float));
		cudaMemset(yd_clust_sizes,0,K*sizeof(int));

		//call centroid update kernel
		kMeansCentroidUpdate<<<(N+TPB-1)/TPB,TPB>>>(d_datapoints,d_clust_assn,d_centroids,d_clust_sizes);
		kMeansCentroidUpdate<<<(N+TPB-1)/TPB,TPB>>>(yd_datapoints,yd_clust_assn,yd_centroids,yd_clust_sizes);

		cur_iter+=1;
	}
	
	end=clock();
	

	double q=(((double)(end-start)))/CLOCKS_PER_SEC  ;
	printf("TIME : %lf",q);

	cudaFree(d_datapoints);
	cudaFree(d_clust_assn);
	cudaFree(d_centroids);
	cudaFree(d_clust_sizes);
	cudaFree(yd_datapoints);
	cudaFree(yd_clust_assn);
	cudaFree(yd_centroids);
	cudaFree(yd_clust_sizes);

	free(h_centroids);
	free(h_datapoints);
	free(h_clust_sizes);
	free(y_centroids);
	free(y_datapoints);
	free(y_clust_sizes);


	return 0;
}