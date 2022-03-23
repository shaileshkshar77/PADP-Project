#include<bits/stdc++.h>
#include<time.h>

struct point {
	double x, y;
	int cluster = -1;
};


int main()
{
	clock_t start,end;
    
    std::vector<point> init; 
	unsigned int t=100, k=30,n=64000; 
	
	for(int i=0;i<n;i++)
	{
		point tmp;
        tmp.x=(float) rand() / (double)RAND_MAX;
        tmp.y=(float) rand() / (double)RAND_MAX;

		init.push_back(tmp);
	}

	auto tmp = std::max_element(init.begin(), init.end(), [](point a, point b) {return a.x < b.x; });
	int max_x = tmp->x;
	tmp = std::max_element(init.begin(), init.end(), [](point a, point b) {return a.y < b.y; });
	int max_y = tmp->y;
	tmp = std::min_element(init.begin(), init.end(), [](point a, point b) {return a.x < b.x; });
	int min_x = tmp->x;
	tmp = std::min_element(init.begin(), init.end(), [](point a, point b) {return a.y < b.y; });
	int min_y = tmp->y;

	std::vector<point> centers(k);
	for (unsigned int i = 0; i < k; i++) {
		centers[i].x = (float) rand() / (double)RAND_MAX;
		centers[i].y = (float) rand() / (double)RAND_MAX;
	}
	
	start=clock();

	for (unsigned int i = 0; i < t; i++) {
		for (unsigned int j = 0; j < init.size(); j++) {
			double* dists = new double[k];
			for (unsigned int p = 0; p < k; p++) {
				double a = std::abs(init[j].y - centers[p].y);	
				double b = std::abs(init[j].x - centers[p].x);	
				dists[p] = std::sqrt(std::pow(a, 2) + std::pow(b, 2));	
			}
			init[j].cluster = std::min_element(dists, dists + k) - dists;
			delete[] dists;
		}
		std::unique_ptr<double[]> sum_x(new double[k], std::default_delete<double[]>());
		std::unique_ptr<double[]> sum_y(new double[k], std::default_delete<double[]>());
		std::unique_ptr<int[]> count = std::make_unique<int[]>(k);
		for (unsigned int p = 0; p < k; p++) {
			sum_x[p] = 0;
			sum_y[p] = 0;
			count[p] = 0;
		}
		for (unsigned int f = 0; f < init.size(); f++) {
			sum_x[init[f].cluster] += init[f].x;
			sum_y[init[f].cluster] += init[f].y;
			count[init[f].cluster]++;
		}
		// set new centers to average coordinate of points in cluster
		for (unsigned int f = 0; f < k; f++) {
			centers[f].x =sum_x[f] / count[f];
			centers[f].y = sum_y[f] / count[f];
		}
		
	}
	end=clock();
	for (unsigned int i = 0; i < k; i++) {
		std::cout << centers[i].x << " " << centers[i].y << "\n";
	}
	double q=(((double)(end-start)))/CLOCKS_PER_SEC ;
	std::cout<<std::fixed<<std::setprecision(8)<<q<<end;
	
}
