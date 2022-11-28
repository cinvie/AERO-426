#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <vector>

using namespace std;
using namespace Eigen;

int image_width = 48000;
int pixel_width = 248;
double focal_length = 543.45;

double Position_Estimation(double focal_length, double real_image_width, double image_width_in_pixels) {
	double distance = (real_image_width * focal_length) / image_width_in_pixels;
	return distance;
}

vector<double> xy_distance(double theta, double image_distance) {
	double phi = (90 - theta) * (M_PI / 180);
	double x_distance = abs(cos(phi) * image_distance);
	double y_distance = abs(tan(phi) * image_distance);
	vector<double> xy_distance;
	xy_distance.push_back(x_distance);
	xy_distance.push_back(y_distance);
	return xy_distance;
}

vector<double> linspace(double init, double end, double num) {
	vector<double> linspace_array;
	
	double step = (end - init) / (num - 1);
	
	for (int i = 0; i < num - 1; ++i) {
		linspace_array.push_back(init + step * i);
	}
	linspace_array.push_back(end);
	return linspace_array;
}

int main() {
	vector <double> pixel_width_arr = linspace(50.0, 500.0, 100);
	vector<double> image_dist_list;
	int image_distance = 100000;

	for (double pixels : pixel_width_arr) {
		if (image_distance > 45000) {
			int percent_change = (int((1 * pixels) / 100));
			double num = rand() % (percent_change + percent_change + 1) - percent_change;
			image_distance = Position_Estimation(focal_length, image_width, pixels);
			image_dist_list.push_back(image_distance + num);
		}
		else {
			cout << "Warning! Distance is less than 45km from target HLS." << endl;
		}
	}

	vector<double> theta = linspace(0.0, 45.0, 46);

	vector<vector<double>> XY_distance;
	vector<double> X_distance;
	vector<double> Y_distance;
	XY_distance.push_back(X_distance);
	XY_distance.push_back(Y_distance);
	for (double i : theta) {
		vector<double> distance = xy_distance(i, image_distance);
		XY_distance[0].push_back(distance[0]);
		XY_distance[1].push_back(distance[1]);
	}
}