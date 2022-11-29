#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <vector>

using namespace std;
using namespace Eigen;

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
	srand(time(nullptr));

	vector<double> pixel_width_arr = linspace(50.0, 500.0, 50);
	vector<double> image_dist_list;

	double image_width = 48000.0; // meters
	double pixel_width = 248.0; // pixels
	double focal_length = 543.45;  //mm
	double theta = 25.0; //degrees

	double image_distance = 100000.0;

	vector<vector<double>> XY_distance;
	vector<double> X_distance;
	vector<double> Y_distance;
	XY_distance.push_back(X_distance);
	XY_distance.push_back(Y_distance);

	for (double pixels : pixel_width_arr) {
		if (image_distance > 50000) {
			double percent_change = (float((0.1 * pixels) / 100));
			
			//double num = rand() % (percent_change + percent_change + 1) - percent_change;
			double num = (-percent_change) + (double)(rand()) / ((double)(RAND_MAX/(percent_change+percent_change))); // generate random number between +/- 0.1% of the pixel 
			double pixels_num = pixels + num;
			image_distance = Position_Estimation(focal_length, image_width, pixels_num);
			vector<double> x_y_distance = xy_distance(theta, image_distance);

			image_dist_list.push_back(image_distance);

			cout << "Distance to HLS from target image: " << image_distance << "km" << endl;

			XY_distance[0].push_back(x_y_distance[0]);
			XY_distance[1].push_back(x_y_distance[1]);

			cout << "Distance to HLS from target image (X Coordinate): " << x_y_distance[0] << endl;
			cout << "Distance to HLS from target image (Y Coordinate): " << x_y_distance[1] << endl;
			cout << endl;
		}
		else {
			cout << "Warning! Distance is less than 50km from target HLS." << endl;
		}
	}
}
