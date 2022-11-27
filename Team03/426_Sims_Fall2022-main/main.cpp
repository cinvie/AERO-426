//
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <random>

#include "model.h"

using namespace std;

int main(int argc, char* argv[])
{
	vector<shared_ptr<Entity>> entities;

	auto moon = make_shared<MoonEntity>();
	auto landingSite = make_shared<Entity>();
	auto lander = make_shared<LanderEntity>();
	auto rover1 = make_shared<RoverEntity>();


	entities.push_back(moon);
	entities.push_back(lander);
	entities.push_back(rover1);

	vector<Model*> models;

	models.push_back(new PropulsionModel(lander));
	models.push_back(new HeatTransferModel(lander));

	double simLength = 100; //seconds
	double updateFreq = 50; //Hz

	// Probably don't modify past here.
	for (auto model : models)
	{
		model->init();
	}

	std::uniform_real_distribution<double> unif((1/updateFreq)*0.6, (1 / updateFreq) * 1.4);
	std::default_random_engine re;
	for (double simtime = 0.0; simtime < simLength; )
	{
		double dt = unif(re);
		for (auto model : models)
		{
			model->update(dt);
		}
		simtime += dt;
	}

	std::cout << "Closing program." << std::endl;
	return 0;
}
