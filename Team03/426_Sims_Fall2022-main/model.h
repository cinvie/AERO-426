// C++ std libraries

#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <memory>

#include "entity.h"

using namespace std;

// Abstract base class; don't modify
class Model
{
private:
	shared_ptr<Entity> en;

public:
	Model(shared_ptr<Entity> _en)
	{
		en = _en;
	}
	virtual void init() = 0;
	virtual void update(double dt_seconds) = 0;
};


class PropulsionModel : public Model
{
public:
	PropulsionModel(shared_ptr<Entity> _en) : Model(_en) {};
	virtual void init() override
	{

	};
	virtual void update(double dt_seconds) override
	{
		cout << "PropulsionModel dt: " << dt_seconds << endl;
	};
};


class HeatTransferModel : public Model
{
public:
	HeatTransferModel(shared_ptr<Entity> _en) : Model(_en) {};
	virtual void init() override
	{

	};
	virtual void update(double dt_seconds) override
	{
		cout << "HeatTransferModel dt: " << dt_seconds << endl;
	};
};