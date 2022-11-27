// C++ std libraries

#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <mutex>

using namespace std;
using namespace Eigen;


class Entity
{
private:
	mutex en_mutex;
	// xyz meters
	Vector3d Location = { 0, 0, 0 };
	// Quaternion corresponding to WorldToBody rotation.
	// Using .toRotationMatrix() is recommended.
	Quaterniond Rotation = { 1,0,0,0 };
	// xyz meters/second
	Vector3d Velocity = { 0,0,0 };
	// xyz rad/second (direction is axis, magnitude is rate)
	Vector3d AngVelocity = { 0,0,0 };
public:
	Vector3d GetLocation()
	{
		return Location;
	};
	Quaterniond GetRotation()
	{
		return Rotation;
	};
	Vector3d GetVelocity()
	{
		return Velocity;
	};
	Vector3d GetAngVelocity()
	{
		return AngVelocity;
	};
};


class LanderEntity : public Entity
{
public:

};

class MoonEntity : public Entity
{
public:

};

class RoverEntity : public Entity
{
public:

};