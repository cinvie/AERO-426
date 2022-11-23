#include <cmath>
#include <iostream>

struct Vec {
    double x;
    double y;
    double z;

    Vec& operator-=(const Vec& rhs) {
        this->x -= rhs.x;
        this->y -= rhs.y;
        this->z -= rhs.z;
        return *this;
    }

    friend Vec operator-(Vec lhs, const Vec& rhs) {
        lhs -= rhs;
        return lhs;
    }

    double dot(Vec b) {
        return this->x * b.x + this->y * b.y + this->z * b.z;
    }

    double norm() {
        return std::sqrt(std::pow(this->x, 2) + std::pow(this->y, 2)  + std::pow(this->z, 2));
    }
};

bool visible(
    Vec rh /* HLS from south pole in m */,
    Vec r /* Target from south pole in m */,
    double terrain_angle = 0 /* Angle between horizontal and horizon in radians */)
{
    // Moon radius in m
    double R = 1737e3;

    // Moon center vector
    auto rc = Vec {0, 0, -R};

    // Vector from HLS to target
    auto rhr = r - rh;

    // Vector from moon center to HLS
    auto rch = rh - rc;

    // Cone angle
    double cone = std::acos(rch.dot(rhr) / (rch.norm() * rhr.norm()));

    // Azimuth angle is the complement of the cone angle
    double azimuth = 3.14159265 / 2 - cone;

    // Is azimuth angle above terrain angle?
    return azimuth > terrain_angle;
}

int main() {
    // Not visible
    Vec rh = Vec {0, 0, 0};
    Vec rg = Vec {0, 0, -1737e3};
    std::cout << visible(rh, rg) << std::endl;

    // Not visible
    rh = Vec {-1737e3, 0, -1737e3};
    rg = Vec {0, 0, 0};
    std::cout << visible(rh, rg) << std::endl;

    // Visible
    rh = Vec {-1737e3, 0, -1737e3};
    rg = Vec {-2 * 1737e3, 0, -2 * 1737e3};
    std::cout << visible(rh, rg) << std::endl;

    // Visible
    // North pole
    rh = Vec {0, 0, -2 * 1737e3};
    rg = Vec {0, 0, 0};
    std::cout << visible(rh, rg) << std::endl;
}
