cmake_minimum_required(VERSION 3.16)

project(Sims_426)

add_executable(simulation
	main.cpp)

target_include_directories(simulation PUBLIC
	eigen)