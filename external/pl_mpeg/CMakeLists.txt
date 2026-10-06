cmake_minimum_required(VERSION 3.22)

project(pl_mpeg)

add_library(pl_mpeg INTERFACE)

target_include_directories(
	pl_mpeg 
	INTERFACE 
	"include/"
)