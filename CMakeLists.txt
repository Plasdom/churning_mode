cmake_minimum_required(VERSION 3.13)

project(churn LANGUAGES CXX)

if (NOT TARGET bout++::bout++)
  find_package(bout++ REQUIRED)
endif()
# set(CMAKE_MODULE_PATH "/Users/power8/Documents/01_code/04_bout/build-bout")
# find_package(bout++ REQUIRED)

bout_add_example(churn
  SOURCES churn.cxx
  DATA_DIRS data fromfile
  EXTRA_FILES generate.py)
# add_executable(churn churn.cxx)