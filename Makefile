# Makefile for OpenMP Image Filtering Course Project

CXX = g++
CXXFLAGS = -O3 -fopenmp -Wall -Wextra

TARGET = image_filter
SRC = image_filter.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
