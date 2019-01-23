#ifndef _SIFT_HPP
#define _SIFT_HPP

#include <array>
#include <vector>
#include <cmath>// for sqrt
#include <algorithm>// for fill
#include <numeric>// for inner_product
#include <opencv2/core/core.hpp>

// #include <iostream>// for debug

#define DESCIPTOR_SZ 4
class feature{
public:
	int x;
	int y;
	float theta;
	std::array<float, DESCIPTOR_SZ* DESCIPTOR_SZ*8> _descriptor;
	feature()
	{
		std::fill(_descriptor.begin(), _descriptor.end(), 0);
	}
	void normalize(){
		// auto summ = std::sqrt(std::inner_product(_descriptor.begin(), _descriptor.end(), _descriptor.begin(), 0.));
		auto summ = std::sqrt(feature::match(*this, *this));
		if(!summ) return;
		for(float& u : _descriptor){
			u = std::min(u/summ, 0.2f);
		}
		summ = std::sqrt(feature::match(*this, *this));
		if(!summ) return;
		for(auto& u : _descriptor){
			u /= summ;
			// std::cerr << u << " ";
		}
		// std::cerr << std::endl;
	}
	static float match(feature const& a, feature const& b){
		return std::inner_product(a._descriptor.begin(), a._descriptor.end(), b._descriptor.begin(), 0.);
	}
};

void featureLoc(int w, int s, float o, cv::Mat3b const& img, std::vector<feature>& outLoc);

#endif