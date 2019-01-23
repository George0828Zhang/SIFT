#include <vector>
#include <iostream>// for cin,cout,cerr,endl
#include <chrono>// for time evaluation
#include <cmath>// for pow
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SIFT.hpp"

cv::Point dir(float theta, float r){
	float x = r*std::cos(theta/180. * M_PI);
	float y = r*std::sin(theta/180. * M_PI);
	return cv::Point(x, y);
}

int match(std::vector<feature> const& base, feature const& data){
	int ifirst, isecond;
	float first = 0., second = 0.;
	for(int i = 0; i < base.size(); i++){
		float cur = feature::match(base[i], data);
		// std::cerr << cur << std::endl;
		if(cur > first){
			first = cur;
			ifirst = i;
		}else if(cur > second){
			second = cur;
			isecond = i;
		}
	}

	if(second==1) return -1;
	else if((1.-first)/(1.-second) < 0.64){
		return ifirst;
	}else return -1;
}

int main(int argc, char** argv){
	auto start = std::chrono::system_clock::now();
	cv::Mat3b image1, image2;
	image1 = cv::imread( argv[1], cv::IMREAD_COLOR );
	if ( !image1.data )
	{
		printf("No image data \n");
		return -1;
	}

	cv::Mat3b big_image;
	cv::resize(image1, big_image, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
	std::vector<feature> sift_descriptors1;
	featureLoc(4, 3, 1.6, big_image, sift_descriptors1);
	

	auto end = std::chrono::system_clock::now(); 
    std::chrono::duration<double> elapsed_seconds = end-start; 
    std::cerr << "[Info] Elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cerr << "[Info] Found feautures: " << sift_descriptors1.size() << "\n";

    for(auto& p : sift_descriptors1){
    	cv::Scalar out[] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};
		cv::circle(image1, cv::Point(p.x / 2, p.y / 2), 3, out[0], -1);
		// float r = 20.;
		// cv::Point from(p.x / 2, p.y / 2);
		// cv::arrowedLine(image, from, from + dir(p.theta, r), out[1], 2, 8, 0, 0.2);
	}


	image2 = cv::imread( argv[2], cv::IMREAD_COLOR );
	if ( !image2.data )
	{
		printf("No image2 data \n");
		return -1;
	}

	cv::resize(image2, big_image, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
	std::vector<feature> sift_descriptors2;
	featureLoc(4, 3, 1.6, big_image, sift_descriptors2);
	

	cv::Mat3b matcher = cv::Mat3b::zeros(std::max(image2.rows, image1.rows), image1.cols+image2.cols);
	image1.copyTo(matcher(cv::Rect(0, 0, image1.cols, image1.rows)));
	image2.copyTo(matcher(cv::Rect(image1.cols, 0, image2.cols, image2.rows)));

    for(auto& p : sift_descriptors2){
    	cv::Scalar out[] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};
		cv::circle(matcher, cv::Point(image1.cols+p.x / 2, p.y / 2), 3, out[0], -1);

		int index = match(sift_descriptors1, p);
		if(index != -1){
			cv::Point from(sift_descriptors1[index].x / 2, sift_descriptors1[index].y / 2);
			cv::Point to(image1.cols+p.x / 2, p.y / 2);

			cv::line(matcher, from, to, out[0], 1, 8, 0);
		}
	}




	cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
	cv::imshow("Display Image", matcher);
	cv::waitKey(0);	
}