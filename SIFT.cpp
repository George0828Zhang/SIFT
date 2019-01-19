#include <vector>
#include <iostream>// for cin,cout,cerr,endl
#include <cassert>
#include <chrono>// for time evaluation
#include <cmath>// for pow
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// #define ARMA_DONT_USE_WRAPPER
// // #define ARMA_NO_DEBUG
// #include <armadillo>

// using vector = std::vector;
using Mato = cv::Mat_<float>;
constexpr float THRES_CONTRAST = 0.03;
constexpr float THRES_EDGE = 12.1;

void octave(int s, float o, cv::Mat1b const& img, std::vector<Mato>& gaussians, std::vector<Mato>& out){
	float k = std::pow(2., 1./s);
	Mato img_o(img);

	gaussians.resize(s+3);
	for(int i = 0; i < s + 3; i++){
		cv::GaussianBlur(img_o, gaussians[i], cv::Size(), o, o, cv::BORDER_DEFAULT);
		o = k * o;
		if(i>0){
			out.push_back(gaussians[i] - gaussians[i-1]);
		}
	}
}

void diff_x(Mato const& in, Mato& out){
	Mato right = in.clone();
	Mato left = in.clone();
	int c = in.cols, r = in.rows;

	right(cv::Rect(1, 0, c - 1, r)).copyTo(right(cv::Rect(0, 0, c - 1, r)));
	left(cv::Rect(0, 0, c - 1, r)).copyTo(right(cv::Rect(1, 0, c - 1, r)));

	out = (right - left) / 2;
}
void diff_y(Mato const& in, Mato& out){
	Mato up = in.clone();
	Mato down = in.clone();
	int c = in.cols, r = in.rows;

	down(cv::Rect(0, 1, c, r - 1)).copyTo(down(cv::Rect(0, 0, c, r - 1)));
	up(cv::Rect(0, 0, c, r - 1)).copyTo(up(cv::Rect(0, 1, c, r - 1)));

	out = (down - up) / 2;
}


void featureLoc(int w, int s, float o, cv::Mat3b const& img, std::vector<cv::Point3i>& outLoc){
	cv::Mat1b top;	
	cv::cvtColor(img, top, cv::COLOR_BGR2GRAY);
	cv::Mat1b response_map = cv::Mat1b::zeros(img.size());
	cv::Rect ROI(0, 0, img.cols, img.rows);
	auto element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(3, 3) );

	for(int i = 0; i < w; i++){
		float scale = 1<<i;

		std::vector<Mato> gaussians;
		std::vector<Mato> DoG;
		octave(s, o, top, gaussians, DoG);

		std::vector<Mato> erotion(s+2), dilation(s+2);
		for(int j = 0; j < s+2; j++){
			cv::dilate( DoG[j], dilation[j], element );
			cv::erode( DoG[j], erotion[j], element );
		}

		std::vector<Mato> DoDoG(s+2);
		for(int j = 0; j < s+2; j++){
			if(j==0){
				DoDoG[j] = (DoG[1] - DoG[0]) / 1;
			}else if(j==s+1){
				DoDoG[j] = (DoG[s+1] - DoG[s]) / 1;
			}else
				DoDoG[j] = (DoG[j+1] - DoG[j-1]) / 2;
		}

		// cv::Mat1b local_response = cv::Mat1b::zeros(top.size());
		for(int j = 1; j < s+1; j++){
			cv::Mat1b keymap = ( 
				  ( (DoG[j] == erotion[j]) & (DoG[j] <= erotion[j-1]) & (DoG[j] <= erotion[j+1]) ) \
				| ( (DoG[j] == dilation[j]) & (DoG[j] >= dilation[j-1]) & (DoG[j] >= dilation[j+1]) ) );

			Mato Dx, Dy, Do = DoDoG[j];
			Mato Dxx, Dyy, Doo, Dxy, Dxo, Dyo;
			
			diff_x(DoG[j], Dx);
			diff_y(DoG[j], Dy);			
			diff_x(Dx, Dxx);
			diff_y(Dx, Dxy);
			diff_y(Dy, Dyy);
			Doo = (DoDoG[j+1] - DoDoG[j-1]) / 2;
			diff_x(Do, Dxo);
			diff_y(Do, Dyo);

			Mato traceH = Dxx + Dyy;
			Mato detH = Dxx.mul(Dyy) - Dxy.mul(Dxy);
			cv::Mat1b Edge_response = (traceH.mul(traceH) / (detH+1e-7)) < THRES_EDGE;

			Mato Lx, Ly, Lmag, Ltheta;
			diff_x(gaussians[j], Lx);
			diff_y(gaussians[j], Ly);
			cv::magnitude(Lx, Ly, Lmag);
			cv::phase(Lx, Ly, Ltheta, true);// measured in degree

			std::vector<cv::Point> keypoints;
			cv::findNonZero(keymap, keypoints);
			for(auto& p : keypoints){
				cv::Mat_<float> sec_derivative = (cv::Mat_<float>(3,3) << \
					Dxx(p), Dxy(p), Dxo(p), \
					Dxy(p), Dyy(p), Dyo(p), \
					Dxo(p), Dyo(p), Doo(p));
				cv::Mat_<float> first_derivative = (cv::Mat_<float>(3,1) << Dx(p), Dy(p), Do(p));
				// cv::Mat_<float> x_hat = sec_derivative.inv(cv::DECOMP_SVD) * -first_derivative;
				cv::Mat_<float> x_hat;
				cv::solve(sec_derivative, -first_derivative, x_hat, cv::DECOMP_SVD);
				
				if(std::abs(DoG[j](p) + 0.5*first_derivative.dot(x_hat)) >= THRES_CONTRAST * 255. \
					&& detH(p) > 0 && Edge_response(p)){

					cv::Point X_plum((p.x + x_hat[0][0]) * scale, (p.y + x_hat[1][0]) * scale);
					if(X_plum.inside(ROI)){
						if (response_map(X_plum)) continue;
						response_map(X_plum) = 255;
						outLoc.push_back(cv::Point3i((p.x + x_hat[0][0]) * scale, (p.y + x_hat[1][0]) * scale, j - 1));
					}
				}
			}
		}

		// response_map |= local_response;

		cv::Mat1b tmp;
		cv::pyrDown(top, tmp );
		top = tmp;
	}

	// cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
	// cv::imshow("Display Image", response_map);
	// cv::waitKey(0);	
}

int main(int argc, char** argv){
	auto start = std::chrono::system_clock::now();
	cv::Mat3b image;
	image = cv::imread( argv[1], cv::IMREAD_COLOR );
	if ( !image.data )
	{
		printf("No image data \n");
		return -1;
	}

	std::vector<cv::Point3i> feature_loc;
	featureLoc(3, 3, 1.6, image, feature_loc);

	

	auto end = std::chrono::system_clock::now(); 
    std::chrono::duration<double> elapsed_seconds = end-start; 
    std::cerr << "[Info] Elapsed time: " << elapsed_seconds.count() << "s\n";

    for(auto& p : feature_loc){
    	cv::Scalar out[] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};
		cv::circle(image, cv::Point(p.x, p.y), 2.5, out[2], -1);
	}
	cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
	cv::imshow("Display Image", image);
	cv::waitKey(0);	
}
