#include <vector>
// #include <array>
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
	left(cv::Rect(0, 0, c - 1, r)).copyTo(left(cv::Rect(1, 0, c - 1, r)));

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
void diff2_x(Mato const& in, Mato& out){
	Mato right = in.clone();
	Mato left = in.clone();
	int c = in.cols, r = in.rows;

	right(cv::Rect(1, 0, c - 1, r)).copyTo(right(cv::Rect(0, 0, c - 1, r)));
	left(cv::Rect(0, 0, c - 1, r)).copyTo(left(cv::Rect(1, 0, c - 1, r)));

	out = right - 2*in  + left;
}
void diff2_y(Mato const& in, Mato& out){
	Mato up = in.clone();
	Mato down = in.clone();
	int c = in.cols, r = in.rows;

	down(cv::Rect(0, 1, c, r - 1)).copyTo(down(cv::Rect(0, 0, c, r - 1)));
	up(cv::Rect(0, 0, c, r - 1)).copyTo(up(cv::Rect(0, 1, c, r - 1)));

	out = down - 2*in + up;
}

using feature = cv::Point3i;
void keyPoint(/*Mato const& gaussian, */
	Mato const& mag, 
	Mato const& phase, 
	cv::Point const& at, 
	float scale, 
	float coords_scale,
	std::vector<feature>& out)
{	
	cv::Rect ROI(0, 0, mag.cols, mag.rows);
	std::vector<float> hist(36, 0);
	int maxr = 0;
	int ksize = 4;
	int up, down, left, right;
	if(!at.inside(ROI)) return;
	// std::cerr << at.x  << " " << at.y << " " << mag.rows << " " << mag.cols << std::endl;
	up = std::min(ksize, at.y);
	down = std::min(ksize, mag.rows - 1 - at.y);
	left = std::min(ksize, at.x);
	right = std::min(ksize, mag.cols - 1 - at.x);

	// cv::Mat G = cv::getGaussianKernel(2*ksize+1, 1.5 * scale, phase.type() );	
	Mato G = Mato::zeros(2*ksize+1, 2*ksize+1);
	G[ksize][ksize] = 255;
	cv::GaussianBlur(G, G, cv::Size(), 1.5*scale, 1.5*scale, cv::BORDER_DEFAULT);


	Mato Weight = Mato::zeros(2*ksize+1, 2*ksize+1);
	// std::cerr << at.x - left << " " << at.y - up << " " << left + right + 1 << " " << up + down + 1 << std::endl;

	mag(cv::Rect(at.x - left, at.y - up, left + right + 1, up + down + 1)).copyTo(\
		Weight(cv::Rect(ksize - left, ksize - up, left + right + 1, up + down + 1)));

	Weight = Weight.mul(G);
	

	for(int dx = 0; dx < 2*ksize+1; dx++){
		for(int dy = 0; dy < 2*ksize+1; dy++){
			if(Weight[dy][dx]>0){
				int binval = (int)(phase[at.y - up + dy][at.x - left + dx] / 10)%36;
				// std::cerr << binval << std::endl;
				hist[binval] += Weight[dy][dx];
				if(hist[binval] > hist[maxr]){
					maxr = binval;
				}
			}
		}
	}

	for(int i = 0; i < 36; i++){
		if(hist[i] > hist[maxr] * 0.8){
			float next = i==35 ? 2*hist[35]-hist[34] : hist[i+1];
			float prev = i==0 ? 2*hist[0]-hist[1] : hist[i-1];
			float dH = 0.5 * (next - prev);
			float d2H = next - 2*hist[i] + prev;

			int maximum = d2H ? std::round(i*10 - dH/d2H) : i*10;

 			out.push_back(cv::Point3i(at.x * coords_scale, at.y * coords_scale, maximum));
		}
	}
}

void featureLoc(int w, int s, float o, cv::Mat3b const& img, std::vector<feature>& outLoc){
	cv::Mat1b top;	
	cv::cvtColor(img, top, cv::COLOR_BGR2GRAY);
	cv::Mat1b response_map = cv::Mat1b::zeros(img.size());
	cv::Rect ROI(0, 0, img.cols, img.rows);
	auto element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size(3, 3) );

	for(int i = 0; i < w; i++){
		float coords_scale = 1<<i;

		std::vector<Mato> gaussians;
		std::vector<Mato> DoG;
		octave(s, o, top, gaussians, DoG);

		std::vector<Mato> erotion(s+2), dilation(s+2);
		for(int j = 0; j < s+2; j++){
			cv::dilate( DoG[j], dilation[j], element );
			cv::erode( DoG[j], erotion[j], element );
		}

		for(int j = 1; j < s+1; j++){
			float scale = o * std::pow(2., (float)j/s + i);

			cv::Mat1b keymap = ( 
				  ( (DoG[j] == erotion[j]) & (DoG[j] <= erotion[j-1]) & (DoG[j] <= erotion[j+1]) ) \
				| ( (DoG[j] == dilation[j]) & (DoG[j] >= dilation[j-1]) & (DoG[j] >= dilation[j+1]) ) );

			Mato Dx, Dy, Do;
			Mato Dxx, Dyy, Doo, Dxy, Dxo, Dyo;
			
			diff_x(DoG[j], Dx);
			diff_y(DoG[j], Dy);
			diff2_x(DoG[j], Dxx);
			diff2_y(DoG[j], Dyy);
			Do = (DoG[j+1] - DoG[j-1]) / 2;			
			Doo = DoG[j+1] - 2*DoG[j]  + DoG[j-1];
			diff_y(Dx, Dxy);
			diff_x(Do, Dxo);
			diff_y(Do, Dyo);

			Mato traceH = Dxx + Dyy;
			Mato detH = Dxx.mul(Dyy) - Dxy.mul(Dxy);
			cv::Mat1b non_Edge_response = (traceH.mul(traceH) / (detH)) < THRES_EDGE;


			Mato Lx, Ly, Lmag, Ltheta;
			diff_x(gaussians[j], Lx);
			diff_y(gaussians[j], Ly);
			cv::magnitude(Lx, Ly, Lmag);
			cv::phase(Lx, Ly, Ltheta, true);// measured in degrees, 0~360
			// cv::imshow("Display Image", Lmag / 255);
			// cv::waitKey(0);

			// double min, max;
			// cv::minMaxLoc(Ltheta, &min, &max);
			// std::cerr << min << " " << max << std::endl;


			std::vector<cv::Point> keypoints;
			cv::findNonZero(keymap, keypoints);
			for(auto& p : keypoints){
				cv::Mat_<float> sec_derivative = (cv::Mat_<float>(3,3) << \
					Dxx(p), Dxy(p), Dxo(p), \
					Dxy(p), Dyy(p), Dyo(p), \
					Dxo(p), Dyo(p), Doo(p));
				cv::Mat_<float> first_derivative = (cv::Mat_<float>(3,1) << Dx(p), Dy(p), Do(p));
				cv::Mat_<float> x_hat;
				// x_hat = sec_derivative.inv(cv::DECOMP_SVD) * -first_derivative;
				cv::solve(sec_derivative, -first_derivative, x_hat, cv::DECOMP_SVD);

				if(cv::countNonZero(x_hat > 0.5) > 0){
					continue;
				}
				
				if(std::abs(DoG[j](p) + 0.5*first_derivative.dot(x_hat)) >= THRES_CONTRAST * 255. \
					&& detH(p) > 0 && non_Edge_response(p)){

					cv::Point X_plum(p.x + x_hat[0][0], p.y + x_hat[1][0]);
					keyPoint(Lmag, Ltheta, X_plum, scale, coords_scale, outLoc);

					// cv::Point X_plum((p.x + x_hat[0][0]) * coords_scale, (p.y + x_hat[1][0]) * coords_scale);
					// if(X_plum.inside(ROI)){
					// 	if (response_map(X_plum)) continue;
					// 	response_map(X_plum) = 255;
	
					// 	outLoc.push_back(cv::Point3i((p.x + x_hat[0][0]) * coords_scale, (p.y + x_hat[1][0]) * coords_scale, j - 1));
					// }
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

cv::Point dir(float theta, float r){
	float x = r*std::cos(theta/180. * M_PI);
	float y = r*std::sin(theta/180. * M_PI);
	return cv::Point(x, y);
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

	// TODO: upsample once
	cv::Mat3b big_image;
	cv::resize(image, big_image, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
	// cv::pyrUp(image, big_image);
	std::vector<feature> sift_features;
	featureLoc(4, 3, 1.6, big_image, sift_features);

	

	auto end = std::chrono::system_clock::now(); 
    std::chrono::duration<double> elapsed_seconds = end-start; 
    std::cerr << "[Info] Elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cerr << "[Info] Found feautures: " << sift_features.size() << "\n";

    for(auto& p : sift_features){
    	cv::Scalar out[] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};
		// cv::circle(image, cv::Point(p.x, p.y), 3, out[0], -1);
		float r = 20.;
		cv::Point from(p.x / 2, p.y / 2);
		cv::arrowedLine(image, from, from + dir(p.z, r), out[1], 2, 8, 0, 0.2);
	}
	cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
	cv::imshow("Display Image", image);
	cv::waitKey(0);	
}
