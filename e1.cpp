#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <algorithm>

#include "../matplotlib-cpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main(int, char**)
{
	cv::Mat img = cv::imread("e1.png", CV_LOAD_IMAGE_GRAYSCALE);
	int w = img.cols / 10;
	std::vector< double > v1, v2, laplacians;
	size_t c_max = 0;
	double score_max = 0;
	for ( size_t c = 0; c < img.cols - w; c++ ) {
		cv::Mat roi = img(cv::Rect(c, 0, w, img.rows));

		cv::Mat lap;
		cv::Laplacian(roi, lap, -1);
		laplacians.push_back(cv::norm(lap, cv::NORM_L1));

		cv::Mat result;
		cv::matchTemplate( img, roi, result, CV_TM_CCOEFF );

		v1.push_back(cv::norm(result, cv::NORM_L1));
		if (v1.back() + laplacians.back() > score_max) {
			c_max = c;
			score_max = v1.back() + laplacians.back();
		}

		cv::matchTemplate( img, roi, result, CV_TM_CCOEFF );
		v2.push_back(cv::norm(result, cv::NORM_L1));
	}

	std::cout << c_max << "\n";
	cv::Mat roi = img(cv::Rect(c_max, 0, w, img.rows));
	cv::imwrite("best.png", roi);

	double M1 = *std::max_element(v1.begin(), v1.end());
	for ( auto && x : v1 ) {
		x /= M1;
	}
	std::vector<int> x1(v1.size());
	std::iota(x1.begin(), x1.end(), 0);

	double M2 = *std::max_element(v2.begin(), v2.end());
	for ( auto && x : v2 ) {
		x /= M2;
	}
	std::vector<int> x2(v2.size());
	std::iota(x2.begin(), x2.end(), 0);

	double M3 = *std::max_element(laplacians.begin(), laplacians.end());
	for ( auto && x : laplacians ) {
		x /= M3;
	}

	cv::Mat result;
	cv::matchTemplate( img, roi, result, CV_TM_CCOEFF );
	double gold = cv::norm(result, cv::NORM_L1);
	std::cout << "e1 " << 1 << "\n";

	cv::Mat img2 = cv::imread("e2.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::matchTemplate( img2, roi, result, CV_TM_CCOEFF );
	std::cout << "e2 " << cv::norm(result, cv::NORM_L1) / gold << "\n";

	cv::Mat img3 = cv::imread("e3.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::matchTemplate( img3, roi, result, CV_TM_CCOEFF );
	std::cout << "e3 " << cv::norm(result, cv::NORM_L1) / gold << "\n";

	cv::Mat img4 = cv::imread("e4.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::matchTemplate( img4, roi, result, CV_TM_CCOEFF );
	std::cout << "e4 " << cv::norm(result, cv::NORM_L1) / gold << "\n";	

	cv::Mat img5 = cv::imread("e5.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::matchTemplate( img5, roi, result, CV_TM_CCOEFF );
	std::cout << "e5 " << cv::norm(result, cv::NORM_L1) / gold << "\n";		

	cv::Mat img6 = cv::imread("e6.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::matchTemplate( img6, roi, result, CV_TM_CCOEFF );
	std::cout << "e6 " << cv::norm(result, cv::NORM_L1) / gold << "\n";		

	plt::plot(x1, v1);
	plt::plot(x1, laplacians);
	//plt::plot(x2, v2);
	plt::show();
	return 0;
}

