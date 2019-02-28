/**
    This file is part of sgm. (https://github.com/dhernandez0/sgm).

    Copyright (c) 2016 Daniel Hernandez Juarez.

    sgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

**/

#ifndef DISPARITY_METHOD_H_
#define DISPARITY_METHOD_H_

#include <stdint.h>
#include <opencv2/opencv.hpp>
#include "util.h"
#include "configuration.h"
#include "costs.h"
#include "hamming_cost.h"
#include "median_filter.h"
#include "cost_aggregation.h"
#include "debug.h"
#include "demo.h"

//#ifdef OPENCV
//#include "opencv2/ximgproc/disparity_filter.hpp"
//#endif


void init_disparity_method(const uint8_t _p1, const uint8_t _p2);
//cv::Mat compute_disparity_method(cv::Mat left, cv::Mat right, float *elapsed_time_ms, const char* directory, const char* fname);
cv::Mat compute_disparity_method(cv::Mat left, cv::Mat right, float *elapsed_time_ms, int path_agg);
//cv::Mat compute_disparity_method_4(cv::Mat left, cv::Mat right, float *elapsed_time_ms);
//cv::Mat compute_disparity_method(IplImage *left, IplImage *right, float *elapsed_time_ms);
//IplImage *compute_disparity_method(IplImage *left, IplImage *right, float *elapsed_time_ms);
//IplImage *post_disparity_filter(IplImage *ipl_dis_l, IplImage *ipl_dis_r, double lambda, double sigma);
void post_disparity_filter(cv::Mat& filtered_disp, cv::Mat& left_disp, cv::Mat& right_disp, cv::Mat& im_l, cv::Ptr<cv::ximgproc::DisparityWLSFilter>& wls_filter);


double disparity_2_millimeter(int disp, double focal_in_pixel, double baseline_in_millimeter, struct_poly* param_poly);

cv::Mat disparity_2_millimeter(cv::Mat& mat_dis, double focal_in_pixel, double baseline_in_millimeter);
//cv::Mat disparity_2_millimeter(cv::Mat& mat_dis, cv::Mat& mat_bunmo);
//bool undistort_and_rectify(IplImage *ipl_l, IplImage *ipl_r, struct_stereo *param_stereo);

cv::Ptr<cv::ximgproc::DisparityWLSFilter> initDisparityWLSFilter(int wsize, int max_disp, double lambda, double sigma);
void finish_disparity_method();
static void free_memory();

#endif /* DISPARITY_METHOD_H_ */
