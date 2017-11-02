#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <iostream>
#include <string.h>
#include <vector>
#include <math.h>

#include <windows.h>
#include <chrono>
#include <time.h>

using namespace std;
using namespace chrono;
using namespace cv;
using namespace cv::xfeatures2d;

#define blue  CV_RGB(0,0,255)
#define green CV_RGB(0,255,0)
#define red   CV_RGB(255,0,0)
#define white CV_RGB(255,255,255)
#define black CV_RGB(0,0,0)

struct greater_sort
{
	template<class T>
	bool operator()(T const &a, T const &b) const { return a > b; }
};

// Core functions
void rearrange_local_OpticalFlow(vector<Mat> &frames, int index_begin, int size, vector<int> &indexes);
vector<Mat> filter_OpticalFlow(vector<Mat> frames);
vector<Mat> filter_Histogram(vector<Mat> frames);
vector<vector<pair<double, int>>> compare_histograms(vector<Mat> frames, vector<MatND> histos);
vector<int> trace_frames(vector<bool> &indexes_check, vector<vector<pair<double, int>>> compare_arrays, double traceback_threshold, bool greater_compare);
vector<MatND> calculate_Histogram(vector<Mat> frames);

// Complimentary functions for video and frames manipulation
void save_output_video(vector<Mat> frames, string output_video, int fps, Size video_size, bool hsv2bgr);
void save_frames(vector<Mat> frames, string output_folder, bool hsv2bgr);
void save_frames_with_index(vector<Mat> frames, vector<int> indexes, string output_folder, bool hsv2bgr);
vector<Mat> extract_frames(string input_video, bool extract_hsv);
vector<Mat> extract_frames_folder(string input_folder, bool extract_hsv);

// Testing functions
vector<vector<pair<double, int>>> compare_FLANN_matcher(vector<Mat> frames);
void rearrange_local_FLANN(vector<Mat> &frames, int index_begin, int size, vector<int> &indexes);
void rearrange(vector<Mat> &frames, vector<int> &indexes);
bool calcOpticalFlowDirection(Mat frame1, Mat frame2);
void drawHist(Mat src, string windowName);