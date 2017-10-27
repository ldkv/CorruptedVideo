#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"

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

void save_output_video(vector<Mat> frames, string output_name, int fps, Size video_size);
void save_frames(vector<Mat> frames, string output_folder, bool hsv2bgr);
vector<Mat> extract_frames(string input_video, bool extract_hsv);
void drawHist(Mat src, string windowName);