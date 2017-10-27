// Corrupted Video.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <string.h>
#include <math.h>

#include <windows.h>
#include <chrono>
#include <time.h>

using namespace std;
using namespace std::chrono;
using namespace cv;

int main()
{
	std::string videoname = "corrupted_video.mp4";
	VideoCapture cap(videoname);			// capture the video
	VideoWriter output_cap("output.avi",	// Setup output video
		CV_FOURCC('D', 'I', 'V', 'X'),
		cap.get(CV_CAP_PROP_FPS),
		cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH),
			cap.get(CV_CAP_PROP_FRAME_HEIGHT)));

	if (!cap.isOpened())	// if not success, exit program
	{
		cout << "Cannot open the video" << endl;
		system("pause");
		return 0;
	}
	else
		cout << "Opened video: " << videoname << ", FOURCC = " << cap.get(CV_CAP_PROP_FOURCC) << endl;

	int nbFrames = 0;
	int fps = 0;		// Set FPS > 0 to show video and save output
	double desired_time = (1.0 / fps) * 1000.;
	
	auto current_time = high_resolution_clock::now();
	auto previous_time = high_resolution_clock::now();
	auto elapsed_time = duration_cast<milliseconds>(current_time - previous_time).count();

	vector<Mat> frames;

	bool video_finished = false;

	while (!video_finished)
	{
		Mat frame;

		// read frame
		cap >> frame;
		nbFrames++;

		if (frame.rows > 0 && frame.cols > 0)
		{
			frames.push_back(frame);
			if (fps > 0)
			{
				imshow("VIDEO", frame);
				output_cap.write(frame);
			}
			previous_time = high_resolution_clock::now();
			// Save each frame into images
			//sstring filename = "./frames/Frame" + to_string(nbFrames) + ".png";
			//imwrite(filename, frame);
		}
		else
			video_finished = true;

		switch (waitKey(1)) // 10ms
		{
			case 27: // 'esc' key has been pressed, exit program
				cvDestroyAllWindows();
				cap.release();
				return 0;
			case 99: // 'c'
				break;
			case 'r':
				cvDestroyAllWindows();
				cap.release();
				cap.open(videoname);
				break;
			default:
				break;
		}

		// if FPS for video output is defined => regulate FPS
		if (fps > 0)
		{
			current_time = high_resolution_clock::now();
			elapsed_time = duration_cast<milliseconds>(current_time - previous_time).count();

			if (elapsed_time < desired_time)
				// Wait for this duration to achieve desired FPS
				Sleep((DWORD)(desired_time - elapsed_time));
			// set previous = current
			previous_time = current_time;
		}
	}

	int channels[] = { 0, 1 };
	MatND hist;
	// Quantize the hue to 30 levels
	// and the saturation to 32 levels
	int hbins = 30, sbins = 32;
	int histSize[] = { hbins, sbins };
	// hue varies from 0 to 179, see cvtColor
	float hranges[] = { 0, 180 };
	// saturation varies from 0 (black-gray-white) to
	// 255 (pure spectrum color)
	float sranges[] = { 0, 256 };
	const float* ranges[] = { hranges, sranges };
	calcHist(&frames[0], frames.size(), channels, Mat(), // do not use mask
		hist, 2, histSize, ranges,
		true, // the histogram is uniform
		false);

	system("pause");
	return 0;
}

