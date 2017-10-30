#include "stdafx.h"

#include "corrupted_video.h"

// Create a video from an array of frames and store it locally
void save_output_video(vector<Mat> frames, string output_video, int fps, Size video_size, bool hsv2bgr)
{
	VideoWriter output_cap(output_video, CV_FOURCC('D', 'I', 'V', 'X'), fps, video_size);
	for each (auto frame in frames)
	{
		// Clone to avoid overwrite on the same address
		Mat new_frame = frame.clone();
		if (hsv2bgr)
			cvtColor(new_frame, new_frame, CV_HSV2BGR);
		output_cap.write(new_frame);
	}
}

// Store all the frames locally at output_folder
// hsv2bgr = true if original frames in hsv => convert to BGR beforehand
void save_frames(vector<Mat> frames, string output_folder, bool hsv2bgr)
{
	int nbFrames = 0;
	for each (auto frame in frames)
	{
		// Clone to avoid overwrite on the same address
		Mat new_frame = frame.clone();
		if (hsv2bgr)
			cvtColor(new_frame, new_frame, CV_HSV2BGR);
		string filename = output_folder + to_string(nbFrames) + ".png";
		imwrite(filename, new_frame);
		nbFrames++;
	}
}

// Generate an array of frames from a video
// extract_hsv = true to have frames in hsv
vector<Mat> extract_frames(string input_video, bool extract_hsv)
{
	// Capture the video - needs opencv_ffmpeg331_64.dll to extract mp4 - vS needs to be in Release
	VideoCapture cap(input_video);
	vector<Mat> frames;
	if (!cap.isOpened())	// if not success, exit program
	{
		cout << "Cannot open the video" << endl;
		return frames;
	}
	else
		cout << "Opened video: " << input_video << ", FPS = " << cap.get(CV_CAP_PROP_FPS) << endl;
	
	int nbFrames = 0;

	while (nbFrames < 10000)
	{
		// read frame
		Mat frame;
		cap >> frame;
		if (frame.data)
		{
			if (extract_hsv)
				cvtColor(frame, frame, CV_BGR2HSV);
			frames.push_back(frame);
		}
		else
			break;
	}
	cout << nbFrames << endl;
	return frames;
}

// Generate an array of frames from a images folder
// extract_hsv = true to have frames in hsv
vector<Mat> extract_frames_folder(string input_folder, bool extract_hsv)
{
	vector<Mat> frames;
	int nbFrames = 0;
	
	while (nbFrames < 10000)
	{
		// read frame
		Mat frame = imread(input_folder + to_string(nbFrames) + ".png");
		//waitKey();
		//cout << input_folder + to_string(nbFrames) + ".png" << endl;
		//imshow("frame", frame);
		nbFrames++;
		if (frame.data)
		{
			if (extract_hsv)
				cvtColor(frame, frame, CV_BGR2HSV);
			frames.push_back(frame);
		}
		else
			break;
	}
	cout << nbFrames << endl;
	return frames;
}

vector<int> trace_frames(vector<bool> &indexes_check, vector<vector<pair<double, int>>> compare_histos)
{
	vector<int> indexes_order;
	int nbFrames = indexes_check.size();
	int current_index = 0;
	double traceback_threshold = 0.9f;
	bool finished_traceback = false;
	// Trace frames until the end on one side
	while (!finished_traceback)
	{
		int j = 0;
		bool found_new_index = false;
		while (!found_new_index && j < nbFrames)
		{
			double value = compare_histos[current_index][j].first;
			int ind = compare_histos[current_index][j].second;
			if (indexes_check[ind] == true && value > traceback_threshold)
			{
				current_index = ind;
				indexes_check[current_index] = false;
				indexes_order.push_back(current_index);
				found_new_index = true;
			}
			else if (value > traceback_threshold)
				j++;
			else
			{
				j = nbFrames;
				finished_traceback = true;
			}
		}
	}
	return indexes_order;
}

// Draw an 1-dimension histogram of an image in RGB
void drawHist(Mat src, string windowName)
{
	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	split(src, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	/// Display
	namedWindow(windowName, CV_WINDOW_AUTOSIZE);
	imshow(windowName, histImage);

	/// Display
	namedWindow("Original " + windowName, CV_WINDOW_AUTOSIZE);
	imshow("Original " + windowName, src);
}

int test()
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
				imshow("VIDEO", frame);
			previous_time = high_resolution_clock::now();

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

