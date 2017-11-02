#include "stdafx.h"

#include "corrupted_video.h"

void rearrange(vector<Mat> &frames, vector<int> &indexes)
{
	int nbFrames = frames.size();
	int count_positive = 0, count_negative = 0;
	cout << endl;
	for (int i = 0; i < nbFrames - 1; i++)
	{
		cout << i << " | ";
		//if (true)
		if (calcOpticalFlowDirection(frames[i], frames[i + 1]))
		{
			cout << indexes[i] << " - " << indexes[i + 1] << " = POSITIVE" << endl;
			count_positive++;
		}
		else
		{
			cout << indexes[i] << " - " << indexes[i + 1] << " = nega" << endl;
			swap(frames[i], frames[i + 1]);
			swap(indexes[i], indexes[i + 1]);
			count_negative++;
		}
	}
	cout << "POSITIVE = " << count_positive << endl;
	cout << "NEGATIVE = " << count_negative << endl;

}

void rearrange_local_FLANN(vector<Mat> &frames, int index_begin, int size, vector<int> &indexes)
{
	int nbFrames = frames.size();
	Mat gray;
	cvtColor(frames[index_begin], gray, CV_BGR2GRAY);

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;
	Ptr<SURF> detector = SURF::create(minHessian);
	vector<vector<KeyPoint>> keypoints(nbFrames);
	for (int i = index_begin; i < index_begin + size; i++)
		detector->detect(frames[i], keypoints[i]);

	//-- Step 2: Calculate descriptors (feature vectors)
	Ptr<SURF> extractor = SURF::create();
	vector<Mat> descriptors(nbFrames);
	for (int i = index_begin; i < index_begin + size; i++)
		extractor->compute(frames[i], keypoints[i], descriptors[i]);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	vector<double> avg_x(nbFrames);
	avg_x[index_begin] = 0;
	for (int i = index_begin + 1; i < index_begin + size; i++)
	{
		vector< DMatch > match;
		matcher.match(descriptors[index_begin], descriptors[i], match);

		double min_dist = 100;
		//-- Quick calculation of min distance between keypoints
		for each (auto m in match)
			if (m.distance < min_dist) 
				min_dist = m.distance;
			
		int count = 0;
		double diff_x = 0;
		vector<Point2f> goodCors, nextCors;
		for each (auto m in match)
		{
			if (m.distance <= 3 * min_dist)
			{
				Point2f p1(keypoints[index_begin][m.queryIdx].pt);
				Point2f p2(keypoints[i][m.trainIdx].pt);
				goodCors.push_back(p1);
				nextCors.push_back(p2);
				diff_x += (p2 - p1).x;
				count++;
			}
		}
		diff_x /= count;
		// translation + rotation only
		Mat T = estimateRigidTransform(nextCors, goodCors, false); // false = rigid transform, no scaling/shearing
		avg_x[i] = (T.rows < 2) ? diff_x : T.at<double>(0, 2);
	}

	for (int i = index_begin; i < index_begin + size - 1; i++)
		for (int j = i + 1; j < index_begin + size; j++)
			if (avg_x[i] < avg_x[j])
			{
				swap(frames[i], frames[j]);
				swap(indexes[i], indexes[j]);
				swap(avg_x[i], avg_x[j]);

			}
	
	//sort(avg_x.begin(), avg_x.end());
	cout << "--- SORT FLANN ---" << endl;
	for (int i = index_begin; i < index_begin + size; i++)
		cout << i << " | " << avg_x[i] << " | " << indexes[i] << endl;
	cout << "--- SORT FLANN ---" << endl;
}

void rearrange_local_OpticalFlow(vector<Mat> &frames, int index_begin, int size, vector<int> &indexes)
{
	int maxPoints = 500;	// 223
	int blockSize = 3;
	double qualityLevel = 0.001;
	double minDistance = 10;
	bool useHarrisDetector = false;
	double k = 0.04;
	
	map<int, double> avg_order;
	for (int i = index_begin; i < index_begin + size; i++)
		avg_order[i] = index_begin == 0 ? 0 : i - index_begin;
	
	for (int index_ref = index_begin; index_ref < index_begin + size; index_ref++)
	{
		Mat gray;
		vector<Point2f> goodPts;
		vector<pair<double, int>> avg_x;

		cvtColor(frames[index_ref], gray, CV_BGR2GRAY);
		goodFeaturesToTrack(gray, goodPts, maxPoints, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);

		for (int i = index_begin; i < index_begin + size; i++)
		{
			if (i == index_ref)
			{
				avg_x.push_back(make_pair(0, i));
				continue;
			}

			vector<Point2f> nextPts, goodCors, nextCors;
			std::vector<uchar> status(goodPts.size());
			cv::Mat err;
			calcOpticalFlowPyrLK(frames[index_ref], frames[i], goodPts, nextPts, status, err);
			double diff_x = 0, avg_dx = 0, avg_dy = 0, min_dx = 1000, min_dy = 1000;
			int count = 0;
			for (int k = 0; k < status.size(); k++)
			{
				if (status[k])
				{
					double dx = abs(nextPts[k].x - goodPts[k].x);
					double dy = abs(nextPts[k].y - goodPts[k].y);
					avg_dx += dx;
					avg_dy += dy;
					if (dx < min_dx)
						min_dx = dx;
					if (dy < min_dy)
						min_dy = dy;
					count++;
				}
			}
			
			count = count > 0 ? count : 1;
			avg_dx /= count;
			avg_dy /= count;
			int multiplier_detect = 150;
			min_dx = min(min_dx*multiplier_detect, avg_dx);
			min_dy = min(min_dy*multiplier_detect, avg_dy);
			//cout << "Min dx  = " << min_dx << endl;
			//cout << "avg dx  = " << avg_dx << endl;

			count = 0;
			for (int k = 0; k < status.size(); k++)
			{
				double dx = abs(nextPts[k].x - goodPts[k].x);
				double dy = abs(nextPts[k].y - goodPts[k].y);
				
				if (status[k] && dx < avg_dx && dy < avg_dy)
				{
					goodCors.push_back(goodPts[k]);
					nextCors.push_back(nextPts[k]);
					count++;
					diff_x += nextPts[k].x - goodPts[k].x;
				}
			}
			// Weighted on the number of matching points
			count = count > 0 ? (count*count) : 1;
			diff_x /= count;

			// translation + rotation only
			Mat T = estimateRigidTransform(nextCors, goodCors, false); // false = rigid transform, no scaling/shearing
			diff_x = (T.rows < 2) ? diff_x : T.at<double>(0, 2);
			
			avg_x.push_back(make_pair(diff_x, i));
		}
		sort(avg_x.begin(), avg_x.end());
		for (int k = 0; k < avg_x.size(); k++)
			avg_order[avg_x[k].second] += k;
	}

	for (int i = index_begin; i < index_begin + size - 1; i++)
	{
		for (int j = i + 1; j < index_begin + size; j++)
		{
			if (avg_order[i] > avg_order[j])
			{
				swap(frames[i], frames[j]);
				swap(indexes[i], indexes[j]);
				swap(avg_order[i], avg_order[j]);

			}
		}
	}

	std::cout << "--- SORT OPTICAL FLOW ---" << endl;
	for (int i = index_begin; i < index_begin + size; i++)
		std::cout << i << " | " << avg_order[i] << " | " << indexes[i] << endl;

	/*for (int i = index_begin; i < index_begin + size; i++)
	{
		avg_order[i] /= 5;
		cout << i << " | " << avg_order[i] << endl;
	}*/

	std::cout << "--- SORT OPTICAL FLOW ---" << endl;
}

bool calcOpticalFlowDirection(Mat frame1, Mat frame2)
{
	Mat gray1;
	Mat gray2;
	vector<Point2f> goodNext, prevPoints, nextPoints;

	int maxPoints = 50;
	int blockSize = 3;
	double qualityLevel = 0.01;
	double minDistance = 10;
	bool useHarrisDetector = false;
	double k = 0.04;

	cvtColor(frame1, gray1, CV_BGR2GRAY);
	cvtColor(frame2, gray2, CV_BGR2GRAY);

	goodFeaturesToTrack(gray1, prevPoints, maxPoints, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
	
	std::vector<uchar> status(prevPoints.size());
	cv::Mat err;
	calcOpticalFlowPyrLK(frame1, frame2, prevPoints, nextPoints, status, err);
	double diff_x = 0;
	double diff_y = 0;
	for (int i = 0; i < nextPoints.size(); i++)
	{
		if (status[i])
		{
			diff_x += nextPoints[i].x - prevPoints[i].x;
			diff_y += nextPoints[i].y - prevPoints[i].y;
		}
	}
	diff_x /= nextPoints.size();
	diff_y /= nextPoints.size();
	//printf("-- diff x = %f \n", diff_x);
	return (diff_x >= 0);
}

vector<vector<pair<double, int>>> compare_histograms(vector<Mat> frames, vector<MatND> histos)
{
	int nbFrames = frames.size();						// number of frames
	vector<vector<pair<double, int>>> compare_histos;	// store histogram comparisons of frame by frame
	compare_histos.resize(nbFrames);

	int compare_method = 0;			// CV_COMP_CORREL - Correlation
	for (int i = 0; i < nbFrames; i++)
	{
		double max = 0, second_max = 0;
		int ind_max = 0, ind_2nd_max = 0;
		for (int j = 0; j < nbFrames; j++)
		{
			if (i == j)
			{
				compare_histos[i].push_back(make_pair(0, j));
				continue;
			}
			compare_histos[i].push_back(make_pair(compareHist(histos[i], histos[j], compare_method), j));
			if (compare_histos[i][j].first > max)
			{
				second_max = max;
				ind_2nd_max = ind_max;
				max = compare_histos[i][j].first;
				ind_max = j;
			}
			else if (compare_histos[i][j].first > second_max)
			{
				second_max = compare_histos[i][j].first;
				ind_2nd_max = j;
			}
		}
		sort(compare_histos[i].begin(), compare_histos[i].end(), greater_sort());
		//cout << i << " | max = " << max << " - " << ind_max << " | 2nd max = " << second_max << " - " << ind_2nd_max << endl;
	}
	return compare_histos;
}

vector<vector<pair<double, int>>> compare_FLANN_matcher(vector<Mat> frames)
{
	int nbFrames = frames.size();

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 1000;
	Ptr<SURF> detector = SURF::create(minHessian);
	vector<vector<KeyPoint>> keypoints;
	for each (auto frame in frames)
	{
		vector<KeyPoint> keypoint;
		detector->detect(frame, keypoint);
		keypoints.push_back(keypoint);
	}

	//-- Step 2: Calculate descriptors (feature vectors)
	Ptr<SURF> extractor = SURF::create();
	vector<Mat> descriptors;
	for (int i = 0; i < nbFrames; i++)
	{
		Mat desc;
		extractor->compute(frames[i], keypoints[i], desc);
		descriptors.push_back(desc);
	}

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	vector<vector<pair<double, int>>> avg_distances;
	avg_distances.resize(nbFrames);
	for (int i = 0; i < nbFrames; i++)
	{
		for (int j = 0; j < nbFrames; j++)
		{
			if (i == j)
			{
				avg_distances[i].push_back(make_pair(1, j));
				continue;
			}
			vector< DMatch > match;
			matcher.match(descriptors[i], descriptors[j], match);
			double avg_dist = 0;
			for each (auto m in match)
				avg_dist += m.distance;
			avg_dist /= match.size();
			avg_distances[i].push_back(make_pair(avg_dist, j));
		}
		sort(avg_distances[i].begin(), avg_distances[i].end());
	}
	return avg_distances;
}

// Create a video from an array of frames and store it locally
void save_output_video(vector<Mat> frames, string output_video, int fps, Size video_size, bool hsv2bgr)
{
#ifdef _DEBUG
	// THE CODE IS COMPILING IN DEBUG MODE - CANNOT OUPUT VIDEO
	return;
#endif
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

void save_frames_with_index(vector<Mat> frames, vector<int> indexes, string output_folder, bool hsv2bgr)
{
	int nbFrames = 0;
	for each (auto frame in frames)
	{
		// Clone to avoid overwrite on the same address
		Mat new_frame = frame.clone();
		if (hsv2bgr)
			cvtColor(new_frame, new_frame, CV_HSV2BGR);
		string filename = output_folder + to_string(nbFrames) + "_Frame" + to_string(indexes[nbFrames]) + ".png";
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

vector<int> trace_frames(vector<bool> &indexes_check, vector<vector<pair<double, int>>> compare_arrays, double traceback_threshold, bool greater_compare)
{
	vector<int> indexes_order;
	int nbFrames = indexes_check.size();
	int current_index = 0;
	bool finished_traceback = false;
	// Trace frames until the end on one side
	while (!finished_traceback)
	{
		int j = 0;
		while (j < nbFrames)
		{
			double value = compare_arrays[current_index][j].first;
			int ind = compare_arrays[current_index][j].second;
			if (indexes_check[ind] == true && (greater_compare ? value > traceback_threshold : value < traceback_threshold) )
			{
				current_index = ind;
				indexes_check[current_index] = false;
				indexes_order.push_back(current_index);
				break;
			}
			else if (greater_compare ? value > traceback_threshold : value < traceback_threshold)
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

