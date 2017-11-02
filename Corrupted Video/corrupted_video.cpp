#include "stdafx.h"

#include "corrupted_video.h"

/*
  Function: LOCAL SORTING METHOD USING OPTICAL FLOW
	- Calculate points of interest on each frame
	- Project these points using optical flow to other frames and sorting based on displacement
	- Calculate and sort average ranking of each frame to find the final order
  Parameters:
	- frames		: input array to pull frames from
	- index_begin	: position of first frame
	- size			: number of frames to be sorted
	- indexes		: for debugging
  Returns:
	- frames		: array with sorted local frames
*/
void rearrange_local_OpticalFlow(vector<Mat> &frames, int index_begin, int size, vector<int> &indexes)
{
	// Parameters for function goodFeaturesToTrack
	int maxPoints = 500;
	int blockSize = 3;
	double qualityLevel = 0.001;
	double minDistance = 10;
	bool useHarrisDetector = false;
	double k = 0.04;

	// Average order of each frame
	map<int, double> avg_order;
	for (int i = index_begin; i < index_begin + size; i++)
		avg_order[i] = index_begin == 0 ? 0 : i - index_begin;

	//--- Each frame is used for reference to calculate optical flow
	for (int index_ref = index_begin; index_ref < index_begin + size; index_ref++)
	{
		Mat gray;
		vector<Point2f> goodPts;
		vector<pair<double, int>> avg_x;

		// Find points of interest in the reference frame
		cvtColor(frames[index_ref], gray, CV_BGR2GRAY);
		goodFeaturesToTrack(gray, goodPts, maxPoints, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);

		for (int i = index_begin; i < index_begin + size; i++)
		{
			// average dx = 0 if same frame
			if (i == index_ref)
			{
				avg_x.push_back(make_pair(0, i));
				continue;
			}

			// Calculate projected points using optical flow on other frames
			vector<Point2f> nextPts, goodCors, nextCors;
			std::vector<uchar> status(goodPts.size());
			cv::Mat err;
			calcOpticalFlowPyrLK(frames[index_ref], frames[i], goodPts, nextPts, status, err);

			// Limit number of good matchings
			double diff_x = 0, avg_dx = 0, avg_dy = 0, min_dx = 1000, min_dy = 1000;
			int count = 0;	// number of points successfully projected
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

			count = 0;	// number interesting points
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
			count = count > 0 ? count : 1;
			diff_x /= count;

			// Calculate overall transformation: translation + rotation only
			Mat T = estimateRigidTransform(nextCors, goodCors, false); // false = rigid transform, no scaling/shearing
																	   //diff_x = (T.rows < 2) ? diff_x : T.at<double>(0, 2);
			avg_x.push_back(make_pair(diff_x, i));

			// Limit number of good points to project for next frame - experimental
			goodPts.clear();
			goodPts = goodCors;
		}

		// Sort results based on average dx and calculate new average order
		sort(avg_x.begin(), avg_x.end());
		for (int k = 0; k < avg_x.size(); k++)
			avg_order[avg_x[k].second] += k;
	}

	//--- Sort frames based on their average order and swap accordingly
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

	// Debugging
	/*std::cout << "--- SORT OPTICAL FLOW ---" << endl;
	for (int i = index_begin; i < index_begin + size; i++)
		std::cout << i << " | " << avg_order[i] << " | " << indexes[i] << endl;
	std::cout << "--- SORT OPTICAL FLOW ---" << endl;*/
}

/*
  Function: Filter method using Optical Flow
	- Compare each frame with all other frames
	- Good frame matches with more than 10% other frames
	- Others are noises
  Parameters:
	- frames		: input frames array
  Returns:
	- filter_frames	: filtered frames
*/
vector<Mat> filter_OpticalFlow(vector<Mat> frames)
{
	// Parameters for function goodFeaturesToTrack
	int maxPoints = 500;
	int blockSize = 3;
	double qualityLevel = 0.001;
	double minDistance = 10;

	//--- Filter parameters
	int nbFrames = frames.size();
	float status_threshold = 0.9f;		// threshold for matching status points: greater is good
	float good_match_threshold = 0.1f;	// If frame match with more than 10% images => good
	vector<int> index_noises;			// Indexes of noise frames

	// Look for a source image (not noise) and filter all noises basing on this source
	// Source : high number of matching points
	// Noise : low matchings with other images
	for (int index = 0; index < nbFrames; index++)
	{
		Mat gray;
		vector<Point2f> goodPts;

		// Find points of interest in the reference frame
		cvtColor(frames[index], gray, CV_BGR2GRAY);
		goodFeaturesToTrack(gray, goodPts, maxPoints, qualityLevel, minDistance);
		int nbGoodPts = goodPts.size();
		int count_good_matches = 0;
		for (int i = 0; i < nbFrames; i++)
		{
			// average dx = 0 if same frame
			if (i == index)
				continue;

			// Calculate projected points using optical flow on other frames
			vector<Point2f> nextPts;
			std::vector<uchar> status(nbGoodPts);
			cv::Mat err;
			calcOpticalFlowPyrLK(frames[index], frames[i], goodPts, nextPts, status, err);

			int count = 0;	// number of points successfully projected
			for (int k = 0; k < status.size(); k++)
				if (status[k])
					count++;
			double percentage_match = count * 1.0f / nbGoodPts;
			// Good image - matching with another
			if (percentage_match > status_threshold)
			{
				count_good_matches++;
				if (count_good_matches * 1.0f / nbFrames > good_match_threshold)
					break;
			}
		}
		if (count_good_matches * 1.0f / nbFrames > good_match_threshold)
			index_noises.push_back(index);
	}

	//--- Remove all noises
	vector<Mat> filter_frames;
	if (index_noises.size() == 0)
		return frames;
	cout << index_noises.size() << " noise frames!" << endl;

	int index = 0;
	for (int i = 0; i < nbFrames; i++)
	{
		if (i != index_noises[index])
			filter_frames.push_back(frames[i]);
		else if (index < index_noises.size() - 1)
			index++;
	}
	return filter_frames;
}

/*
  Function: Filter method using Histogram Comparison
  Parameters:
	- frames		: input frames array
  Returns:
	- filter_frames	: filtered frames
*/
vector<Mat> filter_Histogram(vector<Mat> frames)
{
	vector<Mat> filter_frames;
	vector<MatND> histos = calculate_Histogram(frames);

	int nbFrames = frames.size();
	int compare_method = 0;			// CV_COMP_CORREL - Correlation
	float histo_threshold = 0.2f;	// Threshold for Correlation method
	float noise_threshold = 0.3f;	// Noises expected at < 30% number of frames
	vector<int> index_noises;		// Indexes of noise frames
	int index = 0;

	// Look for a source image (not noise) and filter all noises basing on this source
	// Source : high number of matching histograms
	// Noise : low matchings with other images
	while (index < nbFrames)
	{
		int count_noises = 0;
		index_noises.clear();
		for (int i = 0; i < nbFrames; i++)
		{
			double compare = compareHist(histos[index], histos[i], compare_method);
			if (compare < histo_threshold)
			{
				count_noises++;
				index_noises.push_back(i);
			}
		}
		if (count_noises / nbFrames < noise_threshold)
			break;
		else
			index++;
	}

	// Cannot find a source image (all matchings < threshold) => quit
	if (index >= nbFrames)
	{
		cout << "No source image found, all are noises, nani??? Not possible..." << endl;
		return filter_frames;
	}

	// No noise frame found
	if (index_noises.size() == 0)
	{
		cout << "No noise. Nice!" << endl;
		return frames;
	}
	cout << index_noises.size() << " noise frames!" << endl;

	//--- Remove all noise frames and their histograms
	index = 0;
	for (int i = 0; i < nbFrames; i++)
	{
		if (i != index_noises[index])
			filter_frames.push_back(frames[i]);
		else if (index < index_noises.size() - 1)
			index++;
	}
	return filter_frames;
}

/*
  Function: Calculate sorted histogram comparisons matrix frame by frame (NxN)
  Parameters:
	- frames	: input array
	- histos	: histogram results for frames
  Returns: matrix NxN type vector<vector<pair<double, int>>>
	- double	: histogram comparison value
	- int		: corresponding index - stored for easier sorting and tracing later
*/
vector<vector<pair<double, int>>> compare_histograms(vector<Mat> frames, vector<MatND> histos)
{
	int nbFrames = frames.size();
	vector<vector<pair<double, int>>> compare_histos(nbFrames);

	int compare_method = 0;			// CV_COMP_CORREL - Correlation
	for (int i = 0; i < nbFrames; i++)
	{
		for (int j = 0; j < nbFrames; j++)
		{
			// Same frame (not interested) => result = 1 but set to 0 for sorting purpose
			if (i == j)
			{
				compare_histos[i].push_back(make_pair(0, j));
				continue;
			}
			compare_histos[i].push_back(make_pair(compareHist(histos[i], histos[j], compare_method), j));
		}
		sort(compare_histos[i].begin(), compare_histos[i].end(), greater_sort());
	}
	return compare_histos;
}

/*
  Function: Trace all frames basing on their comparison matrix
	- Take frame 0 (arbitrarily) as first reference
	- Trace all nearest frames to one end until finding no more
	- We found the left (or right) part of frame 0
  Parameters:
	- indexes_check			: array marking which frame is available
	- compare_arrays		: matrix of comparison results frame by frame (from Histogram or FLANN)
	- traceback_threshold	: threshold to stop looking for next frame
	- greater_compare		: comparison using '>' (true) or '<' (false)
  Returns:
	- indexes_order			: sorted indexes tracing on one side of the original frame (0)
*/
vector<int> trace_frames(vector<bool> &indexes_check, vector<vector<pair<double, int>>> compare_arrays, double traceback_threshold, bool greater_compare)
{
	vector<int> indexes_order;
	int nbFrames = indexes_check.size();
	int current_index = 0;
	bool finished_traceback = false;

	//--- Trace frames until the end on one side
	while (!finished_traceback)
	{
		int j = 0;
		while (j < nbFrames)
		{
			double value = compare_arrays[current_index][j].first;
			int ind = compare_arrays[current_index][j].second;
			bool compare_value = greater_compare ? value > traceback_threshold : value < traceback_threshold;
			// This frame is available and satisfy value comparison threshold
			if (indexes_check[ind] == true && compare_value)
			{
				indexes_order.push_back(ind);	// Take this frame
				indexes_check[ind] = false;		// This frame no longer available
				current_index = ind;			// Use this frame to find next frame
				break;
			}
			else if (compare_value)
				j++;
			else
			{
				// We passed the comparison threshold => reaching one end of the video => other frames are on the other end => finished tracing!
				finished_traceback = true;
				break;
			}
		}
	}
	return indexes_order;
}

//--- Calculate histograms for each frame
vector<MatND> calculate_Histogram(vector<Mat> frames)
{
	// Parameters for histograms HSV in 2D
	int h_bins = 50, s_bins = 60;
	int histSize[] = { h_bins, s_bins };
	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 256 };
	const float* ranges[] = { hranges, sranges };
	int channels[] = { 0, 1 };

	vector<MatND> histos;
	for each (auto frame in frames)
	{
		MatND hist;
		calcHist(&frame, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
		normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());
		histos.push_back(hist);
	}
	return histos;
}

//--- Calculate matrix of sorted FLANN matcher distance frame by frame (NxN)
// EXPERIMENTAL
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

//--- Create a video from an array of frames and store it locally
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

//--- Save all frames locally at output_folder
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

//--- Save all frames locally with indexes in file name
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

//--- Generate an array of frames from a video
vector<Mat> extract_frames(string input_video, bool extract_hsv)
{
	// Capture the video - needs opencv_ffmpeg331_64.dll to extract mp4 - Release mode
	VideoCapture cap(input_video);
	vector<Mat> frames;

	if (!cap.isOpened())
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
			// extract_hsv = true to have frames in hsv
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

//--- Generate an array of frames from an images folder
vector<Mat> extract_frames_folder(string input_folder, bool extract_hsv)
{
	vector<Mat> frames;
	int nbFrames = 0;

	while (nbFrames < 10000)
	{
		// read frame
		Mat frame = imread(input_folder + to_string(nbFrames) + ".png");
		nbFrames++;
		if (frame.data)
		{
			// extract_hsv = true to have frames in hsv
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

//--- Test fucntion: draw an 1-dimension histogram of an image in RGB
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

//--- Test method - not working
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

//--- Local sorting method using FLANN matcher - not working
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

//--- Test method for Optical Flow
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