#include "stdafx.h"

#include "corrupted_video.h"

int main_histos();
int maingoodFeatures();

int main_histos()
{
	string videoname = "corrupted_video.mp4";
	string input_folder = "./frames/Frame";

	vector<Mat> frames;
	bool extract_from_video = false;
	if (extract_from_video)	// Extract frames from original video
		frames = extract_frames(videoname, true);
	else					// Extract frames from an images folder
		frames = extract_frames_folder(input_folder, true);
	
	if (frames.size() == 0)
		return 0;

	//save_output_video(frames, "input_original.avi", 25, frames[0].size(), true);
	//save_frames(frames, "./frames/Frame", true);

	// Parameters for histograms HSV in 2D
	int h_bins = 50, s_bins = 60;
	int histSize[] = { h_bins, s_bins };
	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 256 };
	const float* ranges[] = { hranges, sranges };
	int channels[] = { 0, 1 };

	// Vector results of histogram calculation for all frames
	vector<MatND> histos;
	for each (auto frame in frames)
	{
		MatND hist;
		calcHist(&frame, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
		normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());
		histos.push_back(hist);
	}
	
	int nbFrames = histos.size();
	int compare_method = 0;			// CV_COMP_CORREL - Correlation
	float histo_threshold = 0.2f;	// Threshold for Correlation method
	float noise_threshold = 0.3f;	// Noises expected at < 30% number of frames
	int index = 0;
	vector<int> index_noises;		// Indexes of noise frames
	// Loop to find a source (not noise) image and find all the noises
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

	// Cannot find a source image => algo failed => quit
	if (index >= nbFrames)
		return 0;

	// Remove all noise frames and their histograms
	vector<Mat> filter_frames;
	vector<MatND> filter_histos;
	index = 0;
	
	// if no noise file add -1 to the array to avoid vector error
	if (index_noises.size() == 0)
		index_noises.push_back(-1);

	for (int i = 0; i < nbFrames; i++)
	{
		if (i != index_noises[index])
		{
			filter_frames.push_back(frames[i]);
			filter_histos.push_back(histos[i]);
		}
		else if (index < index_noises.size() - 1)
			index++;
	}
	//save_frames(filter_frames, "./frames/filter/Frame", true);

	nbFrames = filter_frames.size();					// new number of frames
	vector<vector<pair<double, int>>> compare_histos;	// store histogram comparisons of frame by frame
	
	compare_histos = compare_histograms(filter_frames, filter_histos);
	vector<bool> indexes_check(nbFrames, true);
	indexes_check[0] = false;
	vector<int> indexes_first, indexes_reverse;
	double traceback_threshold = 0.9f;
	bool compare_greater = true;

	//-- Sort indexes
	indexes_first = trace_frames(indexes_check, compare_histos, traceback_threshold, compare_greater);
	indexes_reverse = trace_frames(indexes_check, compare_histos, traceback_threshold, compare_greater);

	//-- Join sorted indexes
	reverse(indexes_first.begin(), indexes_first.end());
	indexes_first.push_back(0);
	for each(auto ind in indexes_reverse)
		indexes_first.push_back(ind);

	//-- Show indexes left out, not sorted
	for (int i = 0; i < nbFrames; i++)
	{
		if (indexes_check[i])
		{
			cout << i << endl;
			int nearest_ind = compare_histos[i][0].second;
			for (int k = 0; k < indexes_first.size(); k++)
				if (indexes_first[k] == nearest_ind)
				{
					indexes_first.insert(indexes_first.begin() + k + 1, i);
					break;
				}
		}
	}

	//-- Join frames
	vector<Mat> arranged_frames;
	for each (auto ind in indexes_first)
	{
		cout << ind << " - ";
		cvtColor(filter_frames[ind], filter_frames[ind], CV_HSV2BGR);
		arranged_frames.push_back(filter_frames[ind]);
	}
	cout << endl;
	
	//rearrange_local_OpticalFlow(arranged_frames, 11, 5, indexes_first);
	//rearrange_local_FLANN(arranged_frames, 0, 6, indexes_first);

	int size_local = 5;
	for (int i = 30; i < nbFrames - size_local; i++)
		rearrange_local_OpticalFlow(arranged_frames, i, size_local, indexes_first);
	
	
	/*char next;
	do
	{
		rearrange(arranged_frames, indexes_first);
		cout << "NEXT ? ";
		cin >> next;
	} while (next == 'y');
*/
	for (int i = 0; i < nbFrames; i++)
		putText(arranged_frames[i], to_string(indexes_first[i]), cvPoint(200, 200), FONT_HERSHEY_COMPLEX, 5, white, 5, CV_AA);
	//save_frames_with_index(arranged_frames, indexes_first, "./frames/arranged/", false);
	
	// Save output video only works in Release
	save_output_video(arranged_frames, "output_original.avi", 30, arranged_frames[0].size(), false);
	save_output_video(arranged_frames, "output_test_1FPS.avi", 1, arranged_frames[0].size(), false);

	reverse(arranged_frames.begin(), arranged_frames.end());
	save_output_video(arranged_frames, "output_reversed.avi", 30, arranged_frames[0].size(), false);
	save_output_video(arranged_frames, "output_test_res_1FPS.avi", 1, arranged_frames[0].size(), false);
	
	waitKey();
	system("pause");
	return 0;
}

//======================================================================
void draw(Mat img, vector<Point2f> points, string nameWindow, Scalar color)
{
	int point_radius = 5;
	for (auto &point : points)
	{
		cv::circle(img, point, point_radius, color);
	}
	imshow(nameWindow, img);
}

int maingoodFeatures()
//int main()
{
	string frame1 = "./frames/Frame20.png";
	string frame2 = "./frames/Frame30.png";
	Mat prevInput = imread(frame1);
	Mat nextInput = imread(frame2);
	Mat prevGray;
	Mat nextGray;
	vector<Point2f> goodNext, nextPoints;
	vector<Point2f> prevPoints;

	int maxPoints = 30;
	int blockSize = 3;
	double qualityLevel = 0.01;
	double minDistance = 10;
	bool useHarrisDetector = false;
	double k = 0.04;

	cvtColor(prevInput, prevGray, CV_BGR2GRAY);
	cvtColor(nextInput, nextGray, CV_BGR2GRAY);

	goodFeaturesToTrack(prevGray, prevPoints, maxPoints, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
	goodFeaturesToTrack(nextGray, goodNext, maxPoints, qualityLevel, minDistance, Mat(), blockSize);

	//draw(nextInput, goodNext, frame2, blue);

	std::vector<uchar> status(prevPoints.size());
	cv::Mat err;
	calcOpticalFlowPyrLK(prevInput, nextInput, prevPoints, nextPoints, status, err);
	double diff_x = 0;
	double diff_y = 0;
	for (int i = 0; i < nextPoints.size(); i++)
	{
		if (status[i])
		{
			arrowedLine(nextInput, prevPoints[i], nextPoints[i], cv::Scalar(0, 0, 255));
			diff_x += nextPoints[i].x - prevPoints[i].x;
			diff_y += nextPoints[i].y - prevPoints[i].y;
		}
	}
	diff_x /= nextPoints.size();
	diff_y /= nextPoints.size();
	printf("-- diff x = %f \n", diff_x);
	printf("-- diff y = %f \n", diff_y);

	draw(prevInput, prevPoints, frame1, blue);
	draw(nextInput, prevPoints, frame2, blue);
	draw(nextInput, nextPoints, "movement", green);

	waitKey();
	system("pause");
	return 0;
}

int main()
{
	main_histos();
	//maingoodFeatures();
}


int mainFLANN()
{
	string videoname = "corrupted_video.mp4";
	string input_folder = "./frames/Frame";

	vector<Mat> frames;
	bool extract_from_video = false;
	if (extract_from_video)	// Extract frames from original video
		frames = extract_frames(videoname, true);
	else					// Extract frames from an images folder
		frames = extract_frames_folder(input_folder, true);

	if (frames.size() == 0)
		return 0;

	//save_output_video(frames, "input_original.avi", 25, frames[0].size(), true);
	//save_frames(frames, "./frames/Frame", true);

	// Parameters for histograms HSV in 2D
	int h_bins = 50, s_bins = 60;
	int histSize[] = { h_bins, s_bins };
	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 256 };
	const float* ranges[] = { hranges, sranges };
	int channels[] = { 0, 1 };

	// Vector results of histogram calculation for all frames
	vector<MatND> histos;
	for each (auto frame in frames)
	{
		MatND hist;
		calcHist(&frame, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
		normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());
		histos.push_back(hist);
	}

	int nbFrames = histos.size();
	int compare_method = 0;			// CV_COMP_CORREL - Correlation
	float histo_threshold = 0.2f;	// Threshold for Correlation method
	float noise_threshold = 0.3f;	// Noises expected at < 30% number of frames
	int index = 0;
	vector<int> index_noises;		// Indexes of noise frames
									// Loop to find a source (not noise) image and find all the noises
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

	// Cannot find a source image => algo failed => quit
	if (index >= nbFrames)
		return 0;

	// Remove all noise frames and their histograms
	vector<Mat> filter_frames;
	vector<MatND> filter_histos;
	index = 0;

	// if no noise file add -1 to the array to avoid vector error
	if (index_noises.size() == 0)
		index_noises.push_back(-1);

	for (int i = 0; i < nbFrames; i++)
	{
		if (i != index_noises[index])
		{
			filter_frames.push_back(frames[i]);
			filter_histos.push_back(histos[i]);
		}
		else if (index < index_noises.size() - 1)
			index++;
	}
	//save_frames(filter_frames, "./frames/filter/Frame", true);
	nbFrames = filter_frames.size();

	vector<vector<pair<double, int>>> avg_distances = compare_FLANN_matcher(filter_frames);
	double traceback_threshold = 0.2f;
	
	vector<bool> indexes_check(nbFrames, true);
	indexes_check[0] = false;
	vector<int> indexes_first, indexes_reverse;
	

	waitKey();
	system("pause");
	return 0;
}