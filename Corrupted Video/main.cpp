#include "stdafx.h"

#include "corrupted_video.h"

struct greater_sort
{
	template<class T>
	bool operator()(T const &a, T const &b) const { return a > b; }
};

int main()
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
	compare_histos.resize(nbFrames);
	
	for (int i = 0; i < nbFrames; i++)
	{	
		double max = 0, second_max = 0;
		int ind_max = 0, ind_2nd_max = 0;
		for (int j = 0; j < nbFrames; j++)
		{
			compare_histos[i].push_back(make_pair(compareHist(filter_histos[i], filter_histos[j], compare_method),j));
			if (compare_histos[i][j].first == 1.0f)
				compare_histos[i][j].first = 0;
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
		cout << i << " | max = " << max << " - " << ind_max << " | 2nd max = " << second_max << " - " << ind_2nd_max << endl;
	}

	vector<bool> indexes_check(nbFrames, true);
	indexes_check[0] = false;
	vector<int> indexes_first, indexes_reverse;
	double traceback_threshold = 0.9f;

	indexes_first = trace_frames(indexes_check, compare_histos);
	indexes_reverse = trace_frames(indexes_check, compare_histos);
	reverse(indexes_reverse.begin(), indexes_reverse.end());
		
	for (int i = 0; i < nbFrames; i++)
	{
		if (indexes_check[i])
			cout << i << endl;
	}
	vector<Mat> arranged_frames;
	for each (auto ind in indexes_reverse)
	{
		if (ind == 24 || ind == 45 ||ind == 75)
			cout << " *** " << ind << " *** ";
		else
			cout << ind << " - ";
		putText(filter_frames[ind], to_string(ind), cvPoint(200, 200),
			FONT_HERSHEY_COMPLEX, 5, cvScalar(0, 0, 255), 5, CV_AA);
		arranged_frames.push_back(filter_frames[ind]);
		if (ind == 53)
		{
			cout << " *** 75 *** ";
			putText(filter_frames[75], "75", cvPoint(200, 200),
				FONT_HERSHEY_COMPLEX, 5, cvScalar(0, 0, 255), 5, CV_AA);
			arranged_frames.push_back(filter_frames[75]);
		}
	}
	arranged_frames.push_back(filter_frames[0]);
	cout << "*** 0 *** ";
	for each (auto ind in indexes_first)
	{
		if (ind == 24 || ind == 45 || ind == 75)
			cout << " *** " << ind << " *** ";
		else
			cout << ind << " - ";
		putText(filter_frames[ind], to_string(ind), cvPoint(200, 200),
			FONT_HERSHEY_COMPLEX, 5, cvScalar(0, 0, 255), 5, CV_AA);
		arranged_frames.push_back(filter_frames[ind]);
	}

	// Save output video only works in Release
	if (0)
	{
		save_output_video(arranged_frames, "output_original.avi", 1, arranged_frames[0].size(), true);
		reverse(arranged_frames.begin(), arranged_frames.end());
		save_output_video(arranged_frames, "output_reversed.avi", 1, arranged_frames[0].size(), true);
	}
	
	/*drawHist(frames[0], "Frame0");
	drawHist(frames[10], "Frame10");
	drawHist(frames[17], "Frame17");
	drawHist(frames[36], "Frame36");
	drawHist(frames[42], "Frame42");
	drawHist(frames[46], "Frame46");
	drawHist(frames[78], "Frame78");*/
	//drawHist(frames[], "Frame");
	
	waitKey();
	system("pause");
	return 0;
}