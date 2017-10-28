#include "stdafx.h"

#include "corrupted_video.h"

struct greater_sort
{
	template<class T>
	bool operator()(T const &a, T const &b) const { return a > b; }
};

int main()
{
	// Extract frames from original video
	string videoname = "corrupted_video.mp4";
	vector<Mat> frames = extract_frames(videoname, true);
	//save_output_video(frames, "output.avi", 30, frames[0].size(), true);
	//save_frames(frames, "./frames/Frame", true);

	// Parameters for histograms HSV in 2D
	// Quantize the hue to 50 levels and the saturation to 60 levels
	int h_bins = 50, s_bins = 60;
	int histSize[] = { h_bins, s_bins };
	// hue varies from 0 to 179
	float hranges[] = { 0, 180 };
	// saturation varies from 0 (black-gray-white) to 255
	float sranges[] = { 0, 256 };
	const float* ranges[] = { hranges, sranges };
	// Compute the histogram from the 0-th and 1-st channels
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
	bool found_src = false;			
	int index = 0;
	vector<int> index_noises;		// Indexes of noise frames
	// Loop to find a source (not noise) image and find all the noises
	while (!found_src && index < nbFrames)
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
			found_src = true;
		else
			index++;
	}

	// Cannot find a source image => algo failed => quit
	if (index >= nbFrames)
		return 0;

	// Remove all noise frames and their histograms
	vector<Mat> filter_frames;
	vector<MatND> filter_histos;
	for (int i = 0; i < nbFrames; i++)
	{
		if (i != index_noises[index])
		{
			filter_frames.push_back(frames[i]);
			filter_histos.push_back(histos[i]);
		}
		else
			index++;
	}
	//save_frames(filter_frames, "./frames/filter/Frame", true);

	nbFrames -= index_noises.size();		// new number of frames
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

	vector<Mat> arranged_frames;
	for each (auto ind in indexes_reverse)
		arranged_frames.push_back(filter_frames[ind]);
	arranged_frames.push_back(filter_frames[0]);
	for each (auto ind in indexes_first)
		arranged_frames.push_back(filter_frames[ind]);

	save_output_video(arranged_frames, "fixed_output.avi", 30, arranged_frames[0].size(), true);	
	reverse(arranged_frames.begin(), arranged_frames.end());
	save_output_video(arranged_frames, "reverse_fixed_output.avi", 30, arranged_frames[0].size(), true);
	//// Trace frames until the end on one side
	//while (!finished_traceback)
	//{
	//	int j = 0;
	//	bool found_new_index = false;
	//	while (!found_new_index && j < nbFrames)
	//	{
	//		double value = compare_histos[current_index][j].first;
	//		int ind = compare_histos[current_index][j].second;
	//		if (indexes_check[ind] == true && value > traceback_threshold)
	//		{
	//			current_index = ind;
	//			indexes_check[current_index] = false;
	//			indexes_first_order.push_back(current_index);
	//			found_new_index = true;
	//		}
	//		else if (value > traceback_threshold)
	//			j++;
	//		else
	//		{
	//			j = nbFrames;
	//			finished_traceback = true;
	//		}
	//	}
	//}

	
	// Trace frames until the end on the other side
	/*while (!finished_traceback)
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
				indexes_second_order.push_back(current_index);
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
	}*/
	
	
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