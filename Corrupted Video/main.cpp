#include "stdafx.h"

#include "corrupted_video.h"

int main()
{
	//--- Extract frames
	string videoname;
	cout << "Enter video name : ";
	cin >> videoname;
	//string videoname = "corrupted_video.mp4";
	string input_folder = "./frames/Frame";
	vector<Mat> frames;
	bool extract_from_video = true;
	bool extract_hsv = true;
	if (extract_from_video)	// Extract frames from original video
		frames = extract_frames(videoname, extract_hsv);
	else					// Extract frames from an images folder
		frames = extract_frames_folder(input_folder, true);

	if (frames.size() == 0)
		return 0;

	//--- PART 1: FILTERING NOISES
	//--- Remove all noise frames and their histograms
	//vector<Mat> temp_frames = filter_OpticalFlow(frames);
	vector<Mat> filter_frames = filter_Histogram(frames);
	vector<MatND> filter_histos = calculate_Histogram(filter_frames);
	//save_frames(filter_frames, "./frames/filter/Frame", true);

	//--- PART 2: SORTING FRAMES - PREMILINARY
	int nbFrames = filter_frames.size();

	// Histogram comparisons matrix frame by frame (NxN)
	vector<vector<pair<double, int>>> compare_histos = compare_histograms(filter_frames, filter_histos);
	double traceback_threshold = 0.9f;
	bool compare_greater = true;

	//--- Test sorting method with FLANN matcher comparison
	// vector<vector<pair<double, int>>> avg_matches = compare_FLANN_matcher(filter_frames);
	// double traceback_threshold = 0.2f;
	// bool compare_greater = false;

	//--- Sorting using histograms matrix
	// Indexes to mark which frame already sorted
	// true: frame available - false: frame taken
	vector<bool> indexes_check(nbFrames, true);
	indexes_check[0] = false;
	vector<int> indexes_first, indexes_reverse;

	// Take frame 0 arbitrarily and trace nearest frames to the 'left'
	indexes_first = trace_frames(indexes_check, compare_histos, traceback_threshold, compare_greater);
	// Trace nearest frames to the 'right'
	indexes_reverse = trace_frames(indexes_check, compare_histos, traceback_threshold, compare_greater);

	// Join sorted indexes (index 0 in the middle of indexes_first and reverse)
	reverse(indexes_first.begin(), indexes_first.end());
	indexes_first.push_back(0);
	for each(auto ind in indexes_reverse)
		indexes_first.push_back(ind);

	// Find not sorted indexes and arbitrarily adding them to their highest histogram matchings
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

	// Join frames basing on their sorted indexes
	vector<Mat> arranged_frames;
	cout << "SORTED FRAMES ORDER : ";
	for each (auto ind in indexes_first)
	{
		cout << ind << " - ";
		cvtColor(filter_frames[ind], filter_frames[ind], CV_HSV2BGR);
		arranged_frames.push_back(filter_frames[ind]);
	}
	cout << endl;

	//--- PART 3: LOCAL REFINING TO SORT FRAMES IN EXACT ORDER OF MOVEMENT

	//rearrange_local_OpticalFlow(arranged_frames, 11, 5, indexes_first);
	//rearrange_local_FLANN(arranged_frames, 0, 6, indexes_first);

	//--- Refining using otpical flow method
	// Not working as intended (yet)
	int size_local = 5;
	for (int i = 0; i < 35 - size_local; i++)
		rearrange_local_OpticalFlow(arranged_frames, i, size_local, indexes_first);

	/*char next;
	do
	{
	rearrange(arranged_frames, indexes_first);
	cout << "NEXT ? ";
	cin >> next;
	} while (next == 'y');
	*/

	//--- PART 4: OUTPUT
	for (int i = 0; i < nbFrames; i++)
		putText(arranged_frames[i], to_string(indexes_first[i]), cvPoint(200, 200), FONT_HERSHEY_COMPLEX, 5, white, 5, CV_AA);
	//save_frames_with_index(arranged_frames, indexes_first, "./frames/arranged/", false);

	// Save output video only works in Release
	save_output_video(arranged_frames, "output_original.avi", 25, arranged_frames[0].size(), false);
	save_output_video(arranged_frames, "output_test_1FPS.avi", 1, arranged_frames[0].size(), false);
	// Output with reversed order since we don't know the exact flow of original video
	reverse(arranged_frames.begin(), arranged_frames.end());
	save_output_video(arranged_frames, "output_reversed.avi", 25, arranged_frames[0].size(), false);
	save_output_video(arranged_frames, "output_test_1FPS_reversed.avi", 1, arranged_frames[0].size(), false);

	system("pause");
	return 0;
}