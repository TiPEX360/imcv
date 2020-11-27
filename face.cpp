/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>

using namespace std;


/** Global variables */
cv::String cascade_name = "frontalface.xml";
cv::CascadeClassifier cascade;


std::vector<std::string> str_split(const std::string &line, char delimiter) {
	string haystack = line;
	vector<std::string> tokens;
	size_t pos;
	while ((pos = haystack.find(delimiter)) != std::string::npos) {
		tokens.push_back(haystack.substr(0, pos));
		haystack.erase(0, pos + 1);
	}
	// Push the remaining chars onto the vector
	tokens.push_back(haystack);
	return tokens;
}

vector<cv::Rect> readGroundTruth(string path) {
	ifstream file(path, ifstream::in);
	string line;
	
	vector<cv::Rect> truths;
	getline(file, line);
	while(!file.eof()) {
		vector<string> tokens = str_split(line, ' ');
		truths.push_back(cv::Rect(stoi(tokens[0]), stoi(tokens[1]), stoi(tokens[2]), stoi(tokens[3])));
		getline(file, line);
	}
	file.close();

	return truths;
}

float calcIOU(cv::Rect detected, vector<cv::Rect> truths) {
	float intersect = 0;
	float area = 0;
	float rectUnion = 0;
	float iou = 0;
	for(int truth = 0; truth < truths.size(); truth++) {
		if((detected & truths[truth]).area() > 0) {
			intersect += (detected & truths[truth]).area();
			area += detected.area() + truths[truth].area();
		}
	}

	if(area > 0) {
		rectUnion = area - intersect;
		iou = intersect / rectUnion;
	}
	return iou;
}

/** @function detectAndDisplay */
void detectAndDisplay(cv::Mat frame, string truthsPath)
{
	std::vector<cv::Rect> detected;
	cv::Mat frame_gray;
	float threshold = 0.3f;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, detected, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, cv::Size(50, 50), cv::Size(500,500) );

    // 3. Print number of Faces found
	std::cout << detected.size() << std::endl;

	//Draw ground truth
	vector<cv::Rect> truths = readGroundTruth(truthsPath);
	for(int i = 0; i < truths.size(); i++) {
		rectangle(frame, cv::Point(truths[i].x, truths[i].y), cv::Point(truths[i].x + truths[i].width, truths[i].y + truths[i].height), cv::Scalar(0, 0, 255), 2);
	}

	int positives = truths.size();
	int noDetected = detected.size();
	
	int truePositives = 0;
	int falsePositives = noDetected - truePositives;
	int trueNegatives = 0;
	int falseNegatives = 0;
	for(int i = 0; i < detected.size(); i++) {
		float iou = calcIOU(detected[i], truths);
		if(iou > threshold) truePositives++;
	}
	falseNegatives = positives - truePositives;

	cout  << "True Positives: " << truePositives << endl << "False Positives: " << noDetected - truePositives << endl;
	
	cout << "False Negatives: " << falseNegatives << endl;
	float tpr = (float)truePositives / (float)(truePositives + falseNegatives);
	cout << "TPR = " << tpr << endl;
	float f1 = (float)truePositives / (float)(truePositives + 0.5f*(float)(falsePositives+falseNegatives));
	cout << "f1-score = " << f1 << endl;

	// 4. Draw box around faces found
	for( int i = 0; i < detected.size(); i++ )
	{
		rectangle(frame, cv::Point(detected[i].x, detected[i].y), cv::Point(detected[i].x + detected[i].width, detected[i].y + detected[i].height), cv::Scalar( 0, 255, 0 ), 2);
	}

}

/** @function main */
int main( int argc, const char** argv )
{
    // 1. Read Input Image
	cv::Mat frame = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay(frame, argv[2]);

	// 4. Save Result Image
	cv::imwrite( "detected.jpg", frame );

	return 0;
}

