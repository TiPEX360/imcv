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
cv::String cascade_name = "dartcascade/cascade.xml";
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

vector<cv::Rect> readGroundTruths(string truthPath, string imgPath) {
	ifstream file(truthPath, ifstream::in);
	string line;
	
	vector<cv::Rect> truths;
	getline(file, line);
	while(!file.eof()) {
		vector<string> tokens = str_split(line, ',');
		if(tokens[5].compare(imgPath) == 0) {
			truths.push_back(cv::Rect(stoi(tokens[1]), stoi(tokens[2]), stoi(tokens[3]), stoi(tokens[4])));
		}
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
void detectAndDisplay(cv::Mat frame, string truthsPath, string imgPath)
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
	vector<cv::Rect> truths = readGroundTruths(truthsPath, imgPath);
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

cv::Mat gaussian(cv::Mat frame) {

	cv::Mat result = frame.clone();
	cv::Mat kernel(3, 3, CV_8SC1, new signed char[9]{1, 2, 1, 2, 4, 2, 1, 2, 1});
	for(int y = 0; y < frame.rows; y++) {
		for(int x = 0; x < frame.cols; x++) {
			// Apply kernel
			unsigned int sum = 0;
			for(int j = -1; j < 2; j++) {
				for(int i = -1; i < 2; i++) {
					int kernelX, kernelY;
					if((x + i) < 0) kernelX = frame.cols - 1;
					else if((x + i) == frame.cols) kernelX = 0;
					else kernelX = x + i;
					if((y + j) < 0) kernelY = frame.rows - 1;
					else if((y + j) == frame.rows) kernelY = 0;
					else kernelY = y + j;

					unsigned char cell = frame.at<uchar>(kernelY, kernelX);//0 - 255
					sum += (cell * kernel.at<schar>(j + 1, i + 1))/16;//-512 - 512
				}
			}
			result.at<uchar>(y, x) = sum;
		}
	}
	return result;
}

cv::Mat sobel(cv::Mat frame, cv::Mat* gradient) {

	cv::Mat mag = frame.clone();
	cv::Mat sobelX(3, 3, CV_8SC1, new signed char[9]{-1, 0, 1, -2, 0, 2, -1, 0, 1});
	cv::Mat sobelY(3, 3, CV_8SC1, new signed char[9]{-1, -2, -1, 0, 0, 0, 1, 2, 1});
	for(int y = 0; y < frame.rows; y++) {
		for(int x = 0; x < frame.cols; x++) {
			// Apply kernel
			signed int deltaY = 0;
			signed int deltaX = 0;
			for(int j = -1; j < 2; j++) {
				for(int i = -1; i < 2; i++) {
					int kernelX, kernelY;
					if((x + i) < 0) kernelX = frame.cols - 1;
					else if((x + i) == frame.cols) kernelX = 0;
					else kernelX = x + i;
					if((y + j) < 0) kernelY = frame.rows - 1;
					else if((y + j) == frame.rows) kernelY = 0;
					else kernelY = y + j;

					unsigned char cell = frame.at<uchar>(kernelY, kernelX);//0 - 255
					deltaY += cell * sobelY.at<schar>(j + 1, i + 1);//-512 - 512
					deltaX += cell * sobelX.at<schar>(j + 1, i + 1);//-512 - 512
				}
			}
			uchar magnitude = sqrt(pow(deltaY, 2) + pow(deltaX, 2));
			float w = cvFastArctan(deltaY, deltaX) * (255.0f/360.0f);
			mag.at<uchar>(y, x) = magnitude;
			if(gradient != NULL) {
				gradient->at<uchar>(y, x) = w;
			}
		}
	}
	return mag;
}

cv::Mat thresholdFilter(cv::Mat frame, unsigned char low, unsigned char high) {
	cv::Mat frameCpy = frame.clone();
	for(int y = 0; y < frame.rows; y++) {
		for(int x = 0; x < frame.cols; x++) {
			if(frame.at<uchar>(y, x) < low || frame.at<uchar>(y, x) > high) frameCpy.at<uchar>(y, x) = 0;
			else frameCpy.at<uchar>(y, x) = 255; 
		}
	}
	return frameCpy;
}

cv::Mat houghCircle(cv::Mat gradient, int minRad, int maxRad, unsigned char peakThreshold, int minDistance) {
	cv::Mat houghSpace(gradient.rows, gradient.cols, CV_)
}

/** @function main */
int main( int argc, const char** argv )
{
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	string argv1 = argv[1];		//Run on all images
	if(argv1.compare("-A") == 0) {
		for(int i = 0; i < 16; i++) {
			string outputPath = "./detected/dart" + to_string(i) + ".jpg";
			string imgPath = "dart" + to_string(i) + ".jpg";
			cv::Mat frame = cv::imread(imgPath, CV_LOAD_IMAGE_COLOR);
			detectAndDisplay(frame, argv[2], imgPath);
			cout << outputPath << endl;
			cv::imwrite(outputPath, frame);
		}
	} 
	else {
		// 1. Read Input Image
		cv::Mat frame = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
		cv::Mat frame_gray;
		cv::cvtColor( frame, frame_gray, CV_BGR2GRAY );
		cv::Mat gaussianFrame = gaussian(frame_gray);
		cv::Mat gradientFrame = frame_gray.clone();
		cv::Mat sobelFrame = sobel(gaussianFrame, &gradientFrame);
		cv::Mat threshold = thresholdFilter(sobelFrame, 100, 255);
		cv::imwrite("edges.jpg", threshold);

		// 2. Load the Strong Classifier in a structure called `Cascade'
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
		detectAndDisplay(frame, argv[2], argv[1]);
		cv::imwrite( "detected.jpg", frame );
	}

	// 3. Detect Faces and Display Result

	// 4. Save Result Image

	return 0;
}
