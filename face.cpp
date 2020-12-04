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
			if(y > 0 && y < frame.rows - 1 && x > 0 && x < frame.cols - 1) {
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
			} else {
				mag.at<uchar>(y, x) = 0;
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

vector<array<int, 3>> houghCircle(cv::Mat edges, cv::Mat gradient, int minRad, int maxRad, unsigned char peakThreshold, int minDistance) {
	cv::Mat houghSpace(3, new int[3]{gradient.rows, gradient.cols, maxRad - minRad + 1}, CV_32SC1, cv::Scalar::all(0)); // need +1?

	int maxPeak = 0;
	for(int gY = 5; gY < gradient.rows - 5; gY++) {
		for(int gX = 5; gX < gradient.cols - 5; gX++) {
			if(edges.at<uchar>(gY, gX) == 255) {
				float theta = gradient.at<uchar>(gY, gX) * (360.0f/255.0f);
				for(int r = minRad; r <= maxRad; r++) {

					float a0 = gX + r * cos(theta * CV_PI / 180.0f);
					float b0 = gY + r * sin(theta * CV_PI / 180.0f);
					float a1 = gX - r * cos(theta * CV_PI / 180.0f);
					float b1 = gY - r * sin(theta * CV_PI / 180.0f);
					// cout << a0 << " " << b0 << endl;
					if((b0 >= 0 && b0 < gradient.rows) && (a0 >= 0 && a0 < gradient.cols)) {
						houghSpace.at<signed int>(b0, a0, r - minRad) += 1;
						if(houghSpace.at<signed int>(b0, a0, r - minRad) > maxPeak) maxPeak = houghSpace.at<signed int>(b0, a0, r - minRad);
						// cout << b0 << " " << a0 << endl;
					}
					if((b1 >= 0 && b1 < gradient.rows) && (a1 >= 0 && a1 < gradient.cols)) {
						houghSpace.at<signed int>(b1, a1, r - minRad) += 1;
						if(houghSpace.at<signed int>(b1, a1, r - minRad) > maxPeak) maxPeak = houghSpace.at<signed int>(b1, a1, r - minRad);
					}
				}
			}
		}
	}
	//normalize and find peaks
	vector<array<int, 3>> peaks;
	for(int b = 0; b < gradient.rows; b++) {
		for(int a = 0; a < gradient.cols; a++) {
			for(int r = minRad; r < maxRad; r++) {
				//normalize
				houghSpace.at<signed int>(b, a, r - minRad) *= 255.0f / (float)maxPeak;
				//find peak
				if(houghSpace.at<signed int>(b, a, r - minRad) > peakThreshold) peaks.push_back(array<int, 3>{b, a, r});
			}
		}
	}

	return peaks;
}

vector<array<int, 2>> houghLines(cv::Mat edges, unsigned char peakThreshold) {
	cv::Mat houghSpace(2, new int[2]{max(edges.rows, edges.cols), 360}, CV_32SC1, cv::Scalar::all(0)); // need +1?

	int maxPeak = 0;
	for(int gY = 5; gY < edges.rows - 5; gY++) {
		for(int gX = 5; gX < edges.cols - 5; gX++) {
			if(edges.at<uchar>(gY, gX) == 255) {
				// for(int row = 0; row < edges.rows; row++) {
				// 	for(int theta = 0; theta < edges.cols; theta++) {

				// 	}
				// }
				for(int t = 0; t < 180; t++) {
					float theta = t;
					float rho = gX*cos(theta * CV_PI / 180.0f) + gY*sin(theta * CV_PI/180.0f);
					if( rho < 0) {
						rho *= -1;
						theta = 180 + theta;
					}
					// cout << rho << " " << theta << endl;
					houghSpace.at<signed int>((int)floor(rho), (int)round(theta)) += 1;
					if(houghSpace.at<signed int>(rho, theta) > maxPeak) maxPeak = houghSpace.at<signed int>(rho, theta);
					// cout << rho << " " << theta << endl;
				}
			}
		}	
	}
	vector<array<int, 2>> lines;
	for(int r = 0; r < houghSpace.rows; r++) {
		for(int t = 0; t < houghSpace.cols; t++) {
			houghSpace.at<signed int>(r, t) *= 255.0f / (float)maxPeak;
			if(houghSpace.at<signed int>(r, t) > peakThreshold) lines.push_back(array<int, 2>{r, t});
		}
	}
	return lines;
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
		// equalizeHist( frame_gray, frame_gray );
		//TRY NORMALISING HISTOGRAM FIRST!
		cv::Mat gaussianFrame = gaussian(frame_gray);
		cv::Mat gradientFrame = frame_gray.clone();
		cv::Mat sobelFrame = sobel(gaussianFrame, &gradientFrame);
		cv::Mat threshold = thresholdFilter(sobelFrame, 200, 255);

		// vector<array<int, 3>> circles = houghCircle(threshold, gradientFrame, 0, 500, 160, 0);
		// for(int i = 0; i < circles.size(); i++) {
		// 	cv::circle(frame, cv::Point(circles[i][1], circles[i][0]), circles[i][2], cv::Scalar(255, 0, 0), 5);
		// }
		vector<array<int, 2>> lines = houghLines(threshold, 200);
		// for(int i = 0; i < lines.size(); i++) {
		// 	// cout << lines[i][0] << " " << lines[i][1] << endl;
		// }
		cv::imwrite("edges.jpg", threshold);


		// 2. Load the Strong Classifier in a structure called `Cascade'
		if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
		detectAndDisplay(frame, argv[2], argv[1]);
		cv::imwrite( "detected.jpg", frame );
		cout << "here" << endl;
	}

	// 3. Detect Faces and Display Result

	// 4. Save Result Image

	return 0;
}
