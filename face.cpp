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
using namespace cv;


/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;


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

vector<Rect> readGroundTruth(string path) {
	ifstream file(path, ifstream::in);
	string line;
	
	vector<Rect> truths;
	getline(file, line);
	while(!file.eof()) {
		vector<string> tokens = str_split(line, ' ');
		truths.push_back(Rect(stoi(tokens[0]), stoi(tokens[1]), stoi(tokens[2]), stoi(tokens[3])));
		getline(file, line);
	}
	file.close();

	return truths;
}

float calcIOU(vector<Rect> faces, vector<Rect> truths) {
	float intersect = 0;
	float area = 0;
	for(int face = 0; face < faces.size(); face++) {
		for(int truth = 0; truth < truths.size(); truth++) {
			intersect += (faces[face] & truths[truth]).area();
			area += faces[face].area() + truths[truth].area();
		}
	}
	int rUnion = area - intersect;
	return (intersect / rUnion);
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame, string truthsPath)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

    // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

	//Draw ground truth
	vector<Rect> truths = readGroundTruth(truthsPath);
	for(int i = 0; i < truths.size(); i++) {
		rectangle(frame, Point(truths[i].x, truths[i].y), Point(truths[i].x + truths[i].width, truths[i].y + truths[i].height), Scalar(0, 0, 255), 2);
	}
	cout << "IOU = " << calcIOU(faces, truths) << endl;

    // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

}

/** @function main */
int main( int argc, const char** argv )
{
    // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay(frame, argv[2]);

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

