// mywish.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
int H_MIN = 0;
int H_MAX = 255;
int S_MIN = 0;
int S_MAX = 255;
int V_MIN = 0;
int V_MAX = 255;
int thresh = 40;
const string trackbarWindowName = "Trackbars";
void createTrackbars(){


	namedWindow(trackbarWindowName, 0);
	char TrackbarName[50];
	printf(TrackbarName, "H_MIN", H_MIN);
	printf(TrackbarName, "H_MAX", H_MAX);
	printf(TrackbarName, "S_MIN", S_MIN);
	printf(TrackbarName, "S_MAX", S_MAX);
	printf(TrackbarName, "V_MIN", V_MIN);
	printf(TrackbarName, "V_MAX", V_MAX);
	createTrackbar("H_MIN", trackbarWindowName, &H_MIN, H_MAX, NULL);
	createTrackbar("H_MAX", trackbarWindowName, &H_MAX, H_MAX, NULL);
	createTrackbar("S_MIN", trackbarWindowName, &S_MIN, S_MAX, NULL);
	createTrackbar("S_MAX", trackbarWindowName, &S_MAX, S_MAX, NULL);
	createTrackbar("V_MIN", trackbarWindowName, &V_MIN, V_MAX, NULL);
	createTrackbar("V_MAX", trackbarWindowName, &V_MAX, V_MAX, NULL);
	createTrackbar("THRESHOLD", trackbarWindowName, &thresh, 255, NULL);											//if thresh=100
}
int main(int argc, _TCHAR* argv[])
{
	Mat img, gray,HSV,resources,img1,canny_output,towncentre,obstacles;
	createTrackbars();
	while (1)
	{

		img = imread("C:/Users/SoumyaGourab/Documents/Visual Studio 2013/Projects/mywish/14.jpg");
		if (img.empty())
		{
			cout << "The image could not be loaded";
			return -1;
		}
		
		//imshow("Raw", img);
		
		//IplImage* gray = cvCreateImage(cvGetSize(img), 8, 1);
		cvtColor(img, gray, CV_BGR2GRAY);
		cvtColor(img, HSV, COLOR_BGR2HSV);
		Canny(gray, canny_output, thresh, thresh * 2, 3);
		imshow("Canny", canny_output);
		inRange(HSV, Scalar(30, 241, 0), Scalar(30, 255, 205), resources);
		inRange(HSV, Scalar(10, 0, 0), Scalar(16, 255, 255), towncentre);
		inRange(HSV, Scalar(89, 241, 185), Scalar(93,255, 201), obstacles);
		//threshold(gray, gray, thresh, 255, THRESH_BINARY);
		imshow("gray", resources);
		imshow("gray2", towncentre);
		imshow("gray3", obstacles);

		vector< vector<Point> > contours;
		vector< vector<Point> > entry;
		vector< vector<Point> > water;
		vector<Vec4i> hierarchy;
		findContours(resources, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		findContours(towncentre, entry, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		findContours(obstacles, water, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		cout << contours.size();
		cout << entry.size();
		cout << water.size();
		vector<Moments> m_food(contours.size());
		vector<Moments> m_ckpt(entry.size());
		vector<Moments> m_water(water.size());
		
		for (int i = 0; i < contours.size(); i++)
		{
			m_food[i] = moments(contours[i], false);
		}
		for (int i = 0; i < entry.size(); i++)
		{
			m_ckpt[i] = moments(entry[i], false);
		}
		for (int i = 0; i < water.size(); i++)
		{
			m_water[i] = moments(water[i], false);
		}

		vector<Point2f> mc_food(contours.size());
		vector<Point2f> mc_ckpt(entry.size());
		vector<Point2f> mc_water(water.size());
		
		for (int i = 0; i < contours.size(); i++)
		{
			mc_food[i] = Point2f(m_food[i].m10 / m_food[i].m00, m_food[i].m01 / m_food[i].m00);
			circle(img, mc_food[i], 3, Scalar(0,255,0));
				
			//cout << mc_food[i] << endl;
		}
		for (int i = 0; i < entry.size(); i++)
		{
			mc_ckpt[i] = Point2f(m_ckpt[i].m10 / m_ckpt[i].m00, m_ckpt[i].m01 / m_ckpt[i].m00);
			circle(img, mc_ckpt[i], 15, Scalar(255, 255, 255));

			//cout << mc_food[i] << endl;
		}
		for (int i = 0; i < water.size(); i++)
		{
			mc_water[i] = Point2f(m_water[i].m10 / m_water[i].m00, m_water[i].m01 / m_water[i].m00);
			circle(img, mc_water[i], 5, Scalar(0, 0, 0));

			//cout << mc_food[i] << endl;
		}

		Mat draw_tri = Mat::zeros(resources.size(), CV_8UC3);
		Mat draw_sqr = Mat::zeros(resources.size(), CV_8UC3);
		vector<Point> shape;
		vector<Point2f> mc_tri(contours.size());
		vector<Point2f> mc_sqr(contours.size());
		int j = 0,k = 0;
		
		for (int i = 0; i< contours.size(); i++)
		{
			Scalar color = Scalar(255, 255, 255);
			approxPolyDP(contours[i], shape, arcLength(Mat(contours[i]), true)*0.05, true);
			if (shape.size() == 3)
			{
				drawContours(draw_tri, contours, i, color, CV_FILLED);
				mc_tri[j] = Point2f(m_food[i].m10 / m_food[i].m00, m_food[i].m01 / m_food[i].m00);
				circle(img, mc_tri[j++], 10, Scalar(0, 0, 255));
			}
			else if (shape.size() == 4)
			{
				drawContours(draw_sqr, contours, i, color, CV_FILLED);
				mc_sqr[k] = Point2f(m_food[i].m10 / m_food[i].m00, m_food[i].m01 / m_food[i].m00);
				circle(img, mc_sqr[k++], 10, Scalar(255, 0, 0));
			}
		}

		imshow("Contours_tri", draw_tri);
		imshow("Contours_sqr", draw_sqr);
		
		if (waitKey(10) == 27)
		{
			cout << endl;
			for (int i = 0; i < contours.size(); i++)
			{
				//cout << mc_food[i] << endl;
				cout << mc_tri[i] << endl;
				
			}
			for (int i = 0; i < contours.size(); i++)
			{
				//cout << mc_food[i] << endl;
				cout << mc_sqr[i] << endl;

			}
			for (int i = 0; i < entry.size(); i++)
			{
				cout << mc_ckpt[i] << endl;
			}
			for (int i = 0; i < water.size(); i++)
			{
				cout << mc_water[i] << endl;
			}
			break;
		}		
		imshow("Raw", img);
	}
	return 0;
}
