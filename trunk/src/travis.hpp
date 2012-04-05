#ifndef TRAVIS_H
#define TRAVIS_H

	/**
    * \defgroup Travis
    * Travis (TRAcking VISion library)
	*
	* Author: Santiago Morante
	*
	* Copyright (c) 2012 Universidad Carlos III de Madrid (http://www.uc3m.es)
	*
	* License: LGPL v3 or later. License available at GNU (http://www.gnu.org/licenses/lgpl.html)
	*
	* For more license information see attached files: LicenseOpencv.txt and LicenseTravis.txt
	*/

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace cv;

  /**
	* @ingroup Travis
    * This function detects moving areas on live video.
    */

void OptFlowDetec();

  /**
	* @ingroup Travis
    * Reactive navigation by dinamic Otsu thresholding method. 
    */

void NaviOtsu();

  /**
	* @ingroup Travis
    * Navigation by checking color floor and finding similar on 3 directions. 
    */

void NaviColor();

  /**
	* @ingroup Travis
    * This function contains 3 methods to follow objects by color.
    * @param option is the algorithm selector (1 = Camshift,2 = Meanshift,3 = Colormax).
    * @param accuracy is the required precission (1 = Low, 2 = Medium, 3 = High).
    * @param selection is the rectangle that contains colors to find.
    * @param selectedImage is the image to select color to track.
    *  
    */

void SearchColor(int option, int accuracy ,Rect selection, IplImage* selectedImage);

  /**
	* @ingroup Travis
    * It finds similar images using a template.
    * @param accuracy is the required precission (1 = Low, 2 = Medium, 3 = High).
    * @param selection is the rectangle that contains small image to find.
    * @param selectedImage is the image to select small image to track.
    *  
    */

void TemplateMatching(int accuracy ,Rect selection, IplImage* selectedImage);

  /**
	* 
    * It changes matches vector to points vector 
    * matches is the vector of matches.
    * kpts_train is the vector of keypoints of selection.
    * kpts_query is the vector of keypoints of video.
    * pts_train is the vector of matched points of selection.
    * pts_query is the vector of matched points of video.
    *  
    */

void matches2points(const vector<vector<DMatch> >& matches, const vector<KeyPoint>& kpts_train,
                    const vector<KeyPoint>& kpts_query, vector<Point2f>& pts_train,
                    vector<Point2f>& pts_query);

  /**
	* @ingroup Travis
    * Object recognition by using SURF algorithm. It also indicates the most probably centroid.
    * @param accuracy is the required precission (1 = Low, 2 = Medium, 3 = High).
    * @param selectedImage is the template image to be found.
    *  
    */

void SurfCentroid(int accuracy, IplImage* selectedImage);

  /**
	* @ingroup Travis
    * Object recognition by using SURF algorithm. It also calculates the homography.
    * @param accuracy is the required precission (1 = Low, 2 = Medium, 3 = High).
    * @param selectedImage is the template image to be found.
    *  
    */

void SurfHomography(int accuracy, IplImage* selectedImage);

#endif // TRAVIS_H
