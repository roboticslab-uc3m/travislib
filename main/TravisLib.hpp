// -*- mode:C++; tab-width:4; c-basic-offset:4; indent-tabs-mode:nil -*-

#ifndef __TRAVIS_LIB__
#define __TRAVIS_LIB__

/**
 * 
 * @ingroup travis_libraries
 * \defgroup TravisLib TravisLib
 *
 * @brief The TravisLib library provides basic 2D image feature extraction auxiliary
 * functions for the \ref segEx module and others.
 * 
 * <hr>
 *
 * This file can be edited at $TRAVIS_ROOT/main/TravisLib.hpp
 *
 */

#include <stdio.h>  // just printf

//#include "highgui.h" // to show windows
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

/**
 * @ingroup Travis
 *
 * The Travis class implements all the algorithms on a single image.
 * 
 */
class Travis {
protected:
    bool isQuiet;

public:

    /**
     * Constructor.
     */
    Travis() {}

    /**
     * Configure the object and make it connect to a colorXxx module by port name.
     * @param quiet suppress messages displayed upon success/failure.
     * @return true if the object and connection was created successfully.
     */
    bool open(const cv::Mat& image, bool quiet=true);
};

/**
 * @ingroup TravisLib
 * This function gets the biggest contour.
 */
vector <Point> getBiggestContour(const Mat image);

/**
 * @ingroup TravisLib
 * This function calculates X and Y.
 */
void calcLocationXY(float& locX, float& locY, const vector <Point> biggestCont);

/**
 * @ingroup TravisLib
 * This function calculates the mask.
 */
void calcMask(Mat& mask, const vector <Point> biggestCont);

/**
 * @ingroup TravisLib
 * This function calculates the area.
 */
void calcArea(float& area, const vector <Point> biggestCont);

/**
 * @ingroup TravisLib
 * This function calculates the rectangularity.
 */
void calcRectangularity(float& rectangularity, const vector <Point> biggestCont);

/**
 * @ingroup TravisLib
 * This function calculates the mass center.
 */
void calcMassCenter(float& massCenterLocX, float& massCenterLocY , const vector <Point> biggestCont);

/**
 * @ingroup TravisLib
 * This function calculates the aspect ratio.
 */
void calcAspectRatio(float& aspectRatio, float& axisFirst, float& axisSecond ,const vector <Point> biggestCont);

/**
 * @ingroup TravisLib
 * This function calculates the solidity.
 */
void calcSolidity(float& solidity, const vector <Point> biggestCont);

/**
 * @ingroup TravisLib
 * This function calculates the HSV mean and std deviation.
 */
void calcHSVMeanStdDev(const Mat image, const Mat mask, float& hue_mean, float& hue_stddev,
                       float& saturation_mean, float& saturation_stddev,
                       float& value_mean, float& value_stddev);

/**
 * @ingroup TravisLib
 * This function calculates the HSV peak color.
 */
void calcHSVPeakColor(const Mat image, const Mat mask, float& hue_mode, float& hue_peak,
                       float& value_mode, float& value_peak);

/**
 * @ingroup TravisLib
 * This function calculates the moments.
 */
void calcMoments(Mat& theHuMoments, const vector <Point> biggestCont );

/**
 * @ingroup TravisLib
 * This function calculates the arc length.
 */
void calcArcLength(float& arc, const vector <Point> biggestCont );

/**
 * @ingroup TravisLib
 * This function calculates the circle.
 */
void calcCircle(float& radius, const vector <Point> biggestCont );

#endif  // __TRAVIS_LIB __

