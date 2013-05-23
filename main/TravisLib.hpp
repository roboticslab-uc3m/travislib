// -*- mode:C++; tab-width:4; c-basic-offset:4; indent-tabs-mode:nil -*-

#ifndef __TRAVIS_LIB_HPP__
#define __TRAVIS_LIB_HPP__

/**
 * 
 * \defgroup travis_libraries TravisLib
 *
 * @brief The TravisLib library provides basic 2D image feature extraction auxiliary
 * functions for different modules.
 *
 * \defgroup travis_functions TravisFuncs
 *
 * @brief The TravisFuncs library provides basic 2D image feature extraction auxiliary
 * functions.
 *
 * <hr>
 *
 * This file can be edited at $TRAVIS_ROOT/TravisLib.hpp
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
 * @ingroup travis_libraries
 *
 * \defgroup Travis
 *
 * @brief The Travis class implements all the algorithms on a single image.
 *
 * The Travis class implements all the algorithms on a single image.
 * 
 */
class Travis {
protected:
    /** Store the verbosity level. */
    bool _quiet;

    /** Store the image in cv::Mat format. */
    cv::Mat _img;

public:

    /**
     * Travis class constructor.
     * @param quiet suppress messages displayed upon success/failure.
     */
    Travis(bool quiet=true) : _quiet(quiet) {}

    /**
     * Set the image in cv::Mat format.
     * @param image the image to set, in cv::Mat format.
     * @return true if the object was set successfully.
     */
    bool setCvMat(const cv::Mat& image);

    /**
     * Get the image in cv::Mat format.
     * @return the image, in cv::Mat format.
     */
    cv::Mat& getCvMat();

    /**
     * Binarize the image.
     * @param algorithm i.e. "redMinusGreen".
     * @param threshold i.e. 50.
     */
    void binarize(const char* algorithm, const double threshold);
};

/**
 * @ingroup travis_functions
 * This function gets the biggest contour.
 */
vector <Point> getBiggestContour(const Mat image);

/**
 * @ingroup travis_functions
 * This function calculates X and Y.
 */
void calcLocationXY(float& locX, float& locY, const vector <Point> biggestCont);

/**
 * @ingroup travis_functions
 * This function calculates the mask.
 */
void calcMask(Mat& mask, const vector <Point> biggestCont);

/**
 * @ingroup travis_functions
 * This function calculates the area.
 */
void calcArea(float& area, const vector <Point> biggestCont);

/**
 * @ingroup travis_functions
 * This function calculates the rectangularity.
 */
void calcRectangularity(float& rectangularity, const vector <Point> biggestCont);

/**
 * @ingroup travis_functions
 * This function calculates the mass center.
 */
void calcMassCenter(float& massCenterLocX, float& massCenterLocY , const vector <Point> biggestCont);

/**
 * @ingroup travis_functions
 * This function calculates the aspect ratio.
 */
void calcAspectRatio(float& aspectRatio, float& axisFirst, float& axisSecond ,const vector <Point> biggestCont);

/**
 * @ingroup travis_functions
 * This function calculates the solidity.
 */
void calcSolidity(float& solidity, const vector <Point> biggestCont);

/**
 * @ingroup travis_functions
 * This function calculates the HSV mean and std deviation.
 */
void calcHSVMeanStdDev(const Mat image, const Mat mask, float& hue_mean, float& hue_stddev,
                       float& saturation_mean, float& saturation_stddev,
                       float& value_mean, float& value_stddev);

/**
 * @ingroup travis_functions
 * This function calculates the HSV peak color.
 */
void calcHSVPeakColor(const Mat image, const Mat mask, float& hue_mode, float& hue_peak,
                       float& value_mode, float& value_peak);

/**
 * @ingroup travis_functions
 * This function calculates the moments.
 */
void calcMoments(Mat& theHuMoments, const vector <Point> biggestCont );

/**
 * @ingroup travis_functions
 * This function calculates the arc length.
 */
void calcArcLength(float& arc, const vector <Point> biggestCont );

/**
 * @ingroup travis_functions
 * This function calculates the circle.
 */
void calcCircle(float& radius, const vector <Point> biggestCont );

#endif  // __TRAVIS_LIB_HPP__

