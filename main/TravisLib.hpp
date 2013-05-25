// -*- mode:C++; tab-width:4; c-basic-offset:4; indent-tabs-mode:nil -*-

#ifndef __TRAVIS_LIB_HPP__
#define __TRAVIS_LIB_HPP__

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
 * \defgroup travis_library TravisLib
 *
 * @brief Contains a single class, called Travis.
 *
 * Contains a single class, called Travis.
 */

/*
 * @ingroup travis_library
 *
 * @brief The Travis class implements all the algorithms on a single image.
 */
class Travis {
public:

    /**
     * Travis class constructor.
     * @param quiet suppress messages displayed upon success/failure.
     * @param overwrite will not make a copy (faster, less memory), but will overwrite the image you pass.
     */
    Travis(bool quiet=true, bool overwrite=true) : _quiet(quiet), _overwrite(overwrite) {}

    /**
     * Set the image in cv::Mat format.
     * @param image the image to set, in cv::Mat format.
     * @return true if the object was set successfully.
     */
    bool setCvMat(const cv::Mat& image);

    /**
     * Binarize the image.
     * @param algorithm implemented: "redMinusGreen", "greenMinusRed".
     * @param threshold i.e. 50.
     */
    void binarize(const char* algorithm, const double& threshold);

    /**
     * Use findContours to get what we use as blobs.
     * @param maxNumBlobs the number of max blobs to keep, the rest get truncated.
     */
    void blobize(const int& maxNumBlobs, const int& vizualization);

    /**
     * This function calculates X and Y.
     */
    void getBlobsXY(vector <Point>& locations);

    /**
     * Get the image in cv::Mat format.
     * @return the image, in cv::Mat format.
     */
    cv::Mat& getCvMat();

protected:
    /** Store the verbosity level. */
    bool _quiet;

    /** Store the overwrite parameter. */
    bool _overwrite;

    /** Store the image in cv::Mat format. */
    cv::Mat _img;

    /** Store the binary image in cv::Mat format. */
    cv::Mat _imgBin;

    /** Store the contours (blob contours). */
    vector < vector <Point> > _contours;
};

/**
 * @ingroup travis_functions
 * Can be used as a comparison function object for sorting.
 */
bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 );

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

