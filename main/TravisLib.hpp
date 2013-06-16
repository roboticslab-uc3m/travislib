// -*- mode:C++; tab-width:4; c-basic-offset:4; indent-tabs-mode:nil -*-

#ifndef __TRAVIS_LIB_HPP__
#define __TRAVIS_LIB_HPP__

#include <stdio.h>  // just printf and fprintf

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
//#include "highgui.h" // to show windows

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
     * Morphologically closing the binarized image.
     * @param closure i.e. 4 for a 100x100 image, 15 for higher resolution.
     */
    void morphClosing(const int& closure);

    /**
     * Use findContours to get what we use as blobs.
     * @param maxNumBlobs the number of max blobs to keep, the rest get truncated.
     */
    void blobize(const int& maxNumBlobs);

    /**
     * This function calculates X and Y as moments directly extracted from the stored contours.
     * @param locations returned.
     */
    bool getBlobsXY(vector <Point>& locations);

    /**
     * This function calculates ALPHA.
     * @param method 0=box, 1=ellipse.
     * @param angles returned.
     */
    bool getBlobsAngle(const int& method, vector <double>& angles);

    /**
     * This function calculates HSV.
     * @param hues returned.
     * @param vals returned.
     * @param sats returned.
     */
    bool getBlobsHSV(vector <double>& hue, vector <double>& val, vector <double>& sat);

    /**
     * Get the image in cv::Mat format.
     * @param image
     * @param vizualization param, 0=None, 1=Contour.
     * @return the image, in cv::Mat format.
     */
    cv::Mat& getCvMat(const int& image, const int& vizualization);

    /**
     * Release _img and _imgBin3 to prevent memory leaks.
     */
    void release();

protected:
    /** Store the verbosity level. */
    bool _quiet;

    /** Store the overwrite parameter. */
    bool _overwrite;

    /** Store the image in cv::Mat format. */
    cv::Mat _img;

    /** Store the binary image in cv::Mat format. */
    cv::Mat _imgBin;

    /** Store the binary image fit for 3 layer sending in cv::Mat format. */
    cv::Mat _imgBin3;

    /** Store the contours (blob contours). */
    vector < vector <Point> > _contours;

    /** Store the box. */
    vector < RotatedRect > _minRotatedRects;

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

