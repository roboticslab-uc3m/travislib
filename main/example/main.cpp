// -*- mode:C++; tab-width:4; c-basic-offset:4; indent-tabs-mode:nil -*-

#include <stdio.h>

#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <TravisLib.hpp>

int main(int argc, char *argv[]) {

    if (argc!=2) {
        printf( "Usage: travisExample [imageName]\n");
        return -1;
    }
    char* inImageName = argv[1];
    cv::Mat inImage = cv::imread(inImageName, 1);

    // travisCrop(0,0,50,50,inImage);  // pre-treat by cropping

    vector<cv::Point> blobsXY;
    vector<double> blobsAngle;
    vector<double> blobsArea, blobsSolidity;
    vector<double> blobsAspectRatio, blobsAxisFirst, blobsAxisSecond, blobsRectangularity;
    vector<double> blobsHue,blobsSat,blobsVal,blobsHueStdDev,blobsSatStdDev,blobsValStdDev;
    // \begin{Use of Travis}
    Travis travis(false, false);  // ::Travis(quiet=true, overwrite=true);
    if( !travis.setCvMat(inImage) ) return -1;
    if( !travis.binarize("canny") ) return -1;      // Choose between
    //if( !travis.binarize("greenMinusRed",20) ) return -1; //   the different
    //if( !travis.binarize("hue",0,10) ) return -1;    //   overloadings. :)
    travis.morphClosing(4);
    travis.blobize(2);  // max 2 blobs
    travis.getBlobsXY(blobsXY);
    if (! travis.getBlobsAngle(1,blobsAngle) ) return -1; // method: 0=box, 1=ellipse; note check for return as can break
    travis.getBlobsArea(blobsArea);
    travis.getBlobsSolidity(blobsSolidity);
    travis.getBlobsRectangularity(blobsRectangularity); // note: getBlobsAngle(...) must be called before, it computes minRects!
    travis.getBlobsAspectRatio(blobsAspectRatio,blobsAxisFirst,blobsAxisSecond);
        // note: getBlobsAngle(...) must be called before, it computes minRects!
    travis.getBlobsHSV(blobsHue,blobsSat,blobsVal,blobsHueStdDev,blobsSatStdDev,blobsValStdDev);
    cv::Mat outImage = travis.getCvMat(1,3);  // image: 0=color, 1=bw; vizualize: 0=None, 1=contour, 2=box, 3=both
    travis.release();  // Use to free memory and avoid leaks!
    // \end{Use of Travis}
    for( int i = 0; i < blobsXY.size(); i++)
        printf("XY %d: %d, %d.\n", i+1, blobsXY[i].x, blobsXY[i].y);
/*    for( int i = 0; i < blobsAngle.size(); i++)
        printf("Angle %d: %f.\n", i+1, blobsAngle[i]);
    for( int i = 0; i < blobsArea.size(); i++)
        printf("Area %d: %f.\n", i+1, blobsArea[i]);
    for( int i = 0; i < blobsSolidity.size(); i++)
        printf("Solidity %d: %f.\n", i+1, blobsSolidity[i]);
    for( int i = 0; i < blobsSolidity.size(); i++)
        printf("Rectangularity %d: %f.\n", i+1, blobsRectangularity[i]);
    for( int i = 0; i < blobsAspectRatio.size(); i++)
        printf("AspectRatio AxisFirst AxisSecond %d: %f, %f, %f.\n",
            i+1, blobsAspectRatio[i], blobsAxisFirst[i], blobsAxisSecond[i]);
    for( int i = 0; i < blobsHue.size(); i++)
        printf("HSV HSVstdDevs %d: %f, %f, %f; %f %f %f.\n",
            i+1, blobsHue[i], blobsSat[i], blobsVal[i], blobsHueStdDev[i],blobsSatStdDev[i],blobsValStdDev[i]);*/

    cv::namedWindow( "Input image", CV_WINDOW_AUTOSIZE );
    cv::namedWindow( "Output image", CV_WINDOW_AUTOSIZE );
    imshow( "Input image", inImage );
    imshow( "Output image", outImage );
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
    cv::waitKey(0);
    printf( "Done. Bye!\n");

    return 0;
}

