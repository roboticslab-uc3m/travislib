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

    vector<cv::Point> blobsXY;
    vector<double> blobsAngle;
    // \begin{Use of Travis}
    Travis travis(false, false);  // for quiet and overwrite just use: Travis travis;
    if( !travis.setCvMat(inImage) ) return -1;
    travis.binarize("redMinusGreen",50);
    travis.morphClosing(4);
    travis.blobize(3);  // max 3 blobs
    travis.getBlobsXY(blobsXY);
    bool ok = travis.getBlobsAngle(1, blobsAngle);  // method: 0=box, 1=ellipse; note check for return as can break
    cv::Mat outImage = travis.getCvMat(0,3);  // image: 0=color, 1=bw; vizualize: 0=None, 1=contour, 2=box, 3=both
    travis.release();  // Use to free memory and avoid leaks!
    // \end{Use of Travis}
    for( int i = 0; i < blobsXY.size(); i++)
        printf("XY %d: %d, %d.\n", i+1, blobsXY[i].x, blobsXY[i].y);
    for( int i = 0; i < blobsAngle.size(); i++)
        printf("Angle %d: %f.\n",i+1,blobsAngle[i]);

    cv::namedWindow( "Input image", CV_WINDOW_AUTOSIZE );
    cv::namedWindow( "Output image", CV_WINDOW_AUTOSIZE );
    imshow( "Input image", inImage );
    imshow( "Output image", outImage );
    cv::waitKey(0);
    printf( "Done. Bye!\n");

    return 0;
}

