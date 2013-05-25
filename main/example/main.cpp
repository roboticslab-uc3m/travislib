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

    vector<cv::Point> blobCentroids;
    // \begin{Use of Travis}
    Travis travis(false, false);  // for quiet and overwrite just use: Travis travis;
    if( !travis.setCvMat(inImage) ) return -1;
    travis.binarize("redMinusGreen",50);
    travis.blobize(3, 1);  // max 3 blobs, vizualize: 0=None, 1=Contour, 2=?
    travis.getBlobsXY(blobCentroids);
    cv::Mat outImage = travis.getCvMat();
    // \end{Use of Travis}
    for( int i = 0; i < blobCentroids.size(); i++)
        printf("Centroid %d: %d, %d.\n",i+1,blobCentroids[i].x,blobCentroids[i].y);

    cv::namedWindow( "Input image", CV_WINDOW_AUTOSIZE );
    cv::namedWindow( "Output image", CV_WINDOW_AUTOSIZE );
    imshow( "Input image", inImage );
    imshow( "Output image", outImage );
    cv::waitKey(0);
    printf( "Done. Bye!\n");

    return 0;
}
