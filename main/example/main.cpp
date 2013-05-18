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
    char* imageName = argv[1];
    cv::Mat image = cv::imread(imageName, 1);

    Travis travis;  // travis(bool quiet=true); set false for verbosity
    if( !travis.setCvMat(image) ) return -1;

    cv::Mat outImage = travis.getCvMat();

    //cvtColor( outImage, outImage, CV_BGR2GRAY );

    cv::namedWindow( imageName, CV_WINDOW_AUTOSIZE );
    cv::namedWindow( "Output image", CV_WINDOW_AUTOSIZE );

    imshow( imageName, image );
    imshow( "Output image", outImage );

    cv::waitKey(0);
    printf( "Done. Bye!\n");
    return 0;
}
