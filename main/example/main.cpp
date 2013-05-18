// -*- mode:C++; tab-width:4; c-basic-offset:4; indent-tabs-mode:nil -*-

#include <stdio.h>

#include <cv.h>
#include <cvaux.h>
#include <highgui.h>

#include <TravisLib.hpp>

int main(int argc, char *argv[]) {
    if (argc!=2) {
        printf( "Usage: travisExample [imageName]\n");
        return -1;
    }
    char* imageName = argv[1];
    cv::Mat image = cv::imread(imageName, 1);

    if (!image.data) {
        printf( "[error] No image data.\n " );
        return -1;
    }
    Travis travis;
    travis.setCvMat(image);

    cv::Mat outImage = travis.getCvMat();

    cv::Mat gray_image;
    cvtColor( image, gray_image, CV_BGR2GRAY );

    //imwrite( "../../images/Gray_Image.jpg", gray_image );

    cv::namedWindow( imageName, CV_WINDOW_AUTOSIZE );
    cv::namedWindow( "Gray image", CV_WINDOW_AUTOSIZE );

    imshow( imageName, image );
    imshow( "Gray image", gray_image );

    cv::waitKey(0);
    printf( "Done. Bye!\n");
    return 0;
}
