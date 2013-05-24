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

    // \begin{Use of Travis}
    Travis travis(false);  // for quiet just use: Travis travis;
    if( !travis.setCvMat(inImage) ) return -1;
    travis.binarize("redMinusGreen",50);
    travis.setMaxNumBlobs(3);
    cv::Mat outImage = travis.getCvMat();
    // \end{Use of Travis}

    cv::namedWindow( "Input image", CV_WINDOW_AUTOSIZE );
    cv::namedWindow( "Output image", CV_WINDOW_AUTOSIZE );
    imshow( "Input image", inImage );
    imshow( "Output image", outImage );
    cv::waitKey(0);
    printf( "Done. Bye!\n");

    return 0;
}
