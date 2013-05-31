// -*- mode:C++; tab-width:4; c-basic-offset:4; indent-tabs-mode:nil -*-

#include "TravisLib.hpp"

/************************************************************************/

bool Travis::setCvMat(const cv::Mat& image) {
    if (!_quiet) printf("[Travis] in: setCvMat(...)\n");
    if (!image.data) {
        fprintf(stderr,"[Travis] error: No image data.\n");
        return false;
    }
    if (!_overwrite) _img = image.clone();  // safer
    else _img = image;  // faster and less memory
    return true;
}

/************************************************************************/

void Travis::binarize(const char* algorithm, const double& threshold) {
    if (strcmp(algorithm,"redMinusGreen")==0) {
        if (!_quiet) printf("[Travis] in: binarize(redMinusGreen, %f)\n",threshold);
        cv::Mat bgrChannels[3];
        cv::split(_img, bgrChannels);
        cv::subtract(bgrChannels[2], bgrChannels[1], _imgBin);  // BGR
        cv::threshold(_imgBin, _imgBin, threshold, 255, 3);
    } else if (strcmp(algorithm,"redMinusBlue")==0) {
        if (!_quiet) printf("[Travis] in: binarize(redMinusBlue, %f)\n",threshold);
        cv::Mat bgrChannels[3];
        cv::split(_img, bgrChannels);
        cv::subtract(bgrChannels[2], bgrChannels[0], _imgBin);  // BGR
        cv::threshold(_imgBin, _imgBin, threshold, 255, 3);
    } else if (strcmp(algorithm,"greenMinusRed")==0) {
        if (!_quiet) printf("[Travis] in: binarize(greenMinusRed, %f)\n",threshold);
        cv::Mat bgrChannels[3];
        cv::split(_img, bgrChannels);
        cv::subtract(bgrChannels[1], bgrChannels[2], _imgBin);  // BGR
        cv::threshold(_imgBin, _imgBin, threshold, 255, 3);
    } else if (strcmp(algorithm,"greenMinusBlue")==0) {
        if (!_quiet) printf("[Travis] in: binarize(greenMinusBlue, %f)\n",threshold);
        cv::Mat bgrChannels[3];
        cv::split(_img, bgrChannels);
        cv::subtract(bgrChannels[1], bgrChannels[0], _imgBin);  // BGR
        cv::threshold(_imgBin, _imgBin, threshold, 255, 3);
    } else if (strcmp(algorithm,"blueMinusRed")==0) {
        if (!_quiet) printf("[Travis] in: binarize(blueMinusRed, %f)\n",threshold);
        cv::Mat bgrChannels[3];
        cv::split(_img, bgrChannels);
        cv::subtract(bgrChannels[0], bgrChannels[2], _imgBin);  // BGR
        cv::threshold(_imgBin, _imgBin, threshold, 255, 3);
    } else if (strcmp(algorithm,"blueMinusGreen")==0) {
        if (!_quiet) printf("[Travis] in: binarize(blueMinusGreen, %f)\n",threshold);
        cv::Mat bgrChannels[3];
        cv::split(_img, bgrChannels);
        cv::subtract(bgrChannels[0], bgrChannels[1], _imgBin);  // BGR
        cv::threshold(_imgBin, _imgBin, threshold, 255, 3);
    } else fprintf(stderr,"[Travis] warning: Unrecognized algorithm: %s.\n",algorithm);
    cv::Mat outChannels[3];
    outChannels[0] = _imgBin;
    outChannels[1] = _imgBin;
    outChannels[2] = _imgBin;
    cv::merge(outChannels, 3, _imgBin3);
    return;
}

/************************************************************************/

void Travis::blobize(const int& maxNumBlobs) {
    if (!_quiet) printf("[Travis] in: blobize(%d)\n", maxNumBlobs);

    //dilate(_imgBin, _imgBin, Mat(),Point(-1,-1),1);
    //erode(_imgBin, _imgBin, Mat(),Point(-1,-1),1);
    dilate(_imgBin, _imgBin, Mat(), Point(-1,-1), 15);
    erode(_imgBin, _imgBin, Mat(), Point(-1,-1), 15);

    // [thanks getBiggestContour from smorante] note: here jgvictores decides to avoid Canny
    findContours( _imgBin, _contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    if (!_quiet) printf("[Travis] # of found contours: %zd.\n", _contours.size());
    
    // [thanks http://stackoverflow.com/questions/13495207/opencv-c-sorting-contours-by-their-contourarea]
    // default to sort by size (to keep the biggest, xD)
    std::sort( _contours.begin(), _contours.end(), compareContourAreas);

    // Now truncate
    if (_contours.size() > maxNumBlobs)
        _contours.erase( _contours.begin()+maxNumBlobs, _contours.end() );

}

/************************************************************************/
bool Travis::getBlobsXY(vector <Point>& locations) {
    if (!_quiet) printf("[Travis] in: getBlobsXY(...)\n");

    // we have the number of actual blobs in _contours.size()

    // This method seems less accurate and less efficient, breaks if < 5 (px?)
    /*for ( int i = 0; i < _contours.size(); i++ ) {
        RotatedRect minEllipse = fitEllipse( Mat(_contours[i]) );
        locations.push_back( minEllipse.center );
    }*/

    // [thanks http://areshopencv.blogspot.com.es/2011/09/finding-center-of-gravity-in-opencv.html]
    vector<Moments> mu( _contours.size() );
    for( int i = 0; i < _contours.size(); i++ ) {
        mu[i] = moments( Mat(_contours[i]), false );
    }
    vector<Point2f> mc( _contours.size() );
    for( int i = 0; i < _contours.size(); i++ ) {
        mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
        locations.push_back( mc[i] );
    }

    return true;
}

/************************************************************************/
bool Travis::getBlobsAngle(const int& method, vector <double>& angles) {
    if (!_quiet) printf("[Travis] in: getBlobsBoxAngle(%d,...)\n", method);

    for( int i = 0; i < _contours.size(); i++ ) {
        //Rect sqCont = boundingRect( Mat(_contours[i]) );
        //RotatedRect sqCont = boundingRect( Mat(_contours[i]) );
        
        if (method == 0) {  // box
            // [thanks http://felix.abecassis.me/2011/10/opencv-bounding-box-skew-angle/]
            _minRotatedRects.push_back( minAreaRect( Mat(_contours[i]) ) );
            /*double angle = minRotatedRect.angle;
            if (angle < -45.) angle += 90.;  // it just tends to go (-90,0)
            angles.push_back( angle );*/
            angles.push_back( _minRotatedRects[_minRotatedRects.size()-1].angle+90.0 );
        } else if (method == 1) {  // ellipse
        // hopefully people will see this return false as a warning and treat before error.
            if (_contours[i].size() < 5) {
                fprintf(stderr,"[Travis] warning: returning false as ellipse would break with < 5 points.\n");
                return false;  // else fitEllipse would cause break exit.
            }
            // [thanks smorante]
            _minRotatedRects.push_back( fitEllipse( Mat(_contours[i]) ) );
            //?//if (angle < -45.) angle += 90.;
            angles.push_back( _minRotatedRects[_minRotatedRects.size()-1].angle );        
        }

    }
    return true;
}

/************************************************************************/

cv::Mat& Travis::getCvMat(const int& image, const int& vizualization) {
    if (!_quiet) printf("[Travis] in: getCvMat(%d,%d)\n",image,vizualization);

    if ( image == 1 ) _returnImg = _imgBin3;
    else _returnImg = _img;

    if (( vizualization == 1 )||( vizualization == 3 )) {  // Contour
        RNG rng(12345);
        for( int i = 0; i < _contours.size(); i++ ) {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( _returnImg, _contours, i, color, 1, 8, CV_RETR_LIST, 0, Point() );
        }
    }

    if (( vizualization == 2 )||( vizualization == 3 )) {  // Box
        for(int i=0;i<_minRotatedRects.size();i++) {
            cv::Point2f vertices[4];
            _minRotatedRects[i].points(vertices);
            for(int i = 0; i < 4; ++i)
                cv::line(_returnImg, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 0), 1, CV_AA);
        }
    }

    return _returnImg;  // image == 0, etc
}

/************************************************************************/
/************************************************************************/
/************************************************************************/

// comparison function object
bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
    double i = fabs( contourArea(cv::Mat(contour1)) );
    double j = fabs( contourArea(cv::Mat(contour2)) );
    return ( i < j );
}

/************************************************************************/

vector <Point> getBiggestContour(const Mat image){
    //variables
    Mat grayImg, cannyImg;
    vector < vector <Point> > contours;
    vector < vector <Point> > biggest;

    //converting to grayscale
    cvtColor(image,grayImg,CV_BGR2GRAY);

    //canny filter and dilation to fill little holes
    Canny(grayImg,cannyImg, 30,100);
    dilate(cannyImg, cannyImg, Mat(),Point(-1,-1),1);

    //finding all contours
    findContours(cannyImg,contours,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);

    //finding biggest contour
    int maxSize=0;
    int indexCont=0;

    for(int i=0 ; i< contours.size() ; i++){
            if(contours[i].size() > maxSize){
                maxSize=contours[i].size();
                indexCont=i;
                }
     }

    biggest.push_back(contours[indexCont]);
    return biggest[0];
}

void calcLocationXY(float& locX, float& locY, const vector <Point> biggestCont){

    RotatedRect minEllipse;

    //fitting ellipse around contour
    minEllipse= fitEllipse(Mat(biggestCont));

    //getting the center
    locX = minEllipse.center.x;
    locY = minEllipse.center.y;

}

void calcMask(Mat& mask, const vector <Point> biggestCont){

    vector < vector <Point> > listCont;
    vector <Point> biggestCH;

    //doing convexhull
    convexHull(biggestCont,biggestCH);
    listCont.push_back(biggestCH);

    //drawing in mask
    drawContours(mask, listCont,-1, Scalar(255),CV_FILLED);
}

void calcArea(float& area, const vector <Point> biggestCont){

    //setting area
    area = contourArea(biggestCont);
}

void calcRectangularity(float& rectangularity, const vector <Point> biggestCont){
    //RotatedRect minEllipse;
    float areaObj;
    float areaRect;

    //calc area of contour
    areaObj = contourArea(biggestCont);

    Rect sqCont = boundingRect(biggestCont);
    //calc area rect
    areaRect = sqCont.area();

    //setting parameter
    rectangularity = areaObj/areaRect;
}

void calcMassCenter(float& massCenterLocX, float& massCenterLocY , const vector <Point> biggestCont){

    Moments mu;
    Point2f mc;

    //calc moments of contour
    mu = moments(biggestCont,false);

    //calc mass center with moments
    mc = Point2f(mu.m10/mu.m00, mu.m01/mu.m00);

    massCenterLocX = mc.x;
    massCenterLocY = mc.y;
}

void calcAspectRatio(float& aspectRatio, float& axisFirst, float& axisSecond ,const vector <Point> biggestCont){

    RotatedRect minEllipse;
    Point2f vertices[4];

    //extracting axis from vertices
    minEllipse = fitEllipse(Mat(biggestCont));
    minEllipse.points(vertices);
    float dist[2];
    for (int i = 0; i < 2; i++){
        dist[i]=std::sqrt(pow((vertices[i].x - vertices[(i+1)%4].x),2)+pow((vertices[i].y - vertices[(i+1)%4].y),2));
    }

    //setting parameter
    aspectRatio = dist[0]/dist[1];
    axisFirst=dist[0];
    axisSecond=dist[1];
}


void calcSolidity(float& solidity, const vector <Point> biggestCont){

    vector <Point> biggestCH;
    float areaCont;
    float areaCH;

    //doing convexhull
    convexHull(biggestCont,biggestCH);

    areaCont= contourArea(biggestCont);
    areaCH = contourArea(biggestCH);
    solidity= areaCont/areaCH;

}

void calcHSVMeanStdDev(const Mat image, const Mat mask, float& hue_mean, float& hue_stddev,
                       float& saturation_mean, float& saturation_stddev,
                       float& value_mean, float& value_stddev){

    Mat hsvImage;
    cvtColor(image, hsvImage, CV_BGR2HSV);

    // Separate the image in 3 places ( H - S - V )
     vector<Mat> hsv_planes;
     split( hsvImage, hsv_planes );

   // calculating mean and stddev
     Scalar h_mean, h_stddev;
     cv::meanStdDev(hsv_planes[0], h_mean, h_stddev,mask);

     Scalar s_mean, s_stddev;
     cv::meanStdDev(hsv_planes[1], s_mean, s_stddev, mask);

     Scalar v_mean, v_stddev;
     cv::meanStdDev(hsv_planes[2], v_mean, v_stddev, mask);

     //setting values
     hue_mean = h_mean[0];
     hue_stddev = h_stddev[0];
     saturation_mean = s_mean[0];
     saturation_stddev = s_stddev[0];
     value_mean = v_mean[0];
     value_stddev = v_stddev[0];

}


void calcHSVPeakColor(const Mat image, const Mat mask, float& hue_mode, float& hue_peak,
                       float& value_mode, float& value_peak) {

    Mat hsvImage;
    cvtColor(image, hsvImage, CV_BGR2HSV);

    // Separate the image in 3 places ( H - S - V )
     vector<Mat> hsv_planes;
     split( hsvImage, hsv_planes );

    // number of bins for each variable
    int h_bins = 180;
//    int s_bins = 255;
    int v_bins = 255;

    // hue varies from 0 to 180, the other 0-255. Ranges
    float h_ranges[] = { 0, 180 };
//    float s_ranges[] = { 0, 255 };
    float v_ranges[] = { 0, 255 };

    const float* h_histRange = { h_ranges };
//    const float* s_histRange = { s_ranges };
    const float* v_histRange = { v_ranges };

    // image to keep the histogram
    MatND h_hist;
//    MatND s_hist;
    MatND v_hist;

    //calculation
    calcHist(&hsv_planes[0], 1, 0, mask, h_hist, 1, &h_bins, &h_histRange);
//    calcHist(&hsv_planes[1], 1, 0, mask, s_hist, 1, &s_bins, &s_histRange);
    calcHist(&hsv_planes[2], 1, 0, mask, v_hist, 1, &v_bins, &v_histRange);

    // finding highest peak
    float h_mode= 0;
    int h_maxPixels = 0;

    for(int i=0; i< h_bins; i++){
        if(h_hist.at<float>(i) > h_maxPixels){
            h_mode=i;
            h_maxPixels=h_hist.at<float>(i);
        }
    }

    float h_peak= 0;

    for(int i=h_bins; i>0; i--){
        if((int)h_hist.at<float>(i) > 0){
            //cout << (int)h_hist.at<float>(i);
            h_peak=i;
            break;
        }
    }

//    float s_peak= 0;
//    int s_maxPixels = 0;

//    for(int i=0; i< s_bins; i++){
//        if(s_hist.at<float>(i) > s_maxPixels){
//            s_peak=i;
//            s_maxPixels=s_hist.at<float>(i);
//        }
//    }

    float v_mode= 0;
    int v_maxPixels = 0;

    for(int i=0; i< v_bins; i++){
        if(v_hist.at<float>(i) > v_maxPixels){
            v_mode=i;
            v_maxPixels=v_hist.at<float>(i);
        }
    }

    float v_peak= 0;

    for(int i=v_bins; i>0; i--){
        if((int)v_hist.at<float>(i) > 0){
            //cout << (int)v_hist.at<float>(i);
            v_peak=i;
            break;
        }
    }

    //setting
    hue_mode=h_mode;
    hue_peak=h_peak;
//    saturation_peak=s_peak;
    value_mode=v_mode;
    value_peak=v_peak;
}

void calcMoments(Mat& theHuMoments, const vector <Point> biggestCont ){

    Moments mu;
    mu = cv::moments(biggestCont);
    HuMoments(mu,theHuMoments);

}

void calcArcLength(float& arc, const vector <Point> biggestCont ){

    Mat tranf = Mat(biggestCont);
    arc=arcLength(tranf,true);
}

void calcCircle(float& radius, const vector <Point> biggestCont ){

    Point2f center;
    float rad=0;
    minEnclosingCircle(Mat(biggestCont),center,rad);
    radius=rad;
}


