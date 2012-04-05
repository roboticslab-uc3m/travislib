// travis.cpp
// Author: Santiago Morante
// Copyright (c) 2012 Universidad Carlos III de Madrid (http://www.uc3m.es)
// License: LGPL v3 or later. License available at GNU (http://www.gnu.org/licenses/lgpl.html)
// For more license information see attached files: LicenseOpencv.txt and LicenseTravis.txt

#include "travis.hpp"

void OptFlowDetec(){

    // this function uses Lucas-Kanade algorithm to detect points of movement comparing two frames.

    VideoCapture cap;
    cap.open(0);
    Mat gray, prevGray, image;
    vector<Point2f> points[2];
    RotatedRect square;

    namedWindow("Motion",0);

    while(1)
    {
        Mat frame;
        cap.read(frame);
        if( frame.empty() )
            break;

        frame.copyTo(image);
        cvtColor(image, gray, CV_BGR2GRAY);

        // the points are recalculated is there are less than ten.

        if(points[1].size() < 10)
        {
            goodFeaturesToTrack(gray, points[1], 100, 0.1, 1);
            cornerSubPix(gray, points[1], Size(10,10), Size(-1,-1), TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.01));
        }

        else if( !points[0].empty() )
        {
            vector<uchar> status;
            vector<float> err;
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, Size(15,15),3,
                                 TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 30, 0.01),0);

            // the for loop looks for valid points (the ones which x and y coordenates have changed)

            size_t i, k;
            for( i = k = 0; i < points[1].size(); i++ )
            {
                if( status[i] && (abs(points[0][i].x-points[1][i].x)+(abs(points[0][i].y-points[1][i].y)) > 1))
                {
                    points[1][k++] = points[1][i];
                    circle( image, points[1][i], 3, Scalar(0,255,255));
                }
            }
            points[1].resize(k);

            // we fit an ellipse around the motion points

            if(points[1].size() > 10)
            {
                square = fitEllipse(points[1]);
                rectangle(image,square.boundingRect(),Scalar(0,255,0),2);
            }
        }

        imshow("Motion", image);

        std::swap(points[1], points[0]);
        swap(prevGray, gray);

        char c = waitKey(33);
        if(c==27) break;
    }

    cap.release();
    image.release();
    gray.release();
    prevGray.release();
}

void NaviOtsu()
{

    //Otsu method for thresholding is used here to detect floor - no floor

    VideoCapture cap (0);
    Mat frame, extra, frame_bn;
    cap.read(frame);
    Rect selection(Point((frame.cols/2)-10 ,frame.rows-1  ),Point((frame.cols/2)+10 ,frame.rows-20));
    Rect left(Point((frame.cols*0.25)-10 ,(frame.rows*0.5)-10  ),Point((frame.cols*0.25)+10 ,(frame.rows*0.5)+10));
    Rect right(Point((frame.cols*0.75)-10 ,(frame.rows*0.5)-10  ),Point((frame.cols*0.75)+10 ,(frame.rows*0.5)+10));
    Rect center(Point((frame.cols*0.5)-10 ,(frame.rows*0.5)-10  ),Point((frame.cols*0.5)+10 ,(frame.rows*0.5)+10));

    double maxval_l, maxval_r, maxval_c, minval_l, minval_r, minval_c, maxval_s, minval_s;
    Point minloc_l, minloc_r, minloc_c, maxloc_l, maxloc_r, maxloc_c, maxloc_s, minloc_s;

    int forward, turn_r, turn_l;
    namedWindow("Otsu Navigation Live",0);
    namedWindow("Otsu Robot Vision",0);

    while(1)
    {
            cap.read(frame);
            if( frame.empty() )
                break;

        cvtColor(frame, extra, CV_BGR2GRAY);

        dilate(extra, extra, Mat());
        erode(extra, extra, Mat());


        threshold(extra, extra, 30, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        GaussianBlur(extra,frame_bn, Size(5,5),0);

// we "cut" the image to compare the floor color with 3 possible paths

        Mat roi_l(frame_bn,left);
        Mat roi_r(frame_bn, right);
        Mat roi_c(frame_bn, center);
        Mat roi_s(frame_bn,selection);

        minMaxLoc(roi_l,&minval_l,&maxval_l,&minloc_l,&maxloc_l);
        minMaxLoc(roi_r,&minval_r,&maxval_r,&minloc_r,&maxloc_r);
        minMaxLoc(roi_c,&minval_c,&maxval_c,&minloc_c,&maxloc_c);
        minMaxLoc(roi_s,&minval_s,&maxval_s,&minloc_s,&maxloc_s);

        forward = abs(maxval_c - maxval_s);
        turn_r  = abs(maxval_r - maxval_s);
        turn_l  = abs(maxval_l - maxval_s);

        rectangle(frame,selection, Scalar(255,255,0));



        if (turn_l < forward && turn_l < turn_r)
             rectangle(frame,left,Scalar(0,255,255),2);

        else if (turn_r < forward && turn_r < turn_l)
             rectangle(frame,right,Scalar(0,255,255),2);
        else{
            rectangle(frame,center,Scalar(0,255,255),2);}

        imshow("Otsu Navigation Live",frame);
        imshow("Otsu Robot Vision", frame_bn);

        char c = waitKey(33);
        if( c==27) break;

    }
        cap.release();
        frame.release();
        extra.release();
        frame_bn.release();
        destroyAllWindows();
}

void NaviColor()
{

    // here we use adaptative backprojections

    VideoCapture cap;
    cap.open(0);
    Mat frame ,image,hsv, hue, mask, hist, backproj, pyr;
    cap.read(frame);
    Mat hist_l, hist_r, hist_c, hist_f;
    Rect selection(Point((frame.cols/2)-20 ,frame.rows-1  ),Point((frame.cols/2)+20 ,frame.rows-40));
    Rect left(Point((frame.cols*0.25)-20 ,(frame.rows*0.5)-20  ),Point((frame.cols*0.25)+20 ,(frame.rows*0.5)+20));
    Rect right(Point((frame.cols*0.75)-20 ,(frame.rows*0.5)-20  ),Point((frame.cols*0.75)+20 ,(frame.rows*0.5)+20));
    Rect center(Point((frame.cols*0.5)-20 ,(frame.rows*0.5)-20  ),Point((frame.cols*0.5)+20 ,(frame.rows*0.5)+20));
    double centcomp, leftcomp, rightcomp;


    int hsize = 30;
    float hranges[] = {0,180};
    const float* phranges = hranges;
    int ch[] = {0, 0};

    namedWindow("Navicolor Robot Vision",0);
    namedWindow("Navicolor Live",0);


    while(1)
    {

                if(!cap.read(frame) )
                 break;
                frame.copyTo(image);

                GaussianBlur(image,pyr,Size(11,11),0);

                cvtColor(pyr, hsv, CV_BGR2HSV);

// data from live video

                inRange(hsv, Scalar(0, 30,10),Scalar(180, 256, 256), mask);
                hue.create(hsv.size(), hsv.depth());
                mixChannels(&hsv, 1, &hue, 1, ch, 1);

// data from floor

                Mat roi(hue,selection), maskroi(mask,selection);
                calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                normalize(hist, hist, 0, 255, CV_MINMAX);

// here we mix floor histogram with live video backproject

                calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
                backproj &= mask;

                dilate(backproj,backproj,Mat());
                erode(backproj,backproj,Mat());

                Mat roi_f(backproj,selection);
                Mat roi_l(backproj,left);
                Mat roi_r(backproj, right);
                Mat roi_c(backproj, center);

				calcHist(&roi_f, 1, 0, Mat(), hist_f, 1,&hsize, &phranges);
                calcHist(&roi_l, 1, 0, Mat(), hist_l, 1,&hsize, &phranges);
                calcHist(&roi_r, 1, 0, Mat(), hist_r, 1,&hsize, &phranges);
                calcHist(&roi_c, 1, 0, Mat(), hist_c, 1,&hsize, &phranges);

                centcomp = compareHist(hist_f,hist_c,CV_COMP_CHISQR);
                leftcomp = compareHist(hist_f,hist_l,CV_COMP_CHISQR);
                rightcomp = compareHist(hist_f,hist_r,CV_COMP_CHISQR);

                rectangle(frame,selection, Scalar(255,255,0));

                if (leftcomp < centcomp && leftcomp < rightcomp)
                    rectangle(frame,left,Scalar(0,255,255),2);

                else if (rightcomp < centcomp && rightcomp < leftcomp)
                    rectangle(frame,right,Scalar(0,255,255),2);

                else(centcomp <= leftcomp && centcomp <= rightcomp)
                    rectangle(frame,center,Scalar(0,255,255),2);

                imshow("Navicolor Robot Vision", backproj);
                imshow("Navicolor Live", frame);

                char c = waitKey(33);
                if(c== 27) break;

    }

        cap.release();
        frame.release();
        image.release();
        hsv.release();
        hue.release();
        mask.release();
        hist.release();
        backproj.release();
        pyr.release();
        destroyAllWindows();

}

void SearchColor(int option, int accuracy ,Rect selection, IplImage* selectedImage)
{

    VideoCapture cap;
        cap.open(0);

    Rect trackWindow;
    RotatedRect trackBox;
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;
    double minval,maxval;
    Point minloc,maxloc;
    int trackObject= -1;
    Mat histimg = Mat::zeros(200, 320, CV_8UC3);


    namedWindow("Backproject",0);
    namedWindow( "Histogram", 0 );
    namedWindow("Search color",0);
    cvNamedWindow( "Reference", 0 );

    Mat frame, hsv, hsv_f, hue,hue_f, mask, mask_f, hist, pyr, timg, backproj,
            image, backproj2;

    cvShowImage("Reference", selectedImage);
    while(1)
    {
            cap >> frame;
            if( frame.empty() )
                break;
            frame.copyTo(image);

            cvtColor(Mat (selectedImage), hsv, CV_BGR2HSV);
            cvtColor(image,hsv_f, CV_BGR2HSV);

            // if a region has been selected, calculate histogram

            if( trackObject )
            {

                inRange(hsv, Scalar(0, 30, 10), Scalar(180, 256, 256), mask);
                inRange(hsv_f, Scalar(0, 30, 10), Scalar(180, 256, 256), mask_f);

                int ch[] = {0, 0};
                int ch_f[] = {0, 0};

                hue.create(hsv.size(), hsv.depth());
                hue_f.create(hsv_f.size(), hsv_f.depth());

                mixChannels(&hsv, 1, &hue, 1, ch, 1);
                mixChannels(&hsv_f, 1 ,&hue_f,1 ,ch_f, 1);

                if( trackObject < 0 )
                {

// ROI region of interest, is the area where we are interested to calculate histogram

                    Mat roi(hue, selection), maskroi(mask, selection);

                    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);

                    normalize(hist, hist, 0, 255, CV_MINMAX);

                    trackWindow = selection;
                    trackObject = 1;

                    histimg = Scalar::all(0);
                    int binW = histimg.cols / hsize;
                    Mat buf(1, hsize, CV_8UC3);
                    for( int i = 0; i < hsize; i++ )
                        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
                    cvtColor(buf, buf, CV_HSV2BGR);

                    for( int i = 0; i < hsize; i++ )
                    {
                        int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
                        rectangle( histimg, Point(i*binW,histimg.rows),
                                   Point((i+1)*binW,histimg.rows - val),
                                   Scalar(buf.at<Vec3b>(i)), -1, 8 );
                    }

                  }
// calcbackproject is called with histogram from template and hue from frame (where we are searching)

                calcBackProject(&hue_f, 1, 0, hist, backproj, &phranges);
                backproj &= mask_f;

// when nothing is detected, the focus of searching is moved to the center of the screen.

                if(trackWindow.height<=0 || trackWindow.width<=0)
                {
                    trackWindow.height= frame.rows/2;
                    trackWindow.width= frame.cols/2;
                }

                GaussianBlur(backproj,backproj,Size(11,11),0);

// three options: camshift, meanshift, and colormax

                if(option==1)
                {
                   trackBox = CamShift(backproj, trackWindow, TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

                   if( trackWindow.area() <= 1 )
                   {
                       int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
                       trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                       trackWindow.x + r, trackWindow.y + r) &
                                     Rect(0, 0, cols, rows);
                    }

                    rectangle( image, trackBox.boundingRect(), Scalar(0,0,255), 3, CV_AA );
                }

                if(option==2){
                    meanShift(backproj,trackWindow,TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER,10,1));
                    rectangle( image, trackWindow, Scalar(0,0,255), 3, CV_AA );
                }

                if(option==3)
                {

// pyrdown reduces resolution, pyrup the opposite. That reduces small noise

                    pyrDown(backproj, pyr, Size(backproj.cols/2, backproj.rows/2));
                    pyrUp(pyr, timg, backproj.size());

// gaussianblur is used to blur :-) the image, because we are looking for maximum points of color,
// so if a "false" positive appears, we blur it

                    GaussianBlur(timg,pyr,Size(11,11),0);

                    minMaxLoc(pyr,&minval,&maxval,&minloc,&maxloc);

                    if(accuracy == 1 && maxval>240)
                       circle(image,maxloc,20, Scalar(0,255,255),2);
                    else if(accuracy == 2 && maxval>247)
                       circle(image,maxloc,20, Scalar(0,255,255),2);
                    else if(accuracy == 3 && maxval>252)
                       circle(image,maxloc,20, Scalar(0,255,255),2);

                }

                cvtColor( backproj, backproj2, CV_GRAY2BGR );
                rectangle( backproj2, trackBox.boundingRect(), Scalar(0,0,255), 3, CV_AA );
                imshow("Backproject", backproj2);

            }


                imshow("Search color",image);
                imshow( "Histogram", histimg );

                char c= waitKey(33);
                if(c==27) break;
        }

// we clean the captures, images, and windows at the end

        cap.release();
        image.release();
        backproj.release();
        backproj2.release();
        cvReleaseImage(&selectedImage);
        frame.release();
        hsv.release();
        hue.release();
        hue_f.release();
        mask.release();
        mask_f.release();
        hist.release();
        pyr.release();
        timg.release();
        backproj.release();
        backproj2.release();
        cvDestroyAllWindows();

}

void TemplateMatching(int accuracy ,Rect selection, IplImage* selectedImage)
{

    IplImage	*img, *res, *reference;
    CvPoint	minloc, maxloc;
    double	minval=1, maxval=0;
    int         trackObject = -1;
    int         img_width, img_height, reference_width, reference_height, res_width, res_height;

    CvCapture* capture = cvCreateCameraCapture(0) ;

    cvNamedWindow( "Reference",0 );
    cvShowImage("Template Matching",0);


    cvShowImage( "Reference", selectedImage );

    while(1)
    {
        img = cvQueryFrame( capture);

        if(!img)
             break;

        if(trackObject)
        {
            if(trackObject<0)
            {
// roi is the area of the template that we are looking for

                cvSetImageROI(selectedImage,selection);
                reference = cvCreateImage(cvGetSize(selectedImage),selectedImage->depth, selectedImage->nChannels);
                cvCopy(selectedImage,reference);
                cvResetImageROI(selectedImage);

                reference_width  = reference->width;
                reference_height = reference->height;
                img_width  = img->width;
                img_height = img->height;
                res_width  = img_width - reference_width + 1;
                res_height = img_height - reference_height + 1;
                res = cvCreateImage( cvSize( res_width, res_height ), IPL_DEPTH_32F, 1 );

                trackObject = 1;

            }
// this functions is the responsible of compare template and small parts of image

            cvMatchTemplate( img, reference, res, CV_TM_SQDIFF_NORMED);
            cvMinMaxLoc( res, &minval, &maxval, &minloc, &maxloc, 0 );

            if(accuracy == 1 && minval < 0.15)
                cvRectangle(img, cvPoint( minloc.x, minloc.y ), cvPoint( minloc.x + reference_width, minloc.y + reference_height ), cvScalar( 0, 0, 255),2, 8, 0 );

            else if(accuracy == 2 && minval < 0.1)
                cvRectangle(img, cvPoint( minloc.x, minloc.y ), cvPoint( minloc.x + reference_width, minloc.y + reference_height ), cvScalar( 0, 0, 255),2, 8, 0 );

            else if(accuracy == 3 && minval < 0.07)
                cvRectangle(img, cvPoint( minloc.x, minloc.y ), cvPoint( minloc.x + reference_width, minloc.y + reference_height ), cvScalar( 0, 0, 255),2, 8, 0 );

            cvShowImage("Template Matching",img);
        }

        char c = cvWaitKey(33);
        if(c==27) break;
    }

    cvReleaseCapture( &capture );
    cvReleaseImage(&img);
    cvReleaseImage(&res);
    cvDestroyAllWindows();
}

void matches2points(const vector<vector<DMatch> >& matches, const vector<KeyPoint>& kpts_train,
                    const vector<KeyPoint>& kpts_query, vector<Point2f>& pts_train,
                    vector<Point2f>& pts_query)
{
  pts_train.clear();
  pts_query.clear();

  for (size_t k = 0; k < matches.size(); k++)
  {
    for (size_t i = 0; i < matches[k].size(); i++)
    {
        const DMatch& match = matches[k][i];
        pts_query.push_back(kpts_query[match.queryIdx].pt);
        pts_train.push_back(kpts_train[match.trainIdx].pt);
    }
  }

}

void SurfCentroid(int accuracy, IplImage* selectedImage)
{
    CvCapture* capture = cvCreateCameraCapture(0) ;
    IplImage* image_gray = cvCreateImage(cvGetSize(selectedImage), IPL_DEPTH_8U,1),
            * reference = cvCreateImage(cvSize(320,240),IPL_DEPTH_8U,1),
            *img, *frame;

    Mat img_matches;
    int first=1, x_max=0,y_max=0, x_med=0, y_med=0, valid=0;

// converting to grayscale

    cvCvtColor(selectedImage,image_gray,CV_RGB2GRAY);
    cvResize(image_gray,reference);

    SurfFeatureDetector detector(400);
    vector<KeyPoint> keypoints1, keypoints2;
    keypoints1.clear();
    detector.detect(reference, keypoints1);

    SurfDescriptorExtractor extractor;
    Mat descriptors1, descriptors2;
    extractor.compute(reference, keypoints1, descriptors1);
    vector<Point2f> detectedPointsVideo;
    vector<Point2f> detectedPointsSelection;

    BruteForceMatcher<L2<float> > matcher;
    vector<vector<DMatch> > supervector;

    namedWindow("Surf Average", CV_WINDOW_AUTOSIZE);

    while(1)
    {
        img = cvQueryFrame( capture);

        if(!img)
             break;
        if(first)
        {
            int width = img->width;
            int height = img->height;
            frame = cvCreateImage( cvSize( width, height ), IPL_DEPTH_8U, 1 );
            first=0;
        }

        cvCvtColor( img, frame, CV_RGB2GRAY );
        keypoints2.clear();
        detector.detect(frame, keypoints2);
        extractor.compute(frame, keypoints2, descriptors2);
        supervector.clear();

        // Different levels of accuracy change max distance allowed

        if(accuracy == 1)
            matcher.radiusMatch(descriptors1, descriptors2,supervector,0.27);
        else if(accuracy == 2)
            matcher.radiusMatch(descriptors1, descriptors2,supervector,0.15);
        else if(accuracy == 3)
            matcher.radiusMatch(descriptors1, descriptors2,supervector,0.09);

        drawMatches(reference, keypoints1, frame, keypoints2, supervector, img_matches,Scalar(0,255,255),Scalar(255,0,0),
                vector<vector<char> >(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        detectedPointsVideo.clear();
        detectedPointsSelection.clear();

        matches2points(supervector,keypoints1,keypoints2, detectedPointsSelection, detectedPointsVideo);

        if (detectedPointsSelection.size() > 7)
        {
            x_max=y_max=x_med=y_med=valid=0;

            for(size_t k=0; k < detectedPointsSelection.size(); k++)
            {
                if((int)(detectedPointsVideo[k].x) > 0 && (int)(detectedPointsVideo[k].y) > 0)
                {
                    x_max = x_max + (int)(detectedPointsVideo[k].x);
                    y_max = y_max + (int)(detectedPointsVideo[k].y);
                    valid++;
                }
            }
            if(!valid)
                valid=1;
            x_med= (x_max / valid)+reference->width;
            y_med= y_max / valid;

            circle(img_matches,Point(x_med,y_med),50,Scalar(255,0,255),2);
        }

        imshow("Surf Average",img_matches);

        char c=cvWaitKey(33);
        if(c==27) break;
    }

    cvReleaseCapture( &capture );
    cvReleaseImage(&reference);
    cvReleaseImage(&img);
    cvReleaseImage(&frame);
    keypoints1.clear();
    keypoints2.clear();
    descriptors1.release();
    descriptors2.release();
    detectedPointsVideo.clear();
    detectedPointsSelection.clear();
    matcher.clear();
    supervector.clear();
    img_matches.release();
    cvDestroyAllWindows();
}

void SurfHomography(int accuracy, IplImage* selectedImage){

    Mat referencebig = Mat(selectedImage);
    Mat reference;
    resize(referencebig,reference,Size(referencebig.cols/2,referencebig.rows/2));
    Mat frame;
    double min_dist = 100;

    VideoCapture capture;
    capture.open(0);

    vector<KeyPoint> keypoints1, keypoints2;
    SurfDescriptorExtractor extractor;
    SurfFeatureDetector detector( 400 );

    Mat descriptors1, descriptors2;
    detector.detect( reference, keypoints1 );
    extractor.compute( reference, keypoints1, descriptors1 );

    FlannBasedMatcher matcher;
    vector< DMatch > matches;
    vector< DMatch > good_matches;
    Mat img_matches;
    vector<Point2f> obj;
    vector<Point2f> scene;
    vector<Point2f> scene_corners(4);
    vector<Point2f> obj_corners(4);

    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( reference.cols, 0 );
    obj_corners[2] = cvPoint( reference.cols, reference.rows );
    obj_corners[3] = cvPoint( 0, reference.rows );

    while(1){
        capture.read(frame);
        detector.detect( frame, keypoints2 );
        extractor.compute( frame, keypoints2, descriptors2 );

  if(descriptors2.elemSize()>0)
  {
      matcher.clear();
      matches.clear();
      good_matches.clear();

    matcher.match( descriptors1, descriptors2, matches );

    for( int i = 0; i < descriptors1.rows; i++ )
        { double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
        }

    for( int i = 0; i < descriptors1.rows; i++ )
        { if( matches[i].distance <(4/accuracy)*min_dist )
            good_matches.push_back( matches[i]);
         }

    drawMatches( reference, keypoints1, frame, keypoints2,
               good_matches, img_matches, Scalar(32,32,2), Scalar(123,45,3),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

     obj.clear();
     scene.clear();

     for( int i = 0; i < (int)(good_matches.size()); i++ ){
            obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
            scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
     }

    if(scene.size()>4 && obj.size()>0){
            Mat H = findHomography( obj, scene, CV_RANSAC );
            perspectiveTransform( obj_corners, scene_corners, H);
            line( img_matches, scene_corners[0] + Point2f( reference.cols, 0), scene_corners[1] + Point2f( reference.cols, 0), Scalar(0, 255, 0), 4 );
            line( img_matches, scene_corners[1] + Point2f( reference.cols, 0), scene_corners[2] + Point2f( reference.cols, 0), Scalar( 0, 255, 0), 4 );
            line( img_matches, scene_corners[2] + Point2f( reference.cols, 0), scene_corners[3] + Point2f( reference.cols, 0), Scalar( 0, 255, 0), 4 );
            line( img_matches, scene_corners[3] + Point2f( reference.cols, 0), scene_corners[0] + Point2f( reference.cols, 0), Scalar( 0, 255, 0), 4 );
         }

        imshow( "Surf Homography", img_matches );
        char c=waitKey(33);
        if(c==27){break;}
      }
    }
    capture.release();
	reference.release();
	referencebig.release();
	frame.release();
    keypoints1.clear();
    keypoints2.clear();
    descriptors1.release();
    descriptors2.release();
    matcher.clear();
    img_matches.release();
    cvDestroyAllWindows();
}
