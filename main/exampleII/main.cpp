// -*- mode:C++; tab-width:4; c-basic-offset:4; indent-tabs-mode:nil -*-

#include <stdio.h>

#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>

#include <TravisLib.hpp>

int main(int argc, char *argv[]) {

    if (argc!=5) {
        cout << "Usage: travisExampleII [firstWord] [SecondWord] [firstValue] [SecondValue]" << endl;
        return -1;
    }
    string path="../testImages/";
    string db= path + "database.txt";
    vector<string> infiles;
//    string wordTofindOne = "top";
//    string wordTofindTwo = "right";
    string wordTofindOne = argv[1];
    string wordTofindTwo = argv[2];

    float trainingAverageX=atof(argv[3]);
    float trainingAverageY=atof(argv[4]);
    vector<float> singleX;
    vector<float> singleY;

    ifstream ifs;
    ifs.open(db.c_str());
    if(!ifs.is_open()){
            cout << "Cannot open filelist " << db << endl;
    }
    else{
        while(!ifs.eof()){
            getline(ifs,db);
            if(db!=""){
                    infiles.push_back(db);
            }
         }
    }
    ifs.close();

    for (int i = 0; i < infiles.size(); i++)
    {
        string singleFile = infiles.at(i);
        std::size_t foundOne = singleFile.find(wordTofindOne);
        std::size_t foundTwo= singleFile.find(wordTofindTwo);

        if (foundOne!=std::string::npos && foundTwo!=std::string::npos){
          cout << "file: " << singleFile << endl;
          Mat inImage = cv::imread(path + singleFile, 1);
          vector<cv::Point> blobsXY;

          //\begin{Use of Travis}
          Travis travis(false, false);  // ::Travis(quiet=true, overwrite=true);
          if( !travis.setCvMat(inImage) ) return -1;
          if( !travis.binarize("grayscale") ) return -1;      // Choose between
          travis.morphClosing(4);
          travis.blobize(1);  // max 1 blob
          travis.getBlobsXY(blobsXY);
          cv::Mat outImage = travis.getCvMat(0,3);  // image: 0=color, 1=bw; vizualize: 0=None, 1=contour, 2=box, 3=both
          travis.release();  // Use to free memory and avoid leaks!
          // \end{Use of Travis}

          cout << "(X,Y) : (" << blobsXY[0].x << ","<< blobsXY[0].y << ")" << endl << endl;
          //adding to vector
          singleX.push_back(blobsXY[0].x);
          singleY.push_back(blobsXY[0].y);

//          imshow( "Input image", inImage );
//          imshow( "Output image", outImage );
//          cv::waitKey(0);
        }
    }

    //final calculation X
    float sumSquareX=0;
    float rootSumSquareX=0;
    for(int i=0; i< singleX.size();i++){
        sumSquareX=sumSquareX+pow(singleX.at(i)-trainingAverageX,2);
    }

    rootSumSquareX=sqrt(sumSquareX);
    cout << "Root-Sum-Square of X: " << rootSumSquareX << endl;

    //final calculation Y
    float sumSquareY=0;
    float rootSumSquareY=0;
    for(int i=0; i< singleY.size();i++){
        sumSquareY=sumSquareY+pow(singleY.at(i)-trainingAverageY,2);
    }

    rootSumSquareY=sqrt(sumSquareY);
    cout << "Root-Sum-Square of Y: " << rootSumSquareY << endl;

    return 1;
}

