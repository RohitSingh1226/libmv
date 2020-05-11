#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
using namespace std;
using namespace cv;
int main( int argc, char** argv ){
  
  if(argc<2){
    cout<< endl;
    return 0;
  }
  
  Rect2d roi;
  Mat frame;
  
  Ptr<Tracker> tracker = TrackerKCF::create();
  
  std::string video = argv[1];
  VideoCapture cap("C:\Users\faker\OneDrive\Desktop\video0.mov"); //use video in repo and add location
  
  cap >> frame;
  roi=selectROI("tracker",frame);
  
  if(roi.width==0 || roi.height==0)
    return 0;
  
  tracker->init(frame,roi);
  
  printf("Start the tracking process, press ESC to quit.\n");
  for ( ;; ){
    
    cap >> frame;
    
    if(frame.rows==0 || frame.cols==0)
      break;
    
    tracker->update(frame,roi);
    
    rectangle( frame, roi, Scalar( 255, 0, 0 ), 2, 1 );
    
    imshow("tracker",frame);
    
    if(waitKey(1)==27)break;
  }
  return 0;
}