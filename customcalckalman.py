import cv2
import argparse
import sys
import math
import numpy as np



keep_processing = True
selection_in_progress = False 
fullscreen = False


parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
parser.add_argument("-r", "--rescale", type=float, help="rescale image by this factor", default=1.0)
parser.add_argument('video_file', metavar='video_file', type=str, nargs='?', help='specify optional video file')
args = parser.parse_args()



#cropping the desired region

boxes = []
current_mouse_position = np.ones(2, dtype=np.int32)

def on_mouse(event, x, y, flags, params):

    global boxes
    global selection_in_progress

    current_mouse_position[0] = x
    current_mouse_position[1] = y

    if event == cv2.EVENT_LBUTTONDOWN:
        boxes = []

        sbox = [x, y]
        selection_in_progress = True
        boxes.append(sbox)

    elif event == cv2.EVENT_LBUTTONUP:

        ebox = [x, y]
        selection_in_progress = False
        boxes.append(ebox)




def center(points):
    x = np.float32((points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0)
    y = np.float32((points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0)
    return np.array([np.float32(x), np.float32(y)], np.float32)





def nothing(x):
    pass


try:


    if not(args.video_file):
        import camera_stream
        cap = camera_stream.CameraVideoStream()
    else:
        cap = cv2.VideoCapture(0) 

except:


    print("INFO: camera_stream class not found - camera input may be buffered")
    cap = cv2.VideoCapture()



windowName = "Kalman Object Tracking" 
windowName2 = "Hue histogram back projection" 
windowNameSelection = "initial selected region"


# kalmann filter object
# calculated for my environment

kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]],np.float32)

kalman.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [0,0,0,1]],np.float32)

'''kalman.transitionMatrix = np.array([[1,0,1,0],
                                    [0,1,0,1],
                                    [0,0,1,0],
                                    [1,0,0,1]],np.float32)'''

kalman.processNoiseCov = np.array([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,0],
                                   [0,0,0,1]],np.float32) * 0.03

measurement = np.array((2,1), np.float32)
prediction = np.zeros((2,1), np.float32)

print("\nObservation blue color mai im showing")
print("Prediction is in the green color")


if (((args.video_file) and (cap.open(str(args.video_file))))
    or (cap.open(args.camera_to_use))):



    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowName2, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowNameSelection, cv2.WINDOW_NORMAL)

    #might need thresholding for more accuracy
    #ill remove this in the final thing
#8879606320


    s_lower = 60
    cv2.createTrackbar("s lower", windowName2, s_lower, 255, nothing)
    s_upper = 255
    cv2.createTrackbar("s upper", windowName2, s_upper, 255, nothing)
    v_lower = 32
    cv2.createTrackbar("v lower", windowName2, v_lower, 255, nothing)
    v_upper = 255
    cv2.createTrackbar("v upper", windowName2, v_upper, 255, nothing)



    cv2.setMouseCallback(windowName, on_mouse, 0)
    cropped = False


    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    while (keep_processing):



        if (cap.isOpened):
            ret, frame = cap.read()

           

            if (args.rescale != 1.0):
                frame = cv2.resize(frame, (0, 0), fx=args.rescale, fy=args.rescale)

        
    #seeing how much time kalmanntakes 
        start_t = cv2.getTickCount()

        # get parameters from track bars

        s_lower = cv2.getTrackbarPos("s lower", windowName2)
        s_upper = cv2.getTrackbarPos("s upper", windowName2)
        v_lower = cv2.getTrackbarPos("v lower", windowName2)
        v_upper = cv2.getTrackbarPos("v upper", windowName2)

        # cropping waala part

        if (len(boxes) > 1) and (boxes[0][1] < boxes[1][1]) and (boxes[0][0] < boxes[1][0]):
            crop = frame[boxes[0][1]:boxes[1][1],boxes[0][0]:boxes[1][0]].copy()

            h, w, c = crop.shape   #
            if (h > 0) and (w > 0):
                cropped = True

                

                hsv_crop =  cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)


                mask = cv2.inRange(hsv_crop, np.array((0., float(s_lower),float(v_lower))), np.array((180.,float(s_upper),float(v_upper))))
                # mask = cv2.inRange(hsv_crop, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

                

                crop_hist = cv2.calcHist([hsv_crop],[0, 1],mask,[180, 255],[0,180, 0, 255])
                cv2.normalize(crop_hist,crop_hist,0,255,cv2.NORM_MINMAX)

                

                track_window = (boxes[0][0],boxes[0][1],boxes[1][0] - boxes[0][0],boxes[1][1] - boxes[0][1])

                cv2.imshow(windowNameSelection,crop)

            

            boxes = []

        ##mouse position can be changed

        if (selection_in_progress):
            top_left = (boxes[0][0], boxes[0][1])
            bottom_right = (current_mouse_position[0], current_mouse_position[1])
            cv2.rectangle(frame,top_left, bottom_right, (0,255,0), 2)

        # if we have a selected region

        if (cropped):

           

            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            

            img_bproject = cv2.calcBackProject([img_hsv],[0,1],crop_hist,[0,180,0,255],1)
            cv2.imshow(windowName2,img_bproject)

            ##camshift for better results

            ret, track_window = cv2.CamShift(img_bproject, track_window, term_crit)

            
            x,y,w,h = track_window
            frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)

            

            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            

           

            kalman.correct(center(pts))

           

            prediction = kalman.predict()
            print(prediction[0],prediction[1],prediction[2])
            

            

            frame = cv2.rectangle(frame, (prediction[0]-(0.5*w),prediction[1]-(0.5*h)), (prediction[0]+(0.5*w),prediction[1]+(0.5*h)), (0,255,0),2)

        else:



            img_hsv =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



            mask = cv2.inRange(img_hsv, np.array((0., float(s_lower),float(v_lower))), np.array((180.,float(s_upper),float(v_upper))))

            cv2.imshow(windowName2,mask)
             



        cv2.imshow(windowName,frame)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN & fullscreen)

        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

       



        key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF


        if (key == ord('x')):
            keep_processing = False
        elif (key == ord('f')):
            fullscreen = not(fullscreen)

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.")


