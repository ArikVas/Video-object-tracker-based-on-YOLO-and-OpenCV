# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 16:59:47 2019

@author: Arik
"""

from YOLO_API import InitializeYOLO, performDetect, DrawBoundingBoxes
import cv2
import sys
import numpy as np
import csv
import pickle

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')#version of OpenCV(needed for the trackers)
winname1="fixed"#name of window displaying the video
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']#list of possible trackers
tracker_type = tracker_types[0]#chosen tracker
DataFile = "C:/Users/Arik/Documents/Python_Scripts/darknet/build/darknet/x64/data/coco.data"#path to data file - needed for YOLO
ClassesFile = 'C:/Users/Arik/Documents/Python_Scripts/detector/yolov3.txt'#path to list of classes file
WeightsFile = "yolov3-tiny.weights"#'C:/Users/Arik/Documents/Python_Scripts/darknet/build/darknet/x64/yolov3-spp.weights'
# #path to weights file
CfgFile = "yolov3-tiny.cfg"#'C:/Users/Arik/Documents/Python_Scripts/darknet/build/darknet/x64/yolov3-spp.cfg'#path to CFG gile
VideoFile = 'C:/Users/Arik/Documents/Python_Scripts/C0001.mp4'#'C:/Users/Arik/Pictures/20181001_125852.jpg'#'C:/Users/Arik/Documents/Python_Scripts/tracker/venice.mp4'#'MOT17-11.mp4'#'C:/Users/Arik/Documents/Python_Scripts/tracker/mov1_edit.avi'#'C:/Users/Arik/Pictures/20180928_084707.mp4'
VidOrImg = 'V'#'V' - the given file is a video, 'I' - the given file is an image
conf_threshold = 0.5#confidence level threshold
nms_threshold = 0.1#non-max-surpression threshold
DetectPeriod = 10#number of frames after which a detection with YOLO is performed again
MaxDetectNum = 20#maximal allowed number of detections
PicRotate = 0#1 - rotate the picture or frame, 0 - don't rotate
isSave = 1#1 - save the image or video, 0 - don't save
SavePath = 'matan/YOLO2.avi'#[ath and name of the file the video or image will be saved to
WriteFPS = 24#number of frames per second
showImage = False#True - show each frame
makeImageOnly = False
initOnly = False
PrintPeriod = 20#defines the number of iterations before each "keep alive" printing
#Width_YOLO, Height_YOLO = 2048, 1080
Width_YOLO, Height_YOLO = 3840, 2160#width and height of image
WindowWidth, WindowHeight = 900, 900#width and height of convolutional window
FR = 25.0#frame rate
VidLength = 291
frame_no = 3300#/(FR*VidLength)#index of first frame for detection and tracking
MaxDist = 20#number of pixels in which a detection is allowed to be found from its predcessor
VarFileName = 'matan/ObjectsLocation.pkl'
IOULim = 0.2
isLoadList = 1
DetectionListFN = 'matan/ObjectDetection_Every10Frames.pkl'
# isInitialize = 1
# isSaveInitialization = 1
# InitializeFileName = 'matan/InitVar.pkl
# read class names from text file
#classes = None
multiTracker = []
with open(ClassesFile, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
 
# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


def createTrackerByName(trackerType):# Set up tracker.
    if int(minor_ver) < 3:
        tracker = 'cv2.Tracker_create(tracker_type)'
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
    return tracker


def draw_bounding_box(img, boxes, ActiveInd=0):#label, x, y, x_plus_w, y_plus_h):
    for ind, currentDetection in enumerate(boxes):
        label = currentDetection['label']
        #print(currentDetection)
        #print('\n')
        x1 = currentDetection['TopLeft_x'][-1]
        y1 = currentDetection['TopLeft_y'][-1]
        x2 = currentDetection['BottomRight_x'][-1]
        y2 = currentDetection['BottomRight_y'][-1]
        if label == "person":
            boxColor = (int(0), int(255), int(0))
        else:
            boxColor = (int(255), int(0), int(0))

        cv2.rectangle(img, (x1, y1), (x2, y2), boxColor, 2)

        cv2.putText(img, label+'_'+str(ActiveInd[ind]), (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, boxColor, 2)
    return img


def closest_point(NewPoint, OriginalPoints, MaxDist):
    OriginalPoints = np.asarray(OriginalPoints)
    #print(OriginalPoints.shape)
    #print(NewPoint.shape)
    dist_2 = np.sqrt(np.sum((OriginalPoints.T - NewPoint)**2, axis=1))
    #print(dist_2)
    MinDist = min(dist_2)
    if MinDist <= MaxDist:
        return np.argmin(dist_2)
    else:
        return -1


def IOU(oldBoxes, newBox, IOULim):
    L = len(oldBoxes[0])
    xA = np.maximum(np.asarray(oldBoxes[0]), newBox[0])
    yA = np.maximum(np.asarray(oldBoxes[1]), newBox[1])
    xB = np.minimum(np.asarray(oldBoxes[2]), newBox[2])
    yB = np.minimum(np.asarray(oldBoxes[3]), newBox[3])

    # compute the area of intersection rectangle
    interArea = np.multiply(np.maximum(0, xB - xA + 1), np.maximum(0, yB - yA + 1))

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = np.multiply((np.asarray(oldBoxes[2]) - np.asarray(oldBoxes[0]) + 1), (np.asarray(oldBoxes[3]) - np.asarray(oldBoxes[1]) + 1))
    boxBArea = (newBox[2] - newBox[0] + 1) * (newBox[3] - newBox[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = np.divide(interArea, boxAArea + boxBArea - interArea)

    # find the maximum IOU. If it is within the allowed limit, return the index of the box
    MaxIou = max(iou)
    if MaxIou >= IOULim:
        return np.argmax(iou)
    else:
        return -1.0


def CreateDetectionList(image, boxes):
    DetectionList = []
    multiTracker = []
    for i, bbox in enumerate(boxes):
        tracker = createTrackerByName(tracker_type)
        multiTracker.append(tracker)  # , image, tuple(bbox))
        x = bbox['TopLeft_x']
        y = bbox['TopLeft_y']
        w = bbox['BottomRight_x'] - x
        h = bbox['BottomRight_y'] - y
        temp = [x, y, w, h]
        ok = multiTracker[i].init(image, tuple(temp))
        # tempDict={'TopLeft_x':[x],'TopLeft_y':[y],'BottomRight_x':[bbox['BottomRight_x']],'BottomRight_y':[bbox['BottomRight_y']],'isActive':1}
        DetectionList.append(bbox)
        DetectionList[i]['TopLeft_x'] = [x]
        DetectionList[i]['TopLeft_y'] = [y]
        DetectionList[i]['BottomRight_x'] = [x + w]
        DetectionList[i]['BottomRight_y'] = [y + h]
    return DetectionList, multiTracker, ok


def DetectAndAssign(frame, DetectionList, boxes, tracker_type):
    multiTracker = []
    # xCenter0 = (np.array([l['BottomRight_x'][-1] for l in DetectionList])+np.array([l['TopLeft_x'][-1] for l in DetectionList]))/2
    # yCenter0 = (np.array([l['TopLeft_y'][-1] for l in DetectionList])+np.array([l['BottomRight_y'][-1] for l in DetectionList]))/2
    ActiveInd, xTopOld, yTopOld, yBottomOld, xBottomOld = [], [], [], [], []
    for DL in DetectionList:
        xTopOld.append(DL['TopLeft_x'][-1])
        yTopOld.append(DL['TopLeft_y'][-1])
        xBottomOld.append(DL['BottomRight_x'][-1])
        yBottomOld.append(DL['BottomRight_y'][-1])
        DL['isActive'] = 0
    for i, bbox in enumerate(boxes):
        xTopNew = bbox['TopLeft_x']
        yTopNew = bbox['TopLeft_y']
        xBottomNew = bbox['BottomRight_x']
        yBottomNew = bbox['BottomRight_y']
        ClosestInd = IOU([xTopOld, yTopOld, xBottomOld, yBottomOld], [xTopNew, yTopNew, xBottomNew, yBottomNew], IOULim)
        # find the closest previous detection. If there is no close detection, create a new item in the list
        if ClosestInd < 0:
            DetectionList.append(bbox)
            ActiveInd.append(len(DetectionList) - 1)
            DetectionList[-1]['TopLeft_x'] = [xTopNew]
            DetectionList[-1]['TopLeft_y'] = [yTopNew]
            DetectionList[-1]['BottomRight_x'] = [xBottomNew]
            DetectionList[-1]['BottomRight_y'] = [yBottomNew]
            DetectionList[-1]['isActive'] = 1
        else:
            DetectionList[ClosestInd]['TopLeft_x'].append(xTopNew)
            DetectionList[ClosestInd]['TopLeft_y'].append(yTopNew)
            DetectionList[ClosestInd]['BottomRight_x'].append(xBottomNew)
            DetectionList[ClosestInd]['BottomRight_y'].append(yBottomNew)
            # DetectionList[ClosestInd] = bbox
            ActiveInd.append(ClosestInd)
            DetectionList[ClosestInd]['isActive'] = 1
        tracker = createTrackerByName(tracker_type)
        multiTracker.append(tracker)
        temp = [xTopNew, yTopNew, xBottomNew - xTopNew, yBottomNew - yTopNew]
        ok = multiTracker[i].init(frame, tuple(temp))
    return DetectionList, ActiveInd, multiTracker, ok


def TrackerUpdate(frame, DetectionList, ActiveInd, multiTracker):
    # print(i)
    for i, mt in enumerate(multiTracker):
        ok, temp = mt.update(frame)
        ind = ActiveInd[i]
        DetectionList[ind]['TopLeft_x'].append(round(temp[0]))
        DetectionList[ind]['TopLeft_y'].append(round(temp[1]))
        DetectionList[ind]['BottomRight_x'].append(round(temp[0]) + round(temp[2]))
        DetectionList[ind]['BottomRight_y'].append(round(temp[1]) + round(temp[3]))
        if not ok:
            print('Something is wrong with the tracker')
            return ok
    return DetectionList, multiTracker


if VidOrImg == 'V':
    video = cv2.VideoCapture(VideoFile)
    if not video.isOpened():
        print("could not open video\n")
        sys.exit()
    video.set(1, frame_no)
    ok, image = video.read()

    if not ok:
        print("could not open video\n")
        sys.exit()
    print('video opened')
else:
    image = cv2.imread(VideoFile)


if PicRotate == 1:
    image = np.flip(np.transpose(image, (1, 0, 2)), 0)

Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392
# if isInitialize == 1:
#     netMain, metaMain = InitializeYOLO(CfgFile, WeightsFile, DataFile)
#     if isSaveInitialization == 1:
#         with open(InitializeFileName, 'wb') as f:
#             pickle.dump([netMain, metaMain], f)
# else:
#     with open(InitializeFileName, 'rb') as f:
#         netMain, metaMain = pickle.load(f)

image = cv2.resize(image, (Width_YOLO, Height_YOLO), interpolation=cv2.INTER_LINEAR)
#image = cv2.resize(image, (Height,Width))
if showImage == 1:
    cv2.imshow(winname1, image)
    cv2.waitKey(1)

image = image[:, :, ::-1]

if isSave == 1:
    outWrite = cv2.VideoWriter(SavePath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), WriteFPS, (Width, Height))
    
#boxes,class_id,confidence=dn(WeightsFile, CfgFile,image,scale,Width,Height,1)
if isLoadList == 1:
    with open(DetectionListFN, 'rb') as f:  # Python 3: open(..., 'wb')
        AllVideoDetectionList = pickle.load(f)
    boxes = AllVideoDetectionList[0]
else:
    netMain, metaMain = InitializeYOLO(CfgFile, WeightsFile, DataFile)
    boxes = performDetect(netMain, metaMain, image, conf_threshold)

if showImage == 1:
    cv2.namedWindow(winname1, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname1, WindowWidth, WindowHeight)
count = 0

DetectionList, multiTracker, ok = CreateDetectionList(image, boxes)

print(DetectionList)
    #print(bbox)
    #print(DetectionList)
    #print(i)
    #image = draw_bounding_box(image.copy(), boxes, np.arange(len(boxes),dtype=int))#= DrawBoundingBoxes(image.copy(), boxes)
YOLO_IMAGE_RESULT = draw_bounding_box(image.copy(), DetectionList, np.arange(len(boxes), dtype=int))#= DrawBoundingBoxes(image.copy(), boxes)

if showImage == 1:
    cv2.imshow(winname1, YOLO_IMAGE_RESULT[:, :, ::-1])
    cv2.waitKey(1)
    cv2.resizeWindow(winname1, WindowWidth, WindowHeight)
#cv2.imshow(winname1, image)

#cv2.waitKey(1)
# save output image to disk
if isSave == 1:
    cv2.imwrite("object-detection.jpg", YOLO_IMAGE_RESULT[:, :, ::-1])
#    cv2.imwrite("object-detection.jpg", image[:,:,::-1])

#for item in DetectionList:
#   item.update({"isActive":1})#add an entry that states whether the detection is active at the current period
ActiveInd = np.arange(len(DetectionList), dtype=int)
while True and VidOrImg == 'V':
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break
        
    count = count+1
    frame = frame[:, :, ::-1]
    # Start timer
    timer = cv2.getTickCount()
    if count % PrintPeriod == 0:
        print(count)
        #print('\n')
    if count % DetectPeriod == 0:
        multiTracker = []
        BoxInd = int(count/DetectPeriod)
        if isLoadList == 1 and BoxInd <= len(AllVideoDetectionList)-1:
            boxes = AllVideoDetectionList[BoxInd]
        else:
            boxes = performDetect(netMain, metaMain, frame, conf_threshold)
        DetectionList, ActiveInd, multiTracker, ok = DetectAndAssign(frame, DetectionList, boxes, tracker_type)
        if not ok:
            break
#        #print(DetectionList)
    else:
        DetectionList, multiTracker = TrackerUpdate(frame, DetectionList, ActiveInd, multiTracker)

    frame_YOLO = draw_bounding_box(frame.copy(), [DetectionList[i] for i in ActiveInd], ActiveInd)
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
 
    # Display tracker type on frame
    cv2.putText(frame_YOLO, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
     
    # Display FPS on frame
    cv2.putText(frame_YOLO, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
 
    # Display result
    if showImage == 1:
        cv2.imshow(winname1, frame_YOLO)
    
    if isSave == 1:
        outWrite.write(frame_YOLO[:, :, ::-1])
    
    # Exit if ESC pressed
    k = cv2.waitKey(1) #& 0xff
    if k == 27:
        break

with open(VarFileName, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(DetectionList, f)

if isSave == 1:
    outWrite.release()


cv2.destroyAllWindows()
