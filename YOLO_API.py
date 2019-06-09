'''
This script is for creating an YOLO API, you should only import the function InitializeYOLO, performDetect and DrawBoundingBoxes.
In your script, you should copy the class YOLO_Thread (threading.Thread) which is at the bottom of this file.
'''



import numpy as np
import threading
from ctypes import *
import math
import random
import os
import cv2

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd,  "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value '"+tmp+"' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # print(os.environ.keys())
            # print("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print("Environment variables indicated a CPU run, but we didn't find `"+winNoGPUdll+"`. Trying a GPU run anyway.")
else:
    lib = CDLL("./darknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug= False):
    """
    Performs the meat of the detection
    """
    #pylint: disable= C0321
    # im = load_image(image, 0, 0)
    #import cv2
    #custom_image_bgr = cv2.imread(image) # use: detect(,,imagePath,)
    #custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
    #custom_image = cv2.resize(custom_image,(lib.network_width(net), lib.network_height(net)), interpolation = cv2.INTER_LINEAR)
    #import scipy.misc
    #custom_image = scipy.misc.imread(image)
    im, arr = array_to_image(image)		# you should comment line below: free_image(im)
    if debug: print("Loaded image")
    num = c_int(0)
    if debug: print("Assigned num")
    pnum = pointer(num)
    if debug: print("Assigned pnum")
    predict_image(net, im)
    if debug: print("did prediction")
    #dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, 0) # OpenCV
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)
    if debug: print("Got dets")
    num = pnum[0]
    if debug: print("got zeroth index of pnum")
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    if debug: print("did sort")
    res = []
    if debug: print("about to range")
    for j in range(num):
        if debug: print("Ranging on "+str(j)+" of "+str(num))
        if debug: print("Classes: "+str(meta), meta.classes, meta.names)
        for i in range(meta.classes):
            if debug: print("Class-ranging on "+str(i)+" of "+str(meta.classes)+"= "+str(dets[j].prob[i]))
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                if debug:
                    print("Got bbox", b)
                    print(nameTag)
                    print(dets[j].prob[i])
                    print((b.x, b.y, b.w, b.h))
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    if debug: print("did range")
    res = sorted(res, key=lambda x: -x[1])
    if debug: print("did sort")
    # free_image(im)
    if debug: print("freed image")
    free_detections(dets, num)
    if debug: print("freed detections")

    # Return only the interesting objects objects
    res_cpy = res.copy() # copy the detection result for the remove process
    # ObjectToDetect = ["person", "bicycle", "car", "motorbike", "bus", "truck",
    #                   "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    #                   "zebra", "giraffe", "kite"]
    ObjectToDetect = ["person", "car", "motorbike", "bus", "truck"]
    for detection in res_cpy:
        label = detection[0]
        if(label not in ObjectToDetect): # if the detected object is in our objective list
            res.remove(detection)

    return res


# netMain = None
# metaMain = None
altNames = None

def InitializeYOLO(configPath = "C:/Users/HATAL99/Desktop/darknet-master/build/darknet/x64/yolov3-spp.cfg", weightPath = "yolov3-spp.weights",
                  metaPath= "C:/Users/HATAL99/PycharmProjects/YOLOpython/darknet/build/darknet/x64/data/coco.data"):
    """
    Convenience function to handle the detection and returns of objects.

    Parameters
    ----------------

    configPath: str
        Path to the configuration file. Raises ValueError if not found

    weightPath: str
        Path to the weights file. Raises ValueError if not found

    metaPath: str
        Path to the data file. Raises ValueError if not found


    Returns
    ----------------------
    netMain, metaMain (input of the performDetect function)

    """
    # global metaMain, netMain, altNames #pylint: disable=W0603
    global altNames
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `"+os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `"+os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `"+os.path.abspath(metaPath)+"`")
    # if netMain is None:
    netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    # if metaMain is None:
    metaMain = load_meta(metaPath.encode("ascii"))
    if altNames is None:
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    return netMain, metaMain

def performDetect(netMain, metaMain, image_RGB, thresh=0.25):
    """
    Convenience function to handle the detection.
    Return the Detection BoundingBoxes as a list of dict.
    Each element of the list is a dict with the key as next:
        - Detection_arrays[IdxBBox]['label'] = vehicle/person
        - Detection_arrays[IdxBBox]['TopLeft_x'] = x coordinate of the TopLeft Corner of the bounding box (openCV standard)
        - Detection_arrays[IdxBBox]['TopLeft_y'] = y coordinate of the TopLeft Corner of the bounding box (openCV standard)
        - Detection_arrays[IdxBBox]['BottomRight_x'] = x coordinate of the BottomRight Corner of the bounding box (openCV standard)
        - Detection_arrays[IdxBBox]['BottomRight_y'] = x coordinate of the BottomRight Corner of the bounding box (openCV standard)
        - Detection_arrays[IdxBBox]['Prediction_Score'] = Score of the prediction, between 0 and 1

    Parameters
    ----------------
    netMain: Output of the InitializeYOLO function

    metaMain: Output of the InitializeYOLO function

    image_RGB: image array opened with cv2 and cvtColor to RGB

    thresh: float (default= 0.25)
        The detection threshold

    """
    # Do the detection
    detections = detect(netMain, metaMain, image_RGB, thresh)
    Detection_arrays = [] # initializes
    for detection in detections:
        currentDetection = {}
        currentDetection['label'] = detection[0]
        currentDetection['Prediction_Score'] = detection[1]
        # get the bbox (the returned result by detect is the center)
        bounds = detection[2]
        x1, y1, x2, y2 = bounds[:]
        x1 = max(np.round(x1 - x2/2).astype(int), 5)
        y1 = max(np.round(y1 - y2/2).astype(int), 5)
        x2 = min(np.round(x1 + x2).astype(int), image_RGB.shape[1])
        y2 = min(np.round(y1 + y2).astype(int), image_RGB.shape[0])
        currentDetection['TopLeft_x'] = x1
        currentDetection['TopLeft_y'] = y1
        currentDetection['BottomRight_x'] = x2
        currentDetection['BottomRight_y'] = y2
        Detection_arrays.append(currentDetection)

    return Detection_arrays

def DrawBoundingBoxes(Image, Detection_arrays, IsBGR=False):
    '''The function get an Image and an array of Detection as returned in performDetect and draw them.'''
    for currentDetection in Detection_arrays:
        label = currentDetection['label']
        x1 = currentDetection['TopLeft_x']
        y1 = currentDetection['TopLeft_y']
        x2 = currentDetection['BottomRight_x']
        y2 = currentDetection['BottomRight_y']

        if IsBGR:
            if label == "person":
                boxColor = (int(0), int(255), int(0))
            else:
                boxColor = (int(0), int(0), int(255))
        else:
            if label == "person":
                boxColor = (int(0), int(255), int(0))
            else:
                boxColor = (int(255), int(0), int(0))

        cv2.rectangle(Image, (x1, y1), (x2, y2), boxColor, 1)

    return Image

# class YOLO_Thread (threading.Thread):
#     '''The thread class of the YOLO Detection. For each inference, create an instance of this class which gets mainly
#     netMain and metaMain (output of InitializeYOLO).
#     The purpose of this thread is to perform YOLO while new frames are getting grabbed in concurrency.'''
#
#
#     def __init__(self, threadID, name, netMain, metaMain, image_RGB, thresh=0.25):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.name = name
#         self.netMain = netMain
#         self.metaMain = metaMain
#         self.image_RGB = image_RGB
#         self.thresh = thresh
#
#     def run(self):
#         Detection_arrays = performDetect(self.netMain, self.metaMain, self.image_RGB, self.thresh)




