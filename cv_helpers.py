__author__ = "Igor Kim"
__credits__ = ["Igor Kim"]
__maintainer__ = "Igor Kim"
__email__ = "igor.skh@gmail.com"
__status__ = "Development"
__date__ = "05/2019"
__license__ = "MIT"

import cv2, pytesseract, imutils, logging
import numpy as np

from skimage.measure import compare_ssim as ssim

from consts import *

logger = logging.getLogger('')
def sort_contours(cnts, method="left-to-right"):
    """
    Source https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/

    Arguments:
        cnts {[type]} -- [description]

    Keyword Arguments:
        method {str} -- [description] (default: {"left-to-right"})
    """
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

def preprocess_image(image):
    """Preprocess image for better contours detection
    Transofrm text blocks into almost rectangles
    
    Arguments:
        image {cv2.image} -- CV2 image object
    
    Returns:
        cv2.image -- Processed CV2 image object
    """
    new_image = image.copy()

    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    new_image = cv2.GaussianBlur(new_image,(15,15), 0)
    ret3, new_image = cv2.threshold(new_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # _, new_image = cv2.threshold(new_image, 0, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((7, 7),np.uint8)
    new_image = cv2.dilate(new_image, kernel, iterations = 3)
    return new_image
    
def preprocess_roi(image, padding=0, scaling_factor=1):
    """Processes image for beter text detection witht Google Tesseract
    Based on: https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
    
    Arguments:
        image {cv2.image} -- cv2 image
    
    Keyword Arguments:
        padding {int} -- padding from all image sides (default: {0})
        scaling_factor {int} -- resize factor (default: {1})
    
    Returns:
        cv2.image -- processed cv2 image object
    """
    roi = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (image.shape[:2][1]*scaling_factor, image.shape[:2][0]*scaling_factor), interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(roi,(3,3),0)
    _,roi = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    roi = cv2.copyMakeBorder(roi, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255,255,255))
    return roi

def find_countours(image):
    """Get contours on the image
    
    Arguments:
        image {cv2.image} -- processed image
    
    Returns:
        list -- contours
    """
    proc_image = image.copy()
    # proc_image = cv2.cvtColor(proc_image, cv2.COLOR_BGR2GRAY)
    cntrs = cv2.findContours(proc_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = imutils.grab_contours(cntrs)
    cntrs = sort_contours(cntrs, method="top-to-bottom")[0]
    return cntrs

def draw_countrous(image, cntrs):
    """Draw regions of interest contours on the image
    
    Arguments:
        image {cv2.image} -- original image
        cntrs {list} -- contours object
    
    Returns:
        cv2.image -- image with drawn contours on it
    """
    out_image = image.copy()
    for c in cntrs:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(out_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out_image

def get_rois(image, cntrs):
    """Extract regions of interestt from the image based on contours
    Original image shall be provided
    
    Arguments:
        image {cv2.image} -- cv2 image object
        cntrs {list} -- [description]
    
    Returns:
        list --  list of cv2 images
    """
    rois = []
    for c in cntrs:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = image[y:y + h, x:x + w]
        rois.append((roi, (y, y + h, x, x+w)))
    return rois

def recognise_rois(rois, padding=0, scaling_factor=1, debug=False):
    """Recognize text on the image roi with Goolge Tesseract
    
    Arguments:
        rois {cv2.image} -- CV2 image object
    
    Keyword Arguments:
        padding {int} -- padding from all  four sides of the image (default: {0})
        scaling_factor {int} -- rescaling factor to resize image (default: {1})
        debug {bool} -- enable debug output (default: {False})
    
    Returns:
        list -- text on the image as list
    """
    values = []
    triggered = False
    for i in range(len(rois)):
        r, pos = rois[i]
        image = preprocess_roi(r, padding, scaling_factor)
        text = pytesseract.image_to_string(image, config=TESSERACT_CONF)
        if triggered:
            logger.debug("ROI {}: {}".format(i, text))
            if debug:
                cv2.imwrite("%s/%d.png"%(DEBUG_FOLDER,i), image)
            values.append((text, pos))
            if len(values) == len(EXPECTED_KEYS)*2:
                break
        triggered = triggered or str(text).lower().find(TRIGGER_WORD) > -1
    if not triggered:
        logger.warning("Trigger not found")
    return values

def get_video_n_frames(filename):
    vidcap = cv2.VideoCapture(filename)
    n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    vidcap.release()
    return n_frames, fps

def open_video_file(filename):
    return cv2.VideoCapture(filename)

def get_video_frame(filename, frame_n):
    vidcap = cv2.VideoCapture(filename)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES,frame_n)
    frame_n = int(vidcap.get(0))
    _, frame = vidcap.read()
    vidcap.release()
    return frame_n, frame

def get_video_frames(filename, output="images", frames=None, middle_frame=None, tolerance=0.98, middle_tolerance=0.7):
    """Get individual frames from the video file, skipping images which
    are similar by structurual similarity (SSIM) by more than tolerance
    
    Arguments:
        filename {str} -- path to video file
    
    Keyword Arguments:
        output {str} -- output folder path (default: {"images"})
        tolerance {float} -- tolerance to skip similar images (default: {0.98})
    
    Returns:
        (int, list) -- (number frames, file names)
    """
    simple_read = False
    if frames is None:
        n_frames, _ = get_video_n_frames(filename)
        frames = range(n_frames)
        simple_read = True
    vidcap = cv2.VideoCapture(filename)
    last_image = None
    frame_list = []
    count = 0
    for f in frames:
        save_fn = "%s/frame%d.png"%(output,f)
        if not simple_read:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES,f)
        success, image = vidcap.read()
        if not success:
            if simple_read:
                break
            else: 
                logger.warning("Failed reading frame %d"%f)
                continue
        if middle_frame is not None and ssim(middle_frame, image, multichannel=True) < middle_tolerance:
            continue
        if last_image is None or ssim(last_image, image, multichannel=True) < tolerance:
            frame_list.append(save_fn)
            count += 1
            cv2.imwrite(save_fn, image) 
            last_image = image.copy()
    vidcap.release()
    return count, frame_list