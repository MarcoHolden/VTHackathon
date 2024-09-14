'''
This file will contains all the method needed by the eyecursorinterface.py
'''
import cv2
import pyautogui
import time

def left_eyebrow(frame, landmarks, frame_w, frame_h):
    #left eyebrow movement action
    left_eye_brow = [landmarks[66], landmarks[69]]
    for landmark in left_eye_brow:
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    if (abs(left_eye_brow[0].y - left_eye_brow[1].y)) < 0.020:
        return True
    return False
        
def right_eyebrow(frame, landmarks, frame_w, frame_h):
    #right eyebrow movement action
    right_eye_brow = [landmarks[296], landmarks[299]]
    for landmark in right_eye_brow:
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    if (abs(right_eye_brow[0].y - right_eye_brow[1].y)) < 0.020:
        return True
    return False

def mouth_open(frame, landmarks, frame_w, frame_h, isOpen:bool):
    #mouth open action
    mouth = [landmarks[13], landmarks[14]]
    for landmark in mouth:
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    if isOpen:
        return 0
    else:
        if (abs(mouth[0].y - mouth[1].y)) >0.1: #0.012:
            isOpen = True
            return 1
        return 2

def right_wink(frame, landmarks, frame_w, frame_h):
    #detect right wink
    right = [landmarks[374], landmarks[386]]
    for landmark in right:
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    if (right[0].y - right[1].y) < 0.020:
        return True
    return False


def left_wink(frame, landmarks, frame_w, frame_h):
    # Detect left wink (blinking)
    left = [landmarks[145], landmarks[159]]
    for landmark in left:
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    current_time = cv2.getTickCount() / cv2.getTickFrequency()
    if ((left[0].y - left[1].y) < 0.020):
        return True
    return False