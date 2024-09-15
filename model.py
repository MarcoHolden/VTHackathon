'''
This file will contains all the method needed by the eyecursorinterface.py
'''
import cv2
import pyautogui
import time

#apply happy face for down scroll
def smile(frame, landmarks, frame_w, frame_h):
    leftWisker = [landmarks[61]]
    rightWisker = [landmarks[291]]#[269, 270, 409, 291, 375, 321, 405]
    for landmark in leftWisker:
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    for landmark in rightWisker:
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))
    #Mark likes 0.12    
    if (abs(leftWisker[0].x - rightWisker[0].x) > 0.12) :
        print("You smiled")
        pyautogui.scroll(-10)

def eyebrows(frame, landmarks, frame_w, frame_h):
    left_eye_brow = [landmarks[66], landmarks[69]]
    right_eye_brow = [landmarks[296], landmarks[299]]
#    for landmarkLeft, landmarkRight in zip(left_eyebrow, right_eyebrow):
#        #left eyebrow
#        xLeft = int(landmarkLeft.x * frame_w)
#        yLeft = int(landmarkLeft.y * frame_h)`
#        cv2.circle(frame, (xLeft, yLeft), 3, (0, 255, 255))
#        #right eyebrow
#        xRight = int(landmarkRight.x * frame_w)
#        yRight = int(landmarkRight.y * frame_h)
#        cv2.circle(frame, (xRight, yRight), 3, (0, 255, 255))

    if (abs(left_eye_brow[0].y - left_eye_brow[1].y) < 0.020) and (abs(right_eye_brow[0].y - right_eye_brow[1].y) < 0.020):
        print("both eyebrow detected")
        pyautogui.scroll(10)

# def left_eyebrow(frame, landmarks, frame_w, frame_h):
#     #left eyebrow movement action
#     left_eye_brow = [landmarks[66], landmarks[69]]
#     for landmark in left_eye_brow:
#         x = int(landmark.x * frame_w)
#         y = int(landmark.y * frame_h)
#         cv2.circle(frame, (x, y), 3, (0, 255, 255))

#     if (abs(left_eye_brow[0].y - left_eye_brow[1].y)) < 0.040:
#         print("left eyebrow detected")
#         #pyautogui.click()

# def right_eyebrow(frame, landmarks, frame_w, frame_h):
#     #right eyebrow movement action
#     right_eye_brow = [landmarks[296], landmarks[299]]
#     for landmark in right_eye_brow:
#         x = int(landmark.x * frame_w)
#         y = int(landmark.y * frame_h)
#         cv2.circle(frame, (x, y), 3, (0, 255, 255))

#     if (abs(right_eye_brow[0].y - right_eye_brow[1].y)) < 0.040:
#         print("right eyebrow detected")
#         #pyautogui.click()

def mouth_open(frame, landmarks, frame_w, frame_h, isOpen:bool):
    #mouth open action
    mouth = [landmarks[13], landmarks[14]]
    for landmark in mouth:
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    if isOpen:
        pyautogui.press('esc')
    else:
        if (abs(mouth[0].y - mouth[1].y)) >0.1: #0.012:
            print("mouth open")
            #pyautogui.hotkey('ctrl','ctrl')
            pyautogui.press('ctrl', presses=2)
            isOpen = True

def right_wink(frame, landmarks, frame_w, frame_h):
    #detect right wink
    right = [landmarks[374], landmarks[386]]
    left = [landmarks[145], landmarks[159]]
    for landmark in right:
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    if (abs(right[0].y - right[1].y) < 0.012) and (abs(left[0].y - left[1].y) > 0.013):
        print("right wink")
        pyautogui.mouseDown(button='right')

def left_wink(frame, landmarks, frame_w, frame_h, click_interval, last_click_time):
    # Detect left wink (blinking)
    left = [landmarks[145], landmarks[159]]
    right = [landmarks[374], landmarks[386]]
    for landmark in left:
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    current_time = cv2.getTickCount() / cv2.getTickFrequency()
    if (abs(left[0].y - left[1].y) < 0.02) and (abs(right[0].y - right[1].y) > 0.015):
        if current_time - last_click_time >= click_interval:
            print("wink")
            pyautogui.click()
            last_click_time = current_time  # Update last click time
        else:
            print("long wink")
            pyautogui.mouseDown(button='left')
    return last_click_time