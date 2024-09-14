import numpy as np
import tkinter as tk
import random
import cv2
import pyautogui
import mediapipe as mp
from threading import Thread
from queue import Queue, Empty
import model
import time


class CircleMenuApp:
    def __init__(self):
        self.root = tk.Tk()
        self.menu = CircleMenu(self.root)
        self.root.mainloop()

class CircleMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Clickable Circles")
        self.canvas = tk.Canvas(root, width=800, height=600, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.circles = []
        self.start_time = time.time()  # Time when the first circle is created
        self.create_circle()  # Create the first circle

    def create_circle(self):
        # Remove any existing circles
        self.canvas.delete("circle")
        self.circles.clear()
        
        # Generate new circle position and radius
        x = random.randint(100, 700)
        y = random.randint(100, 500)
        radius = 20
        circle = self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill='orange', outline='orange', tags="circle")
        self.circles.append((circle, (x, y, radius)))
        self.canvas.tag_bind(circle, '<Button-1>', self.on_circle_click)
        self.start_time = time.time()  # Update start time

    def on_circle_click(self, event):
        item = self.canvas.find_closest(event.x, event.y)
        click_time = time.time() - self.start_time  # Time since the circle appeared
        print(f"Circle clicked! Time taken: {click_time:.2f} seconds")
        self.create_circle()  # Create a new circle

    def get_circle_data(self):
        print(self.circles)
        return self.circles


class EyeTrackingThread(Thread):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.daemon = True
        self.cam = cv2.VideoCapture(0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.screen_w, self.screen_h = pyautogui.size()
        self.alpha = 0.5
        self.prev_x, self.prev_y = 0.5, 0.5
        self.isOpen = False
        self.last_click_time = 0
        self.click_interval = 0.5

    def run(self):
        while True:
            ret, frame = self.cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = self.face_mesh.process(rgb_frame)
            landmark_points = output.multi_face_landmarks

            frame_h, frame_w, _ = frame.shape
            if landmark_points:
                landmarks = landmark_points[0].landmark
                right_iris_landmarks = [473, 474, 475, 476]
                avg_x = sum(landmarks[idx].x for idx in right_iris_landmarks) / len(right_iris_landmarks)
                avg_y = sum(landmarks[idx].y for idx in right_iris_landmarks) / len(right_iris_landmarks)
                
                self.move_mouse(self.screen_w, avg_x, self.screen_h, avg_y)

                #left eyebrow movement
                #model.left_eyebrow(frame,landmarks, frame_w, frame_h)
                #right eyebrow movement
                #model.right_eyebrow(frame, landmarks, frame_w, frame_h)
                #mouth open is action
                model.mouth_open(frame, landmarks, frame_w, frame_h, self.isOpen)
                #right wink
                model.right_wink(frame, landmarks, frame_w,frame_h)
                # Detect left wink (blinking)
                self.last_click_time = model.left_wink(frame, landmarks,frame_w,frame_h, self.click_interval, self.last_click_time)

                
                # Put cursor position in queue for main thread to process
                cursor_x, cursor_y = pyautogui.position()
                self.queue.put((cursor_x, cursor_y))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cam.release()
        cv2.destroyAllWindows()

    def move_mouse(self, screen_width, x, screen_height, y):
        x = self.alpha * x + (1 - self.alpha) * self.prev_x
        y = self.alpha * y + (1 - self.alpha) * self.prev_y

        self.prev_x, self.prev_y = x, y

        screen_x = x * screen_width
        screen_y = y * screen_height
        screen_x = (5 * (screen_x - screen_width / 2) + screen_width / 2)
        screen_y = (5 * (screen_y - screen_height / 2) + screen_height / 2)

        screen_x = np.clip(screen_x, 0, screen_width - 1)
        screen_y = np.clip(screen_y, 0, screen_height - 1)

        pyautogui.moveTo(screen_x, screen_y)

def main():
    queue = Queue()
    
    # Start the EyeTrackingThread
    eye_tracking_thread = EyeTrackingThread(queue)
    eye_tracking_thread.start()

    # Run the Tkinter GUI in the main thread
    tk_app = CircleMenuApp()

    while True:
        try:
            circle = tk_app.menu.get_circle_data()
            if is_circle_clicked(circle):
                print(f"Cursor clicked a Circle!")
        except Empty:
            print("here")
            pass

def is_circle_clicked(circle):
    if circle.clicked:
            return True
    return False

if __name__ == "__main__":
    main()