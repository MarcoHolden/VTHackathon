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


class CircleMenuApp():
    def __init__(self):
        self.root = tk.Tk()
        self.menu = CircleMenu(self.root)
        self.root.mainloop()
class CircleMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Clickable Circles")
        self.screen_width = pyautogui.size()[0]
        self.screen_height = pyautogui.size()[1]
        self.canvas = tk.Canvas(root, width=self.screen_width, height=self.screen_height, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.circles = []
        self.reaction_times = []
        self.start_time = time.time()
        self.pattern = [
            (self.screen_width / 2, self.screen_height / 2+20),  # Center
            (self.screen_width / 8, self.screen_height / 8),  # Top left
            (7 * self.screen_width / 8, self.screen_height / 8),  # Top right
            (self.screen_width / 8, 7 * self.screen_height / 8 - 16),  # Bottom left
            (7 * self.screen_width / 8, 7 * self.screen_height / 8 - 16)   # Bottom right
        ]
        self.pattern_index = 0
        self.cycle_count = 0
        self.max_cycles = 2  # Repeat the pattern twice
        self.show_instructions()

    def show_instructions(self):
        self.canvas.delete("all")
        self.canvas.create_text(self.screen_width / 2, self.screen_height / 2,
                                text="Move the cursor with your head and wink with your left eye to click the circles.",
                                fill="white", font=("Arial", 16),
                                anchor='center')
        self.root.after(5000, self.create_circle)  # Show instructions for 5 seconds, then start the pattern

    def create_circle(self):
        if self.cycle_count < self.max_cycles:
            # Get the current position from the pattern
            x, y = self.pattern[self.pattern_index]
            radius = 20
            circle = self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill='orange', outline='orange', tags="circle")
            self.circles.append((circle, (x, y, radius)))
            self.canvas.tag_bind(circle, '<Button-1>', self.on_circle_click)

            self.start_time = time.time()

            # Move to the next pattern position
            self.pattern_index = (self.pattern_index + 1) % len(self.pattern)
            
            # Increase cycle count if the pattern is completed
            if self.pattern_index == 0:
                self.cycle_count += 1

        else:
            self.canvas.delete("all")
            self.canvas.config(bg='black')  # Change canvas background to black after completing the pattern
            self.canvas.create_text(self.screen_width / 2, self.screen_height / 2,
                                    text="Pattern completed. Thank you!",
                                    fill="white", font=("Arial", 16),
                                    anchor='center')

    def on_circle_click(self, event):
        clicked_circle = self.canvas.find_closest(event.x, event.y)[0]
        for circle_id, (x, y, radius) in self.circles:
            if circle_id == clicked_circle:
                self.canvas.delete(circle_id)
                self.circles.remove((circle_id, (x, y, radius)))
                click_time = time.time() - self.start_time
                self.reaction_times.append(click_time)
                print(f"Circle clicked! Time taken: {click_time:.2f} seconds")
                # Create the next circle after a click
                self.root.after(500, self.create_circle)  # Short delay before creating the next circle
                break

    def calculate_average_reaction_time(self):
        if len(self.reaction_times) >= 3:
            # Use the last 3 reaction times for a more responsive average
            return np.mean(self.reaction_times[-3:])
        elif self.reaction_times:
            return np.mean(self.reaction_times)
        return float('inf')



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
            
            
            left_eye_landmarks = [33, 133, 144, 145, 153, 154, 155, 159, 160, 161, 163, 173]
            right_eye_landmarks = [362, 382, 383, 384, 385, 386, 387, 388, 390, 398]
            left_iris_landmarks = [468, 469, 470, 471]
            right_iris_landmarks = [473, 474, 475, 476]
            MOUTH_LANDMARKS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
    

            frame_h, frame_w, _ = frame.shape
            if landmark_points:
                landmarks = landmark_points[0].landmark
                avg_x = sum(landmarks[idx].x for idx in right_iris_landmarks) / len(right_iris_landmarks)
                avg_y = sum(landmarks[idx].y for idx in right_iris_landmarks) / len(right_iris_landmarks)
                
                self.move_mouse(self.screen_w, avg_x, self.screen_h, avg_y)

                #left eyebrow movement
                #model.left_eyebrow(frame,landmarks, frame_w, frame_h)
                #right eyebrow movement
                #model.right_eyebrow(frame, landmarks, frame_w, frame_h)

                #eveybrow raise
                model.eyebrows(frame,landmarks, frame_w, frame_h)
                #mouth open is action
                model.mouth_open(frame, landmarks, frame_w, frame_h, self.isOpen)
                #detect Smile
                model.smile(frame, landmarks,frame_w,frame_h)
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
    
    eye_tracking_thread = EyeTrackingThread(queue)
    eye_tracking_thread.start()

    root = tk.Tk()
    circle_menu = CircleMenu(root)

    best_alpha = 0.5
    min_reaction_time = float('inf')
    alpha_step = 0.05  # Reduced step size for finer adjustments
    min_reaction_time_threshold = 2.0
    update_interval = 500  # Update every 500ms


    def update_alpha():
        nonlocal best_alpha, min_reaction_time

        average_reaction_time = circle_menu.calculate_average_reaction_time()
        print(f"Average reaction time: {average_reaction_time:.2f} seconds")

        if average_reaction_time < min_reaction_time:
            min_reaction_time = average_reaction_time
            best_alpha = eye_tracking_thread.alpha
            print(f"New best reaction time: {min_reaction_time:.2f} seconds with alpha: {best_alpha}")

        if average_reaction_time < min_reaction_time_threshold:
            eye_tracking_thread.alpha = best_alpha
            print(f"Optimized alpha value: {eye_tracking_thread.alpha}")
        else:
            # Adjust alpha based on the difference from the best reaction time
            adjustment = (min_reaction_time - average_reaction_time) * alpha_step
            new_alpha = min(1.0, max(0.1, eye_tracking_thread.alpha + adjustment))
            if new_alpha != eye_tracking_thread.alpha:
                eye_tracking_thread.alpha = new_alpha
                print(f"Adjusted alpha: {eye_tracking_thread.alpha}")

        # Process any pending cursor positions
        while not queue.empty():
            try:
                cursor_x, cursor_y = queue.get_nowait()
                print(f"Cursor position: ({cursor_x}, {cursor_y})")
            except Empty:
                break

        # Schedule the next update
        root.after(update_interval, update_alpha)

    # Start the update loop
    root.after(update_interval, update_alpha)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        # Cleanup
        eye_tracking_thread.join()

if __name__ == "__main__":
    main()