# Import necessary libraries
import cv2
import pyautogui
import time
import os
from datetime import datetime
import numpy as np
import win32gui

# Settings
CHECK_EVERY = 2  # Check every 2 seconds
FOLDER = "exam_logs"  # Folder to save evidence
EXAM_WINDOW_TITLE = "Online Exam Monitoring"  # Title of your exam window

# Create folder if needed
if not os.path.exists(FOLDER):
    os.mkdir(FOLDER)

# Store previous screenshot
last_pic = None
alerts = []

def take_picture():
    """Take screenshot and prepare it for analysis"""
    pic = pyautogui.screenshot()
    return cv2.cvtColor(np.array(pic), cv2.COLOR_RGB2BGR)

def check_difference(new, old):
    """Compare two screenshots to see if something important changed"""
    if old is None:
        return True
    diff = cv2.absdiff(new, old)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    changed = cv2.countNonZero(thresh)
    total = gray.shape[0] * gray.shape[1]
    return changed/total > 0.01

def save_alert(what_happened):
    """Record when something important happens"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alerts.append(f"{now} - {what_happened}")
    print(f"ALERT: {what_happened}")

def save_pic(pic, reason):
    """Save screenshot as evidence"""
    filename = f"{FOLDER}/{datetime.now().strftime('%H%M%S')}_{reason}.jpg"
    cv2.imwrite(filename, pic)

def is_exam_window_active():
    """Check if the exam window is the active window"""
    try:
        window = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(window)
        return EXAM_WINDOW_TITLE in title
    except:
        return False

def is_browser_fullscreen():
    """Check if browser is in fullscreen mode"""
    try:
        window = win32gui.GetForegroundWindow()
        placement = win32gui.GetWindowPlacement(window)
        # Check if window is maximized (common fullscreen behavior)
        return placement[1] == win32con.SW_SHOWMAXIMIZED
    except:
        return False

# Main program
print("Exam monitoring started. Press CTRL+C to stop.")
last_active = time.time()
last_window_check = time.time()

try:
    while True:
        current_time = time.time()
        current_pic = take_picture()

        # Check window activity every 5 seconds to reduce overhead
        if current_time - last_window_check > 5:
            if not is_exam_window_active():
                save_pic(current_pic, "window_switch")
                save_alert("User switched from exam window")
            last_window_check = current_time

        # Check for screen changes (like answers being entered)
        if check_difference(current_pic, last_pic):
            last_active = current_time
            save_pic(current_pic, "activity")
            last_pic = current_pic

        # Check for prolonged inactivity
        if current_time - last_active > 30:  # 30 seconds of no activity
            save_alert("No activity detected for 30 seconds")
            last_active = current_time  # Reset timer

        time.sleep(CHECK_EVERY)

except KeyboardInterrupt:
    print("\nStopping monitor...")

# Save all alerts to report
with open(f"{FOLDER}/report.txt", "w") as f:
    f.write("\n".join(alerts))
print(f"Report saved to {FOLDER}/report.txt")
