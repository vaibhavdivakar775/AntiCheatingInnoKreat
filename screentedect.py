# Import necessary libraries - these are toolkits we need
import cv2          # Computer vision library for image processing
import pyautogui    # Lets us take screenshots of the screen
import time         # For adding delays and tracking time
import os           # For working with files and folders
from datetime import datetime  # For getting current date/time
import numpy as np  # Needed for screenshot conversion
import win32gui     # For getting the active window title (Windows only)

# Settings - these are adjustable values that control how the program works
CHECK_EVERY = 2  # How often to check the screen (in seconds)
FOLDER = "exam_logs"  # Name of folder where we'll save evidence

# Create folder if needed - makes sure we have a place to save our files
if not os.path.exists(FOLDER):  # Checks if folder doesn't exist
    os.mkdir(FOLDER)  # Creates the folder if it's missing

# Store previous screenshot - we'll compare new screenshots to this
last_pic = None  # Starts as None because we don't have a first screenshot yet
alerts = []  # Empty list to store all our alerts/warnings

def take_picture():
    """Take screenshot and prepare it for analysis"""
    pic = pyautogui.screenshot()  # Takes picture of entire screen
    return cv2.cvtColor(np.array(pic), cv2.COLOR_RGB2BGR)

def check_difference(new, old):
    """Compare two screenshots to see if something important changed"""
    if old is None:  # If this is our first screenshot
        return True  # Consider it changed (nothing to compare to)
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

def is_chrome_active():
    """Check if the active window is Chrome (Windows only)"""
    window = win32gui.GetForegroundWindow()
    title = win32gui.GetWindowText(window)
    return "Chrome" in title  # You can make this more strict if needed

# Main program starts here
print("Monitoring started. Press CTRL+C to stop.")
last_active = time.time()

try:
    while True:
        current_pic = take_picture()

        # Check if active app is not Chrome
        if not is_chrome_active():
            save_pic(current_pic, "non_chrome_app")
            save_alert("Non-Chrome application detected")
            time.sleep(CHECK_EVERY)
            continue

        # Check for screen change
        if check_difference(current_pic, last_pic):
            last_active = time.time()
            save_pic(current_pic, "change")
            save_alert("Screen changed")
            last_pic = current_pic

        # Check for inactivity
        if time.time() - last_active > 30:
            save_alert("No activity for 30 seconds")
            last_active = time.time()

        time.sleep(CHECK_EVERY)

except KeyboardInterrupt:
    print("\nStopping monitor...")

# Save all alerts to report
with open(f"{FOLDER}/report.txt", "w") as f:
    f.write("\n".join(alerts))
print(f"Report saved to {FOLDER}/report.txt")
