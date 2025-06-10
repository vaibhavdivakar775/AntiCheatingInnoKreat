# Import necessary libraries - these are toolkits we need
import cv2          # Computer vision library for image processing
import pyautogui    # Lets us take screenshots of the screen
import time         # For adding delays and tracking time
import os           # For working with files and folders
from datetime import datetime  # For getting current date/time

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
    # Convert screenshot to format OpenCV can work with:
    # 1. np.array converts the image to numbers
    # 2. cv2.cvtColor changes color format from RGB to BGR
    return cv2.cvtColor(np.array(pic), cv2.COLOR_RGB2BGR)

def check_difference(new, old):
    """Compare two screenshots to see if something important changed"""
    if old is None:  # If this is our first screenshot
        return True  # Consider it changed (nothing to compare to)
    
    # Calculate difference between current and previous screenshot:
    diff = cv2.absdiff(new, old)  # Gets absolute difference pixel-by-pixel
    
    # Convert to grayscale (black and white) to simplify analysis:
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold - turns grayscale into pure black/white:
    # - Pixels darker than 25 become black (0)
    # - Pixels brighter than 25 become white (255)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    
    # Count how many pixels changed (white pixels in thresh image):
    changed = cv2.countNonZero(thresh)
    # Get total number of pixels in image:
    total = gray.shape[0] * gray.shape[1]  # height × width
    
    # Return True if more than 1% of pixels changed:
    return changed/total > 0.03

def save_alert(what_happened):
    """Record when something important happens"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current time
    alerts.append(f"{now} - {what_happened}")  # Add to alerts list
    print(f"ALERT: {what_happened}")  # Also print to screen

def save_pic(pic, reason):
    """Save screenshot as evidence"""
    # Create filename with timestamp and reason:
    filename = f"{FOLDER}/{datetime.now().strftime('%H%M%S')}_{reason}.jpg"
    cv2.imwrite(filename, pic)  # Save image to file

# Main program starts here
print("Monitoring started. Press CTRL+C to stop.")  # Instructions
last_active = time.time()  # Remember when we last saw activity

try:
    # This loop runs forever until we stop it:
    while True:
        current_pic = take_picture()  # Take new screenshot
        
        # Check if screen changed significantly:
        if check_difference(current_pic, last_pic):
            last_active = time.time()  # Update last active time
            save_pic(current_pic, "change")  # Save evidence
            save_alert("Screen changed")  # Record alert
            last_pic = current_pic  # Remember this screenshot
        
        # Check if too much time passed without changes:
        if time.time() - last_active > 30:  # If 30 seconds inactive
            save_alert("No activity for 30 seconds")
            last_active = time.time()  # Reset timer
        
        time.sleep(CHECK_EVERY)  # Wait before checking again

except KeyboardInterrupt:  # If user presses CTRL+C
    print("\nStopping monitor...")

# After stopping, save all alerts to a report file:
with open(f"{FOLDER}/report.txt", "w") as f:  # Open file for writing
    f.write("\n".join(alerts))  # Combine alerts with newlines between them
print(f"Report saved to {FOLDER}/report.txt")  # Tell user where report is
