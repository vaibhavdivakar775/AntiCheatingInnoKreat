# Import all required libraries
import cv2          # OpenCV for image processing
import numpy as np  # NumPy for numerical operations
import pyautogui    # For taking screenshots
import time         # For time-related functions
import os           # For file/folder operations
from datetime import datetime  # For timestamps
from pyzbar.pyzbar import decode  # For QR/barcode detection

# Configuration section - these are settings you can adjust
CAPTURE_INTERVAL = 2  # Take screenshot every 2 seconds
OUTPUT_DIR = "monitoring_data"  # Folder to save screenshots
SENSITIVITY = 0.01    # 1% change considered significant
MAX_IDLE_TIME = 30    # 30 seconds allowed without activity
FORBIDDEN_APPS = ["chrome.exe", "notepad.exe"]  # Blocked apps

# Create the output folder if it doesn't exist
# exist_ok=True prevents errors if folder already exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize variables to store data
last_screenshot = None  # Will hold the previous screenshot
events = []  # List to store all detected events

def capture_screen():
    """Takes a screenshot and converts it to OpenCV format"""
    # pyautogui.screenshot() captures the entire screen
    # np.array() converts it to a NumPy array
    # cv2.cvtColor converts from RGB to BGR (OpenCV's default format)
    screenshot = pyautogui.screenshot()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def detect_changes(current_frame, last_frame):
    """Compares current and last frame to detect significant changes"""
    # If no previous frame exists (first run), return True
    if last_frame is None:
        return True
    
    # Calculate absolute difference between current and last frame
    diff = cv2.absdiff(current_frame, last_frame)
    # Convert to grayscale for simpler analysis
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Apply threshold - pixels with >25 difference become white (255)
    _, threshold = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    
    # Count how many pixels changed
    changed_pixels = cv2.countNonZero(threshold)
    total_pixels = gray.shape[0] * gray.shape[1]
    # Calculate percentage of changed pixels
    change_percent = changed_pixels / total_pixels
    
    # Return True if change exceeds sensitivity threshold
    return change_percent > SENSITIVITY

def detect_barcodes(frame):
    """Detects and decodes any QR/barcodes in the frame"""
    # pyzbar's decode function finds and reads barcodes
    barcodes = decode(frame)
    if barcodes:
        # Return list of decoded barcode contents
        return [barcode.data.decode("utf-8") for barcode in barcodes]
    return []

def check_running_apps():
    """Checks if any forbidden applications are running (Windows only)"""
    try:
        # Run Windows tasklist command to get running processes
        output = os.popen('tasklist').read()
        # Check if any forbidden apps are in the tasklist
        return [app for app in FORBIDDEN_APPS if app in output]
    except:
        return []  # Return empty list if command fails

def log_event(event_type, details=""):
    """Records an event with timestamp and details"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    event = {
        "timestamp": timestamp,
        "type": event_type,
        "details": details
    }
    events.append(event)
    # Also print to console for immediate feedback
    print(f"[{timestamp}] ALERT: {event_type} - {details}")

def save_screenshot(frame, reason):
    """Saves screenshot with timestamp and reason in filename"""
    filename = f"{OUTPUT_DIR}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{reason}.jpg"
    cv2.imwrite(filename, frame)
    return filename

def main():
    global last_screenshot
    # Record when activity last occurred
    last_active = time.time()

    print("Starting exam monitoring. Press Ctrl+C to stop...")
    
    try:
        while True:
            # 1. Capture current screen
            current_frame = capture_screen()
            
            # 2. Detect if screen content changed significantly
            changed = detect_changes(current_frame, last_screenshot)
            if changed:
                # Update last active time
                last_active = time.time()
                # Save screenshot of the change
                filename = save_screenshot(current_frame, "change")
                log_event("SCREEN_CHANGE", f"Saved: {filename}")
                # Update last screenshot
                last_screenshot = current_frame
            
            # 3. Check if user has been idle too long
            if time.time() - last_active > MAX_IDLE_TIME:
                log_event("IDLE_TOO_LONG", f"{MAX_IDLE_TIME} seconds")
                # Reset idle timer
                last_active = time.time()
            
            # 4. Check for barcodes/QR codes
            barcodes = detect_barcodes(current_frame)
            if barcodes:
                log_event("BARCODE_DETECTED", f"Content: {barcodes}")
            
            # 5. Check for forbidden applications
            forbidden_running = check_running_apps()
            if forbidden_running:
                log_event("FORBIDDEN_APP", f"Running: {forbidden_running}")
            
            # Wait before next capture
            time.sleep(CAPTURE_INTERVAL)
    
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    
    # Save all events to JSON report
    with open(f"{OUTPUT_DIR}/report.json", "w") as f:
        json.dump(events, f, indent=2)
    print(f"Report saved to {OUTPUT_DIR}/report.json")

if __name__ == "__main__":
    main()