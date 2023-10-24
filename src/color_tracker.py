"""
color_tracker.py
----------------
This script implements object tracking in real-time video using color contours.
It uses basic OpenCV functions for a lightweight and fast implementation.
"""

import cv2
import numpy as np
import logging

# Logging
logging.basicConfig(level=logging.INFO)

class Contour:
    """
    Class to store contour information with color

    Usage Example:
    --------------
    my_contour = Contour(contour, 'blue')
    """
    def __init__(self, contour, color):
        """
        Initializes the class.
        :param contour: Contour found in a frame.
        :param color: color of the target to be detected.
        """
        self.contour = contour
        self.color = color

class ContourDetector:
    """
    Class to detect contours in a video frame

    Usage Example:
    --------------
    detector = ContourDetector(color_ranges)
    detected_contours = detector.detect(frame)
    """
    def __init__(self, color_ranges, min_area=3000):
        """
        Initializes the class.
        :param color_ranges: list of tuples with name, lower and upper bounds for color range detection.
        :param min_area: minimum area of a contour to be considered for plotting.   
        """
        self.color_ranges = color_ranges
        self.min_area = min_area
    
    def detect(self, frame):
        """Method to detect contours in a frame
        :param frame: frame on which to find color countours.
        :return: List of contours of detected objects.
        """
        if frame.size == 0:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert frame from BGR to HSV color space
        contours = []
        for color, lower, upper in self.color_ranges:  # Iterate over each color range
            mask = cv2.inRange(hsv, lower, upper) # Create a mask for each color range
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours in the mask
            for c in cnts:   # Iterate over each contour
                if cv2.contourArea(c) > self.min_area:  # Filter contours by area
                    contours.append(Contour(cv2.convexHull(c), color)) # Append convex hull of contour with color to contours list
        return contours

class ContourDrawer:
    """
    Class to draw contours on a video frame

    Usage Example:
    --------------
    drawer = ContourDrawer()
    drawer.draw(frame, detected_contours)
    """
    def draw(self, frame, contours):
        """
        Function to draw contours on a frame
        :param frame: Video frame.
        :param contours: List of contours to be drawn on the frame.
        :return: void
        """
        for contour in contours: # Iterate over each contour and draw it on the frame
            cv2.drawContours(frame, [contour.contour], 0, (0,255,255), 3)


class Visualizer:
    """
    Class to read video frames and detect and draw contours

    Usage Example:
    --------------
    visualizer = Visualizer(0, detector, drawer)
    visualizer.run()
    """
    def __init__(self, channel, detector, drawer):
        """
        Initializes the class.
        :param channel: Channel number for video capture (e.g. 0 for default webcam).
        :param detector: Instance of the ContourDetector class.
        :param drawer: Instance of the ContourDrawer class.
        """
        self.channel = channel
        self.detector = detector
        self.drawer = drawer

    def run(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and show output.
        :return: void
        """

        try:
            capture = cv2.VideoCapture(self.channel) # Open video capture object   
            if not capture.isOpened():
                raise Exception("Could not open video device") 
        
            while capture.isOpened(): # Run until stream is out of frames or stop is True
                ret, frame = capture.read() # Read next frame
                if not ret:
                    break

                contours = self.detector.detect(frame) # Detect contours in the frame
                self.drawer.draw(frame, contours) # Draw contours on the frame

                cv2.imshow('frame', frame) # Show the output frame


                key = cv2.waitKey(1) & 0xff # Check for user key press
                if key == ord('q'):
                    break
                # Add other key bindings as needed     
            
        except Exception as e:
            logging.error(f"An error occurred: {e}")   
        finally:
            cv2.destroyAllWindows()
            capture.release() # When everything is done, release the capture


# Adjust these parameters according to your needs
color_ranges = [('blue', np.array([100,100,20]), np.array([125,255,255]))]

detector = ContourDetector(color_ranges)
drawer = ContourDrawer()
visualizer = Visualizer(0, detector, drawer)
visualizer.run()
