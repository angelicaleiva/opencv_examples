"""
object_tracker.py
-----------------
This script uses the YOLO5 model to perform object tracking in real-time video.
"""

import cv2
import torch
import numpy as np
import logging
from torch import hub

logging.basicConfig(level=logging.INFO)

class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a webcam using Opencv.
    Usage Example:
    --------------
    detector = ObjectDetection(channel)
    detector.detect_objects()
    """

    def __init__(self, channel):
        """
        Initializes the class.
        :param channel: channel on which the webcam is working.
        """
        self.channel = channel # Store the channel parameter
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available, otherwise use CPU
        self.model = self.load_model().to(self.device) # Load the Yolo5 model from Pytorch Hub
        self.classes = self.model.names   # Store the class labels     
        self.x_shape, self.y_shape = None, None  # Store x and y dimensions for later use

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        with torch.no_grad():
            results = self.model(frame) # Make inferences on the frame
        labels = results.xyxyn[0][:, -1].cpu().detach().numpy() # Get the object labels
        cord = results.xyxyn[0][:, :-1].cpu().detach().numpy() # Get the object coordinates
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Plots boxes and labels on frame.
        :param results: inferences made by model
        :param frame: frame on which to make the plots
        :return: new frame with boxes and labels plotted.
        """
        labels, cord = results  # Get the object labels and coordinates from the model results
        n = len(labels)

        # Check if x_shape and y_shape have been set, and if not, set them to the frame's shape
        if self.x_shape is None and self.y_shape is None:
            self.x_shape, self.y_shape = frame.shape[1], frame.shape[0]

        bgr = (0, 255, 0) # color of the box
            
        # Loop through each label and coordinate
        for i in range(n):
            row = cord[i]
            label = self.class_to_label(labels[i])
            
            # If score is less than 0.2 or the label is not "person" we avoid making a prediction.
            if row[4] < 0.2 or label != "person":
                continue
            
                       
            # Get the coordinates of the box
            x1 = int(row[0]*self.x_shape)
            y1 = int(row[1]*self.y_shape)
            x2 = int(row[2]*self.x_shape)
            y2 = int(row[3]*self.y_shape)
            

            # Plot the boxes with the given color
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            
            # Add the label to the frame
            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2)
        
        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and show output.
        :return: void
        """    
        try:
            # Open the video stream
            capture = cv2.VideoCapture(self.channel)
            if not capture.isOpened():
                raise Exception("Could not open video device")

        
            # Keep looping while there are frames left in the stream
            while capture.isOpened():
                # Read the next frame
                ret, frame = capture.read()
                if not ret:
                    break
            
                # Get the model predictions for this frame
                results = self.score_frame(frame)
                
                # Plot the boxes and labels on the frame
                frame_with_boxes = self.plot_boxes(results, frame)
            
                # Visualize the frame
                cv2.imshow('frame', frame_with_boxes)
                
                # Check if the user has pressed the "q" key to stop the video
                if (cv2.waitKey(1) & 0xff == ord('q')):
                    break
        
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            
        finally:
            # Clean up
            capture.release()
            cv2.destroyAllWindows()


# Adjust these parameters according to your needs
CHANNEL = 0  # Webcam channel

person_detector = ObjectDetection(CHANNEL)
with torch.no_grad():
    person_detector()