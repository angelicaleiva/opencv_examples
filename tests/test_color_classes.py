import unittest
import cv2
import numpy as np
from src.color_tracker import Contour, ContourDetector, ContourDrawer  # Make sure these imports are correct based on your file structure

class TestContour(unittest.TestCase):
    """
    Test case for the Contour class in color_tracker.py
    """

    def test_initialization(self):
        """
        Test the __init__ method of the Contour class.
        """
        # Create a contour object
        contour = Contour(np.array([[0, 0], [1, 1]]), 'blue')

        # Check if the color attribute is set correctly
        self.assertEqual(contour.color, 'blue')

        # Check if the contour attribute is set correctly
        np.testing.assert_array_equal(contour.contour, np.array([[0, 0], [1, 1]]))

class TestContourDetector(unittest.TestCase):
    """
    Test case for the ContourDetector class in color_tracker.py
    """

    def setUp(self):
        """
        Setup that runs before each test case.
        """
        # Define a blue color range for testing
        self.color_ranges = [('blue', np.array([100, 100, 20]), np.array([125, 255, 255]))]
        
        # Create a ContourDetector object
        self.detector = ContourDetector(self.color_ranges)

    def test_initialization(self):
        """
        Test the __init__ method of the ContourDetector class.
        """
        # Check if the min_area attribute is set correctly
        self.assertEqual(self.detector.min_area, 3000)

        # Check if the color_ranges attribute is set correctly
        self.assertEqual(self.detector.color_ranges, self.color_ranges)

    def test_detect(self):
        """
        Test the detect method of the ContourDetector class.
        """
        # Create a black frame of shape 100x100x3
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Add a blue square in the frame
        frame[20:80, 20:80] = [255, 0, 0]  
        
        # Use the detect method to find contours in the frame
        contours = self.detector.detect(frame)
        
        # Check if at least one contour is detected
        self.assertTrue(len(contours) > 0)

    def test_detect_with_empty_frame(self):
        """
        Test the detect method with an empty frame.
        """
        frame = np.zeros((0, 0, 3), dtype=np.uint8)
        contours = self.detector.detect(frame)
        self.assertEqual(len(contours), 0)


class TestContourDrawer(unittest.TestCase):
    """
    Test case for the ContourDrawer class in color_tracker.py
    """

    def test_draw(self):
        """
        Test the draw method of the ContourDrawer class.
        """
        # Create a ContourDrawer object
        drawer = ContourDrawer()

        # Create a black frame of shape 100x100x3
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create a test Contour object
        contour = Contour(np.array([[25, 25], [25, 75], [75, 75], [75, 25]]), 'blue')

        # Use the draw method to draw the contour on the frame
        drawer.draw(frame, [contour])

        # Here you could add assertions to verify that the contour is drawn correctly on the frame

if __name__ == '__main__':
    unittest.main()