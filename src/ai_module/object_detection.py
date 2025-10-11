import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LaneDetector:
    def __init__(self):
        self.previous_lanes = []
        self.canny_low = 50
        self.canny_high = 150
        self.hough_threshold = 50
        self.min_line_length = 50
        self.max_line_gap = 30
        
    def detect_lanes(self, image: np.ndarray) -> Optional[List]:
        """Detect lane lines in the image using computer vision"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blur, self.canny_low, self.canny_high)
            
            # Create region of interest mask
            height, width = edges.shape
            mask = np.zeros_like(edges)
            polygon = np.array([[
                (0, height),
                (width // 2, height // 2),
                (width, height)
            ]], np.int32)
            cv2.fillPoly(mask, polygon, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # Detect lines using Hough Transform
            lines = cv2.HoughLinesP(
                masked_edges, 
                1, 
                np.pi/180, 
                self.hough_threshold,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap
            )
            
            if lines is not None:
                self.previous_lanes = lines
                return lines
            else:
                return self.previous_lanes if self.previous_lanes else []
                
        except Exception as e:
            logger.error(f"Error in lane detection: {e}")
            return self.previous_lanes if self.previous_lanes else []
    
    def calculate_steering_angle(self, lines: List, image_width: int) -> float:
        """Calculate steering angle based on detected lanes"""
        if not lines:
            return 0.0
            
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
            
            if abs(slope) < 0.5:  # Ignore horizontal lines
                continue
                
            if slope < 0:  # Left lane
                left_lines.append(line)
            else:  # Right lane
                right_lines.append(line)
        
        # Calculate average position for left and right lanes
        left_x = np.mean([(line[0][0] + line[0][2]) / 2 for line in left_lines]) if left_lines else image_width * 0.25
        right_x = np.mean([(line[0][0] + line[0][2]) / 2 for line in right_lines]) if right_lines else image_width * 0.75
        
        # Calculate center point between lanes
        lane_center = (left_x + right_x) / 2
        image_center = image_width / 2
        
        # Calculate steering angle (-30 to 30 degrees)
        deviation = (lane_center - image_center) / image_center
        steering_angle = deviation * 30
        
        return np.clip(steering_angle, -30, 30)

class ObjectDetector:
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        
    def detect_obstacles(self, image: np.ndarray) -> List[Tuple]:
        """Detect obstacles in the image using Haar cascades"""
        obstacles = []
        
        try:
            # Load pre-trained car classifier
            car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect cars
            cars = car_cascade.detectMultiScale(gray, 1.1, 3)
            
            for (x, y, w, h) in cars:
                obstacles.append(('car', x, y, w, h))
                
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            
        return obstacles

def main():
    """Test lane detection"""
    detector = LaneDetector()
    print("Lane detector initialized successfully!")
    
    # Test with a sample image (you would replace this with camera input)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    lanes = detector.detect_lanes(test_image)
    print(f"Detected {len(lanes) if lanes else 0} lanes")
    
if __name__ == "__main__":
    main()