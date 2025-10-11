import logging
from typing import Dict, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionMaker:
    def __init__(self):
        self.safe_distance = 30.0  # cm
        self.max_speed = 100
        self.min_speed = 0
        self.current_speed = 0
        self.current_steering = 0
        
    def make_decision(self, sensor_data: Dict, lanes: list, obstacles: list) -> Tuple[int, int]:
        """
        Make driving decision based on sensor data and AI analysis
        
        Returns:
            tuple: (speed, steering_angle)
        """
        try:
            # Default values
            speed = 50
            steering = 0
            
            # Get sensor data
            distance = sensor_data.get('DIST', 100)
            left_ir = sensor_data.get('LIR', 0)
            right_ir = sensor_data.get('RIR', 0)
            
            # Obstacle avoidance priority
            if distance < self.safe_distance:
                speed = 0  # Emergency stop
                steering = 30  # Turn right to avoid
                logger.warning(f"Obstacle too close: {distance}cm - EMERGENCY STOP")
                
            # Lane following if no immediate obstacle
            elif lanes and len(lanes) > 0:
                from .object_detection import LaneDetector
                detector = LaneDetector()
                steering = detector.calculate_steering_angle(lanes, 640)  # Assuming 640px width
                speed = 70  # Moderate speed for lane following
                
            # IR sensor based line following
            else:
                # Simple line following logic based on IR sensors
                ir_difference = left_ir - right_ir
                steering = np.clip(ir_difference / 10, -30, 30)  # Scale IR difference to steering
                speed = 60
                
            # Adjust speed based on conditions
            if distance < 50:  # Slow down when approaching obstacles
                speed = max(20, speed * 0.5)
                
            # Smooth transitions
            self.current_speed = self._smooth_transition(self.current_speed, speed)
            self.current_steering = self._smooth_transition(self.current_steering, steering)
            
            return int(self.current_speed), int(self.current_steering)
            
        except Exception as e:
            logger.error(f"Error in decision making: {e}")
            return 0, 0  # Safe default: stop
    
    def _smooth_transition(self, current: float, target: float, factor: float = 0.3) -> float:
        """Smooth transition between values to avoid jerky movements"""
        return current + (target - current) * factor
    
    def update_parameters(self, safe_distance: float = None, max_speed: int = None):
        """Update decision parameters dynamically"""
        if safe_distance is not None:
            self.safe_distance = safe_distance
        if max_speed is not None:
            self.max_speed = max_speed

def main():
    """Test decision maker"""
    decision_maker = DecisionMaker()
    
    # Test sensor data
    test_sensor_data = {'DIST': 100, 'LIR': 500, 'RIR': 300}
    test_lanes = []  # No lanes detected
    test_obstacles = []  # No obstacles
    
    speed, steering = decision_maker.make_decision(test_sensor_data, test_lanes, test_obstacles)
    print(f"Decision: Speed={speed}, Steering={steering}")

if __name__ == "__main__":
    main()
