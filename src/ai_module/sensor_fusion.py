import numpy as np
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorFusion:
    def __init__(self):
        self.sensor_history = []
        self.max_history = 10
        self.weights = {
            'ultrasonic': 0.4,
            'camera': 0.3,
            'ir_sensors': 0.2,
            'gyroscope': 0.1
        }
    
    def fuse_data(self, ultrasonic_data: Dict, camera_data: List, ir_data: Dict, gyro_data: Dict = None) -> Dict:
        """Fuse data from multiple sensors"""
        try:
            fused_data = {}
            
            # Fuse distance data (ultrasonic is primary)
            if ultrasonic_data and 'distance' in ultrasonic_data:
                fused_data['distance'] = ultrasonic_data['distance']
            elif camera_data:
                # Estimate distance from camera (simplified)
                fused_data['distance'] = self._estimate_distance_from_camera(camera_data)
            else:
                fused_data['distance'] = 100.0  # Default safe distance
            
            # Fuse position/lane data
            if camera_data:
                fused_data['lane_position'] = self._calculate_lane_position(camera_data)
                fused_data['obstacles'] = self._detect_obstacles_from_camera(camera_data)
            else:
                fused_data['lane_position'] = 0.0
                fused_data['obstacles'] = []
            
            # Fuse IR sensor data for line following
            if ir_data:
                fused_data['line_position'] = self._calculate_line_position(ir_data)
                fused_data['ir_confidence'] = self._calculate_ir_confidence(ir_data)
            else:
                fused_data['line_position'] = 0.0
                fused_data['ir_confidence'] = 0.0
            
            # Add gyroscope data if available
            if gyro_data:
                fused_data['orientation'] = gyro_data.get('orientation', 0.0)
                fused_data['acceleration'] = gyro_data.get('acceleration', 0.0)
            
            # Calculate overall confidence
            fused_data['confidence'] = self._calculate_overall_confidence(
                ultrasonic_data, camera_data, ir_data, gyro_data
            )
            
            # Store in history
            self._update_history(fused_data)
            
            return fused_data
            
        except Exception as e:
            logger.error(f"Error in sensor fusion: {e}")
            return {'distance': 100.0, 'lane_position': 0.0, 'confidence': 0.0}
    
    def _estimate_distance_from_camera(self, camera_data: List) -> float:
        """Estimate distance from camera data (simplified)"""
        # This is a simplified implementation
        # In real implementation, you would use object size or stereo vision
        if not camera_data:
            return 100.0
        
        # Mock implementation - return a safe distance
        return 80.0
    
    def _calculate_lane_position(self, camera_data: List) -> float:
        """Calculate lane position from camera data"""
        if not camera_data:
            return 0.0
        
        # Mock implementation - return center position
        return 0.0
    
    def _detect_obstacles_from_camera(self, camera_data: List) -> List:
        """Detect obstacles from camera data"""
        return []  # Mock implementation
    
    def _calculate_line_position(self, ir_data: Dict) -> float:
        """Calculate line position from IR sensors"""
        left_ir = ir_data.get('left', 0)
        right_ir = ir_data.get('right', 0)
        
        if left_ir == 0 and right_ir == 0:
            return 0.0
        
        # Normalize and calculate position (-1.0 to 1.0)
        total = left_ir + right_ir
        if total == 0:
            return 0.0
            
        position = (right_ir - left_ir) / total
        return np.clip(position, -1.0, 1.0)
    
    def _calculate_ir_confidence(self, ir_data: Dict) -> float:
        """Calculate confidence in IR sensor data"""
        left_ir = ir_data.get('left', 0)
        right_ir = ir_data.get('right', 0)
        
        # Higher values indicate better detection
        max_ir_value = 1023  # Assuming 10-bit ADC
        avg_reading = (left_ir + right_ir) / 2
        confidence = avg_reading / max_ir_value
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _calculate_overall_confidence(self, ultrasonic, camera, ir, gyro) -> float:
        """Calculate overall confidence in fused data"""
        confidence = 0.0
        sensors_used = 0
        
        if ultrasonic:
            confidence += self.weights['ultrasonic']
            sensors_used += 1
        if camera:
            confidence += self.weights['camera']
            sensors_used += 1
        if ir:
            confidence += self.weights['ir_sensors']
            sensors_used += 1
        if gyro:
            confidence += self.weights['gyroscope']
            sensors_used += 1
        
        # Normalize by number of sensors used
        if sensors_used > 0:
            confidence /= sensors_used
        
        return confidence
    
    def _update_history(self, fused_data: Dict):
        """Update sensor history for filtering"""
        self.sensor_history.append(fused_data)
        if len(self.sensor_history) > self.max_history:
            self.sensor_history.pop(0)
    
    def get_filtered_data(self) -> Dict:
        """Get filtered sensor data using history"""
        if not self.sensor_history:
            return {}
        
        # Simple moving average filter
        filtered = {}
        for key in self.sensor_history[0].keys():
            if isinstance(self.sensor_history[0][key], (int, float)):
                values = [data[key] for data in self.sensor_history if key in data]
                filtered[key] = np.mean(values) if values else 0.0
            else:
                filtered[key] = self.sensor_history[-1][key]  # Use latest for non-numeric
        
        return filtered

def main():
    """Test sensor fusion"""
    fusion = SensorFusion()
    
    # Test data
    ultrasonic = {'distance': 45.5}
    camera_data = [{'type': 'lane', 'position': 0.2}]
    ir_data = {'left': 450, 'right': 650}
    gyro_data = {'orientation': 5.2, 'acceleration': 0.1}
    
    fused = fusion.fuse_data(ultrasonic, camera_data, ir_data, gyro_data)
    print("Fused Sensor Data:", fused)

if __name__ == "__main__":
    main()