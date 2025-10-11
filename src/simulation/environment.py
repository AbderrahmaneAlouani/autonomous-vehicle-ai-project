import pygame
import numpy as np
import logging
from typing import Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulationEnvironment:
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.road_width = 400
        self.lane_width = self.road_width // 3
        self.car_position = [width // 2, height - 100]
        self.car_speed = 0
        self.car_angle = 0
        self.obstacles = []
        
        # Pygame initialization
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Autonomous Vehicle Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
    def reset(self):
        """Reset simulation to initial state"""
        self.car_position = [self.width // 2, self.height - 100]
        self.car_speed = 0
        self.car_angle = 0
        self.obstacles = []
        self._generate_obstacles()
        
    def _generate_obstacles(self):
        """Generate random obstacles on the road"""
        self.obstacles = []
        for _ in range(5):
            x = np.random.randint(self.width // 2 - self.road_width // 2 + 50, 
                                self.width // 2 + self.road_width // 2 - 50)
            y = np.random.randint(100, self.height - 200)
            self.obstacles.append([x, y, 30, 30])  # x, y, width, height
    
    def update(self, steering: float, speed: float):
        """Update simulation based on control inputs"""
        # Update car position based on speed and steering
        self.car_speed = np.clip(speed, 0, 100)
        self.car_angle += steering * 0.1
        
        # Move car forward
        rad_angle = np.radians(self.car_angle)
        self.car_position[0] += self.car_speed * np.sin(rad_angle) * 0.1
        self.car_position[1] -= self.car_speed * np.cos(rad_angle) * 0.1
        
        # Keep car within road boundaries
        road_left = self.width // 2 - self.road_width // 2
        road_right = self.width // 2 + self.road_width // 2
        
        self.car_position[0] = np.clip(self.car_position[0], road_left + 20, road_right - 20)
        self.car_position[1] = np.clip(self.car_position[1], 50, self.height - 50)
        
        # Check collisions
        collision = self._check_collisions()
        return collision
    
    def _check_collisions(self) -> bool:
        """Check if car collides with any obstacles"""
        car_rect = pygame.Rect(self.car_position[0] - 15, self.car_position[1] - 25, 30, 50)
        
        for obstacle in self.obstacles:
            obstacle_rect = pygame.Rect(obstacle[0], obstacle[1], obstacle[2], obstacle[3])
            if car_rect.colliderect(obstacle_rect):
                return True
        return False
    
    def get_sensor_data(self) -> dict:
        """Get simulated sensor data"""
        # Simulate ultrasonic distance
        distance = 100.0  # Default
        
        # Find closest obstacle in front
        for obstacle in self.obstacles:
            if abs(obstacle[0] - self.car_position[0]) < 50:  # Within lane
                if obstacle[1] > self.car_position[1]:  # In front
                    dist = obstacle[1] - self.car_position[1]
                    if dist < distance:
                        distance = dist
        
        # Simulate IR sensors for line following
        road_center = self.width // 2
        position_from_center = self.car_position[0] - road_center
        
        # IR values based on position (simplified)
        left_ir = max(0, 500 - abs(position_from_center + 50) * 5)
        right_ir = max(0, 500 - abs(position_from_center - 50) * 5)
        
        return {
            'DIST': distance,
            'LIR': left_ir,
            'RIR': right_ir
        }
    
    def render(self):
        """Render the simulation"""
        self.screen.fill((100, 100, 100))  # Gray background
        
        # Draw road
        road_left = self.width // 2 - self.road_width // 2
        pygame.draw.rect(self.screen, (50, 50, 50), 
                        (road_left, 0, self.road_width, self.height))
        
        # Draw lane markings
        for i in range(1, 3):
            lane_x = road_left + i * self.lane_width
            for y in range(0, self.height, 40):
                pygame.draw.rect(self.screen, (255, 255, 255), 
                               (lane_x - 2, y, 4, 20))
        
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, (255, 0, 0), 
                           (obstacle[0], obstacle[1], obstacle[2], obstacle[3]))
        
        # Draw car
        car_color = (0, 0, 255)  # Blue
        car_rect = pygame.Rect(self.car_position[0] - 15, self.car_position[1] - 25, 30, 50)
        pygame.draw.rect(self.screen, car_color, car_rect)
        
        # Draw car direction indicator
        end_x = self.car_position[0] + 30 * np.sin(np.radians(self.car_angle))
        end_y = self.car_position[1] - 30 * np.cos(np.radians(self.car_angle))
        pygame.draw.line(self.screen, (255, 255, 0), 
                        self.car_position, (end_x, end_y), 3)
        
        # Display sensor data
        sensor_data = self.get_sensor_data()
        info_text = f"Speed: {self.car_speed:.1f} | Steering: {self.car_angle:.1f}"
        info_surface = self.font.render(info_text, True, (255, 255, 255))
        self.screen.blit(info_surface, (10, 10))
        
        sensor_text = f"Dist: {sensor_data['DIST']:.1f} | LIR: {sensor_data['LIR']:.0f} | RIR: {sensor_data['RIR']:.0f}"
        sensor_surface = self.font.render(sensor_text, True, (255, 255, 255))
        self.screen.blit(sensor_surface, (10, 50))
        
        pygame.display.flip()
    
    def run_test(self):
        """Run a simple test of the simulation"""
        running = True
        self.reset()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Simple autonomous behavior for testing
            sensor_data = self.get_sensor_data()
            
            # Basic decision making
            if sensor_data['DIST'] < 50:
                steering = 30  # Turn right to avoid
                speed = 20
            else:
                # Follow center based on IR sensors
                steering = (sensor_data['RIR'] - sensor_data['LIR']) / 100
                speed = 50
            
            collision = self.update(steering, speed)
            
            if collision:
                logger.warning("Collision detected! Resetting...")
                self.reset()
            
            self.render()
            self.clock.tick(30)
        
        pygame.quit()

def main():
    """Test simulation environment"""
    env = SimulationEnvironment()
    env.run_test()

if __name__ == "__main__":
    main()