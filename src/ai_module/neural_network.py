import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrivingNN(nn.Module):
    """Neural network for autonomous driving decisions"""
    
    def __init__(self, input_size: int = 5, hidden_size: int = 64, output_size: int = 2):
        super(DrivingNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NeuralNetwork:
    def __init__(self, model_path: str = None):
        self.model = DrivingNN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        if model_path:
            self.load_model(model_path)
    
    def predict(self, sensor_data: np.ndarray) -> Tuple[float, float]:
        """Predict speed and steering based on sensor data"""
        try:
            self.model.eval()
            with torch.no_grad():
                # Convert sensor data to tensor
                input_tensor = torch.FloatTensor(sensor_data).unsqueeze(0)
                output = self.model(input_tensor)
                speed, steering = output[0].numpy()
                return float(speed), float(steering)
        except Exception as e:
            logger.error(f"Error in neural network prediction: {e}")
            return 0.0, 0.0
    
    def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int = 100):
        """Train the neural network"""
        try:
            self.model.train()
            inputs_tensor = torch.FloatTensor(inputs)
            targets_tensor = torch.FloatTensor(targets)
            
            for epoch in range(epochs):
                self.optimizer.zero_grad()
                outputs = self.model(inputs_tensor)
                loss = self.criterion(outputs, targets_tensor)
                loss.backward()
                self.optimizer.step()
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                    
        except Exception as e:
            logger.error(f"Error in training: {e}")
    
    def save_model(self, path: str):
        """Save trained model"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        try:
            self.model.load_state_dict(torch.load(path))
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.warning(f"Could not load model from {path}: {e}")

def main():
    """Test neural network"""
    nn = NeuralNetwork()
    
    # Test prediction with sample data
    test_data = np.array([100, 500, 300, 0.5, 0.2])  # [distance, left_ir, right_ir, lane_deviation, obstacle_close]
    speed, steering = nn.predict(test_data)
    print(f"Neural Network Prediction: Speed={speed:.2f}, Steering={steering:.2f}")

if __name__ == "__main__":
    main()