
"""
AI Module for Autonomous Vehicle
Contains computer vision, object detection, and decision making algorithms.
"""

from .object_detection import LaneDetector
from .decision_maker import DecisionMaker
from .neural_network import NeuralNetwork
from .sensor_fusion import SensorFusion

__version__ = "1.0.0"
__all__ = ['LaneDetector', 'DecisionMaker', 'NeuralNetwork', 'SensorFusion']
