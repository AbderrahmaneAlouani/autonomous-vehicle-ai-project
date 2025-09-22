# Simple autonomous-vehicle-ai-project
Building AI course project 
## Summary

This project implements a small-scale autonomous vehicle capable of:
- Lane following using computer vision
- Obstacle detection and avoidance
- Path planning and decision making
- Real-time control using AI algorithms

## Hardware Components

- Arduino Uno/Mega microcontroller
- Motor driver shield (L298N)
- Ultrasonic sensors for distance measurement
- IR sensors for edge detection
- Raspberry Pi/Jetson Nano as AI processing unit
- USB camera for computer vision
- Chassis, motors, and wheels

## Software Architecture

1. **Arduino Firmware**: Low-level motor control and sensor reading
2. **AI Module**: Python-based computer vision and decision making
3. **Communication**: Serial protocol between Arduino and AI unit

## Installation & Setup

### Arduino Setup
1. Upload the firmware to Arduino using the Arduino IDE
2. Connect all sensors and motors according to circuit diagrams in html file

### AI Module Setup
```bash
cd ai_module
pip install -r requirements.txt

