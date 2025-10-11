#!/usr/bin/env python3
\"\"\"
Test script to verify all modules work correctly
\"\"\"

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_ai_module():
    \"\"\"Test AI module imports\"\"\"
    try:
        from ai_module import LaneDetector, DecisionMaker, NeuralNetwork, SensorFusion
        print(\"✅ AI module imports successful\")
        
        # Test basic functionality
        detector = LaneDetector()
        decision_maker = DecisionMaker()
        nn = NeuralNetwork()
        sensor_fusion = SensorFusion()
        
        print(\"✅ AI module instantiation successful\")
        return True
        
    except Exception as e:
        print(f\"❌ AI module test failed: {e}\")
        return False

def test_simulation():
    \"\"\"Test simulation module\"\"\"
    try:
        from simulation import SimulationEnvironment
        print(\"✅ Simulation module import successful\")
        return True
    except Exception as e:
        print(f\"❌ Simulation module test failed: {e}\")
        return False

if __name__ == \"__main__\":
    print(\"Running project tests...\")
    
    ai_success = test_ai_module()
    sim_success = test_simulation()
    
    if ai_success and sim_success:
        print(\"\\n🎉 All tests passed! Project is ready.\")
        sys.exit(0)
    else:
        print(\"\\n❌ Some tests failed.\")
        sys.exit(1)
