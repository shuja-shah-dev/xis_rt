# test_service.py
"""
Simple test script to isolate GenICam service issues
"""

import sys
import os
import traceback

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import psutil
        print("✓ psutil imported successfully")
    except ImportError as e:
        print(f"✗ psutil import failed: {e}")
        return False
    
    try:
        from harvesters.core import Harvester
        print("✓ Harvesters imported successfully")
    except ImportError as e:
        print(f"✗ Harvesters import failed: {e}")
        return False
    
    try:
        import pycuda.driver as cuda
        cuda.init()
        print("✓ CUDA initialized successfully")
        device_count = cuda.Device.count()
        print(f"  Found {device_count} CUDA device(s)")
    except Exception as e:
        print(f"✗ CUDA initialization failed: {e}")
        return False
    
    try:
        import tensorrt as trt
        print("✓ TensorRT imported successfully")
        print(f"  TensorRT version: {trt.__version__}")
    except ImportError as e:
        print(f"✗ TensorRT import failed: {e}")
        return False
    
    return True

def test_genicam_service():
    """Test GenICam service initialization"""
    print("\nTesting GenICam service initialization...")
    
    try:
        # Add the app directory to Python path if needed
        sys.path.append('.')
        
        from app.core.genicam_service import GenICamService
        print("✓ GenICamService import successful")
        
        # Test creating an instance
        service = GenICamService()
        print("✓ GenICamService instance created successfully")
        
        # Test getting status
        status = service.get_status()
        print("✓ Service status retrieved successfully")
        print(f"  Status: {status}")
        
        # Test cleanup
        service.cleanup_all()
        print("✓ Service cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"✗ GenICam service test failed: {e}")
        traceback.print_exc()
        return False

def test_camera_detection():
    """Test camera detection without starting streams"""
    print("\nTesting camera detection...")
    
    try:
        sys.path.append('.')
        from geni_inference import TensorRTGenICamDetector
        
        # Just test initialization, don't start streaming
        detector = TensorRTGenICamDetector()
        print("✓ TensorRT detector created successfully")
        
        if detector.h and len(detector.h.device_info_list) > 0:
            print(f"✓ Found {len(detector.h.device_info_list)} camera(s)")
            for i, device in enumerate(detector.h.device_info_list):
                vendor = getattr(device, 'vendor', 'Unknown')
                model = getattr(device, 'model', 'Unknown')
                print(f"  Camera {i}: {vendor} {model}")
        else:
            print("! No cameras found")
        
        # Cleanup
        detector._final_cleanup()
        print("✓ Detector cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Camera detection test failed: {e}")
        traceback.print_exc()
        return False

def test_flask_app():
    """Test Flask app creation"""
    print("\nTesting Flask app creation...")
    
    try:
        sys.path.append('.')
        from app import create_app
        
        app = create_app()
        print("✓ Flask app created successfully")
        
        # Test if we can get the service
        from app import get_genicam_service
        service = get_genicam_service()
        
        if service:
            print("✓ GenICam service available in app context")
            status = service.get_status()
            print(f"  Service status: {status}")
        else:
            print("! GenICam service not available in app context")
        
        return True
        
    except Exception as e:
        print(f"✗ Flask app test failed: {e}")
        traceback.print_exc()
        return False

def test_model_files():
    """Check for model files"""
    print("\nChecking model files...")
    
    models_dir = "models"
    if os.path.exists(models_dir):
        engine_files = [f for f in os.listdir(models_dir) if f.endswith('.engine')]
        if engine_files:
            print(f"✓ Found {len(engine_files)} TensorRT engine file(s):")
            for f in engine_files:
                file_path = os.path.join(models_dir, f)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"  - {f} ({file_size:.1f} MB)")
        else:
            print("! No TensorRT engine files found in models directory")
    else:
        print("! Models directory does not exist")

def main():
    """Run all tests"""
    print("=== GenICam Service Test Suite ===\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("Model Files Check", test_model_files),
        ("Camera Detection", test_camera_detection),
        ("GenICam Service", test_genicam_service),
        ("Flask App", test_flask_app),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your setup should work.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)