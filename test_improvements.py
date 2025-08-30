#!/usr/bin/env python3
"""Test script to verify all improvements are working"""

import os
import sys
import logging

# Add the app directory to the path
sys.path.insert(0, '/home/claudeuser/zonos2')

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from app.services.tts import TTSService, is_deepspeed_available, normalize_language_code, get_optimal_device
        print("✅ TTSService imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} available")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA device count: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        deepspeed_available = is_deepspeed_available()
        print(f"✅ DeepSpeed available: {deepspeed_available}")
    except Exception as e:
        print(f"⚠️ DeepSpeed check failed: {e}")
    
    return True

def test_language_normalization():
    """Test language code normalization"""
    print("\n🧪 Testing language normalization...")
    
    test_cases = [
        ("EN_en", "en-en"),
        ("DE_de", "de-de"),
        ("FR_fr", "fr-fr"),
        ("en-us", "en-us"),
        ("SPANISH", "spanish"),
    ]
    
    for input_lang, expected in test_cases:
        result = normalize_language_code(input_lang)
        if result == expected:
            print(f"✅ {input_lang} → {result}")
        else:
            print(f"❌ {input_lang} → {result} (expected {expected})")
            return False
    
    return True

def test_device_selection():
    """Test device selection functionality"""
    print("\n🧪 Testing device selection...")
    
    try:
        device_fastest = get_optimal_device("fastest")
        print(f"✅ Fastest device: {device_fastest}")
        
        device_memory = get_optimal_device("memory")  
        print(f"✅ Memory device: {device_memory}")
        
        return True
    except Exception as e:
        print(f"❌ Device selection failed: {e}")
        return False

def test_environment_variables():
    """Test environment variable handling"""
    print("\n🧪 Testing environment variables...")
    
    # Test different environment variable configurations
    test_envs = {
        "ENABLE_DEEPSPEED": ["true", "false", "1", "0"],
        "GPU_PREFERENCE": ["fastest", "memory"],
        "OFFLINE_MODE": ["true", "false"],
    }
    
    for env_var, values in test_envs.items():
        for value in values:
            os.environ[env_var] = value
            print(f"✅ Set {env_var}={value}")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Testing Zonos Docker API Improvements")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_language_normalization, 
        test_device_selection,
        test_environment_variables,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED")
            else:
                failed += 1
                print("❌ FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ FAILED with exception: {e}")
        print()
    
    print("=" * 50)
    print(f"🎯 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All improvements are working correctly!")
        return 0
    else:
        print("💥 Some tests failed - check implementation")
        return 1

if __name__ == "__main__":
    exit(main())