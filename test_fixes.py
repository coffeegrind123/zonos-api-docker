#!/usr/bin/env python3
"""
Test script to verify the fixes work correctly
"""
import requests
import json

def test_api_endpoints():
    base_url = "http://localhost:8181"
    
    print("🔍 Testing API endpoints...")
    
    try:
        # Test root endpoint
        response = requests.get(f"{base_url}/", timeout=5)
        print(f"✅ Root endpoint: {response.status_code}")
        
        # Test models endpoint
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Models available: {models.get('models', [])}")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            
        # Test a simple TTS request with transformer model
        tts_request = {
            "text": "Hello world test",
            "model": "zonos-v0.1-transformer"
        }
        
        response = requests.post(f"{base_url}/v1/audio/text-to-speech", 
                               json=tts_request, timeout=30)
        if response.status_code == 200:
            print(f"✅ Transformer model TTS: Success (audio length: {len(response.content)} bytes)")
        else:
            print(f"❌ Transformer model TTS failed: {response.status_code}")
            print(response.text)
        
        # Test hybrid model
        tts_request["model"] = "zonos-v0.1-hybrid"
        response = requests.post(f"{base_url}/v1/audio/text-to-speech", 
                               json=tts_request, timeout=30)
        if response.status_code == 200:
            print(f"✅ Hybrid model TTS: Success (audio length: {len(response.content)} bytes)")
        else:
            print(f"❌ Hybrid model TTS failed: {response.status_code}")
            print(response.text)
            
        print("\n🎉 API testing complete!")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("Make sure the container is running on port 8181")

if __name__ == "__main__":
    test_api_endpoints()