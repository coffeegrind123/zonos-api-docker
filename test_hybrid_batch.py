#!/usr/bin/env python3
"""
Test script to verify hybrid model support and batch processing functionality
"""
import requests
import json
import time

def test_hybrid_and_batch_functionality():
    base_url = "http://localhost:8181"
    
    print("🔍 Testing Hybrid Model Support and Batch Processing...")
    
    try:
        # Test 1: Check available models
        print("\n1. Testing available models...")
        response = requests.get(f"{base_url}/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Available models: {models.get('models', [])}")
            has_hybrid = "Zyphra/Zonos-v0.1-hybrid" in models.get('models', [])
            has_transformer = "Zyphra/Zonos-v0.1-transformer" in models.get('models', [])
            print(f"   - Hybrid model available: {has_hybrid}")
            print(f"   - Transformer model available: {has_transformer}")
        else:
            print(f"❌ Failed to get models: {response.status_code}")
            return
        
        # Test 2: Test transformer model (should work)
        print("\n2. Testing transformer model...")
        transformer_request = {
            "text": "Hello, this is a test of the transformer model.",
            "model": "zonos-v0.1-transformer"
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/v1/audio/text-to-speech", 
                               json=transformer_request, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            print(f"✅ Transformer model: Success (audio: {len(response.content)} bytes, time: {end_time-start_time:.1f}s)")
        else:
            print(f"❌ Transformer model failed: {response.status_code}")
            print(response.text)
        
        # Test 3: Test hybrid model (should work with [compile] support)
        print("\n3. Testing hybrid model...")
        hybrid_request = {
            "text": "Hello, this is a test of the hybrid model with SSM backbone.",
            "model": "zonos-v0.1-hybrid"
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/v1/audio/text-to-speech", 
                               json=hybrid_request, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            print(f"✅ Hybrid model: Success (audio: {len(response.content)} bytes, time: {end_time-start_time:.1f}s)")
        else:
            print(f"❌ Hybrid model failed: {response.status_code}")
            print(response.text)
        
        # Test 4: Test batch processing with transformer model
        print("\n4. Testing batch processing...")
        batch_request = {
            "model_choice": "Zyphra/Zonos-v0.1-transformer",
            "texts": [
                "This is the first sentence for batch testing.",
                "Here is the second sentence in the batch.",
                "And finally, this is the third sentence to complete the batch test."
            ],
            "language": "en-us",
            "speaking_rate": 15.0,
            "randomize_seed": True
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/batch/synthesize", 
                               json=batch_request, timeout=120)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch processing: Success!")
            print(f"   - Batch size: {result.get('batch_size', 0)}")
            print(f"   - Model used: {result.get('model_used', 'unknown')}")
            print(f"   - True batch processing: {result.get('batch_processing_used', False)}")
            print(f"   - Total time: {end_time-start_time:.1f}s")
            print(f"   - Average per text: {(end_time-start_time)/len(batch_request['texts']):.1f}s")
            
            # Check individual results
            results = result.get('results', [])
            for i, res in enumerate(results):
                audio_size = len(res.get('audio_base64', '')) * 3 // 4  # Rough size estimate
                print(f"   - Text {i+1}: {audio_size} bytes, seed {res.get('seed', 'unknown')}")
        else:
            print(f"❌ Batch processing failed: {response.status_code}")
            print(response.text)
        
        # Test 5: Performance comparison - sequential vs batch
        print("\n5. Performance comparison (3 texts)...")
        test_texts = [
            "Performance test sentence one.",
            "Performance test sentence two.", 
            "Performance test sentence three."
        ]
        
        # Sequential processing
        print("   Testing sequential processing...")
        sequential_start = time.time()
        sequential_success = 0
        for i, text in enumerate(test_texts):
            req = {"text": text, "model": "zonos-v0.1-transformer"}
            resp = requests.post(f"{base_url}/v1/audio/text-to-speech", 
                               json=req, timeout=30)
            if resp.status_code == 200:
                sequential_success += 1
        sequential_end = time.time()
        sequential_time = sequential_end - sequential_start
        
        # Batch processing
        print("   Testing batch processing...")
        batch_start = time.time()
        batch_req = {
            "model_choice": "Zyphra/Zonos-v0.1-transformer",
            "texts": test_texts,
            "randomize_seed": True
        }
        batch_resp = requests.post(f"{base_url}/batch/synthesize", 
                                 json=batch_req, timeout=60)
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_success = batch_resp.status_code == 200
        
        print(f"\n📊 Performance Results:")
        print(f"   Sequential: {sequential_time:.1f}s ({sequential_success}/{len(test_texts)} successful)")
        print(f"   Batch: {batch_time:.1f}s ({'successful' if batch_success else 'failed'})")
        
        if batch_success and sequential_success > 0:
            speedup = sequential_time / batch_time
            print(f"   🚀 Batch speedup: {speedup:.1f}x faster!")
            
            if speedup > 2.0:
                print("   ✅ Excellent batch performance!")
            elif speedup > 1.5:
                print("   ✅ Good batch performance!")
            else:
                print("   ⚠️ Batch performance could be better")
        
        print("\n🎉 Hybrid model and batch processing testing complete!")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("Make sure the container is running on port 8181")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_hybrid_and_batch_functionality()