from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np

from app.main import app
from app.services.tts import TTSService

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Zonos Text-to-Speech API"

def test_get_models():
    with patch('app.main.get_tts_service') as mock_get_service:
        mock_service = Mock(spec=TTSService)
        mock_service.get_model_names.return_value = [
            "Zyphra/Zonos-v0.1-transformer",
            "Zyphra/Zonos-v0.1-hybrid"
        ]
        mock_get_service.return_value = mock_service
        
        response = client.get("/models")
        assert response.status_code == 200
        assert "models" in response.json()
        assert len(response.json()["models"]) == 2

def test_get_languages():
    with patch('app.main.get_tts_service') as mock_get_service:
        mock_service = Mock(spec=TTSService)
        mock_service.get_supported_languages.return_value = ["en-us", "es-es"]
        mock_get_service.return_value = mock_service
        
        response = client.get("/languages")
        assert response.status_code == 200
        assert "languages" in response.json()
        assert len(response.json()["languages"]) == 2

def test_synthesize_speech():
    with patch('app.main.get_tts_service') as mock_get_service:
        mock_service = Mock(spec=TTSService)
        # Mock audio generation
        mock_service.generate_audio.return_value = (
            (44100, np.zeros(44100).astype(np.float32)),  # 1 second of silence
            42  # seed
        )
        mock_get_service.return_value = mock_service
        
        test_request = {
            "model_choice": "Zyphra/Zonos-v0.1-transformer",
            "text": "Test speech",
            "language": "en-us",
            "emotion_values": [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2],
            "vq_score": 0.78,
            "cfg_scale": 2.0,
            "min_p": 0.15,
            "randomize_seed": False,
            "seed": 42
        }
        
        response = client.post("/synthesize", json=test_request)
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers["x-seed"] == "42"