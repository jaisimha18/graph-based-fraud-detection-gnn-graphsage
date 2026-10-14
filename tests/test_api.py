"""
Unit tests for FastAPI endpoints (src/api/main.py).
"""
import io
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_model_info_endpoint(client):
    """Verify /api/model_info returns active model metadata and threshold."""
    response = client.get("/api/model_info")
    assert response.status_code == 200
    data = response.json()
    assert "active_model" in data
    assert "threshold" in data
    assert "device" in data

def test_ablation_endpoint(client):
    """Verify /api/ablation returns a dictionary (whether empty or populated)."""
    response = client.get("/api/ablation")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)

def test_predict_endpoint_missing_columns(client):
    """Verify /api/predict returns an error when CSV missing required headers."""
    invalid_csv = "col1,col2\nval1,val2"
    file_payload = {"file": ("test.csv", io.BytesIO(invalid_csv.encode("utf-8")), "text/csv")}
    response = client.post("/api/predict", files=file_payload)
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert "Missing required columns" in data["error"]

def test_predict_endpoint_valid_mini_data(client):
    """Verify /api/predict parses valid transaction rows and produces graph nodes & links."""
    valid_csv = (
        "Timestamp,From Bank,Account,To Bank,Account.1,Amount Received,Amount Paid,"
        "Receiving Currency,Payment Currency,Payment Format,Is Laundering\n"
        "2022/09/01 01:20,10,1001,12,1002,5000.0,5000.0,US Dollar,US Dollar,Wire,0\n"
        "2022/09/01 02:15,12,1002,15,1003,4900.0,4900.0,US Dollar,US Dollar,Wire,1\n"
        "2022/09/01 03:00,15,1003,10,1001,4800.0,4800.0,US Dollar,US Dollar,Wire,1\n"
    )
    file_payload = {"file": ("mini.csv", io.BytesIO(valid_csv.encode("utf-8")), "text/csv")}
    response = client.post("/api/predict", files=file_payload)
    assert response.status_code == 200
    data = response.json()
    assert "error" not in data
    assert "nodes" in data
    assert "links" in data
    assert "threshold" in data
    assert len(data["nodes"]) == 3
    assert len(data["links"]) == 3
