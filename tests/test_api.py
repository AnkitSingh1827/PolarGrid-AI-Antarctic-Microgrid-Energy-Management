"""
PolarGrid AI — API Integration Tests
Tests all FastAPI REST endpoints: health, forecast, dispatch, metrics, alerts, optimize.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["project"] == "PolarGrid AI"
    assert "version" in data


def test_api_forecast(client):
    response = client.post("/forecast", json={"hours_ahead": 24, "model_type": "xgboost"})
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "xgboost"
    assert len(data["forecasts"]) == 24
    first_pt = data["forecasts"][0]
    assert "load_kw" in first_pt
    assert "solar_kw" in first_pt
    assert "wind_kw" in first_pt


def test_api_dispatch_rule_based(client):
    payload = {
        "load_kw": 60.0,
        "solar_gen_kw": 15.0,
        "wind_gen_kw": 20.0,
        "temperature": -15.0,
        "battery_soc": 0.65,
        "controller_mode": "rule_based"
    }
    response = client.post("/dispatch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "battery_power_kw" in data
    assert "diesel_generation_kw" in data
    assert "diesel_status" in data
    assert data["diesel_status"] in (0, 1)


def test_api_dispatch_rl_ppo(client):
    payload = {
        "load_kw": 60.0,
        "solar_gen_kw": 15.0,
        "wind_gen_kw": 20.0,
        "temperature": -15.0,
        "battery_soc": 0.65,
        "controller_mode": "rl_ppo"
    }
    response = client.post("/dispatch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "battery_power_kw" in data
    assert "diesel_generation_kw" in data


def test_api_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "fuel_savings" in data
    assert "renewable_utilization" in data
    assert "reliability" in data
    assert "battery" in data
    assert "cost" in data


def test_api_alerts(client):
    response = client.get("/alerts")
    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data
    assert "count" in data
    assert isinstance(data["alerts"], list)


def test_api_optimize(client):
    response = client.post("/optimize", json={"horizon_hours": 12, "solver": "glpk"})
    assert response.status_code == 200
    data = response.json()
    assert "solved_optimally" in data
    assert "solver_used" in data
    assert "schedule" in data
    assert len(data["schedule"]) == 12
    assert "total_cost" in data
