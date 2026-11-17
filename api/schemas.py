"""
PolarGrid AI — API Pydantic Schemas
Request and response models for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# === Health ===
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    project: str = "PolarGrid AI"
    timestamp: str


# === Forecast ===
class ForecastRequest(BaseModel):
    hours_ahead: int = Field(24, ge=1, le=168, description="Number of hours to forecast")
    model_type: str = Field("xgboost", description="Forecasting model type")


class ForecastPoint(BaseModel):
    timestamp: str
    load_kw: float
    solar_kw: float
    wind_kw: float


class ForecastResponse(BaseModel):
    model_type: str
    hours_ahead: int
    forecasts: List[ForecastPoint]
    metrics: Optional[Dict[str, Any]] = None


# === Dispatch ===
class DispatchRequest(BaseModel):
    load_kw: float = Field(..., ge=0, description="Current station load in kW")
    solar_gen_kw: float = Field(0.0, ge=0, description="Available solar generation in kW")
    wind_gen_kw: float = Field(0.0, ge=0, description="Available wind generation in kW")
    temperature: float = Field(-10.0, description="Current temperature in °C")
    battery_soc: float = Field(0.5, ge=0, le=1, description="Current battery SOC")
    controller_mode: str = Field("rule_based", description="Controller mode: rule_based, rl_ppo, rl_sac")


class DispatchResponse(BaseModel):
    battery_power_kw: float
    battery_charge_kw: float
    battery_discharge_kw: float
    diesel_generation_kw: float
    diesel_status: int
    unmet_load_kw: float
    curtailment_kw: float
    controller_mode: str


# === Metrics ===
class MetricsResponse(BaseModel):
    fuel_savings: Dict[str, Any]
    renewable_utilization: Dict[str, Any]
    reliability: Dict[str, Any]
    battery: Dict[str, Any]
    cost: Dict[str, Any]


# === Alerts ===
class AlertItem(BaseModel):
    level: str
    title: str
    message: str
    icon: str


class AlertsResponse(BaseModel):
    alerts: List[AlertItem]
    count: int


# === Optimization ===
class OptimizeRequest(BaseModel):
    horizon_hours: int = Field(24, ge=1, le=168, description="Optimization horizon")
    solver: str = Field("glpk", description="Solver name")


class OptimizeResponse(BaseModel):
    solved_optimally: bool
    solver_used: str
    schedule: List[Dict[str, Any]]
    total_cost: float
