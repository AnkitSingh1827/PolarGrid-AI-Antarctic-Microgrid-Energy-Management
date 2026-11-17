"""
PolarGrid AI — Dashboard Backend API Client
Provides seamless, non-blocking HTTP communication with the FastAPI backend with automatic local fallback.
"""

import socket
import logging
import requests
import streamlit as st
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def is_port_open(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 0.04) -> bool:
    """Ultra-fast raw TCP socket ping (runs in <2ms, never hangs UI)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


@st.cache_data(ttl=5)
def check_api_health(api_url: str = DEFAULT_API_URL, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> Dict[str, Any]:
    """
    Check whether the FastAPI backend is online.
    Cached for 5 seconds to avoid network latency on every Streamlit interaction.
    """
    # Quick socket test first to avoid requests.get hanging when offline
    if not is_port_open(host, port, timeout=0.04):
        return {"online": False, "data": None, "url": api_url}

    try:
        resp = requests.get(f"{api_url}/health", timeout=0.25)
        if resp.status_code == 200:
            return {"online": True, "data": resp.json(), "url": api_url}
    except Exception:
        pass
    return {"online": False, "data": None, "url": api_url}


def dispatch_energy(
    load_kw: float,
    solar_gen_kw: float,
    wind_gen_kw: float,
    temperature: float = -15.0,
    battery_soc: float = 0.5,
    controller_mode: str = "rule_based",
    api_url: str = DEFAULT_API_URL,
) -> Dict[str, Any]:
    """
    Execute energy dispatch.
    First attempts to call FastAPI backend /dispatch.
    If backend is unreachable, gracefully falls back to cached local dispatch controller.
    """
    payload = {
        "load_kw": float(load_kw),
        "solar_gen_kw": float(solar_gen_kw),
        "wind_gen_kw": float(wind_gen_kw),
        "temperature": float(temperature),
        "battery_soc": float(battery_soc),
        "controller_mode": controller_mode,
    }

    # Attempt API call if port is open
    if is_port_open(DEFAULT_HOST, DEFAULT_PORT, timeout=0.04):
        try:
            resp = requests.post(f"{api_url}/dispatch", json=payload, timeout=0.5)
            if resp.status_code == 200:
                res = resp.json()
                res["source"] = "FastAPI Backend (REST API)"
                return res
        except Exception as e:
            logger.debug(f"API dispatch call failed, using local engine: {e}")

    # Local fallback
    if controller_mode in ("rule_based", "rule"):
        from src.dispatch.fallback_controller import run_rule_based_dispatch
        from src.dispatch.controller import load_system_configs
        params = load_system_configs()
        local_res = run_rule_based_dispatch(
            load_kw=load_kw,
            critical_load_kw=load_kw * 0.4,
            solar_gen_kw=solar_gen_kw,
            wind_gen_kw=wind_gen_kw,
            current_soc=battery_soc,
            temp_c=temperature,
            params=params,
        )
    else:
        from src.dispatch.controller import DispatchController
        ctrl = DispatchController(mode=controller_mode)
        ctrl.soc = battery_soc
        local_res = ctrl.dispatch_step(
            load_kw=load_kw,
            critical_load_kw=load_kw * 0.4,
            solar_gen_kw=solar_gen_kw,
            wind_gen_kw=wind_gen_kw,
            temperature=temperature,
        )

    local_res["controller_mode"] = controller_mode
    local_res["source"] = "Local Python Engine (Standby Mode)"
    return local_res
