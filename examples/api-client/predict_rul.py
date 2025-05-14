"""FaultScope Inference API client example.

Demonstrates single prediction, health classification, and batch requests
using the httpx library with proper error handling.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get(
    "FAULTSCOPE_INFERENCE_URL", "http://localhost:8000"
)
API_KEY = os.environ.get("FAULTSCOPE_INFERENCE_API_KEY", "")

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}

# Example sensor readings for a turbofan engine
EXAMPLE_SENSORS: dict[str, float] = {
    "sensor_2": 641.82,
    "sensor_3": 1589.70,
    "sensor_4": 1400.60,
    "sensor_7": 554.36,
    "sensor_8": 2388.02,
    "sensor_9": 9046.19,
    "sensor_11": 47.47,
    "sensor_12": 521.66,
    "sensor_13": 2388.10,
    "sensor_14": 8138.62,
    "sensor_15": 8.4195,
    "sensor_17": 392.0,
    "sensor_20": 38.83,
    "sensor_21": 23.4190,
}


# ---------------------------------------------------------------------------
# Helper: wait for service to be ready
# ---------------------------------------------------------------------------
def wait_for_ready(
    client: httpx.Client,
    timeout_s: float = 60.0,
) -> None:
    """Poll /ready until the service reports models are loaded."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            resp = client.get(f"{BASE_URL}/ready", timeout=5.0)
            data = resp.json()
            if data.get("models_loaded"):
                return
            print(f"  Service not ready yet: {data.get('status')} — retrying…")
        except httpx.RequestError:
            print("  Inference service unreachable — retrying…")
        time.sleep(3.0)
    raise TimeoutError(
        f"Inference service not ready after {timeout_s}s"
    )


# ---------------------------------------------------------------------------
# Example 1: Single RUL prediction
# ---------------------------------------------------------------------------
def predict_remaining_life(
    client: httpx.Client,
    machine_id: str,
    sensor_readings: dict[str, float],
) -> dict[str, Any]:
    """Call /api/v1/predict/remaining-life and return the response body."""
    payload: dict[str, Any] = {
        "machine_id": machine_id,
        "sensor_readings": sensor_readings,
        "operational_setting_1": -0.0007,
        "operational_setting_2": -0.0004,
        "operational_setting_3": 100.0,
    }
    resp = client.post(
        f"{BASE_URL}/api/v1/predict/remaining-life",
        json=payload,
        headers=HEADERS,
        timeout=15.0,
    )
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Example 2: Health status classification
# ---------------------------------------------------------------------------
def predict_health_status(
    client: httpx.Client,
    machine_id: str,
    sensor_readings: dict[str, float],
) -> dict[str, Any]:
    """Call /api/v1/predict/health-status and return the response body."""
    payload: dict[str, Any] = {
        "machine_id": machine_id,
        "sensor_readings": sensor_readings,
    }
    resp = client.post(
        f"{BASE_URL}/api/v1/predict/health-status",
        json=payload,
        headers=HEADERS,
        timeout=15.0,
    )
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Example 3: Batch prediction
# ---------------------------------------------------------------------------
def predict_batch(
    client: httpx.Client,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Call /api/v1/predict/batch for multiple machines at once."""
    resp = client.post(
        f"{BASE_URL}/api/v1/predict/batch",
        json={"samples": samples},
        headers=HEADERS,
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def print_rul_result(result: dict[str, Any]) -> None:
    print(f"Machine: {result['machine_id']}")
    print(f"Predicted RUL: {result['predicted_rul']:.1f} cycles")
    lo = result["confidence_lower"]
    hi = result["confidence_upper"]
    lvl = int(result["confidence_level"] * 100)
    print(f"{lvl}% CI: [{lo:.1f}, {hi:.1f}]")
    print(f"Model version: {result['model_version']}")


def print_health_result(result: dict[str, Any]) -> None:
    print(f"Machine: {result['machine_id']}")
    print(f"Health label: {result['health_label']}")
    print("Probabilities:")
    for label, prob in result["probabilities"].items():
        bar = "█" * int(prob * 20)
        print(f"  {label:<20} {prob * 100:5.1f}%  {bar}")


def print_batch_result(result: dict[str, Any]) -> None:
    print(f"Batch prediction — {result['batch_size']} machines")
    for item in result["results"]:
        print(
            f"  {item['machine_id']}: "
            f"RUL={item['predicted_rul']:.1f}, "
            f"health={item['health_label']}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not API_KEY:
        raise ValueError(
            "Set FAULTSCOPE_INFERENCE_API_KEY environment variable"
        )

    print("=== FaultScope Inference API Client Example ===\n")

    with httpx.Client() as client:
        # Wait until models are loaded (important on first startup)
        print("Waiting for inference service to be ready…")
        wait_for_ready(client)
        print("Service ready.\n")

        # --- RUL prediction ---
        print("[RUL Prediction]")
        try:
            rul = predict_remaining_life(client, "M-001", EXAMPLE_SENSORS)
            print_rul_result(rul)
        except httpx.HTTPStatusError as exc:
            print(f"Error {exc.response.status_code}: {exc.response.text}")
        print()

        # --- Health classification ---
        print("[Health Status]")
        try:
            health = predict_health_status(
                client, "M-001", EXAMPLE_SENSORS
            )
            print_health_result(health)
        except httpx.HTTPStatusError as exc:
            print(f"Error {exc.response.status_code}: {exc.response.text}")
        print()

        # --- Batch prediction ---
        print("[Batch Prediction]")
        batch_samples = [
            {
                "machine_id": f"M-{i:03d}",
                "sensor_readings": {
                    k: v + (i * 5.0) for k, v in EXAMPLE_SENSORS.items()
                },
            }
            for i in range(1, 4)
        ]
        try:
            batch = predict_batch(client, batch_samples)
            print_batch_result(batch)
        except httpx.HTTPStatusError as exc:
            print(f"Error {exc.response.status_code}: {exc.response.text}")

        # --- Error handling demo ---
        print("\n[Error Handling Demo]")
        print("Testing with invalid API key…")
        try:
            bad_headers = {**HEADERS, "X-API-Key": "invalid-key"}
            resp = client.post(
                f"{BASE_URL}/api/v1/predict/remaining-life",
                json={
                    "machine_id": "M-001",
                    "sensor_readings": EXAMPLE_SENSORS,
                },
                headers=bad_headers,
                timeout=10.0,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            print(
                f"  Got expected {exc.response.status_code}: "
                f"{exc.response.json()['detail']}"
            )

        print("Testing with oversized batch (>100 samples)…")
        too_many = [
            {"machine_id": f"M-{i:03d}", "sensor_readings": EXAMPLE_SENSORS}
            for i in range(101)
        ]
        try:
            predict_batch(client, too_many)
        except httpx.HTTPStatusError as exc:
            print(
                f"  Got expected {exc.response.status_code} "
                "(batch size limit exceeded)"
            )


if __name__ == "__main__":
    main()
