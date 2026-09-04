from __future__ import annotations

import requests
import pandas as pd


BASE_URL = "https://floodforecasting.googleapis.com/v1"


class FloodHubClient:
    def __init__(self, api_key: str, timeout: int = 60):
        if not api_key:
            raise ValueError("Flood Hub API key is empty.")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

    def _request(self, method, path, params=None, json_body=None):
        url = BASE_URL + path

        if params is None:
            params = [("key", self.api_key)]
        elif isinstance(params, list):
            params = [("key", self.api_key), *params]
        else:
            params = {**params, "key": self.api_key}

        response = self.session.request(
            method=method,
            url=url,
            params=params,
            json=json_body,
            timeout=self.timeout,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError:
            print("\nFlood Forecasting API error:")
            print(response.status_code)
            print(response.text[:5000])
            raise

        return response.json()

    def search_gauges_by_loop(
        self,
        vertices,
        include_non_quality_verified=True,
        page_size=5000,
    ):
        body = {
            "loop": {
                "vertices": [
                    {"latitude": float(lat), "longitude": float(lon)}
                    for lat, lon in vertices
                ]
            },
            "includeNonQualityVerified": bool(include_non_quality_verified),
            "includeGaugesWithoutHydroModel": False,
            "pageSize": int(page_size),
        }
        return self._request("POST", "/gauges:searchGaugesByArea", json_body=body)

    def get_gauge(self, gauge_id):
        return self._request("GET", f"/gauges/{gauge_id}")

    def get_gauge_model(self, gauge_id):
        return self._request("GET", f"/gaugeModels/{gauge_id}")

    def query_forecasts(self, gauge_ids, issued_time_start=None, issued_time_end=None):
        params = []
        for gauge_id in gauge_ids:
            params.append(("gaugeIds", str(gauge_id)))
        if issued_time_start:
            params.append(("issuedTimeStart", issued_time_start))
        if issued_time_end:
            params.append(("issuedTimeEnd", issued_time_end))
        return self._request("GET", "/gauges:queryGaugeForecasts", params=params)

    def normalize_forecasts(self, payload):
        rows = []
        forecast_map = payload.get("forecasts", {})

        for gauge_id, forecast_set in forecast_map.items():
            for forecast in forecast_set.get("forecasts", []):
                issued_time = forecast.get("issuedTime")
                forecast_gauge = forecast.get("gaugeId", gauge_id)
                for item in forecast.get("forecastRanges", []):
                    rows.append(
                        {
                            "gauge_id": forecast_gauge,
                            "issued_time": pd.to_datetime(issued_time, utc=True),
                            "forecast_start": pd.to_datetime(
                                item.get("forecastStartTime"), utc=True
                            ),
                            "forecast_end": pd.to_datetime(
                                item.get("forecastEndTime"), utc=True
                            ),
                            "value": item.get("value"),
                        }
                    )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "gauge_id",
                    "issued_time",
                    "forecast_start",
                    "forecast_end",
                    "value",
                ]
            )

        df = pd.DataFrame(rows)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.sort_values(["issued_time", "forecast_start"]).reset_index(drop=True)

    def query_latest_flood_status(self, gauge_ids):
        params = [("gaugeIds", str(gauge_id)) for gauge_id in gauge_ids]
        return self._request(
            "GET",
            "/floodStatus:queryLatestFloodStatusByGaugeIds",
            params=params,
        )

    def get_serialized_polygon(self, polygon_id):
        return self._request("GET", f"/serializedPolygons/{polygon_id}")
