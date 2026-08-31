"""Small standard-library HTTP client for the SnowGym JSON service."""

from __future__ import annotations

import json
import uuid
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

JsonObject = dict[str, Any]
API_VERSION = "snowgym.v0"


class SnowGymClient(Protocol):
    """Transport contract consumed by :class:`SnowGymEnv`."""

    def status(self) -> JsonObject: ...

    def reset(
        self,
        seed: int,
        scenario: JsonObject | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> JsonObject: ...

    def step(
        self,
        action: JsonObject,
        *,
        expected_state_hash: str | None = None,
        idempotency_key: str | None = None,
    ) -> JsonObject: ...

    def step_scripted(
        self,
        *,
        expected_state_hash: str | None = None,
        idempotency_key: str | None = None,
    ) -> JsonObject: ...

    def autoplay(
        self,
        max_decisions: int,
        *,
        expected_state_hash: str | None = None,
        idempotency_key: str | None = None,
    ) -> JsonObject: ...


class SnowGymHttpClient:
    """Synchronous JSON-over-HTTP client suitable for correctness rollouts."""

    def __init__(self, base_url: str = "http://127.0.0.1:8787", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> JsonObject:
        return self._request("GET", "/health")

    def status(self) -> JsonObject:
        payload = self._request("GET", "/status")
        self._check_version(payload)
        return payload

    def capabilities(self) -> JsonObject:
        return self._request("GET", "/capabilities")

    def reset(
        self,
        seed: int,
        scenario: JsonObject | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> JsonObject:
        body: JsonObject = {
            "seed": int(seed),
            "idempotencyKey": idempotency_key or f"python-{uuid.uuid4()}",
        }
        if scenario is not None:
            body["scenario"] = scenario
        payload = self._request("POST", "/reset", body)
        self._check_version(payload)
        return payload

    def step(
        self,
        action: JsonObject,
        *,
        expected_state_hash: str | None = None,
        idempotency_key: str | None = None,
    ) -> JsonObject:
        body = self._guarded_body(expected_state_hash, idempotency_key)
        body["action"] = action
        payload = self._request("POST", "/step", body)
        self._check_version({"status": payload.get("info")})
        return payload

    def step_scripted(
        self,
        *,
        expected_state_hash: str | None = None,
        idempotency_key: str | None = None,
    ) -> JsonObject:
        payload = self._request(
            "POST",
            "/step-scripted",
            self._guarded_body(expected_state_hash, idempotency_key),
        )
        self._check_version({"status": payload.get("info")})
        return payload

    def autoplay(
        self,
        max_decisions: int = 10_000,
        *,
        expected_state_hash: str | None = None,
        idempotency_key: str | None = None,
    ) -> JsonObject:
        body = self._guarded_body(expected_state_hash, idempotency_key)
        body["maxDecisions"] = int(max_decisions)
        payload = self._request("POST", "/autoplay", body)
        self._check_version(payload)
        return payload

    @staticmethod
    def _guarded_body(
        expected_state_hash: str | None, idempotency_key: str | None
    ) -> JsonObject:
        body: JsonObject = {
            "idempotencyKey": idempotency_key or f"python-{uuid.uuid4()}"
        }
        if expected_state_hash is not None:
            body["expectedStateHash"] = expected_state_hash
        return body

    def _request(self, method: str, path: str, body: JsonObject | None = None) -> JsonObject:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise SnowGymProtocolError(f"SnowGym returned HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise SnowGymProtocolError(
                f"Cannot reach SnowGym at {self.base_url}; start `npm run snowgym:server`"
            ) from error
        except json.JSONDecodeError as error:
            raise SnowGymProtocolError("SnowGym returned invalid JSON") from error

        if not isinstance(payload, dict):
            raise SnowGymProtocolError("SnowGym response must be a JSON object")
        return payload

    @staticmethod
    def _check_version(payload: JsonObject) -> None:
        status = payload.get("status")
        if not isinstance(status, dict):
            raise SnowGymProtocolError("SnowGym response is missing status metadata")
        version = status.get("apiVersion")
        if version != API_VERSION:
            raise SnowGymProtocolError(f"Expected API {API_VERSION}, received {version!r}")


class SnowGymProtocolError(RuntimeError):
    """Raised when the server is unavailable or violates the JSON contract."""
