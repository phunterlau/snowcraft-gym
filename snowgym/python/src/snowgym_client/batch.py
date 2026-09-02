"""Persistent multi-world SnowGym subprocess client and fixed-batch adapter."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .encoding import (
    GymAction,
    GymObservation,
    encode_action,
    encode_observation,
    make_action_space,
)

BATCH_REQUEST_FORMAT = "snowgym.batch-request.v0"
BATCH_RESPONSE_FORMAT = "snowgym.batch-response.v0"
BATCH_PROTOCOL_VERSION = "snowgym.batch.v0"
PLAN_GROUP_SLOTS = 3
PLAN_FEATURES_PER_GROUP = 38


class BatchOperationError(RuntimeError):
    def __init__(self, message: str, results: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.results = results
        self.advanced_worlds = [
            result["worldId"] for result in results if result.get("status") == 200
        ]


class SnowGymBatchClient:
    def __init__(
        self,
        command: list[str] | None = None,
        *,
        cwd: str | Path | None = None,
    ) -> None:
        repository = Path(__file__).resolve().parents[4]
        self._process = subprocess.Popen(
            command or ["node", "--import", "tsx", "snowgym/batch/main.ts"],
            cwd=Path(cwd) if cwd is not None else repository,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._lock = threading.Lock()
        self._request_index = 0
        self.serialization_seconds = 0.0
        self.io_seconds = 0.0
        self.payload_bytes = 0
        handshake = self.request("handshake", [])
        body = handshake[0].get("body") if handshake else None
        if not isinstance(body, dict) or body.get("protocolVersion") != BATCH_PROTOCOL_VERSION:
            self.close()
            raise RuntimeError("SnowGym batch handshake version mismatch")
        self.capabilities = body

    def request(
        self, operation: str, items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._request_index += 1
            request_id = f"python-{self._request_index}"
            value = {
                "format": BATCH_REQUEST_FORMAT,
                "requestId": request_id,
                "operation": operation,
                "items": items,
            }
            serialization_started = time.perf_counter()
            encoded = json.dumps(value, separators=(",", ":")) + "\n"
            self.serialization_seconds += time.perf_counter() - serialization_started
            self.payload_bytes += len(encoded.encode("utf-8"))
            if self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError("SnowGym batch process pipes are unavailable")
            try:
                io_started = time.perf_counter()
                self._process.stdin.write(encoded)
                self._process.stdin.flush()
                line = self._process.stdout.readline()
                self.io_seconds += time.perf_counter() - io_started
            except (BrokenPipeError, OSError) as error:
                raise RuntimeError("SnowGym batch process pipe failed") from error
            if not line:
                raise RuntimeError(
                    f"SnowGym batch process exited with {self._process.poll()}"
                )
            try:
                serialization_started = time.perf_counter()
                response = json.loads(line)
                self.serialization_seconds += time.perf_counter() - serialization_started
                self.payload_bytes += len(line.encode("utf-8"))
            except json.JSONDecodeError as error:
                raise RuntimeError("SnowGym batch process returned invalid JSON") from error
            if (
                not isinstance(response, dict)
                or response.get("format") != BATCH_RESPONSE_FORMAT
                or response.get("requestId") != request_id
                or response.get("operation") != operation
                or not isinstance(response.get("results"), list)
            ):
                raise RuntimeError("SnowGym batch response contract mismatch")
            return response["results"]

    def close(self) -> None:
        if self._process.poll() is None:
            if self._process.stdin is not None:
                self._process.stdin.close()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                self._process.wait(timeout=5)

    def __enter__(self) -> SnowGymBatchClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class SnowGymBatchEnv:
    """Fixed leading-batch adapter over independent persistent worlds."""

    def __init__(
        self,
        batch_size: int,
        *,
        max_team_units: int = 10,
        client: SnowGymBatchClient | None = None,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.max_team_units = max_team_units
        self.action_space = make_action_space(max_team_units)
        self.client = client or SnowGymBatchClient()
        self._owns_client = client is None
        self.world_ids = [f"world-{index:04d}" for index in range(batch_size)]
        self.raw_observations: list[dict[str, Any] | None] = [None] * batch_size
        self.state_hashes: list[str | None] = [None] * batch_size
        self._observations: list[GymObservation | None] = [None] * batch_size
        self._step_index = 0
        self._plan_activation_index = 0

    def reset(
        self, seeds: list[int], scenarios: list[dict[str, Any]]
    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        if len(seeds) != self.batch_size or len(scenarios) != self.batch_size:
            raise ValueError("reset seeds/scenarios must match batch_size")
        return self._reset_indices(list(range(self.batch_size)), seeds, scenarios)

    def reset_indices(
        self,
        indices: list[int],
        seeds: list[int],
        scenarios: list[dict[str, Any]],
    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        if len(indices) != len(seeds) or len(indices) != len(scenarios):
            raise ValueError("selective reset inputs must have equal length")
        if len(set(indices)) != len(indices) or any(
            index < 0 or index >= self.batch_size for index in indices
        ):
            raise ValueError("selective reset indices are invalid")
        return self._reset_indices(indices, seeds, scenarios)

    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        self._step_index += 1
        items = []
        for index, world_id in enumerate(self.world_ids):
            raw = self.raw_observations[index]
            state_hash = self.state_hashes[index]
            if raw is None or state_hash is None:
                raise RuntimeError("reset() must initialize every batch slot before step()")
            action: GymAction = {
                "action_type": np.asarray(actions["action_type"][index]),
                "target": np.asarray(actions["target"][index], dtype=np.float32),
                "power": np.asarray(actions["power"][index], dtype=np.float32),
            }
            if not self.action_space.contains(action):
                raise ValueError(f"batch action for slot {index} is outside action_space")
            items.append(
                {
                    "worldId": world_id,
                    "body": {
                        "action": encode_action(action, raw, self.max_team_units),
                        "expectedStateHash": state_hash,
                        "idempotencyKey": f"batch-{self._step_index}-{world_id}",
                    },
                }
            )
        results = self.client.request("step", items)
        payloads = self._consume_results(results)
        return (
            self._stack_observations(),
            np.asarray([payload["reward"] for payload in payloads], dtype=np.float32),
            np.asarray([payload["terminated"] for payload in payloads], dtype=np.bool_),
            np.asarray([payload["truncated"] for payload in payloads], dtype=np.bool_),
            [payload["info"] for payload in payloads],
        )

    def step_scripted(
        self,
    ) -> tuple[
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
    ]:
        self._step_index += 1
        items = []
        for index, world_id in enumerate(self.world_ids):
            state_hash = self.state_hashes[index]
            if state_hash is None:
                raise RuntimeError("reset() must initialize every batch slot before step_scripted()")
            items.append(
                {
                    "worldId": world_id,
                    "body": {
                        "expectedStateHash": state_hash,
                        "idempotencyKey": f"batch-scripted-{self._step_index}-{world_id}",
                    },
                }
            )
        payloads = self._consume_results(self.client.request("stepScripted", items))
        return (
            self._stack_observations(),
            np.asarray([payload["reward"] for payload in payloads], dtype=np.float32),
            np.asarray([payload["terminated"] for payload in payloads], dtype=np.bool_),
            np.asarray([payload["truncated"] for payload in payloads], dtype=np.bool_),
            [payload["info"] for payload in payloads],
        )

    def activate_plans(
        self, plan_ids: list[str], plans: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Atomically ground one symbolic plan in every initialized world."""
        if len(plan_ids) != self.batch_size or len(plans) != self.batch_size:
            raise ValueError("plan_ids/plans must match batch_size")
        self._plan_activation_index += 1
        items = []
        for index, world_id in enumerate(self.world_ids):
            state_hash = self.state_hashes[index]
            if state_hash is None:
                raise RuntimeError("reset() must initialize every batch slot before activate_plans()")
            items.append(
                {
                    "worldId": world_id,
                    "body": {
                        "planId": plan_ids[index],
                        "plan": plans[index],
                        "expectedStateHash": state_hash,
                        "idempotencyKey": (
                            f"batch-plan-{self._plan_activation_index}-{world_id}"
                        ),
                    },
                }
            )
        return self._plan_bodies(self.client.request("activatePlan", items))

    def plan_observations(
        self,
    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        """Read current host-resolved plan tensors without advancing any world."""
        items = [{"worldId": world_id} for world_id in self.world_ids]
        bodies = self._plan_bodies(self.client.request("planObservation", items))
        tensors = {
            "plan_groups": np.asarray(
                [body["planGroups"] for body in bodies], dtype=np.float32
            ).reshape(self.batch_size, PLAN_GROUP_SLOTS, PLAN_FEATURES_PER_GROUP),
            "plan_group_mask": np.asarray(
                [body["planGroupMask"] for body in bodies], dtype=np.int8
            ),
        }
        return tensors, bodies

    def plan_teacher_actions(self) -> list[dict[str, Any]]:
        """Read production plan-aware teacher actions at the current world states."""
        items = [{"worldId": world_id} for world_id in self.world_ids]
        results = self.client.request("planTeacherAction", items)
        if len(results) != self.batch_size:
            raise RuntimeError("batch plan teacher result count mismatch")
        failures = [result for result in results if result.get("status") != 200]
        if failures:
            raise BatchOperationError("one or more batch plan teachers failed", results)
        actions: list[dict[str, Any]] = []
        for index, result in enumerate(results):
            body = result.get("body")
            if not isinstance(body, dict) or not isinstance(body.get("action"), dict):
                raise RuntimeError("batch plan teacher payload is missing action")
            status = body.get("status")
            if not isinstance(status, dict) or status.get("stateHash") != self.state_hashes[index]:
                raise RuntimeError("batch plan teacher stateHash does not match world state")
            actions.append(body["action"])
        return actions

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def _reset_indices(
        self, indices: list[int], seeds: list[int], scenarios: list[dict[str, Any]]
    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
        items = [
            {
                "worldId": self.world_ids[index],
                "body": {"seed": seed, "scenario": scenario},
            }
            for index, seed, scenario in zip(indices, seeds, scenarios, strict=True)
        ]
        results = self.client.request("reset", items)
        payloads = self._consume_results(results, indices=indices)
        return self._stack_observations(indices), [payload["status"] for payload in payloads]

    def _consume_results(
        self, results: list[dict[str, Any]], indices: list[int] | None = None
    ) -> list[dict[str, Any]]:
        target_indices = indices if indices is not None else list(range(self.batch_size))
        if len(results) != len(target_indices):
            raise RuntimeError("batch result count mismatch")
        failures = [result for result in results if result.get("status") != 200]
        payloads: list[dict[str, Any]] = []
        for index, result in zip(target_indices, results, strict=True):
            body = result.get("body")
            if result.get("status") != 200 or not isinstance(body, dict):
                continue
            raw = body.get("observation")
            info = body.get("info", body.get("status"))
            if not isinstance(raw, dict) or not isinstance(info, dict):
                raise RuntimeError("batch world payload is missing observation/status")
            state_hash = info.get("stateHash")
            if not isinstance(state_hash, str):
                raise RuntimeError("batch world payload is missing stateHash")
            self.raw_observations[index] = raw
            self.state_hashes[index] = state_hash
            self._observations[index] = encode_observation(
                raw, self.max_team_units, include_unit_masks=True
            )
            payloads.append(body)
        if failures:
            raise BatchOperationError("one or more batch worlds failed", results)
        return payloads

    def _plan_bodies(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(results) != self.batch_size:
            raise RuntimeError("batch plan result count mismatch")
        failures = [result for result in results if result.get("status") != 200]
        if failures:
            raise BatchOperationError("one or more batch plan operations failed", results)
        bodies: list[dict[str, Any]] = []
        for index, result in enumerate(results):
            body = result.get("body")
            if not isinstance(body, dict):
                raise RuntimeError("batch plan payload is missing its body")
            if body.get("stateHash") != self.state_hashes[index]:
                raise RuntimeError("batch plan payload stateHash does not match world state")
            groups = body.get("planGroups")
            mask = body.get("planGroupMask")
            if (
                not isinstance(groups, list)
                or len(groups) != PLAN_GROUP_SLOTS * PLAN_FEATURES_PER_GROUP
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not np.isfinite(value)
                    or value < -1
                    or value > 1
                    for value in groups
                )
            ):
                raise RuntimeError("batch planGroups tensor is invalid")
            if (
                not isinstance(mask, list)
                or len(mask) != PLAN_GROUP_SLOTS
                or any(value not in (0, 1) or isinstance(value, bool) for value in mask)
            ):
                raise RuntimeError("batch planGroupMask tensor is invalid")
            bodies.append(body)
        return bodies

    def _stack_observations(
        self, indices: list[int] | None = None
    ) -> dict[str, np.ndarray]:
        selected = indices if indices is not None else list(range(self.batch_size))
        observations = [self._observations[index] for index in selected]
        if not observations or any(observation is None for observation in observations):
            raise RuntimeError("requested batch observations are not initialized")
        concrete = [observation for observation in observations if observation is not None]
        return {
            name: np.stack([observation[name] for observation in concrete])
            for name in concrete[0]
        }
