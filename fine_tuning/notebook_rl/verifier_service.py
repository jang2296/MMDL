"""Fresh-process verifier facade; only bounded JSON metadata crosses the pipes.

Use as a context manager and install guards in the actor separately, before any
untrusted actor work. The worker allows its private vault and denies benchmark
paths/teacher imports. Neither Python hooks nor same-user subprocesses are an OS
sandbox: native I/O, inherited authority and hostile same-user process inspection
need an external OS isolation policy. There is deliberately no in-process fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import selectors
import signal
import struct
import subprocess
import sys
import threading
import time
import weakref
from collections import OrderedDict
from pathlib import Path

from .reasoning_trace import validate_trace_report
from .repair_contract import RepairContractError, parse_retry_sections, project_solution
from .schema import DiagTarget, VerifierResult, from_json, to_json
from .answers import TRAINING_ANSWER_CONTRACT, extract_eval_aligned, extract_training
from .verifier import RETRY_OUT_OF_CHOICE_TAG, VERSION as VERIFIER_VERSION

VERSION = "notebook-rl-verifier-service-v5"
MAX_FRAME_BYTES = 1024 * 1024
MAX_RESULTS = 256
MAX_SLOTS = 1024
_METHODS = {"grade", "trace", "cause", "diagnosis_target", "route", "grade_diagnosis"}
_CAUSES = {"VISUAL_MISREAD", "VISUAL_RELATION", "NON_VISUAL", "FINALIZATION", "UNKNOWN"}
_ROUTES = {"verified-any-visual-v1", "verified-any-visual-v1+probe"}
_ERRORS = {"INVALID_REQUEST", "UNKNOWN_PROBLEM", "UNISSUED_RESULT", "UNISSUED_TARGET",
           "RESULT_LIMIT", "VERIFIER_ERROR", "VAULT_UNAVAILABLE", "INVALID_RESPONSE_CONTRACT"}
_VERIFIER_SOURCES = ("verifier_service.py", "verifier.py", "answers.py", "v8_parser.py", "observations.py",
                     "attribution.py", "reasoning_trace.py", "repair_contract.py", "circuit_trace.py", "schema.py",
                     "guards.py", "vault.py",
                     "generators/graph.py", "generators/graph_pair.py", "generators/circuit.py",
                     "generators/common.py", "../nvw.py", "../circuit_bridge.py", "../graph_bridge.py",
                     "../ablation.py")
_TRAINING_SOURCES = ("../error_note/answer_contract.py",)


class VerifierServiceError(RuntimeError):
    """A bounded public error code, never a private verifier exception/traceback."""


class _Rejected(Exception):
    pass


def _wire(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _unique(pairs):
    value = {}
    for key, entry in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = entry
    return value


def _decode(raw: bytes):
    def invalid(_):
        raise ValueError("nonfinite JSON")
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=invalid)


def _check_text(value: object, *, identity: bool = False) -> None:
    if not isinstance(value, str) or (identity and (not value.strip() or len(value) > 256)):
        raise _Rejected("INVALID_REQUEST")


def _record(cls, value):
    if type(value) is not dict:
        raise _Rejected("INVALID_REQUEST")
    try:
        return from_json(cls, _wire(value).decode())
    except (TypeError, ValueError):
        raise _Rejected("INVALID_REQUEST") from None


def _slot_names(names) -> None:
    if len(names) > MAX_SLOTS or any(not isinstance(s, str) or not s or len(s) > 256
                                    or "\n" in s or "=" in s for s in names):
        raise _Rejected("RESULT_LIMIT")


def _result(value: VerifierResult) -> None:
    to_json(value)
    if value.slots is not None:
        if value.slots.all_total > MAX_SLOTS:
            raise _Rejected("RESULT_LIMIT")
        _slot_names((*value.slots.wrong_required, *value.slots.conflicting,
                     *value.slots.false_relations, *value.slots.unverified_relations))


def _route(value, size: int):
    if (type(value) not in (list, tuple) or len(value) != 2 or value[0] not in ("VISION_ON", "VISION_OFF")
            or type(value[1]) is not dict):
        raise _Rejected("VERIFIER_ERROR")
    route, data = value
    if (set(data) != {"rule", "observed_evidence", "wrong_required", "false_relations", "complete_required_indices",
                      "probe_evidence", "probe_used", "module_causality_certified"}
            or data["rule"] not in _ROUTES or data["observed_evidence"] not in ("VISUAL", "NON_VISUAL", "UNKNOWN")
            or data["probe_evidence"] not in ("VISUAL", "NON_VISUAL", "UNKNOWN")
            or type(data["probe_used"]) is not bool or data["module_causality_certified"] is not False
            or type(data["wrong_required"]) is not list or len(data["wrong_required"]) > size
            or type(data["false_relations"]) is not list or len(data["false_relations"]) > size
            or type(data["complete_required_indices"]) is not list
            or len(data["complete_required_indices"]) > size):
        raise _Rejected("VERIFIER_ERROR")
    for index in data["complete_required_indices"]:
        if type(index) is not int or not 0 <= index < size:
            raise _Rejected("VERIFIER_ERROR")
    for entry in data["wrong_required"] + data["false_relations"]:
        if (type(entry) is not dict or set(entry) != {"result_index", "wrong_slots"}
                or type(entry["result_index"]) is not int or not 0 <= entry["result_index"] < size
                or type(entry["wrong_slots"]) is not list):
            raise _Rejected("VERIFIER_ERROR")
        _slot_names(entry["wrong_slots"])
    return route, data


class _Issued:
    def __init__(self) -> None:
        self.values: OrderedDict[tuple[str, str, str], None] = OrderedDict()

    def key(self, kind, pid, record):
        return kind, pid, hashlib.sha256(to_json(record).encode()).hexdigest()

    def add(self, kind, pid, record) -> None:
        key = self.key(kind, pid, record)
        self.values[key] = None
        self.values.move_to_end(key)
        if len(self.values) > 4096:
            self.values.popitem(last=False)

    def require(self, kind, pid, record) -> None:
        if self.key(kind, pid, record) not in self.values:
            raise _Rejected("UNISSUED_RESULT" if kind == "result" else "UNISSUED_TARGET")


def _dispatch(verifier, issued: _Issued, request: dict):
    if (type(request) is not dict or set(request) != {"id", "method", "problem_id", "params"}
            or type(request["method"]) is not str or request["method"] not in _METHODS
            or type(request["params"]) is not dict):
        raise _Rejected("INVALID_REQUEST")
    pid, method, args = request["problem_id"], request["method"], request["params"]
    _check_text(pid, identity=True)
    if pid not in verifier.items:
        raise _Rejected("UNKNOWN_PROBLEM")
    if method == "trace":
        fields = {"text", "finish_reason"}
        if set(args) not in (fields, fields | {"response_contract"}) or args["finish_reason"] not in ("stop", "length"):
            raise _Rejected("INVALID_REQUEST")
        _check_text(args["text"])
        contract = args.get("response_contract")
        options = {} if contract is None else {"response_contract": contract}
        report = verifier.trace(pid, args["text"], args["finish_reason"], **options)
        validate_trace_report(report, args["text"], args["finish_reason"], response_contract=contract)
        return report
    if method == "grade":
        fields = {"text", "finish_reason", "mode"}
        if (set(args) not in (fields, fields | {"response_contract"}, fields | {"answer_contract"},
                              fields | {"response_contract", "retry_out_of_choice"})
                or args["finish_reason"] not in ("stop", "length") or args["mode"] not in ("plain", "obs")):
            raise _Rejected("INVALID_REQUEST")
        _check_text(args["text"])
        contract, answer_contract = args.get("response_contract"), args.get("answer_contract")
        rule = args.get("retry_out_of_choice")
        options = ({} if contract is None else {"response_contract": contract}) | (
            {} if answer_contract is None else {"answer_contract": answer_contract}) | (
            {} if rule is None else {"retry_out_of_choice": rule})
        result = verifier.grade(pid, args["text"], args["finish_reason"], args["mode"], **options)
        _result(result)
        # An extracted answer must come from the submitted response, never the vault.
        if answer_contract is not None:
            # Re-extract from public input: neither the reference nor the private rules enter this operation.
            item = verifier.items[pid]
            extractor = extract_training if answer_contract == TRAINING_ANSWER_CONTRACT else extract_eval_aligned
            extracted = extractor(args["text"], args["finish_reason"], verifier.actors[pid].question,
                                  item.choices, item.answer_type)
            if (result.extracted != extracted.answer
                    or answer_contract == TRAINING_ANSWER_CONTRACT and extracted.status != "OK"
                    and result.status != extracted.status
                    or result.verifier_version != VERIFIER_VERSION + "+" + answer_contract):
                raise _Rejected("VERIFIER_ERROR")
        else:
            body = (args["text"] if contract is None else project_solution(args["text"],
                    parse_retry_sections(args["text"], contract_version=contract)))
            if (result.extracted is not None and result.extracted not in body
                    or contract is not None and result.verifier_version != VERIFIER_VERSION + "+" + contract
                    + (RETRY_OUT_OF_CHOICE_TAG if rule is not None else "")):
                raise _Rejected("VERIFIER_ERROR")
        issued.add("result", pid, result)
        return json.loads(to_json(result))
    if method in ("cause", "diagnosis_target"):
        if set(args) != {"result"}:
            raise _Rejected("INVALID_REQUEST")
        result = _record(VerifierResult, args["result"])
        issued.require("result", pid, result)
        value = getattr(verifier, method)(pid, result)
        if method == "cause":
            if value is not None and value not in _CAUSES:
                raise _Rejected("VERIFIER_ERROR")
            return value
        if value is not None:
            _slot_names(value.wrong_slots)
            issued.add("target", pid, value)
            return json.loads(to_json(value))
        return None
    if method == "route":
        if (set(args) != {"results", "rule"} or type(args["results"]) is not list
                or len(args["results"]) > MAX_RESULTS
                or type(args["rule"]) is not str or args["rule"] not in _ROUTES):
            raise _Rejected("INVALID_REQUEST")
        results = [_record(VerifierResult, row) for row in args["results"]]
        for result in results:
            issued.require("result", pid, result)
        return _route(verifier.route(pid, results, args["rule"]), len(results))
    if set(args) != {"text", "target"}:
        raise _Rejected("INVALID_REQUEST")
    _check_text(args["text"])
    target = _record(DiagTarget, args["target"])
    issued.require("target", pid, target)
    reward = verifier.grade_diagnosis(pid, args["text"], target)
    if type(reward) not in (int, float) or reward not in (0.0, 1.0, 1.25):
        raise _Rejected("VERIFIER_ERROR")
    return float(reward)


def _worker_read(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        data = stream.read(size - len(chunks))
        if not data:
            raise EOFError
        chunks.extend(data)
    return bytes(chunks)


def _send(stream, payload: dict) -> None:
    raw = _wire(payload)
    if len(raw) > MAX_FRAME_BYTES:
        raise _Rejected("RESULT_LIMIT")
    stream.write(struct.pack("!I", len(raw)) + raw)
    stream.flush()


def _worker(vault_root: Path, forbidden_roots: list[Path], training_rules_path: Path | None = None,
            training_rules_sha256: str | None = None) -> int:
    output = sys.stdout.buffer
    try:
        # Trusted imports precede the one-shot guard: distribution stdlib files can
        # be hard-linked. Install before opening gold or accepting any actor request.
        from . import attribution, circuit_trace, reasoning_trace, repair_contract, v8_parser, vault
        from .generators import circuit, graph, graph_pair
        from .guards import install_guards
        from .schema import load_protocol
        from .verifier import Verifier
        if training_rules_path is not None:
            from fine_tuning.error_note import answer_contract
            del answer_contract
        del attribution, circuit_trace, reasoning_trace, repair_contract, v8_parser, circuit, graph, graph_pair
        private = [vault_root / "verifier_private.jsonl", vault_root / "lineage.jsonl"]
        if training_rules_path is not None:
            private.append(training_rules_path)
        install_guards(forbidden_roots=forbidden_roots, vault_files=private,
                       allow_vault=True, allow_benchmark=False)
        vault_root = vault._safe_root(vault_root)
        manifest_sha = hashlib.sha256(vault._read(vault_root / "manifest.yaml")).hexdigest()
        verifier = Verifier(vault_root, training_rules_path=training_rules_path,
                            training_rules_sha256=training_rules_sha256)
        actors, lineages = list(verifier.actors.values()), vault.read_lineage(vault_root)
        if set(verifier.items) != {a.problem_id for a in actors} or set(verifier.items) != set(lineages):
            raise ValueError("vault identities differ")
        for actor in actors:
            item, lineage = verifier.items[actor.problem_id], lineages[actor.problem_id]
            if (actor.answer_format != item.answer_type or actor.choices != item.choices
                    or actor.source != lineage.source):
                raise ValueError("vault metadata differs")
        if hashlib.sha256(vault._read(vault_root / "manifest.yaml")).hexdigest() != manifest_sha:
            raise ValueError("vault manifest changed during startup")
        here = Path(__file__).resolve().parent
        sources = _VERIFIER_SOURCES + (_TRAINING_SOURCES if training_rules_path is not None else ())
        metadata = {"version": VERSION, "pid": os.getpid(), "item_count": len(verifier.items),
                    "vault_manifest_sha256": manifest_sha,
                    "max_frame_bytes": MAX_FRAME_BYTES, "python_guards_installed": True,
                    "os_sandbox": False, "protocol_sha256": load_protocol(here / "protocol_v1.yaml")["sha256"],
                    "code_sha256": {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in sources}}
        if training_rules_path is not None:
            metadata["training_rules_sha256"] = training_rules_sha256
        _send(output, {"id": 0, "ok": True, "result": metadata})
    except Exception:
        _send(output, {"id": 0, "ok": False, "error": "VAULT_UNAVAILABLE"})
        return 2
    issued, expected = _Issued(), 1
    while True:
        try:
            length = struct.unpack("!I", _worker_read(sys.stdin.buffer, 4))[0]
            if not 1 <= length <= MAX_FRAME_BYTES:
                return 3
            request = _decode(_worker_read(sys.stdin.buffer, length))
        except EOFError:
            return 0
        except (TypeError, ValueError):
            return 3
        if type(request) is not dict or type(request.get("id")) is not int or request["id"] != expected:
            _send(output, {"id": 0, "ok": False, "error": "INVALID_REQUEST"})
            return 3
        expected += 1
        try:
            value = _dispatch(verifier, issued, request)
            _send(output, {"id": request["id"], "ok": True, "result": value})
        except _Rejected as exc:
            _send(output, {"id": request["id"], "ok": False, "error": str(exc)})
        except RepairContractError:
            _send(output, {"id": request["id"], "ok": False, "error": "INVALID_RESPONSE_CONTRACT"})
        except Exception:
            _send(output, {"id": request["id"], "ok": False, "error": "VERIFIER_ERROR"})


def _stop_process(process: subprocess.Popen) -> None:
    for stream in (process.stdin, process.stdout):
        if stream is not None:
            stream.close()
    if process.poll() is not None:
        process.wait()
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=0.5)
    except ProcessLookupError:
        process.wait()
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait(timeout=2)


class VerifierService:
    """Synchronous, thread-serialized verifier interface used by rollout.build_tree."""
    def __init__(self, vault_root: Path, *, forbidden_roots: list[Path], timeout: float = 30.0,
                 max_request_bytes: int = MAX_FRAME_BYTES, training_rules_path: Path | None = None,
                 training_rules_sha256: str | None = None):
        if (type(timeout) not in (float, int) or not math.isfinite(timeout) or not 0 < timeout <= 120
                or type(max_request_bytes) is not int or not 128 <= max_request_bytes <= MAX_FRAME_BYTES
                or type(forbidden_roots) is not list or any(not isinstance(p, Path) for p in forbidden_roots)):
            raise ValueError("verifier service: invalid explicit bounds/benchmark roots")
        if ((training_rules_path is None) != (training_rules_sha256 is None)
                or training_rules_path is not None and (not isinstance(training_rules_path, Path)
                    or not isinstance(training_rules_sha256, str) or len(training_rules_sha256) != 64
                    or set(training_rules_sha256) - set("0123456789abcdef"))):
            raise ValueError("verifier service: training rules require an explicit path and SHA256")
        self._timeout, self._limit = float(timeout), max_request_bytes
        self._sequence, self._closed, self._lock = 0, False, threading.Lock()
        command = [sys.executable, "-B", "-m", "fine_tuning.notebook_rl.verifier_service", "--worker",
                   "--vault-root", str(Path(vault_root).absolute())]
        if training_rules_path is not None:
            assert training_rules_sha256 is not None
            command.extend(["--training-rules", str(training_rules_path.absolute()),
                            "--training-rules-sha256", training_rules_sha256])
        for root in forbidden_roots:
            command.extend(["--forbidden-root", str(root.absolute())])
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, bufsize=0, close_fds=True,
                                         start_new_session=os.name == "posix")
        self._finalizer = weakref.finalize(self, _stop_process, self._process)
        assert self._process.stdin is not None and self._process.stdout is not None
        os.set_blocking(self._process.stdin.fileno(), False)
        os.set_blocking(self._process.stdout.fileno(), False)
        try:
            metadata = self._response(0, time.monotonic() + self._timeout)
            fields = {"version", "pid", "item_count", "max_frame_bytes", "python_guards_installed", "os_sandbox",
                      "protocol_sha256", "code_sha256", "vault_manifest_sha256"}
            if training_rules_path is not None:
                fields.add("training_rules_sha256")
            if (type(metadata) is not dict or set(metadata) != fields
                    or training_rules_path is not None and metadata["training_rules_sha256"] != training_rules_sha256
                    or metadata["version"] != VERSION or type(metadata["pid"]) is not int
                    or metadata["pid"] != self._process.pid or metadata["pid"] == os.getpid()
                    or type(metadata["item_count"]) is not int or metadata["item_count"] < 1
                    or not isinstance(metadata["vault_manifest_sha256"], str)
                    or len(metadata["vault_manifest_sha256"]) != 64
                    or set(metadata["vault_manifest_sha256"]) - set("0123456789abcdef")
                    or metadata["max_frame_bytes"] != MAX_FRAME_BYTES
                    or metadata["python_guards_installed"] is not True or metadata["os_sandbox"] is not False):
                raise VerifierServiceError("verifier service: INVALID_RESPONSE")
            here = Path(__file__).resolve().parent
            expected = _VERIFIER_SOURCES + (_TRAINING_SOURCES if training_rules_path is not None else ())
            # Code files are public; no actor-side private vault open is needed.
            if (type(metadata["code_sha256"]) is not dict or set(metadata["code_sha256"]) != set(expected)
                    or any(metadata["code_sha256"][name] != hashlib.sha256((here / name).read_bytes()).hexdigest()
                           for name in expected)
                    or metadata["protocol_sha256"] != hashlib.sha256((here / "protocol_v1.yaml").read_bytes()).hexdigest()):
                raise VerifierServiceError("verifier service: CODE_OR_PROTOCOL_MISMATCH")
            self._metadata = metadata
        except _Rejected as exc:
            self.close()
            raise VerifierServiceError("verifier service: " + str(exc)) from None
        except BaseException:
            self.close()
            raise

    @property
    def metadata(self) -> dict:
        return _decode(_wire(self._metadata))

    def _io(self, descriptor: int, data: bytes | int, deadline: float):
        reading = isinstance(data, int)
        if isinstance(data, int):
            count, sending = data, memoryview(b"")
        else:
            count, sending = len(data), memoryview(data)
        pending = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, selectors.EVENT_READ if reading else selectors.EVENT_WRITE)
            position = 0
            while position < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("verifier service: TIMEOUT")
                try:
                    if reading:
                        chunk = os.read(descriptor, count - position)
                        if not chunk:
                            raise VerifierServiceError("verifier service: WORKER_EOF")
                        pending.extend(chunk)
                        position += len(chunk)
                    else:
                        position += os.write(descriptor, sending[position:])
                except BlockingIOError:
                    continue
        return bytes(pending) if reading else None

    def _response(self, request_id: int, deadline: float):
        assert self._process.stdout is not None
        descriptor = self._process.stdout.fileno()
        length = struct.unpack("!I", self._io(descriptor, 4, deadline))[0]
        if not 1 <= length <= MAX_FRAME_BYTES:
            raise VerifierServiceError("verifier service: RESPONSE_LIMIT")
        value = _decode(self._io(descriptor, length, deadline))
        if (type(value) is not dict or type(value.get("id")) is not int or value["id"] != request_id
                or type(value.get("ok")) is not bool
                or set(value) != ({"id", "ok", "result"} if value["ok"] else {"id", "ok", "error"})):
            raise VerifierServiceError("verifier service: INVALID_RESPONSE")
        if not value["ok"]:
            code = value["error"]
            if not isinstance(code, str) or code not in _ERRORS:
                raise VerifierServiceError("verifier service: INVALID_RESPONSE")
            raise _Rejected(code)
        return value["result"]

    def _request(self, method: str, problem_id: str, params: dict):
        with self._lock:
            if self._closed:
                raise VerifierServiceError("verifier service: CLOSED")
            raw = _wire({"id": self._sequence + 1, "method": method, "problem_id": problem_id, "params": params})
            if len(raw) > self._limit:
                raise ValueError("verifier service: REQUEST_LIMIT")
            deadline = time.monotonic() + self._timeout
            assert self._process.stdin is not None
            try:
                self._io(self._process.stdin.fileno(), struct.pack("!I", len(raw)) + raw, deadline)
                self._sequence += 1
                return self._response(self._sequence, deadline)
            except _Rejected as exc:
                raise VerifierServiceError("verifier service: " + str(exc)) from None
            except (OSError, TimeoutError, ValueError, VerifierServiceError):
                self.close()
                raise

    def grade(self, problem_id: str, text: str, finish_reason: str, mode: str, *,
              response_contract: str | None = None, answer_contract: str | None = None,
              retry_out_of_choice: str | None = None) -> VerifierResult:
        """Under answer_contract the actor side holds no question to re-run V8 here; the worker re-extracts,
        and audit_native_sample binds the answer to the public actor by the same V8 extraction."""
        params = {"text": text, "finish_reason": finish_reason, "mode": mode}
        if response_contract is not None:
            params["response_contract"] = response_contract
        if answer_contract is not None:
            params["answer_contract"] = answer_contract
        if retry_out_of_choice is not None:
            params["retry_out_of_choice"] = retry_out_of_choice
        value = self._request("grade", problem_id, params)
        try:
            result = _record(VerifierResult, value)
            _result(result)
            if answer_contract is not None:
                if result.verifier_version != VERIFIER_VERSION + "+" + answer_contract:
                    raise _Rejected("VERIFIER_ERROR")
                return result
            body = (text if response_contract is None else project_solution(text,
                    parse_retry_sections(text, contract_version=response_contract)))
            if (result.extracted is not None and result.extracted not in body
                    or response_contract is not None
                    and result.verifier_version != VERIFIER_VERSION + "+" + response_contract
                    + (RETRY_OUT_OF_CHOICE_TAG if retry_out_of_choice is not None else "")):
                raise _Rejected("VERIFIER_ERROR")
            return result
        except (TypeError, ValueError, _Rejected):
            self.close()
            raise VerifierServiceError("verifier service: INVALID_RESPONSE") from None

    def trace(self, problem_id: str, text: str, finish_reason: str, *,
              response_contract: str | None = None) -> dict:
        params = {"text": text, "finish_reason": finish_reason}
        if response_contract is not None:
            params["response_contract"] = response_contract
        report = self._request("trace", problem_id, params)
        try:
            validate_trace_report(report, text, finish_reason, response_contract=response_contract)
            return report
        except (TypeError, ValueError):
            self.close()
            raise VerifierServiceError("verifier service: INVALID_RESPONSE") from None

    def cause(self, problem_id: str, result: VerifierResult):
        value = self._request("cause", problem_id, {"result": json.loads(to_json(result))})
        if value is not None and (not isinstance(value, str) or value not in _CAUSES):
            self.close()
            raise VerifierServiceError("verifier service: INVALID_RESPONSE")
        return value

    def diagnosis_target(self, problem_id: str, result: VerifierResult) -> DiagTarget | None:
        value = self._request("diagnosis_target", problem_id, {"result": json.loads(to_json(result))})
        try:
            target = _record(DiagTarget, value) if value is not None else None
            if target is not None:
                _slot_names(target.wrong_slots)
            return target
        except (TypeError, ValueError, _Rejected):
            self.close()
            raise VerifierServiceError("verifier service: INVALID_RESPONSE") from None

    def route(self, problem_id: str, results: list[VerifierResult], rule: str):
        if type(results) is not list or len(results) > MAX_RESULTS:
            raise ValueError("verifier service: RESULT_LIMIT")
        value = self._request("route", problem_id, {"results": [json.loads(to_json(r)) for r in results], "rule": rule})
        try:
            return _route(value, len(results))
        except (TypeError, ValueError, _Rejected):
            self.close()
            raise VerifierServiceError("verifier service: INVALID_RESPONSE") from None

    def grade_diagnosis(self, problem_id: str, text: str, target: DiagTarget) -> float:
        value = self._request("grade_diagnosis", problem_id, {"text": text, "target": json.loads(to_json(target))})
        if type(value) not in (float, int) or value not in (0.0, 1.0, 1.25):
            self.close()
            raise VerifierServiceError("verifier service: INVALID_RESPONSE")
        return float(value)

    def close(self) -> None:
        self._closed = True
        self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--vault-root", type=Path, required=True)
    parser.add_argument("--forbidden-root", type=Path, action="append", default=[])
    parser.add_argument("--training-rules", type=Path)
    parser.add_argument("--training-rules-sha256")
    args = parser.parse_args()
    return _worker(args.vault_root, args.forbidden_root, args.training_rules, args.training_rules_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
