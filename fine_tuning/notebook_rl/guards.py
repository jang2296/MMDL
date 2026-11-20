"""One-shot Python audit checks, not an OS sandbox or a native-code boundary."""
from __future__ import annotations

import os
import stat
import sys
import threading
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

_TEACHERS = ("openai", "anthropic", "google.generativeai", "google.genai", "cohere", "mistralai")
_installed = False
_lock = threading.Lock()
_vllm_sdk_methods: tuple[tuple[type, str], ...] = ()


def _deny_sdk_call(*args, **kwargs):
    raise PermissionError("external teacher/API client use blocked; schema dependency only")


def prepare_vllm_schema_dependency() -> dict:
    """Trusted startup only: vLLM needs OpenAI types, never an API client.

    Run after the exact runtime lock check, before guard installation or model
    imports. Refuse preloaded SDKs/clients; this is not a native/network sandbox.
    """
    global _vllm_sdk_methods
    with _lock:
        if _installed or _vllm_sdk_methods:
            raise RuntimeError("schema dependency must be initialized once before guards")
        assert_no_teacher_imports()
        if version("vllm") != "0.11.0" or version("openai") != "3.16.2":
            raise RuntimeError("unreviewed vLLM/OpenAI schema dependency versions")
        sdk = import_module("openai")
        base = import_module("openai._base_client")
        if sdk.__version__ != "3.16.2" or sdk._client is not None:
            raise RuntimeError("pre-existing or mismatched API client dependency")
        clients = {getattr(sdk, name) for name in (
            "OpenAI", "AsyncOpenAI", "AzureOpenAI", "AsyncAzureOpenAI", "BedrockOpenAI", "AsyncBedrockOpenAI")}
        bases = (base.SyncAPIClient, base.AsyncAPIClient)
        methods = tuple((cls, "__init__") for cls in clients | set(bases)) + tuple((cls, "request") for cls in bases)
        if any(not isinstance(cls, type) or not callable(getattr(cls, method, None)) for cls, method in methods):
            raise RuntimeError("unreviewed API client layout")
        # Public constructors run provider setup before the common base constructor.
        for cls, method in methods:
            setattr(cls, method, _deny_sdk_call)
        _vllm_sdk_methods = methods
        return {"dependency": "openai", "version": sdk.__version__, "purpose": "VLLM_SCHEMA_TYPES_ONLY",
                "client_construction_and_requests_blocked": True, "os_network_sandbox": False}


def _teacher(name: str) -> bool:
    if _vllm_sdk_methods and (name == "openai" or name.startswith("openai.")):
        return False
    return any(name == prefix or name.startswith(prefix + ".") for prefix in _TEACHERS)


def assert_no_teacher_imports() -> None:
    if _vllm_sdk_methods and (any(getattr(cls, method) is not _deny_sdk_call for cls, method in _vllm_sdk_methods)
                              or getattr(sys.modules.get("openai"), "_client", None) is not None):
        raise RuntimeError("schema-only API client guard changed")
    if any(_teacher(name) for name in tuple(sys.modules)):
        raise RuntimeError("teacher module already imported")


class _NoTeacherFinder:
    def find_spec(self, fullname: str, path=None, target=None):
        # importlib.import_module can load Python sources without an import audit event.
        if _teacher(fullname):
            raise PermissionError("teacher import blocked")
        return None


def _paths(path: Path) -> tuple[Path, Path]:
    absolute = Path(os.path.abspath(path))
    return absolute, absolute.resolve()


def install_guards(*, forbidden_roots: list[Path], vault_files: list[Path],
                   allow_vault: bool, allow_benchmark: bool) -> None:
    """Install before untrusted work; already-open handles/native I/O remain outside this check."""
    global _installed
    with _lock:
        if _installed:
            raise RuntimeError("guards already installed in this process")
        if type(allow_vault) is not bool or type(allow_benchmark) is not bool:
            raise ValueError("guard permissions must be explicit booleans")
        roots = tuple(pair for root in forbidden_roots for pair in _paths(root))
        vaults = tuple(pair for path in vault_files for pair in _paths(path))
        assert_no_teacher_imports()

        def audit(event: str, args: tuple) -> None:
            if event == "import" and args and isinstance(args[0], str) and _teacher(args[0]):
                raise PermissionError("teacher import blocked")
            if event != "open" or not args:
                return
            raw = args[0]
            try:
                if type(raw) is int:
                    # fdopen and /proc/self/fd aliases must not bypass file policy.
                    candidates = _paths(Path(f"/proc/self/fd/{raw}"))
                    identity = os.fstat(raw)
                else:
                    candidates = _paths(Path(os.fsdecode(os.fspath(raw))))
                    try:
                        identity = candidates[1].stat()
                    except FileNotFoundError:
                        identity = None
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise PermissionError("open path cannot be checked") from exc
            if (identity is not None and stat.S_ISREG(identity.st_mode) and identity.st_nlink > 1
                    and ((roots and not allow_benchmark) or (vaults and not allow_vault))):
                raise PermissionError("hard-linked open path cannot be checked")
            if not allow_benchmark and any(
                path == root or root in path.parents for path in candidates for root in roots
            ):
                raise PermissionError("benchmark path blocked")
            if not allow_benchmark and roots:
                # WSL's case-insensitive mounts and bind aliases need inode ancestry too.
                try:
                    root_ids = set()
                    for root in roots:
                        try:
                            info = root.stat()
                        except FileNotFoundError:
                            continue
                        root_ids.add((info.st_dev, info.st_ino))
                    for parent in (candidates[1], *candidates[1].parents):
                        try:
                            info = parent.stat()
                        except FileNotFoundError:
                            continue
                        if (info.st_dev, info.st_ino) in root_ids:
                            raise PermissionError("benchmark path blocked")
                except OSError as exc:
                    raise PermissionError("benchmark path blocked or cannot be checked") from exc
            if not allow_vault:
                if any(path in vaults for path in candidates):
                    raise PermissionError("vault path blocked")
                if identity is not None:
                    for vault in vaults:
                        try:
                            private = vault.stat()
                        except FileNotFoundError:
                            continue
                        if (identity.st_dev, identity.st_ino) == (private.st_dev, private.st_ino):
                            raise PermissionError("vault path blocked")

        sys.addaudithook(audit)
        sys.meta_path.insert(0, _NoTeacherFinder())
        _installed = True
