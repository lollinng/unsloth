# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows llama.cpp GPU binary selection and launch tuning regressions.

See unslothai/unsloth#5999, #5941, and #5692.
"""

from __future__ import annotations

import subprocess
import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda name = None: __import__("logging").getLogger(
    name or "structlog"
)
sys.modules.setdefault("structlog", _structlog_stub)

_httpx_stub = _types.ModuleType("httpx")
for _exc_name in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))


class _FakeTimeout:
    def __init__(self, *a, **kw):
        pass


_httpx_stub.Timeout = _FakeTimeout
_httpx_stub.Client = type(
    "Client",
    (),
    {
        "__init__": lambda self, **kw: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

_EXE = "llama-server.exe"


def _make_exe(path: Path) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"MZ")
    return path


def _make_windows_layout(root: Path, *, cpu_bin: bool = True, cpu_release: bool = True):
    cpu_bin_path = root / "build" / "bin" / _EXE
    cpu_release_path = root / "build" / "bin" / "Release" / _EXE
    cuda_release_path = root / "build-cuda" / "bin" / "Release" / _EXE
    if cpu_bin:
        _make_exe(cpu_bin_path)
    if cpu_release:
        _make_exe(cpu_release_path)
    _make_exe(cuda_release_path)
    return cpu_bin_path, cpu_release_path, cuda_release_path


@pytest.fixture
def isolated_discovery(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)

    storage_stub = _types.ModuleType("utils.paths.storage_roots")
    storage_stub.studio_root = lambda: home / ".unsloth" / "studio"
    monkeypatch.setitem(sys.modules, "utils.paths.storage_roots", storage_stub)

    import shutil

    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
    return home


def _set_nvidia(monkeypatch, present: bool):
    monkeypatch.setattr(
        LlamaCppBackend, "_nvidia_available", staticmethod(lambda: present)
    )


def test_direct_env_binary_wins_without_nvidia_probe(
    tmp_path, monkeypatch, isolated_discovery
):
    direct = _make_exe(tmp_path / "direct" / _EXE)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LLAMA_SERVER_PATH", str(direct))

    def _boom():
        raise AssertionError("LLAMA_SERVER_PATH should not probe nvidia")

    monkeypatch.setattr(LlamaCppBackend, "_nvidia_available", staticmethod(_boom))

    assert LlamaCppBackend._find_llama_server_binary() == str(direct)


def test_custom_root_prefers_cuda_release_before_cpu_build_bin(
    tmp_path, monkeypatch, isolated_discovery
):
    root = tmp_path / "custom"
    _cpu_bin, _cpu_release, cuda_release = _make_windows_layout(root)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root))
    _set_nvidia(monkeypatch, True)

    assert LlamaCppBackend._find_llama_server_binary() == str(cuda_release)


def test_home_root_prefers_cuda_release_before_cpu_release(
    monkeypatch, isolated_discovery
):
    root = isolated_discovery / ".unsloth" / "llama.cpp"
    _cpu_bin, _cpu_release, cuda_release = _make_windows_layout(
        root, cpu_bin = False
    )
    monkeypatch.setattr(sys, "platform", "win32")
    _set_nvidia(monkeypatch, True)

    assert LlamaCppBackend._find_llama_server_binary() == str(cuda_release)


def test_windows_without_nvidia_keeps_default_cpu_order(
    tmp_path, monkeypatch, isolated_discovery
):
    root = tmp_path / "custom"
    cpu_bin, _cpu_release, _cuda_release = _make_windows_layout(root)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root))
    _set_nvidia(monkeypatch, False)

    assert LlamaCppBackend._find_llama_server_binary() == str(cpu_bin)


def test_linux_discovery_unchanged_and_does_not_probe_nvidia(
    tmp_path, monkeypatch, isolated_discovery
):
    root = tmp_path / "custom"
    linux_bin = _make_exe(root / "build" / "bin" / "llama-server")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root))

    def _boom():
        raise AssertionError("_nvidia_available must not run on Linux")

    monkeypatch.setattr(LlamaCppBackend, "_nvidia_available", staticmethod(_boom))

    assert LlamaCppBackend._find_llama_server_binary() == str(linux_bin)


def test_nvidia_available_true_on_gpu_listing(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: _types.SimpleNamespace(
            returncode = 0,
            stdout = "GPU 0: NVIDIA GeForce RTX 5070 (UUID: GPU-...)",
        ),
    )

    assert LlamaCppBackend._nvidia_available() is True


def test_nvidia_available_false_on_timeout(monkeypatch):
    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd = "nvidia-smi", timeout = 5)

    monkeypatch.setattr(subprocess, "run", _raise)

    assert LlamaCppBackend._nvidia_available() is False


@pytest.mark.parametrize(
    ("use_fit", "gpu_indices", "extra_args", "expected"),
    [
        (False, [0], None, True),
        (False, [0], ["--gpu-layers", "0"], False),
        (False, [0], ["--gpu-layers=-1"], True),
        (False, [0], ["-ngl", "16"], False),
        (False, [0], ["-ngl-1"], True),
        (False, [0], ["--fit", "on"], False),
        (False, [0], ["--fit", "on", "--fit", "off"], True),
        (True, None, ["--fit", "off", "-ngl", "-1"], True),
        (True, None, ["-ngl", "-1"], False),
        (False, None, None, False),
    ],
)
def test_effective_full_gpu_offload_respects_last_wins_extra_args(
    use_fit, gpu_indices, extra_args, expected
):
    assert (
        LlamaCppBackend._effective_full_gpu_offload(
            use_fit = use_fit,
            gpu_indices = gpu_indices,
            extra_args = extra_args,
        )
        is expected
    )


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--threads", "8"],
        ["--threads=8"],
        ["-t", "8"],
        ["-t8"],
    ],
)
def test_extra_args_set_threads_detects_supported_forms(extra_args):
    assert LlamaCppBackend._extra_args_set_threads(extra_args) is True
