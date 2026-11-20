"""Reproduce the original non-LLM OCR, keeping its dependencies out of the training environment."""
from __future__ import annotations

import fcntl
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

from .audit import digest

_UBUNTU = "https://archive.ubuntu.com/ubuntu/pool/"
_DEBS = {
    "main/liba/libarchive/libarchive13_3.6.0-1ubuntu1.9_amd64.deb":
        "979865c53912b227b8d65d208cd5810bcc3c535a042203a68eb1337dadd6d6c4",
    "main/g/giflib/libgif7_5.1.9-2ubuntu0.3_amd64.deb":
        "506682225d0080ad508a1609c93626c8d689046a4fef9cb9739993cf47d2704d",
    "main/g/gcc-12/libgomp1_12.3.0-1ubuntu1~22.04.3_amd64.deb":
        "870c27299185a5dd4accad3b15bf82a7409fd7073cccaa8025875307da4d0ce2",
    "universe/l/leptonlib/liblept5_1.82.0-3build1_amd64.deb":
        "936cf3ce4f192bd33e1ac45921f5f530e7681cfc78d67048f0acd6c2d39f894d",
    "main/o/openjpeg2/libopenjp2-7_2.4.0-6ubuntu0.5_amd64.deb":
        "8efb91572f5f02dcb2bd7ccee580411f9065656523210ac9dfdd5b4c5f07c0b3",
    "universe/t/tesseract/libtesseract4_4.1.1-2.1build1_amd64.deb":
        "1c428586aa9d0b5ef5a8a107f958d7eac99d734b6cb98338488c3aae9e8bd702",
    "main/libw/libwebp/libwebpmux3_1.2.2-2ubuntu0.22.04.2_amd64.deb":
        "9c1cbe1c06be96e91f104d48ac37058ecbc5503bea3e51bd0eb0e6da6462f140",
    "universe/t/tesseract-lang/tesseract-ocr-eng_4.00~git30-7274cfa-1.1_all.deb":
        "a4709bd43f71c83f9171071dddaf87a125f467d5258bd323048d49b054ef5eb8",
    "universe/t/tesseract/tesseract-ocr_4.1.1-2.1build1_amd64.deb":
        "5d33a2cc1484aa5032d07f51a4375d414d6868347691239874f0cfc03280a18f",
}
_WHEEL_URL = ("https://files.pythonhosted.org/packages/84/21/a35af28dcc61f37ed850a2d64c65c701321dfbf25085e469d5559360cbbf/"
              "pillow-12.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl")
_WHEEL_SHA = "78cb2c6865a35ab8ff8b75fd122f6033b92a62c82801110e48ddd6c936a45d91"


def _object(cache, url, sha, offline):
    path = cache / sha
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
        return path
    if offline:
        raise FileNotFoundError(f"v18 OCR offline object missing: {sha}")
    with urllib.request.urlopen(url, timeout=45) as response:
        data = response.read(16 * 1024**2 + 1)
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError(f"v18 OCR upstream digest mismatch: {sha}")
    path.write_bytes(data)
    return path


@lru_cache(maxsize=4)
def _runtime(cache_dir: Path, kind: str, offline: bool) -> Path:
    if kind == "tesseract" and os.environ.get("MMDL_V18_OCR_ROOT"):
        return Path(os.environ["MMDL_V18_OCR_ROOT"]).resolve(strict=True)
    cache = cache_dir / "ocr-v18"
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / kind
    with (cache / "install.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not dest.is_dir():
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                stage = Path(tmp) / kind
                stage.mkdir()
                if kind == "tesseract":
                    for relative, sha in _DEBS.items():
                        deb = _object(cache, _UBUNTU + relative, sha, offline)
                        subprocess.run(["dpkg-deb", "-x", str(deb), str(stage)], check=True)
                else:
                    if sys.version_info[:2] != (3, 12):
                        raise ValueError("ChartQA OCR requires CPython 3.12 on Linux x86_64")
                    wheel = _object(cache, _WHEEL_URL, _WHEEL_SHA, offline)
                    with zipfile.ZipFile(wheel) as archive:
                        archive.extractall(stage)  # Fixed, SHA-verified wheel.
                stage.rename(dest)
    return dest


@lru_cache(maxsize=4)
def _ocr(root: Path):
    from fine_tuning.notebook_rl.exclusion_engines import OCR
    return OCR(root)


def printed_text(image_path: Path, cache_dir: Path, *, offline=False) -> tuple[str, ...]:
    from PIL import Image
    engine = _ocr(_runtime(cache_dir, "tesseract", offline))
    key = digest({"engine": digest(engine.identity), "image": hashlib.sha256(image_path.read_bytes()).hexdigest()})
    cache = cache_dir / "ocr-v18" / "printed"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (key + ".json")
    if path.is_file():
        try:
            saved = json.loads(path.read_text())
            if (isinstance(saved["lines"], list) and all(isinstance(x, str) for x in saved["lines"])
                    and saved["lines"] and saved["sha256"] == digest([key, saved["lines"]])):
                return tuple(saved["lines"])
        except (KeyError, ValueError, TypeError):
            pass
    with Image.open(image_path) as image:
        result = engine.recognize(image)["text"]
    lines = tuple(line.strip() for line in result.splitlines() if line.strip()) or ("",)
    with tempfile.NamedTemporaryFile(dir=cache, mode="w", delete=False) as stream:
        json.dump({"lines": lines, "sha256": digest([key, lines])}, stream)
    os.replace(stream.name, path)
    return lines



def _visibility(image_path, chart, root):
    from PIL import Image
    engine = _ocr(root)
    if engine.identity["packages"]["Pillow"] != "12.3.0":
        raise ValueError("historical ChartQA OCR requires Pillow 12.3.0")
    with Image.open(image_path) as image:
        observed = engine.recognize(image.convert("RGB"))
    return {"engine_type": "NON_LLM_OCR", "engine": engine.identity, "text": observed["text"],
            "tokens": re.findall(r"[+-]?\d[\d,]*(?:\.\d+)?", observed["text"]),
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "annotation_sha256": digest(chart), "status": observed["status"], "source_values_from_ocr": False}


def visibility_receipt(image_path, chart_dict, cache_dir, *, offline=False):
    root = _runtime(cache_dir, "tesseract", offline)
    if importlib.metadata.version("Pillow") == "12.3.0":
        return _visibility(image_path, chart_dict, root)
    python = os.environ.get("MMDL_V18_OCR_PYTHON") or sys.executable
    env = dict(os.environ)
    if python == sys.executable:
        package = _runtime(cache_dir, "pillow-12.3.0", offline)
        env["PYTHONPATH"] = os.pathsep.join([str(package), *sys.path])
    result = subprocess.run([python, "-m", __name__ if __name__ != "__main__" else
        "fine_tuning.error_note.v18_chart_ocr", str(image_path), str(root)], input=json.dumps(chart_dict),
        env=env, text=True, capture_output=True, timeout=80)
    if result.returncode:
        raise ValueError(f"v18 isolated OCR failed: {result.stderr[-1000:]}")
    return json.loads(result.stdout)


if __name__ == "__main__":
    print(json.dumps(_visibility(Path(sys.argv[1]), json.load(sys.stdin), Path(sys.argv[2]))))
