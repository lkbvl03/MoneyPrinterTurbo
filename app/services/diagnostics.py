"""系统自检：给 WebUI 的"Kiểm tra hệ thống" (System Check) 标签页使用。

每一项检查互相独立、内部吞掉所有异常，绝不让某一项检查失败拖垮整个页面——
便携包环境（Windows 打包 python/ffmpeg/git）和 Docker/源码环境的可用检查项
不完全相同，用不到的项目返回 "skip" 而不是报错。
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from loguru import logger

from app.config import config

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_ERROR = "error"
STATUS_SKIP = "skip"

_GIT_TIMEOUT_SECONDS = 15
_SUBPROCESS_TIMEOUT_SECONDS = 10


@dataclass(frozen=True, slots=True)
class CheckResult:
    id: str
    label: str
    status: str
    detail: str = ""


def _portable_root() -> Path | None:
    """便携包场景下，Windows 独立分发的根目录（lib/、Text-to-Speech-Offline/
    所在的那一层），比 `config.root_dir`（MoneyPrinterTurbo 应用目录本身）
    高一级。源码/Docker 运行时通常不存在，调用方需要处理 None。
    """
    candidate = Path(config.root_dir).resolve().parent
    if (candidate / "lib").is_dir():
        return candidate
    return None


def _run(cmd: list[str], timeout: int = _SUBPROCESS_TIMEOUT_SECONDS) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )


def check_python_runtime() -> CheckResult:
    versions = []
    missing = []
    for module_name in ("torch", "onnxruntime", "faster_whisper", "moviepy"):
        try:
            module = __import__(module_name)
            versions.append(f"{module_name} {getattr(module, '__version__', '?')}")
        except Exception as e:
            missing.append(f"{module_name} ({e})")

    if missing:
        return CheckResult(
            "python_runtime",
            "Python & thư viện AI lõi",
            STATUS_ERROR,
            "Thiếu/lỗi: " + "; ".join(missing),
        )
    return CheckResult(
        "python_runtime",
        "Python & thư viện AI lõi",
        STATUS_OK,
        f"Python {platform.python_version()} — " + ", ".join(versions),
    )


def check_ffmpeg() -> CheckResult:
    try:
        from app.utils.utils import get_ffmpeg_binary

        ffmpeg_bin = get_ffmpeg_binary()
        result = _run([ffmpeg_bin, "-version"])
        if result.returncode != 0:
            return CheckResult(
                "ffmpeg", "FFmpeg", STATUS_ERROR,
                f"'{ffmpeg_bin}' chạy lỗi (mã {result.returncode}): {result.stderr[:200]}",
            )
        first_line = (result.stdout or "").splitlines()[0] if result.stdout else ""
        return CheckResult("ffmpeg", "FFmpeg", STATUS_OK, f"{ffmpeg_bin} — {first_line}")
    except FileNotFoundError:
        return CheckResult(
            "ffmpeg", "FFmpeg", STATUS_ERROR,
            "Không tìm thấy FFmpeg khả dụng (kiểm tra IMAGEIO_FFMPEG_EXE / PATH).",
        )
    except Exception as e:
        return CheckResult("ffmpeg", "FFmpeg", STATUS_ERROR, str(e))


def check_llm_connection() -> CheckResult:
    try:
        from app.services import llm

        provider_id = str(config.app.get("llm_provider", "")).lower()
        ok, message, elapsed = llm.test_connection()
        if ok:
            return CheckResult(
                "llm_connection",
                "Kết nối LLM đang chọn",
                STATUS_OK,
                f"{provider_id or '?'} — phản hồi sau {elapsed:.2f}s",
            )
        # `message` từ llm.test_connection() thường đã tự có tiền tố
        # "<provider>: ..." (xem app/services/llm.py raise ValueError),
        # nên không thêm provider_id ở đây nữa để tránh lặp "grok: grok: ...".
        return CheckResult("llm_connection", "Kết nối LLM đang chọn", STATUS_ERROR, message)
    except Exception as e:
        return CheckResult("llm_connection", "Kết nối LLM đang chọn", STATUS_ERROR, str(e))


def check_tts_engine() -> CheckResult:
    tts_server = str(config.ui.get("tts_server", "")).strip().lower()
    label = f"Giọng đọc TTS đang chọn ({tts_server or '?'})"

    if not tts_server:
        return CheckResult("tts_engine", label, STATUS_WARN, "Chưa chọn engine TTS trong Cài đặt giao diện.")

    if tts_server == "piper":
        try:
            from app.services.voice import get_all_piper_voices

            voices = get_all_piper_voices()
            if voices:
                return CheckResult("tts_engine", label, STATUS_OK, f"Tìm thấy {len(voices)} giọng Piper.")
            return CheckResult(
                "tts_engine", label, STATUS_ERROR,
                "Không tìm thấy file giọng (.onnx) nào trong thư mục Piper models_dir đã cấu hình.",
            )
        except Exception as e:
            return CheckResult("tts_engine", label, STATUS_ERROR, str(e))

    if tts_server == "vieneu":
        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir()
            cached_repos = {repo.repo_id for repo in cache_info.repos}
            required = {
                "pnnbao-ump/VieNeu-TTS-v3-Turbo",
                "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX",
            }
            missing = required - cached_repos
            if missing:
                return CheckResult(
                    "tts_engine", label, STATUS_WARN,
                    "Model chưa có trong cache, sẽ tự tải khi dùng lần đầu (cần Internet): "
                    + ", ".join(sorted(missing)),
                )
            return CheckResult("tts_engine", label, STATUS_OK, "Model VieNeu TTS đã có sẵn trong cache máy.")
        except Exception as e:
            return CheckResult("tts_engine", label, STATUS_WARN, f"Không kiểm tra được cache model: {e}")

    # Các engine khác (edge/azure/elevenlabs/chatterbox/sonilo...) cần key/
    # server riêng không kiểm tra sâu ở đây — chỉ xác nhận đã chọn cái nào.
    return CheckResult("tts_engine", label, STATUS_SKIP, "Không kiểm tra chi tiết engine này; sẽ được xác thực khi tạo video.")


def check_whisper_model() -> CheckResult:
    model_size = str(config.whisper.get("model_size", "medium"))
    try:
        from faster_whisper.utils import download_model

        download_model(model_size, local_files_only=True)
        return CheckResult(
            "whisper_model", "Mô hình phụ đề Whisper", STATUS_OK,
            f"Model '{model_size}' đã có sẵn trong cache máy.",
        )
    except Exception:
        return CheckResult(
            "whisper_model", "Mô hình phụ đề Whisper", STATUS_WARN,
            f"Model '{model_size}' chưa có trong cache, sẽ tự tải khi tạo phụ đề lần đầu (cần Internet).",
        )


def check_material_source() -> CheckResult:
    video_source = str(config.app.get("video_source", "")).strip().lower()
    label = f"Nguồn tư liệu video đang chọn ({video_source or '?'})"
    key_field_map = {
        "pexels": "pexels_api_keys",
        "pixabay": "pixabay_api_keys",
        "coverr": "coverr_api_keys",
        "twelvelabs": "twelvelabs_api_keys",
    }
    field = key_field_map.get(video_source)
    if field is None:
        return CheckResult("material_source", label, STATUS_SKIP, "Nguồn tư liệu tự tải lên, không cần API key.")

    keys = config.app.get(field, [])
    if isinstance(keys, str):
        keys = [keys]
    keys = [k for k in keys if str(k).strip()]
    if keys:
        return CheckResult("material_source", label, STATUS_OK, f"Đã cấu hình {len(keys)} API key.")
    return CheckResult("material_source", label, STATUS_ERROR, f"Chưa có API key nào trong '{field}'.")


def check_disk_space() -> CheckResult:
    try:
        storage_dir = Path(config.root_dir) / "storage"
        check_dir = storage_dir if storage_dir.exists() else Path(config.root_dir)
        usage = shutil.disk_usage(check_dir)
        free_gb = usage.free / (1024**3)
        if free_gb < 2:
            return CheckResult("disk_space", "Dung lượng ổ đĩa trống", STATUS_ERROR, f"Chỉ còn {free_gb:.1f} GB, có thể lỗi khi render video dài.")
        if free_gb < 10:
            return CheckResult("disk_space", "Dung lượng ổ đĩa trống", STATUS_WARN, f"Còn {free_gb:.1f} GB.")
        return CheckResult("disk_space", "Dung lượng ổ đĩa trống", STATUS_OK, f"Còn {free_gb:.1f} GB.")
    except Exception as e:
        return CheckResult("disk_space", "Dung lượng ổ đĩa trống", STATUS_SKIP, str(e))


def _resolve_git_binary() -> str | None:
    portable_root = _portable_root()
    if portable_root:
        bundled_git = portable_root / "lib" / "git" / "bin" / "git.exe"
        if bundled_git.is_file():
            return str(bundled_git)
    return shutil.which("git")


def check_git_update() -> CheckResult:
    label = "Phiên bản code & cập nhật GitHub"
    git_bin = _resolve_git_binary()
    if not git_bin:
        return CheckResult("git_update", label, STATUS_SKIP, "Không tìm thấy git (bình thường nếu chạy từ bản build/Docker).")

    repo_dir = config.root_dir
    try:
        is_repo = _run([git_bin, "-C", repo_dir, "rev-parse", "--is-inside-work-tree"], timeout=_GIT_TIMEOUT_SECONDS)
        if is_repo.returncode != 0:
            return CheckResult("git_update", label, STATUS_SKIP, "Thư mục ứng dụng không phải git repo.")

        # "dubious ownership": xảy ra khi thư mục được copy từ máy/tài khoản
        # khác sang, khiến MỌI lệnh git phía dưới thất bại một cách khó hiểu
        # nếu không tự thêm ngoại lệ này trước.
        _run([git_bin, "config", "--global", "--add", "safe.directory", "*"], timeout=_GIT_TIMEOUT_SECONDS)

        fetch = _run([git_bin, "-C", repo_dir, "fetch", "--quiet"], timeout=_GIT_TIMEOUT_SECONDS)
        if fetch.returncode != 0:
            return CheckResult(
                "git_update", label, STATUS_WARN,
                f"Không tải được thông tin mới từ GitHub (có thể do chưa có Internet): {fetch.stderr.strip()[:200]}",
            )

        upstream = _run(
            [git_bin, "-C", repo_dir, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            timeout=_GIT_TIMEOUT_SECONDS,
        )
        if upstream.returncode != 0:
            return CheckResult("git_update", label, STATUS_SKIP, "Nhánh hiện tại chưa gắn với remote nào để so sánh.")

        behind = _run([git_bin, "-C", repo_dir, "rev-list", "--count", "HEAD..@{u}"], timeout=_GIT_TIMEOUT_SECONDS)
        ahead = _run([git_bin, "-C", repo_dir, "rev-list", "--count", "@{u}..HEAD"], timeout=_GIT_TIMEOUT_SECONDS)
        behind_count = int(behind.stdout.strip() or 0) if behind.returncode == 0 else 0
        ahead_count = int(ahead.stdout.strip() or 0) if ahead.returncode == 0 else 0

        if behind_count == 0 and ahead_count == 0:
            return CheckResult("git_update", label, STATUS_OK, "Đang ở bản mới nhất.")

        parts = []
        status = STATUS_OK
        if behind_count > 0:
            status = STATUS_WARN
            parts.append(
                f"Có {behind_count} commit mới trên GitHub chưa cập nhật — "
                "chạy update.bat hoặc Cai-Dat-Tren-May-Moi.exe."
            )
        if ahead_count > 0:
            parts.append(
                f"Có {ahead_count} commit cục bộ chưa đẩy lên — dùng Day-Code-Len-GitHub.bat."
            )
        return CheckResult("git_update", label, status, " ".join(parts))
    except subprocess.TimeoutExpired:
        return CheckResult("git_update", label, STATUS_WARN, "Kiểm tra git quá thời gian chờ (mạng chậm hoặc không có Internet).")
    except Exception as e:
        logger.warning(f"git update check failed: {e}")
        return CheckResult("git_update", label, STATUS_SKIP, str(e))


# Thứ tự hiển thị: nhanh & không cần mạng trước, chậm & cần mạng sau — để
# người dùng thấy kết quả dần dần thay vì màn hình trắng trong lúc chờ.
_ALL_CHECKS: tuple[Callable[[], CheckResult], ...] = (
    check_python_runtime,
    check_ffmpeg,
    check_disk_space,
    check_material_source,
    check_tts_engine,
    check_whisper_model,
    check_llm_connection,
    check_git_update,
)


def run_all_checks() -> list[CheckResult]:
    results = []
    for check_fn in _ALL_CHECKS:
        try:
            results.append(check_fn())
        except Exception as e:
            logger.error(f"diagnostics check {check_fn.__name__} crashed: {e}")
            results.append(
                CheckResult(check_fn.__name__, check_fn.__name__, STATUS_ERROR, f"Lỗi nội bộ: {e}")
            )
    return results
