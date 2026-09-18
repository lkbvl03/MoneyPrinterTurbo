"""自动把本机代码同步到已配置的 Git 远程仓库（fork/personal），后台运行、
不阻塞界面。

场景：主开发机改完代码、推送到自己的 GitHub fork 后，其它"卫星机"不需要
再手动跑 update.bat——WebUI 会在后台定期检查，发现落后时自动 `git pull`，
并在页面上提示"已自动更新，请重启软件"。跟 version_checker.py 检查上游
harry0703 正式版本是两回事：这里跟踪的是用户自己 fork 的最新代码。
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from loguru import logger

from app.config import config

_GIT_TIMEOUT_SECONDS = 20
_PULL_TIMEOUT_SECONDS = 120
# 30 分钟检查一次：足够及时看到主机新推送的代码，又不会让每台卫星机反复
# fetch 造成不必要的网络请求。
_CHECK_TTL_SECONDS = 30 * 60


def _portable_root() -> Optional[Path]:
    candidate = Path(config.root_dir).resolve().parent
    if (candidate / "lib").is_dir():
        return candidate
    return None


def _resolve_git_binary() -> Optional[str]:
    portable_root = _portable_root()
    if portable_root:
        bundled_git = portable_root / "lib" / "git" / "bin" / "git.exe"
        if bundled_git.is_file():
            return str(bundled_git)
    return shutil.which("git")


def _run(cmd: list[str], timeout: int = _GIT_TIMEOUT_SECONDS) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )


@dataclass(frozen=True)
class GitSyncSnapshot:
    """后台同步检查的即时状态，供 WebUI 无阻塞地读取。

    state: idle(还没查过) | up_to_date | updated | update_failed | unavailable
    """

    complete: bool
    state: str = "idle"
    detail: str = ""
    pulled_commits: int = 0


def _do_sync() -> GitSyncSnapshot:
    git_bin = _resolve_git_binary()
    if not git_bin:
        return GitSyncSnapshot(True, "unavailable")

    repo_dir = config.root_dir
    try:
        is_repo = _run(
            [git_bin, "-C", repo_dir, "rev-parse", "--is-inside-work-tree"]
        )
        if is_repo.returncode != 0:
            return GitSyncSnapshot(True, "unavailable")

        # 机器从别的电脑/账号复制过来时，Git 常因为"dubious ownership"
        # 拒绝执行任何命令（含下面的 fetch/pull），先加这条例外规则。
        _run([git_bin, "config", "--global", "--add", "safe.directory", "*"])

        fetch = _run([git_bin, "-C", repo_dir, "fetch", "--quiet"])
        if fetch.returncode != 0:
            return GitSyncSnapshot(
                True, "unavailable", fetch.stderr.strip()[:200]
            )

        upstream = _run(
            [
                git_bin,
                "-C",
                repo_dir,
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{u}",
            ]
        )
        if upstream.returncode != 0:
            # 当前分支没有绑定远程跟踪分支，没有"正确版本"可比较。
            return GitSyncSnapshot(True, "unavailable")

        behind = _run(
            [git_bin, "-C", repo_dir, "rev-list", "--count", "HEAD..@{u}"]
        )
        behind_count = int(behind.stdout.strip() or 0) if behind.returncode == 0 else 0

        if behind_count == 0:
            return GitSyncSnapshot(True, "up_to_date")

        # --autostash：这台机器的 config.toml 等文件通常已经被装机脚本改过
        # 本机专属的路径，属于正常的本地改动；自动存起来、pull 完再放回去，
        # 而不是被"有未提交修改"卡住不动。
        pull = _run(
            [git_bin, "-C", repo_dir, "pull", "--autostash"],
            timeout=_PULL_TIMEOUT_SECONDS,
        )
        if pull.returncode != 0:
            return GitSyncSnapshot(
                True,
                "update_failed",
                (pull.stderr or pull.stdout).strip()[:300],
            )
        return GitSyncSnapshot(True, "updated", pulled_commits=behind_count)
    except subprocess.TimeoutExpired:
        return GitSyncSnapshot(True, "unavailable", "timeout")
    except Exception as e:
        logger.warning(f"git auto-sync failed: {e}")
        return GitSyncSnapshot(True, "unavailable", str(e))


class AsyncGitSyncChecker:
    """跟 version_checker.AsyncUpdateChecker 同样的模式：网络/子进程调用放
    进后台线程，WebUI 只读取缓存的即时快照，避免阻塞页面渲染。"""

    def __init__(
        self,
        check: Callable[[], GitSyncSnapshot] = _do_sync,
        ttl_seconds: float = _CHECK_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._check = check
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._snapshot = GitSyncSnapshot(False)
        self._completed_at: Optional[float] = None
        self._checking = False

    def poll(self) -> GitSyncSnapshot:
        now = self._clock()
        with self._lock:
            cache_is_fresh = (
                self._completed_at is not None
                and now - self._completed_at < self._ttl_seconds
            )
            if cache_is_fresh:
                return self._snapshot
            if self._checking:
                return GitSyncSnapshot(False)

            self._checking = True
            worker = threading.Thread(
                target=self._run_check, name="mpt-git-sync", daemon=True
            )
            worker.start()
        return GitSyncSnapshot(False)

    def _run_check(self) -> None:
        try:
            snapshot = self._check()
        except Exception:
            logger.exception("unexpected error during git auto-sync")
            snapshot = GitSyncSnapshot(True, "unavailable")
        with self._lock:
            self._snapshot = snapshot
            self._completed_at = self._clock()
            self._checking = False


_ASYNC_GIT_SYNC_CHECKER = AsyncGitSyncChecker()


def poll_git_sync() -> GitSyncSnapshot:
    """读取全局后台检查器状态，避免不同 Streamlit 会话各自重复 fetch/pull。"""
    return _ASYNC_GIT_SYNC_CHECKER.poll()
