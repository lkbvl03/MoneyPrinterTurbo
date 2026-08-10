"""
Kho anh/video local, gan tag tu ten file + thu muc cha (khong goi API/LLM
nao - MPT chi lo phan dung/ghep video, khong phu thuoc API key de hoat
dong), ket qua cache trong _tags.json ben trong tung thu muc ti le khung
hinh, sau do dung de tim material khop voi chu de/kich ban - giong vai tro
cua app/services/material.py tim tren Pexels/Pixabay nhung nguon la file
local thay vi API ben ngoai.

Cau truc thu muc (co the chia them thu muc con tuy y de sap xep theo the
loai, se duoc quet de quy):
    storage/media_library/9-16/*.mp4, *.jpg, ...
    storage/media_library/9-16/<the_loai>/*.mp4, *.jpg, ...
    storage/media_library/16-9/*.mp4, *.jpg, ...
    storage/media_library/1-1/*.mp4, *.jpg, ...

Ten file VA ten thu muc con la nguon duy nhat de tim kiem khop noi dung -
dat ten mo ta ro (vi du "thien_nhien/bien_hoanghon.mp4") thi tim moi chinh
xac. Dung dau "_", "-", "." hoac khoang trang de tach cac tu trong ten.
"""

import json
import os
import re
from typing import List

from loguru import logger

from app.models.schema import MaterialInfo, VideoAspect
from app.utils import utils

ASPECT_DIR_NAMES = {
    VideoAspect.portrait: "9-16",
    VideoAspect.landscape: "16-9",
    VideoAspect.square: "1-1",
}

_IMAGE_EXTS = {"jpg", "jpeg", "png", "bmp", "webp"}
_VIDEO_EXTS = {"mp4", "mov", "mkv", "webm"}
_TAGS_FILENAME = "_tags.json"
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_NAME_SPLIT_RE = re.compile(r"[ _\-.]+")


def library_root(create: bool = True) -> str:
    return utils.storage_dir("media_library", create=create)


def aspect_dir(video_aspect: VideoAspect, create: bool = True) -> str:
    name = ASPECT_DIR_NAMES.get(
        VideoAspect(video_aspect), ASPECT_DIR_NAMES[VideoAspect.portrait]
    )
    directory = os.path.join(library_root(create=create), name)
    if create and not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def _tags_path(aspect_directory: str) -> str:
    return os.path.join(aspect_directory, _TAGS_FILENAME)


def _load_tags(aspect_directory: str) -> dict:
    path = _tags_path(aspect_directory)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"failed to read media library tags at {path}: {e}")
        return {}


def _save_tags(aspect_directory: str, tags: dict) -> None:
    with open(_tags_path(aspect_directory), "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False, indent=2)


def _fingerprint(path: str) -> str:
    stat = os.stat(path)
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def _iter_media_files(aspect_directory: str):
    """Quet de quy (bao gom thu muc con), cho phep to chuc kho theo the loai,
    vi du 9-16/thien_nhien/bien.mp4. Key tra ve la duong dan tuong doi so
    voi aspect_directory, dung "/" lam dau phan cach de nhat quan giua cac
    he dieu hanh trong _tags.json."""
    if not os.path.isdir(aspect_directory):
        return
    for root, dirs, files in os.walk(aspect_directory):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for filename in sorted(files):
            if filename == _TAGS_FILENAME or filename.startswith("."):
                continue
            path = os.path.join(root, filename)
            rel_path = os.path.relpath(path, aspect_directory).replace(os.sep, "/")
            ext = utils.parse_extension(filename)
            if ext in _IMAGE_EXTS:
                yield rel_path, path, "image"
            elif ext in _VIDEO_EXTS:
                yield rel_path, path, "video"


def _probe_video_duration(video_path: str) -> float:
    from moviepy.video.io.VideoFileClip import VideoFileClip

    clip = None
    try:
        clip = VideoFileClip(video_path)
        return float(clip.duration or 0)
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass


def _derive_tags_from_path(rel_path: str) -> dict:
    """Suy ra mo ta + tag tu ten file va cac thu muc cha, khong goi AI/API
    nao. Vi du "thien_nhien/bien_hoanghon.mp4" -> tags
    ["thien", "nhien", "bien", "hoanghon"]."""
    parts = rel_path.split("/")
    stem = os.path.splitext(parts[-1])[0]

    tokens: List[str] = []
    seen = set()
    for part in (*parts[:-1], stem):
        for token in _NAME_SPLIT_RE.split(part):
            token = token.strip().lower()
            if token and token not in seen:
                seen.add(token)
                tokens.append(token)

    return {"description": " ".join(tokens), "tags": tokens}


def scan_and_tag_library(video_aspect: VideoAspect, force: bool = False) -> dict:
    """Quet thu muc kho media theo ti le khung hinh, gan tag cho nhung file
    chua duoc tag (hoac da doi noi dung, phat hien qua fingerprint
    size+mtime) dua tren ten file/thu muc - khong goi AI/API nao. Ket qua
    cache trong _tags.json trong chinh thu muc do, nen lan quet sau chi xu
    ly file moi/thay doi.

    force=True bo qua cache, gan tag lai toan bo.
    """
    directory = aspect_dir(video_aspect)
    tags = {} if force else _load_tags(directory)
    stats = {"scanned": 0, "tagged": 0, "skipped": 0, "failed": 0}

    for rel_path, path, kind in _iter_media_files(directory):
        stats["scanned"] += 1
        fingerprint = _fingerprint(path)
        existing = tags.get(rel_path)
        if existing and existing.get("fingerprint") == fingerprint:
            stats["skipped"] += 1
            continue

        try:
            duration = _probe_video_duration(path) if kind == "video" else None
            derived = _derive_tags_from_path(rel_path)

            tags[rel_path] = {
                "fingerprint": fingerprint,
                "type": kind,
                "duration": duration,
                "description": derived["description"],
                "tags": derived["tags"],
            }
            stats["tagged"] += 1
            logger.info(f"tagged media library file: {rel_path} -> {derived['tags']}")
        except Exception as e:
            stats["failed"] += 1
            logger.warning(f"failed to tag media library file {rel_path}: {e}")

    # Don luon entry cua file da bi xoa khoi thu muc, tranh _tags.json
    # phinh to voi du lieu rac theo thoi gian.
    existing_paths = {rel_path for rel_path, _, _ in _iter_media_files(directory)}
    for stale_path in list(tags.keys()):
        if stale_path not in existing_paths:
            del tags[stale_path]

    _save_tags(directory, tags)
    return stats


def library_stats(video_aspect: VideoAspect) -> dict:
    """So luong file da/chua duoc tag - dung cho UI hien trang thai kho."""
    directory = aspect_dir(video_aspect, create=False)
    tags = _load_tags(directory)
    total = 0
    tagged = 0
    for rel_path, _, _ in _iter_media_files(directory):
        total += 1
        entry = tags.get(rel_path)
        if entry and entry.get("fingerprint") == _fingerprint(
            os.path.join(directory, rel_path)
        ):
            tagged += 1
    return {"total": total, "tagged": tagged, "untagged": total - tagged}


def _score_match(search_term: str, entry: dict) -> int:
    haystack = f"{entry.get('description', '')} {' '.join(entry.get('tags', []))}".lower()
    tag_set = {t.lower() for t in entry.get("tags", [])}
    score = 0
    for token in _TOKEN_RE.findall(search_term.lower()):
        if token in tag_set:
            score += 2
        elif token in haystack:
            score += 1
    return score


def search_local_library(
    search_term: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> List[MaterialInfo]:
    """Tim file trong kho khop voi search_term (thuong la chu de/kich ban
    nguoi dung nhap thang, khong qua AI - xem task.py). Diem khop cang cao
    xep truoc. Anh duoc gan thoi luong toi thieu de khong bi loc mat, vi anh
    se duoc chuyen thanh clip zoom rieng khi render (xem
    video.image_to_zoom_clip)."""
    directory = aspect_dir(video_aspect, create=False)
    tags = _load_tags(directory)
    if not tags:
        return []

    scored = []
    for rel_path, entry in tags.items():
        duration = entry.get("duration")
        if entry.get("type") == "video":
            if not duration or duration < minimum_duration:
                continue
        else:
            duration = max(minimum_duration, 3.0)

        score = _score_match(search_term, entry)
        if score <= 0:
            continue

        path = os.path.join(directory, rel_path)
        if not os.path.exists(path):
            continue

        scored.append((score, path, duration))

    scored.sort(key=lambda item: item[0], reverse=True)

    results = []
    for _score, path, duration in scored:
        item = MaterialInfo()
        item.provider = "local_library"
        item.url = path
        item.duration = duration
        results.append(item)
    return results
