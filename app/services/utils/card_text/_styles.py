# app/services/utils/card_text/_styles.py
import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from app.services import video as video_service

from ._primitives import render_card_box


def _resolve_card_font_path(text: str, font_name: Optional[str] = None) -> str:
    # resolve_font_path(None) tra ve font mac dinh (STHeitiMedium.ttc), font
    # nay thieu mot so ky tu co dau tieng Viet (vi du "Ạ") va se render ra o
    # vuong (tofu). Dung lai co che fallback da co san cho phu de de tu dong
    # doi sang font khac trong cung thu muc neu font (mac dinh hoac font
    # nguoi dung chon) thieu ky tu. font_name da duoc video.py xac nhan ton
    # tai truoc khi truyen vao day, nen ham nay khong tu kiem tra lai.
    font_path = video_service.resolve_font_path(font_name)
    return video_service._resolve_subtitle_font_path(font_path, text)


def minimal_white(
    text: str, font_name: Optional[str] = None, font_size: Optional[int] = None
) -> np.ndarray:
    font_path = _resolve_card_font_path(text, font_name)
    return render_card_box(
        text,
        font_path=font_path,
        font_size=font_size or 34,
        text_color=(20, 20, 20, 255),
        background_color=(255, 255, 255, 235),
        border_color=(210, 210, 210, 255),
        border_width=2,
        corner_radius=10,
    )


def bold_yellow_box(
    text: str, font_name: Optional[str] = None, font_size: Optional[int] = None
) -> np.ndarray:
    font_path = _resolve_card_font_path(text, font_name)
    return render_card_box(
        text,
        font_path=font_path,
        font_size=font_size or 40,
        text_color=(20, 20, 20, 255),
        background_color=(255, 205, 0, 255),
        border_color=(20, 20, 20, 255),
        border_width=6,
        corner_radius=4,
    )


def _gradient_background(
    width: int,
    height: int,
    corner_radius: int,
    top_color: Tuple[int, int, int],
    bottom_color: Tuple[int, int, int],
) -> np.ndarray:
    gradient = np.zeros((height, width, 4), dtype=np.uint8)
    for y in range(height):
        ratio = y / max(height - 1, 1)
        gradient[y, :, 0] = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
        gradient[y, :, 1] = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
        gradient[y, :, 2] = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
        gradient[y, :, 3] = 255
    mask_image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask_image).rounded_rectangle(
        (0, 0, width - 1, height - 1), radius=corner_radius, fill=255
    )
    gradient[:, :, 3] = np.asarray(mask_image)
    return gradient


def gradient_pop(
    text: str, font_name: Optional[str] = None, font_size: Optional[int] = None
) -> np.ndarray:
    font_path = _resolve_card_font_path(text, font_name)
    box = render_card_box(
        text,
        font_path=font_path,
        font_size=font_size or 42,
        text_color=(255, 255, 255, 255),
        background_color=(0, 0, 0, 0),
        border_color=None,
        border_width=0,
        corner_radius=20,
    )
    height, width = box.shape[:2]
    background = _gradient_background(
        width, height, corner_radius=20, top_color=(255, 94, 98), bottom_color=(255, 149, 0)
    )
    composed = Image.alpha_composite(
        Image.fromarray(background, "RGBA"), Image.fromarray(box, "RGBA")
    )
    return np.asarray(composed)


def sticky_note(
    text: str, font_name: Optional[str] = None, font_size: Optional[int] = None
) -> np.ndarray:
    font_path = _resolve_card_font_path(text, font_name)
    box = render_card_box(
        text,
        font_path=font_path,
        font_size=font_size or 36,
        text_color=(40, 40, 40, 255),
        background_color=(255, 245, 170, 255),
        border_color=None,
        border_width=0,
        corner_radius=4,
    )
    image = Image.fromarray(box, "RGBA").rotate(
        -3, expand=True, resample=Image.Resampling.BICUBIC
    )
    return np.asarray(image)


def _star_polygon(center, outer_radius, inner_radius, points=12):
    coords = []
    for i in range(points * 2):
        radius = outer_radius if i % 2 == 0 else inner_radius
        angle = math.pi * i / points
        coords.append(
            (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))
        )
    return coords


def comic_burst(
    text: str, font_name: Optional[str] = None, font_size: Optional[int] = None
) -> np.ndarray:
    font_path = _resolve_card_font_path(text, font_name)
    box = render_card_box(
        text,
        font_path=font_path,
        font_size=font_size or 40,
        text_color=(20, 20, 20, 255),
        background_color=(255, 255, 255, 0),
        border_color=None,
        border_width=0,
        corner_radius=0,
    )
    box_h, box_w = box.shape[:2]
    canvas_size = int(max(box_w, box_h) * 1.6)
    star_image = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(star_image)
    center = (canvas_size / 2, canvas_size / 2)
    draw.polygon(
        _star_polygon(
            center, outer_radius=canvas_size / 2 - 4, inner_radius=canvas_size / 3.2
        ),
        fill=(255, 221, 0, 255),
        outline=(20, 20, 20, 255),
        width=4,
    )
    box_image = Image.fromarray(box, "RGBA")
    paste_x = int((canvas_size - box_w) / 2)
    paste_y = int((canvas_size - box_h) / 2)
    star_image.alpha_composite(box_image, (paste_x, paste_y))
    return np.asarray(star_image)


CATEGORY_STYLES: Dict[str, Callable[..., np.ndarray]] = {
    "minimal_white": minimal_white,
    "bold_yellow_box": bold_yellow_box,
    "gradient_pop": gradient_pop,
    "sticky_note": sticky_note,
    "comic_burst": comic_burst,
}
