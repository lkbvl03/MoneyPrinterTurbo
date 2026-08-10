import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# 不区分大小写；内容部分要求至少 1 个非 "]" 字符，否则视为普通文本
# （例如用户手滑打了空的 "[card: ]"，不应该悄悄产出一张空白卡片）。
#
# 插槽号后面可以跟 0 个或多个 key=value 属性（style=/effect=/sound=），让
# ContentStudio 这类只能透传原始脚本文本、无法单独传 card_text_config 的
# 前端，也能在同一个 marker 里直接指定每张卡片的样式，不必依赖 WebUI 的
# 下拉框。属性值不允许包含空格/冒号/"]"，冒号仍然是内容部分的唯一分隔符。
_CARD_MARKER_PATTERN = re.compile(
    r"\[card(?:\s+(\d+))?((?:\s+\w+=[^\s\]:]+)*)\s*:\s*([^\]]+)\]", re.IGNORECASE
)
_CARD_ATTR_PATTERN = re.compile(r"(\w+)=([^\s\]:]+)")


@dataclass(frozen=True)
class CardMarker:
    slot: int
    text: str
    anchor_word_index: int
    style: Optional[str] = None
    effect: Optional[str] = None
    sound: Optional[str] = None
    font: Optional[str] = None
    font_size: Optional[int] = None


def _parse_card_attrs(attrs_text: str) -> dict:
    attrs = {}
    for key, value in _CARD_ATTR_PATTERN.findall(attrs_text):
        key = key.lower()
        if key in ("style", "effect", "sound", "font"):
            attrs[key] = value
        elif key == "font_size":
            try:
                attrs[key] = int(value)
            except ValueError:
                # Gia tri khong phai so - bo qua, de video.py fallback ve
                # co chu mac dinh thay vi lam hong ca marker.
                pass
    return attrs


def _collapse_whitespace(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”"}


def _strip_surrounding_quotes(text: str) -> str:
    # Nguoi dung go tay marker (vi du qua Content Studio) hay boc noi dung
    # trong dau ngoac kep de de doc - bo dau ngoac do di, khong de no hien
    # nguyen van tren the chu.
    if len(text) >= 2 and _QUOTE_PAIRS.get(text[0]) == text[-1]:
        return text[1:-1].strip()
    return text


def extract_card_markers(script: str) -> Tuple[str, List[CardMarker]]:
    """从脚本中提取所有 [card: ...] / [card N: ...] /
    [card N style=.. effect=.. sound=..: ...] 标记，返回去除标记后的干净
    脚本和标记列表。anchor_word_index 是标记在"干净脚本"里、按空格分词
    后紧邻标记之前那个词的 0-based 下标——后续用同一分词方式对齐 Whisper
    逐词时间戳时，必须用完全相同的计数方式，否则下标会对不上。"""
    markers: List[CardMarker] = []
    cleaned_parts: List[str] = []
    cursor = 0
    for match in _CARD_MARKER_PATTERN.finditer(script):
        cleaned_parts.append(script[cursor : match.start()])
        cursor = match.end()

        preceding_text = "".join(cleaned_parts)
        anchor_word_index = max(len(preceding_text.split()) - 1, 0)
        slot = int(match.group(1)) if match.group(1) else 1
        attrs = _parse_card_attrs(match.group(2))
        text = _strip_surrounding_quotes(match.group(3).strip())
        if text:
            markers.append(
                CardMarker(
                    slot=slot,
                    text=text,
                    anchor_word_index=anchor_word_index,
                    style=attrs.get("style"),
                    effect=attrs.get("effect"),
                    sound=attrs.get("sound"),
                    font=attrs.get("font"),
                    font_size=attrs.get("font_size"),
                )
            )
        else:
            # 内容清理后为空（例如只含空白），把原始标记文字保留在脚本里，
            # 不当作有效标记处理。
            cleaned_parts.append(script[match.start() : match.end()])

    cleaned_parts.append(script[cursor:])
    cleaned_script = _collapse_whitespace("".join(cleaned_parts))
    return cleaned_script, markers
