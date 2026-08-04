import re
from dataclasses import dataclass
from typing import List, Tuple

# 不区分大小写；内容部分要求至少 1 个非 "]" 字符，否则视为普通文本
# （例如用户手滑打了空的 "[card: ]"，不应该悄悄产出一张空白卡片）。
_CARD_MARKER_PATTERN = re.compile(
    r"\[card(?:\s+(\d+))?\s*:\s*([^\]]+)\]", re.IGNORECASE
)


@dataclass(frozen=True)
class CardMarker:
    slot: int
    text: str
    anchor_word_index: int


def _collapse_whitespace(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def extract_card_markers(script: str) -> Tuple[str, List[CardMarker]]:
    """从脚本中提取所有 [card: ...] / [card N: ...] 标记，返回去除标记后的
    干净脚本和标记列表。anchor_word_index 是标记在"干净脚本"里、按空格分词
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
        text = match.group(2).strip()
        if text:
            markers.append(
                CardMarker(slot=slot, text=text, anchor_word_index=anchor_word_index)
            )
        else:
            # 内容清理后为空（例如只含空白），把原始标记文字保留在脚本里，
            # 不当作有效标记处理。
            cleaned_parts.append(script[match.start() : match.end()])

    cleaned_parts.append(script[cursor:])
    cleaned_script = _collapse_whitespace("".join(cleaned_parts))
    return cleaned_script, markers
