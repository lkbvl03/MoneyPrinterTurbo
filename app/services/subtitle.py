import json
import os.path
import re
from dataclasses import dataclass
from timeit import default_timer as timer
from typing import List, Optional, Tuple

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
from loguru import logger

from app.config import config
from app.services.utils.card_markers import CardMarker
from app.utils import utils

model_size = config.whisper.get("model_size", "large-v3")
device = config.whisper.get("device", "cpu")
compute_type = config.whisper.get("compute_type", "int8")
model = None

# 找不到已识别词可参考语速时的保底值：大致的中等语速朗读节奏。
_DEFAULT_CHARS_PER_SECOND = 12.0
_MIN_EXTRAPOLATED_DURATION = 0.5

# 找不到已识别词可参考语速时的保底值：卡片标记锚点超出 Whisper 识别范围
# 需要外推时，假设的平均每词耗时（秒）。
_DEFAULT_SECONDS_PER_WORD = 0.4


@dataclass(frozen=True)
class ResolvedCardTiming:
    slot: int
    text: str
    start_time: float


def _flatten_whisper_words(segments):
    """把 Whisper 按 segment 分组的逐词结果展平成一条按时间顺序排列的
    (文字, 开始秒, 结束秒) 列表，丢弃 segment 自身的分段边界——后续按脚本
    词序位置对齐时，只需要每个词自己的时间，不需要 Whisper 认为哪些词属于
    同一段。"""
    words = []
    for segment in segments:
        if not segment.words:
            continue
        for word in segment.words:
            text = word.word.strip()
            if text:
                words.append((text, word.start, word.end))
    return words


def _chars_per_second_from_words(words) -> float:
    if not words:
        return _DEFAULT_CHARS_PER_SECOND
    total_chars = sum(len(text) for text, _, _ in words)
    total_seconds = words[-1][2] - words[0][1]
    if total_chars <= 0 or total_seconds <= 0:
        return _DEFAULT_CHARS_PER_SECOND
    return total_chars / total_seconds


def _seconds_per_word_from_words(words: List[Tuple[str, float, float]]) -> float:
    if len(words) < 2:
        return _DEFAULT_SECONDS_PER_WORD
    total_seconds = words[-1][2] - words[0][1]
    if total_seconds <= 0:
        return _DEFAULT_SECONDS_PER_WORD
    return total_seconds / len(words)


def _resolve_card_marker_timestamps(
    markers: List[CardMarker], whisper_words: List[Tuple[str, float, float]]
) -> List[ResolvedCardTiming]:
    """把 [card N: ...] 标记在清理后脚本里的词序位置（anchor_word_index）
    对应到 Whisper 逐词时间戳——原理和 _align_script_lines_to_word_timestamps
    完全一致：脚本词序和 Whisper 识别出的词序一一对应，只用位置不用文字去对
    齐，避免文本不一致（Whisper 听错、脚本被清理过）导致的错位。

    卡片的出现时间取锚点词的结束时间（而不是开始时间）——锚点词是标记前面
    紧邻的最后一个词，只有等它读完，卡片才应该出现，这样才不会在那个词还
    没读完时就提前弹出。

    锚点超出 Whisper 实际识别到的词数时（识别遗漏、静音检测误判等），按已
    识别部分算出的平均语速继续向后外推，不产生 0 秒占位。
    """
    seconds_per_word = _seconds_per_word_from_words(whisper_words)
    results = []
    for marker in markers:
        if marker.anchor_word_index < len(whisper_words):
            start_time = whisper_words[marker.anchor_word_index][2]
        elif whisper_words:
            last_end = whisper_words[-1][2]
            words_beyond = marker.anchor_word_index - (len(whisper_words) - 1)
            start_time = last_end + words_beyond * seconds_per_word
        else:
            start_time = marker.anchor_word_index * seconds_per_word
        results.append(
            ResolvedCardTiming(slot=marker.slot, text=marker.text, start_time=start_time)
        )
    return results


def _align_script_lines_to_word_timestamps(script_lines, whisper_words):
    """把已经按标点/长度分好的脚本行，按词序位置对应到 Whisper 逐词识别出
    的真实时间戳（第 i 个脚本词直接用第 i 个 Whisper 词的时间），而不是像
    correct() 那样用整句文本相似度去猜该合并哪些片段。

    真实渲染中发现的问题：脚本行边界和 Whisper 自己识别出的分段边界没对齐
    时，整句匹配会把边界词的时间错误地并入相邻的字幕行，导致该词的字幕比
    实际读到的时间提前或延后出现。按位置对齐从根源上避免这个问题——即使
    Whisper 把某个词识别成完全不同的文字，只要词的顺序、数量大致对应，
    这里只用它的时间，不用它的文字，就不会互相污染。

    Whisper 词数不够时，剩余的词按已识别部分的语速连续外推，不使用
    00:00:00,000 占位。
    """
    chars_per_second = _chars_per_second_from_words(whisper_words)
    results = []
    word_cursor = 0
    next_start = 0.0
    for line in script_lines:
        words_in_line = line.split()
        if not words_in_line:
            continue
        start_index = word_cursor
        end_index = word_cursor + len(words_in_line) - 1
        word_cursor += len(words_in_line)

        if start_index < len(whisper_words):
            line_start = whisper_words[start_index][1]
        else:
            line_start = next_start

        if end_index < len(whisper_words):
            line_end = whisper_words[end_index][2]
        else:
            duration = max(len(line) / chars_per_second, _MIN_EXTRAPOLATED_DURATION)
            line_end = line_start + duration

        line_end = max(line_end, line_start + _MIN_EXTRAPOLATED_DURATION)
        results.append((line, line_start, line_end))
        next_start = line_end
    return results


def create(
    audio_file,
    subtitle_file: str = "",
    max_line_length: Optional[int] = None,
    video_script: str = "",
    card_markers: Optional[List[CardMarker]] = None,
):
    global model
    if WhisperModel is None:
        logger.warning("faster_whisper not available, skipping whisper subtitle generation")
        return ""
    if not model:
        model_path = f"{utils.root_dir()}/models/whisper-{model_size}"
        model_bin_file = f"{model_path}/model.bin"
        if not os.path.isdir(model_path) or not os.path.isfile(model_bin_file):
            model_path = model_size

        logger.info(
            f"loading model: {model_path}, device: {device}, compute_type: {compute_type}"
        )
        try:
            model = WhisperModel(
                model_size_or_path=model_path, device=device, compute_type=compute_type
            )
        except Exception as e:
            logger.error(
                f"failed to load model: {e} \n\n"
                f"********************************************\n"
                f"this may be caused by network issue. \n"
                f"please download the model manually and put it in the 'models' folder. \n"
                f"see [README.md FAQ](https://github.com/harry0703/MoneyPrinterTurbo) for more details.\n"
                f"********************************************\n\n"
            )
            return None

    logger.info(f"start, output file: {subtitle_file}")
    if not subtitle_file:
        subtitle_file = f"{audio_file}.srt"

    segments, info = model.transcribe(
        audio_file,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    logger.info(
        f"detected language: '{info.language}', probability: {info.language_probability:.2f}"
    )

    start = timer()
    subtitles = []

    if video_script:
        # 有脚本原文可用时，按词序位置直接对应 Whisper 逐词时间戳，而不是
        # 用 Whisper 自己的分段+标点断句猜测字幕行——见
        # _align_script_lines_to_word_timestamps 的说明：这是真实渲染中发现
        # 的字幕/语音错位问题的根本修复。
        normalized_script = utils.normalize_script_for_subtitle_matching(video_script)
        script_lines = (
            utils.split_string_by_punctuations_and_length(normalized_script, max_line_length)
            if max_line_length
            else utils.split_string_by_punctuations(normalized_script)
        )
        whisper_words = _flatten_whisper_words(segments)
        card_timings = (
            _resolve_card_marker_timestamps(card_markers, whisper_words)
            if card_markers
            else []
        )
        for text, line_start, line_end in _align_script_lines_to_word_timestamps(
            script_lines, whisper_words
        ):
            subtitles.append(
                {"msg": text, "start_time": line_start, "end_time": line_end}
            )
    else:
        card_timings = []

        def recognized(seg_text, seg_start, seg_end):
            seg_text = seg_text.strip()
            if not seg_text:
                return

            msg = "[%.2fs -> %.2fs] %s" % (seg_start, seg_end, seg_text)
            logger.debug(msg)

            subtitles.append(
                {"msg": seg_text, "start_time": seg_start, "end_time": seg_end}
            )

        for segment in segments:
            words_idx = 0
            words_len = len(segment.words)

            seg_start = 0
            seg_end = 0
            seg_text = ""

            if segment.words:
                is_segmented = False
                for word in segment.words:
                    if not is_segmented:
                        seg_start = word.start
                        is_segmented = True

                    # 在真正拼接这个词之前检查：如果加上它会超过字符上限，就先把
                    # 已经累积的文本作为一条独立字幕提交，再让这个词开启新的一段。
                    # 必须在拼接前判断（而不是拼接后再检查），否则每条字幕会被
                    # 多拼进一个词，超出 max_line_length 上限。
                    would_exceed_length = (
                        max_line_length is not None
                        and seg_text.strip()
                        and len((seg_text + word.word).strip()) > max_line_length
                    )
                    if would_exceed_length:
                        recognized(seg_text.strip(), seg_start, seg_end)
                        seg_text = ""
                        seg_start = word.start

                    seg_end = word.end
                    # If it contains punctuation, then break the sentence.
                    seg_text += word.word

                    if utils.str_contains_punctuation(word.word):
                        # remove last char
                        seg_text = seg_text[:-1]
                        if not seg_text:
                            continue

                        recognized(seg_text, seg_start, seg_end)

                        is_segmented = False
                        seg_text = ""

                    if words_idx == 0 and segment.start < word.start:
                        seg_start = word.start
                    if words_idx == (words_len - 1) and segment.end > word.end:
                        seg_end = word.end
                    words_idx += 1

            if not seg_text:
                continue

            recognized(seg_text, seg_start, seg_end)

    end = timer()

    diff = end - start
    logger.info(f"complete, elapsed: {diff:.2f} s")

    idx = 1
    lines = []
    for subtitle in subtitles:
        text = subtitle.get("msg")
        if text:
            lines.append(
                utils.text_to_srt(
                    idx, text, subtitle.get("start_time"), subtitle.get("end_time")
                )
            )
            idx += 1

    sub = "\n".join(lines) + "\n"
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(sub)
    logger.info(f"subtitle file created: {subtitle_file}")
    return card_timings


def file_to_subtitles(filename):
    if not filename or not os.path.isfile(filename):
        return []

    times_texts = []
    current_times = None
    current_text = ""
    index = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            times = re.findall("([0-9]*:[0-9]*:[0-9]*,[0-9]*)", line)
            if times:
                current_times = line
            elif line.strip() == "" and current_times:
                index += 1
                times_texts.append((index, current_times.strip(), current_text.strip()))
                current_times, current_text = None, ""
            elif current_times:
                current_text += line

    # Flush the final block. SRT files whose last subtitle is not followed by a
    # trailing blank line never hit the blank-line branch above, so without this
    # the last subtitle would be silently dropped.
    if current_times:
        index += 1
        times_texts.append((index, current_times.strip(), current_text.strip()))
    return times_texts


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity(a, b):
    distance = levenshtein_distance(a.lower(), b.lower())
    max_length = max(len(a), len(b))
    return 1 - (distance / max_length)


def _estimate_chars_per_second(matched_items) -> float:
    """根据已经成功对齐（含合并/纠正）的字幕片段，估算这段语音的平均语速
    （字符/秒），用于给耗尽 Whisper 片段后剩余的脚本行外推合理时长。"""
    total_chars = 0
    total_seconds = 0.0
    for item in matched_items:
        start_str, _, end_str = item[1].partition(" --> ")
        duration = utils.time_convert_hmsm_to_seconds(
            end_str
        ) - utils.time_convert_hmsm_to_seconds(start_str)
        text = item[2]
        if duration > 0 and text:
            total_chars += len(text)
            total_seconds += duration
    if total_seconds <= 0 or total_chars <= 0:
        return _DEFAULT_CHARS_PER_SECOND
    return total_chars / total_seconds


def correct(subtitle_file, video_script, max_line_length: Optional[int] = None):
    subtitle_items = file_to_subtitles(subtitle_file)
    normalized_script = utils.normalize_script_for_subtitle_matching(video_script)
    # max_line_length 未设置时保持原有的按标点断句；设置时必须用同样的长度
    # 上限拆分脚本作为比对基准，否则下面的合并逻辑会把已经按长度拆好的
    # Whisper 字幕重新拼回一整句原文，超出短视频单行字符上限。
    script_lines = (
        utils.split_string_by_punctuations_and_length(normalized_script, max_line_length)
        if max_line_length
        else utils.split_string_by_punctuations(normalized_script)
    )

    corrected = False
    new_subtitle_items = []
    script_index = 0
    subtitle_index = 0

    while script_index < len(script_lines) and subtitle_index < len(subtitle_items):
        script_line = script_lines[script_index].strip()
        subtitle_line = subtitle_items[subtitle_index][2].strip()

        if script_line == subtitle_line:
            new_subtitle_items.append(subtitle_items[subtitle_index])
            script_index += 1
            subtitle_index += 1
        else:
            combined_subtitle = subtitle_line
            start_time = subtitle_items[subtitle_index][1].split(" --> ")[0]
            end_time = subtitle_items[subtitle_index][1].split(" --> ")[1]
            next_subtitle_index = subtitle_index + 1

            while next_subtitle_index < len(subtitle_items):
                next_subtitle = subtitle_items[next_subtitle_index][2].strip()
                if similarity(
                    script_line, combined_subtitle + " " + next_subtitle
                ) > similarity(script_line, combined_subtitle):
                    combined_subtitle += " " + next_subtitle
                    end_time = subtitle_items[next_subtitle_index][1].split(" --> ")[1]
                    next_subtitle_index += 1
                else:
                    break

            if similarity(script_line, combined_subtitle) > 0.8:
                logger.warning(
                    f"Merged/Corrected - Script: {script_line}, Subtitle: {combined_subtitle}"
                )
                new_subtitle_items.append(
                    (
                        len(new_subtitle_items) + 1,
                        f"{start_time} --> {end_time}",
                        script_line,
                    )
                )
                corrected = True
            else:
                logger.warning(
                    f"Mismatch - Script: {script_line}, Subtitle: {combined_subtitle}"
                )
                new_subtitle_items.append(
                    (
                        len(new_subtitle_items) + 1,
                        f"{start_time} --> {end_time}",
                        script_line,
                    )
                )
                corrected = True

            script_index += 1
            subtitle_index = next_subtitle_index

    # Process the remaining lines of the script.
    # 按长度切分后脚本行数经常多于 Whisper 实际转写出的片段数，导致上面的
    # 合并逻辑提前耗尽 subtitle_items。此时不能再用 00:00:00,000 占位——
    # 那会让这些字幕要么全部堆在片头，要么根本不显示，和实际语音完全对不
    # 上（真实渲染中发现：约 40% 的字幕因此失去同步）。改为按已匹配片段的
    # 实际语速，从上一条已知结束时间连续往后外推。
    chars_per_second = None
    next_start = None
    while script_index < len(script_lines):
        script_line = script_lines[script_index]
        logger.warning(f"Extra script line: {script_line}")
        if subtitle_index < len(subtitle_items):
            new_subtitle_items.append(
                (
                    len(new_subtitle_items) + 1,
                    subtitle_items[subtitle_index][1],
                    script_line,
                )
            )
            subtitle_index += 1
        else:
            if chars_per_second is None:
                chars_per_second = _estimate_chars_per_second(new_subtitle_items)
            if next_start is None:
                next_start = (
                    utils.time_convert_hmsm_to_seconds(
                        new_subtitle_items[-1][1].split(" --> ")[1]
                    )
                    if new_subtitle_items
                    else 0.0
                )
            duration = max(
                len(script_line) / chars_per_second, _MIN_EXTRAPOLATED_DURATION
            )
            start_seconds = next_start
            end_seconds = start_seconds + duration
            new_subtitle_items.append(
                (
                    len(new_subtitle_items) + 1,
                    f"{utils.time_convert_seconds_to_hmsm(start_seconds)} --> "
                    f"{utils.time_convert_seconds_to_hmsm(end_seconds)}",
                    script_line,
                )
            )
            next_start = end_seconds
        script_index += 1
        corrected = True

    if corrected:
        with open(subtitle_file, "w", encoding="utf-8") as fd:
            for i, item in enumerate(new_subtitle_items):
                fd.write(f"{i + 1}\n{item[1]}\n{item[2]}\n\n")
        logger.info("Subtitle corrected")
    else:
        logger.success("Subtitle is correct")


if __name__ == "__main__":
    task_id = "c12fd1e6-4b0a-4d65-a075-c87abe35a072"
    task_dir = utils.task_dir(task_id)
    subtitle_file = f"{task_dir}/subtitle.srt"
    audio_file = f"{task_dir}/audio.mp3"

    subtitles = file_to_subtitles(subtitle_file)
    print(subtitles)

    script_file = f"{task_dir}/script.json"
    with open(script_file, "r") as f:
        script_content = f.read()
    s = json.loads(script_content)
    script = s.get("script")

    correct(subtitle_file, script)

    subtitle_file = f"{task_dir}/subtitle-test.srt"
    create(audio_file, subtitle_file)
