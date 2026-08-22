import json
import warnings
from enum import Enum
from typing import Any, List, Literal, Optional, Union

import pydantic
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import config
from app.services.utils.xfade_transitions import XFADE_TRANSITIONS
from app.services.utils.video_overlay_effects import OVERLAY_EFFECTS

# 忽略 Pydantic 的特定警告
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Field name.*shadows an attribute in parent.*",
)


class VideoConcatMode(str, Enum):
    random = "random"
    sequential = "sequential"


class VideoTransitionMode(str, Enum):
    none = None
    shuffle = "Shuffle"
    fade_in = "FadeIn"
    fade_out = "FadeOut"
    slide_in = "SlideIn"
    slide_out = "SlideOut"
    zoom_in = "ZoomIn"
    zoom_out = "ZoomOut"


class VideoAspect(str, Enum):
    landscape = "16:9"
    portrait = "9:16"
    square = "1:1"

    def to_resolution(self):
        if self == VideoAspect.landscape:
            return 1920, 1080
        elif self == VideoAspect.portrait:
            return 1080, 1920
        elif self == VideoAspect.square:
            return 1080, 1080
        raise ValueError(f"unsupported video aspect: {self}")


class _Config:
    arbitrary_types_allowed = True


@pydantic.dataclasses.dataclass(config=_Config)
class MaterialInfo:
    provider: str = "pexels"
    url: str = ""
    duration: int = 0


class VideoParams(BaseModel):
    """
    {
      "video_subject": "",
      "video_aspect": "横屏 16:9（西瓜视频）",
      "voice_name": "女生-晓晓",
      "bgm_name": "random",
      "font_name": "STHeitiMedium 黑体-中",
      "text_color": "#FFFFFF",
      "font_size": 60,
      "stroke_color": "#000000",
      "stroke_width": 1.5
    }
    """

    video_subject: str
    video_script: str = ""  # Script used to generate the video
    video_terms: Optional[str | list] = None  # Keywords used to generate the video
    video_aspect: Optional[VideoAspect] = VideoAspect.portrait.value
    video_concat_mode: Optional[VideoConcatMode] = VideoConcatMode.random.value
    video_transition_mode: Optional[VideoTransitionMode] = None
    video_transition_style: Optional[str] = None
    video_overlay_effect: Optional[str] = None
    video_clip_duration: Optional[int] = 7
    # 图片素材（上传/本地图库）用缩放动画展示，和下载素材的最大播放时长语义不同：
    # 静态图配文案朗读时常常需要停留更久，不应该被 video_clip_duration 的短时长
    # 上限捆绑。独立字段、无上限，默认值与旧行为一致（原来图片也是走
    # video_clip_duration，默认 7；这里改用图片素材原本更常见的 4 秒默认值）。
    image_clip_duration: Optional[int] = Field(default=4, ge=1)
    card_text_config: Optional[str] = None
    video_clip_speed: Optional[float] = 1.0
    match_materials_to_script: bool = False
    video_count: Optional[int] = 1
    # 批量生成多个视频时，用户需要自己指定成片额外导出到哪个目录、用什么文件名，
    # 而不是只能在 storage/tasks/<task_id> 里用内部 UUID 目录翻找。留空则不导出，
    # 完全保持原有行为；两者互相独立，只设置其中一个也生效。
    output_dir: Optional[str] = None
    output_filename: Optional[str] = None

    video_source: Optional[str] = "pexels"
    video_materials: Optional[List[MaterialInfo]] = (
        None  # Materials used to generate the video
    )
    
    custom_audio_file: Optional[str] = None  # Custom audio file path, will ignore TTS and can still use Whisper subtitles
    video_language: Optional[str] = ""  # auto detect

    voice_name: Optional[str] = ""
    voice_volume: Optional[float] = 1.0
    voice_rate: Optional[float] = 1.0
    bgm_type: Optional[str] = "random"
    bgm_file: Optional[str] = ""
    bgm_volume: Optional[float] = 0.2
    # 视频配乐供应商共用提示词，WebUI 新任务统一写入该字段。保留下面的
    # Sonilo 专用字段以兼容旧任务记录和现有 CLI 参数。
    video_music_prompt: str = Field(default="", max_length=2000)
    sonilo_bgm_prompt: str = Field(default="", max_length=2000)

    subtitle_enabled: Optional[bool] = True
    subtitle_position: Optional[str] = config.ui.get("subtitle_position", "bottom")  # top, bottom, center, custom
    custom_position: float = config.ui.get("custom_position", 70.0)
    font_name: Optional[str] = "STHeitiMedium.ttc"
    text_fore_color: Optional[str] = "#FFFFFF"
    text_background_color: Union[bool, str] = False
    rounded_subtitle_background: bool = False

    font_size: int = 60
    stroke_color: Optional[str] = "#000000"
    stroke_width: float = 1.5
    n_threads: Optional[int] = 2
    paragraph_number: int = Field(default=1, ge=1, le=10)
    video_script_prompt: str = Field(default="", max_length=2000)
    custom_system_prompt: str = Field(default="", max_length=8000)

    @field_validator("video_transition_style")
    @classmethod
    def _validate_video_transition_style(cls, value: Optional[str]) -> Optional[str]:
        # cli.py 的 _transition_style 只保护 CLI 入口；HTTP API 直接从请求 JSON
        # 构造 VideoParams，绕过了那层检查，未校验的字符串会被拼进 ffmpeg
        # filter_complex 字符串。这里在 schema 层强制约束合法取值。
        if value is None or value == "random":
            return value
        if value not in XFADE_TRANSITIONS:
            allowed = ", ".join(("random", *XFADE_TRANSITIONS))
            raise ValueError(
                f"video_transition_style must be one of: {allowed}"
            )
        return value

    @field_validator("video_overlay_effect")
    @classmethod
    def _validate_video_overlay_effect(cls, value: Optional[str]) -> Optional[str]:
        # 与 video_transition_style 同样的道理：HTTP API 绕过 CLI 层的校验，
        # 必须在 schema 层再校验一次，避免非法字符串传入 combine_videos。
        if value is None or value in ("none", "random"):
            return value
        if value not in OVERLAY_EFFECTS:
            allowed = ", ".join(("none", "random", *sorted(OVERLAY_EFFECTS)))
            raise ValueError(f"video_overlay_effect must be one of: {allowed}")
        return value

    @field_validator("output_dir")
    @classmethod
    def _validate_output_dir(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("output_filename")
    @classmethod
    def _validate_output_filename(cls, value: Optional[str]) -> Optional[str]:
        # 只是最终文件名（不带扩展名），不是路径，所以必须拒绝路径分隔符和
        # Windows 保留字符，避免用户输入被拼接进文件系统调用时逃出目标目录
        # 或在 Windows 上创建失败。
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        invalid_chars = set('/\\:*?"<>|')
        if any(ch in invalid_chars for ch in value):
            raise ValueError(
                'output_filename must not contain path separators or any of: / \\ : * ? " < > |'
            )
        return value

    @field_validator("card_text_config")
    @classmethod
    def _validate_card_text_config(cls, value: Optional[str]) -> Optional[str]:
        # card_text_config 是 JSON 字符串（不是嵌套的 pydantic 模型），因为
        # ContentStudio 和 cli.py 两端都是直接拼接/传递原始 JSON 文本，不需要
        # 额外一层 Python 对象转换。这里只做结构和取值合法性校验，校验通过后
        # 原样返回，不做任何规范化改写。
        if value is None or value == "":
            return None

        # 延迟到函数内部导入：card_text 包（经由 _styles.py）在模块级会 import
        # app.services.video，而 video.py 又反过来在模块级 import 本文件
        # （app.models.schema）以获取 VideoParams 等类型，如果把这两个 import
        # 放在文件顶部会形成循环导入（schema -> card_text -> video -> schema），
        # 导致整个模块加载失败。放到这里按需导入可以避免该循环。
        from app.services.utils.card_text import CARD_EFFECTS, CARD_STYLES
        from app.services.utils.card_text._sounds import CARD_SOUNDS

        try:
            slots = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"card_text_config must be valid JSON: {exc}") from exc

        if not isinstance(slots, list):
            raise ValueError("card_text_config must be a JSON array of slot objects")

        allowed_styles = set(CARD_STYLES) | {"random"}
        allowed_effects = set(CARD_EFFECTS) | {"random"}
        allowed_sounds = set(CARD_SOUNDS) | {"auto", "none"}
        seen_slots = set()

        for entry in slots:
            if not isinstance(entry, dict):
                raise ValueError("each card_text_config entry must be a JSON object")

            missing = {"slot", "style", "effect"} - entry.keys()
            if missing:
                raise ValueError(
                    f"card_text_config entry is missing required keys: {sorted(missing)}"
                )

            slot = entry["slot"]
            if not isinstance(slot, int) or slot < 1:
                raise ValueError("card_text_config 'slot' must be a positive integer")
            if slot in seen_slots:
                raise ValueError(f"card_text_config has duplicate slot number: {slot}")
            seen_slots.add(slot)

            style = entry["style"]
            if style not in allowed_styles:
                raise ValueError(
                    f"card_text_config 'style' must be one of: "
                    f"{', '.join(sorted(allowed_styles))}"
                )

            effect = entry["effect"]
            if effect not in allowed_effects:
                raise ValueError(
                    f"card_text_config 'effect' must be one of: "
                    f"{', '.join(sorted(allowed_effects))}"
                )

            sound = entry.get("sound", "auto")
            if sound not in allowed_sounds:
                raise ValueError(
                    f"card_text_config 'sound' must be one of: "
                    f"{', '.join(sorted(allowed_sounds))}"
                )

            font = entry.get("font")
            if font is not None and (not isinstance(font, str) or not font.strip()):
                raise ValueError("card_text_config 'font' must be a non-empty string")

            font_size = entry.get("font_size")
            if font_size is not None and (
                not isinstance(font_size, int) or not (8 <= font_size <= 200)
            ):
                raise ValueError(
                    "card_text_config 'font_size' must be an integer between 8 and 200"
                )

        return value


class SubtitleRequest(BaseModel):
    video_script: str
    video_language: Optional[str] = ""
    voice_name: Optional[str] = "zh-CN-XiaoxiaoNeural-Female"
    voice_volume: Optional[float] = 1.0
    voice_rate: Optional[float] = 1.2
    bgm_type: Optional[str] = "random"
    bgm_file: Optional[str] = ""
    bgm_volume: Optional[float] = 0.2
    subtitle_position: Optional[str] = config.ui.get("subtitle_position", "bottom")
    font_name: Optional[str] = "STHeitiMedium.ttc"
    text_fore_color: Optional[str] = "#FFFFFF"
    text_background_color: Union[bool, str] = False
    rounded_subtitle_background: bool = False
    font_size: int = 60
    stroke_color: Optional[str] = "#000000"
    stroke_width: float = 1.5
    video_source: Optional[str] = "local"
    subtitle_enabled: Optional[str] = "true"


class AudioRequest(BaseModel):
    video_script: str
    video_language: Optional[str] = ""
    voice_name: Optional[str] = "zh-CN-XiaoxiaoNeural-Female"
    voice_volume: Optional[float] = 1.0
    voice_rate: Optional[float] = 1.2
    bgm_type: Optional[str] = "random"
    bgm_file: Optional[str] = ""
    bgm_volume: Optional[float] = 0.2
    video_source: Optional[str] = "local"


class VideoScriptParams:
    """
    {
      "video_subject": "春天的花海",
      "video_language": "",
      "paragraph_number": 1,
      "video_script_prompt": "",
      "custom_system_prompt": ""
    }
    """

    video_subject: Optional[str] = "春天的花海"
    video_language: Optional[str] = ""
    paragraph_number: int = Field(default=1, ge=1, le=10)
    video_script_prompt: str = Field(default="", max_length=2000)
    custom_system_prompt: str = Field(default="", max_length=8000)


class VideoTermsParams:
    """
    {
      "video_subject": "",
      "video_script": "",
      "amount": 5,
      "match_materials_to_script": false
    }
    """

    video_subject: Optional[str] = "春天的花海"
    video_script: Optional[str] = (
        "春天的花海，如诗如画般展现在眼前。万物复苏的季节里，大地披上了一袭绚丽多彩的盛装。金黄的迎春、粉嫩的樱花、洁白的梨花、艳丽的郁金香……"
    )
    amount: Optional[int] = 5
    match_materials_to_script: bool = False


class VideoSocialMetadataParams:
    """
    {
      "video_subject": "A day in Shanghai",
      "video_script": "",
      "language": "auto",
      "platform": "tiktok"
    }
    """

    video_subject: Optional[str] = Field(default="A day in Shanghai", max_length=500)
    video_script: Optional[str] = Field(default="", max_length=8000)
    language: Optional[str] = Field(default="auto", max_length=64)
    platform: Optional[str] = Field(default="tiktok", max_length=64)


class BaseResponse(BaseModel):
    status: int = 200
    message: Optional[str] = "success"
    data: Any = None


class TaskVideoRequest(VideoParams, BaseModel):
    pass


class TaskQueryRequest(BaseModel):
    pass


class VideoScriptRequest(VideoScriptParams, BaseModel):
    pass


class VideoTermsRequest(VideoTermsParams, BaseModel):
    pass


class VideoSocialMetadataRequest(VideoSocialMetadataParams, BaseModel):
    pass


######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
class TaskResponse(BaseResponse):
    class TaskResponseData(BaseModel):
        task_id: str

    data: TaskResponseData

    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "message": "success",
                "data": {"task_id": "6c85c8cc-a77a-42b9-bc30-947815aa0558"},
            },
        }


class TaskStatusData(BaseModel):
    """任务查询对外保证的稳定字段；历史和扩展字段继续原样透传。"""

    model_config = ConfigDict(extra="allow")

    task_id: str
    state: int
    progress: int = 0
    videos: Optional[List[str]] = None
    combined_videos: Optional[List[str]] = None
    failed_stage: Optional[str] = None
    error: Optional[str] = None
    cross_post_state: Optional[
        Literal["pending", "processing", "complete", "failed"]
    ] = None
    cross_post_results: Optional[List[dict[str, Any]]] = None
    cross_post_error: Optional[str] = None


class TaskListData(BaseModel):
    """分页任务列表结构。"""

    tasks: List[TaskStatusData]
    total: int
    page: int
    page_size: int


class TaskQueryResponse(BaseResponse):
    """
    任务查询会返回生成状态和可选的跨平台发布状态。

    生成失败时包含 `failed_stage` 和 `error`；生成完成后如果启用了自动发布，
    `cross_post_state` 会依次进入 pending、processing、complete 或 failed。
    """

    data: TaskStatusData

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": 200,
                    "message": "success",
                    "data": {
                        "task_id": "6c85c8cc-a77a-42b9-bc30-947815aa0558",
                        "state": 1,
                        "progress": 100,
                        "videos": ["/tasks/example/final-1.mp4"],
                        "cross_post_state": "complete",
                        "cross_post_results": [{"success": True}],
                    },
                },
                {
                    "status": 200,
                    "message": "success",
                    "data": {
                        "task_id": "6c85c8cc-a77a-42b9-bc30-947815aa0558",
                        "state": -1,
                        "progress": 30,
                        "failed_stage": "audio",
                        "error": "TTS request timed out",
                    },
                },
            ],
        }
    )


class TaskListResponse(BaseResponse):
    """任务列表使用独立响应模型，避免与单任务查询混用文档结构。"""

    data: TaskListData

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "tasks": [
                        {
                            "task_id": "6c85c8cc-a77a-42b9-bc30-947815aa0558",
                            "state": 4,
                            "progress": 50,
                        }
                    ],
                    "total": 1,
                    "page": 1,
                    "page_size": 10,
                },
            }
        }
    )


class TaskDeletionResponse(BaseResponse):
    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "state": 1,
                    "progress": 100,
                    "videos": [
                        "http://127.0.0.1:8080/tasks/6c85c8cc-a77a-42b9-bc30-947815aa0558/final-1.mp4"
                    ],
                    "combined_videos": [
                        "http://127.0.0.1:8080/tasks/6c85c8cc-a77a-42b9-bc30-947815aa0558/combined-1.mp4"
                    ],
                },
            },
        }


class VideoScriptResponse(BaseResponse):
    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "video_script": "春天的花海，是大自然的一幅美丽画卷。在这个季节里，大地复苏，万物生长，花朵争相绽放，形成了一片五彩斑斓的花海..."
                },
            },
        }


class VideoTermsResponse(BaseResponse):
    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "message": "success",
                "data": {"video_terms": ["sky", "tree"]},
            },
        }


class VideoSocialMetadataResponse(BaseResponse):
    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "title": "A Day in Shanghai You Should Not Miss",
                    "caption": "Save this quick Shanghai inspiration and follow for more short travel ideas.",
                    "hashtags": ["#shorts", "#travel", "#shanghai", "#viral", "#fyp"],
                },
            },
        }


class BgmRetrieveResponse(BaseResponse):
    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "files": [
                        {
                            "name": "4fca18fce7344f3aa824777a40d45c8c.mp3",
                            "size": 1891269,
                            "file": "4fca18fce7344f3aa824777a40d45c8c.mp3",
                        }
                    ]
                },
            },
        }


class BgmUploadResponse(BaseResponse):
    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "message": "success",
                "data": {"file": "4fca18fce7344f3aa824777a40d45c8c.mp3"},
            },
        }

class VideoMaterialRetrieveResponse(BaseResponse):
    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "files": [
                        {
                            "name": "example.mp4",
                            "size": 12345678,
                            "file": "/MoneyPrinterTurbo/resource/videos/example.mp4",
                        }
                    ]
                },
            },
        }

class VideoMaterialUploadResponse(BaseResponse):
    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "message": "success",
                "data": {
                    "file": "/MoneyPrinterTurbo/resource/videos/example.mp4",
                },
            },
        }
