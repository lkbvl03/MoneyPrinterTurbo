import asyncio
import base64
import os
import shutil
import unittest
import sys
import tempfile
import time
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils import utils
from app.services import voice as vs
from app.services import subtitle as subtitle_service
from app.services import task as task_service
from pydub import AudioSegment

temp_dir = utils.storage_dir("temp")

text_en = """
What is the meaning of life? 
This question has puzzled philosophers, scientists, and thinkers of all kinds for centuries. 
Throughout history, various cultures and individuals have come up with their interpretations and beliefs around the purpose of life. 
Some say it's to seek happiness and self-fulfillment, while others believe it's about contributing to the welfare of others and making a positive impact in the world. 
Despite the myriad of perspectives, one thing remains clear: the meaning of life is a deeply personal concept that varies from one person to another. 
It's an existential inquiry that encourages us to reflect on our values, desires, and the essence of our existence.
"""

text_zh = """
预计未来3天深圳冷空气活动频繁，未来两天持续阴天有小雨，出门带好雨具；
10-11日持续阴天有小雨，日温差小，气温在13-17℃之间，体感阴凉；
12日天气短暂好转，早晚清凉；
"""

voice_rate=1.0
voice_volume=1.0
RUN_INTEGRATION_TESTS = os.environ.get("MPT_RUN_INTEGRATION_TESTS", "").lower() in {
    "1",
    "true",
    "yes",
}
                    
class TestVoiceService(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        self.loop.close()

    def test_get_all_azure_voices(self):
        voices = vs.get_all_azure_voices()
        # 数据已从内联字符串迁移到 azure_voices.json，确保仍能完整加载
        self.assertEqual(len(voices), 331)
        # 结果应为 "Name-Gender" 格式且已排序
        self.assertEqual(voices, sorted(voices))
        for v in voices:
            self.assertTrue(v.endswith("-Male") or v.endswith("-Female"))

    def test_get_all_azure_voices_filtered(self):
        filtered = vs.get_all_azure_voices(filter_locals=["zh-CN", "en-US"])
        self.assertTrue(len(filtered) > 0)
        self.assertTrue(
            all(v.startswith(("zh-CN", "en-US")) for v in filtered)
        )

    def test_no_voice_tts_generates_silent_audio_and_subtitle_timeline(self):
        """
        无配音模式不调用任何外部 TTS provider，只生成静音音频作为时间轴占位。
        这里 mock FFmpeg，验证请求参数、输出文件和 legacy 字幕结构都符合后续
        视频合成链路的预期。
        """

        def fake_run(command, capture_output, text, check):
            self.assertEqual(command[0], "/tmp/fake-ffmpeg")
            self.assertIn("anullsrc=r=44100:cl=mono", command)
            Path(command[-1]).write_bytes(b"fake-silent-mp3")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs.utils,
            "get_ffmpeg_binary",
            return_value="/tmp/fake-ffmpeg",
        ), patch.object(vs.subprocess, "run", side_effect=fake_run):
            voice_file = str(Path(tmp_dir) / "silent.mp3")
            sub_maker = vs.tts(
                text="第一句话。Second sentence.",
                voice_name=vs.NO_VOICE_NAME,
                voice_rate=1.0,
                voice_file=voice_file,
            )

            self.assertEqual(Path(voice_file).read_bytes(), b"fake-silent-mp3")

        self.assertIsNotNone(sub_maker)
        self.assertEqual(getattr(sub_maker, "subs", []), ["第一句话", "Second sentence"])
        self.assertEqual(len(getattr(sub_maker, "offset", [])), 2)
        self.assertGreater(vs.get_audio_duration(sub_maker), 0)

    def test_get_audio_duration_accepts_non_mp3_files(self):
        """
        自定义音频（custom_audio_file）常见为 m4a/wav/aac 等非 mp3 格式。
        get_audio_duration 不应因扩展名不是 .mp3 就报 "Invalid target type" 并返回 0，
        而应交给 moviepy(ffmpeg) 读取真实时长。
        """
        for path in ("custom-audio.m4a", "voice.wav", "clip.aac"):
            with patch.object(vs.os.path, "exists", return_value=True), \
                    patch.object(vs, "AudioFileClip") as mock_afc:
                mock_afc.return_value.__enter__.return_value.duration = 28.89
                self.assertEqual(vs.get_audio_duration(path), 28.89)
                mock_afc.assert_called_once_with(path)

    def test_get_audio_duration_missing_file_returns_zero(self):
        """音频文件不存在时安全返回 0，而不是抛异常或读取失败。"""
        with patch.object(vs.os.path, "exists", return_value=False):
            self.assertEqual(vs.get_audio_duration("does-not-exist.m4a"), 0.0)

    def test_no_voice_alias_none_is_supported_temporarily(self):
        """
        兼容 PR #981 曾使用过的 none sentinel，避免少量直接调用 API 的用户
        升级后立即失效。新 UI 和新代码仍统一使用 no-voice。
        """
        self.assertTrue(vs.is_no_voice("none"))
        self.assertTrue(vs.is_no_voice(vs.NO_VOICE_NAME))
        self.assertFalse(vs.is_no_voice(""))

    def test_no_voice_duration_estimates_non_ascii_languages(self):
        """
        无配音没有真实 TTS 音频，只能根据脚本文字估算阅读时间。俄语、阿拉伯语、
        日文假名、韩文等非 ASCII 文本也必须参与估算，不能都落到最短 3 秒。
        """
        russian_text = (
            "Это длинный тестовый сценарий без озвучки. "
            "Он должен получить достаточно времени для чтения субтитров."
        )
        arabic_text = "هذا اختبار طويل بدون تعليق صوتي، ويجب أن يحصل على وقت كاف لقراءة الترجمة."

        self.assertGreater(vs.estimate_no_voice_duration(russian_text), 8.0)
        self.assertGreater(vs.estimate_no_voice_duration(arabic_text), 8.0)

    def test_generate_silent_audio_rejects_missing_output_file(self):
        """
        即使 FFmpeg 进程返回成功，也要确认输出文件真实存在且非空。这样可以把
        异常收敛在 TTS 阶段，而不是拖到后续视频合成阶段才暴露。
        """
        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs.utils,
            "get_ffmpeg_binary",
            return_value="/tmp/fake-ffmpeg",
        ), patch.object(
            vs.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
        ):
            voice_file = str(Path(tmp_dir) / "missing-silent.mp3")

            self.assertFalse(vs.generate_silent_audio(3.0, voice_file))

    def test_empty_voice_name_does_not_enable_no_voice_mode(self):
        """
        空 voice 通常意味着配置缺失或接口参数错误，不能自动切到无配音模式。
        否则用户填错 TTS 配置时也会得到一个“成功”的静音视频，定位成本更高。
        """
        sentinel = object()

        with patch.object(vs, "azure_tts_v1", return_value=sentinel) as azure_tts_v1:
            result = vs.tts(
                text="empty voice should still use the default TTS path",
                voice_name="",
                voice_rate=1.0,
                voice_file="/tmp/empty-voice.mp3",
            )

        self.assertIs(result, sentinel)
        azure_tts_v1.assert_called_once()

    @unittest.skipUnless(
        RUN_INTEGRATION_TESTS,
        "MPT_RUN_INTEGRATION_TESTS not set",
    )
    def test_siliconflow(self):
        # SiliconFlow 的 API Key 存在 [siliconflow].api_key 中，运行时代码也是从
        # config.siliconflow 读取；这里必须使用同一配置源，避免正确配置凭据时
        # 测试仍然被误跳过。
        if not vs.config.siliconflow.get("api_key"):
            self.skipTest("siliconflow_api_key is not configured")

        voice_name = "siliconflow:FunAudioLLM/CosyVoice2-0.5B:alex-Male"
        voice_name = vs.parse_voice_name(voice_name)
        
        async def _do():
            parts = voice_name.split(":")
            if len(parts) >= 3:
                model = parts[1]
                # 移除性别后缀，例如 "alex-Male" -> "alex"
                voice_with_gender = parts[2]
                voice = voice_with_gender.split("-")[0]
                # 构建完整的voice参数，格式为 "model:voice"
                full_voice = f"{model}:{voice}"
                voice_file = f"{temp_dir}/tts-siliconflow-{voice}.mp3"
                subtitle_file = f"{temp_dir}/tts-siliconflow-{voice}.srt"
                sub_maker = vs.siliconflow_tts(
                    text=text_zh, model=model, voice=full_voice, voice_file=voice_file, voice_rate=voice_rate, voice_volume=voice_volume
                )
                if not sub_maker:
                    self.fail("siliconflow tts failed")
                vs.create_subtitle(sub_maker=sub_maker, text=text_zh, subtitle_file=subtitle_file)
                audio_duration = vs.get_audio_duration(sub_maker)
                print(f"voice: {voice_name}, audio duration: {audio_duration}s")
            else:
                self.fail("siliconflow invalid voice name")

        self.loop.run_until_complete(_do())
    
    @unittest.skipUnless(
        RUN_INTEGRATION_TESTS,
        "MPT_RUN_INTEGRATION_TESTS not set",
    )
    def test_azure_tts_v1(self):
        voice_name = "zh-CN-XiaoyiNeural-Female"
        voice_name = vs.parse_voice_name(voice_name)
        print(voice_name)
        
        voice_file = f"{temp_dir}/tts-azure-v1-{voice_name}.mp3"
        subtitle_file = f"{temp_dir}/tts-azure-v1-{voice_name}.srt"
        sub_maker = vs.azure_tts_v1(
            text=text_zh, voice_name=voice_name, voice_file=voice_file, voice_rate=voice_rate
        )
        if not sub_maker:
            self.fail("azure tts v1 failed")
        vs.create_subtitle(sub_maker=sub_maker, text=text_zh, subtitle_file=subtitle_file)
        audio_duration = vs.get_audio_duration(sub_maker)
        print(f"voice: {voice_name}, audio duration: {audio_duration}s")

    def test_azure_tts_v1_supports_legacy_edge_tts_without_boundary(self):
        """
        验证 Azure TTS V1 在旧版 edge_tts 依赖残留时仍可继续工作。

        这个回归场景对应 Windows 便携包更新失败后，现场环境还停留在旧版
        edge_tts 的情况：
        1. `Communicate.__init__()` 不接受 `boundary`
        2. 只有异步 `stream()`，没有 `stream_sync()`
        """

        class _LegacyCommunicate:
            def __init__(self, text, voice, rate="+0%"):
                self.text = text
                self.voice = voice
                self.rate = rate

            async def stream(self):
                yield {"type": "audio", "data": b"legacy-audio"}
                yield {
                    "type": "WordBoundary",
                    "offset": 0,
                    "duration": 10000000,
                    "text": "legacy",
                }

        class _FakeSubMaker:
            def __init__(self):
                self.events = []

            def feed(self, chunk):
                self.events.append(chunk)

            def get_srt(self):
                if not self.events:
                    return ""
                return "1\n00:00:00,000 --> 00:00:01,000\nlegacy\n"

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs.edge_tts, "Communicate", _LegacyCommunicate
        ), patch.object(vs.edge_tts, "SubMaker", _FakeSubMaker):
            voice_file = str(Path(tmp_dir) / "legacy-edge-tts.mp3")
            sub_maker = vs.azure_tts_v1(
                text="legacy edge tts compatibility",
                voice_name="zh-CN-XiaoyiNeural-Female",
                voice_file=voice_file,
                voice_rate=1.0,
            )

            self.assertIsNotNone(sub_maker)
            self.assertEqual(Path(voice_file).read_bytes(), b"legacy-audio")
            self.assertEqual(len(sub_maker.events), 1)
            self.assertEqual(sub_maker.events[0]["type"], "WordBoundary")

    def test_azure_tts_v1_times_out_hanging_stream_sync(self):
        """
        验证 Azure TTS V1 在 edge_tts 同步流卡住时能够快速失败。

        真实现场里，网络异常、服务端限流、voice 语言与文本不匹配时，
        `stream_sync()` 可能长时间不返回，导致 WebUI 任务只停在
        `start, voice name...`。这里用阻塞的 fake stream 复现该场景，
        确认超时保护会让函数结束并返回 None。
        """

        class _HangingCommunicate:
            def __init__(self, text, voice, rate="+0%", boundary=None):
                self.text = text
                self.voice = voice
                self.rate = rate
                self.boundary = boundary

            def stream_sync(self):
                time.sleep(10)
                yield {"type": "audio", "data": b"unreachable"}

        class _FakeSubMaker:
            def feed(self, chunk):
                return None

            def get_srt(self):
                return ""

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs.edge_tts, "Communicate", _HangingCommunicate
        ), patch.object(vs.edge_tts, "SubMaker", _FakeSubMaker), patch.object(
            vs.config,
            "app",
            dict(vs.config.app, edge_tts_timeout=0.05),
        ):
            voice_file = Path(tmp_dir) / "hanging-edge-tts.mp3"
            started_at = time.monotonic()
            sub_maker = vs.azure_tts_v1(
                text="帮我生成一个花开花落的视频",
                voice_name="en-AU-NatashaNeural-Female",
                voice_file=str(voice_file),
                voice_rate=1.0,
            )
            elapsed = time.monotonic() - started_at
            self.assertFalse(voice_file.exists())

        self.assertIsNone(sub_maker)
        self.assertLess(elapsed, 2)

    @unittest.skipUnless(
        RUN_INTEGRATION_TESTS,
        "MPT_RUN_INTEGRATION_TESTS not set",
    )
    def test_azure_tts_v2(self):
        if not vs.config.azure.get("speech_key") or not vs.config.azure.get("speech_region"):
            self.skipTest("Azure speech key or region is not configured")

        voice_name = "zh-CN-XiaoxiaoMultilingualNeural-V2-Female"
        voice_name = vs.parse_voice_name(voice_name)
        print(voice_name)

        async def _do():
            voice_file = f"{temp_dir}/tts-azure-v2-{voice_name}.mp3"
            subtitle_file = f"{temp_dir}/tts-azure-v2-{voice_name}.srt"
            sub_maker = vs.azure_tts_v2(
                text=text_zh,
                voice_name=voice_name,
                voice_file=voice_file,
                voice_rate=1.0,
            )
            if not sub_maker:
                self.fail("azure tts v2 failed")
            vs.create_subtitle(sub_maker=sub_maker, text=text_zh, subtitle_file=subtitle_file)
            audio_duration = vs.get_audio_duration(sub_maker)
            print(f"voice: {voice_name}, audio duration: {audio_duration}s")

        self.loop.run_until_complete(_do())

    def test_azure_tts_v2_ssml_applies_rate_and_escapes_text(self):
        """Azure V2 必须通过 SSML 应用语速，并避免用户文案破坏 XML。"""
        ssml = vs._build_azure_v2_ssml(
            text='A < B & "quoted"',
            voice_name="zh-CN-XiaoxiaoMultilingualNeural",
            voice_rate=1.8,
        )

        self.assertIn('xml:lang="zh-CN"', ssml)
        self.assertIn('rate="1.8"', ssml)
        self.assertIn("A &lt; B &amp; \"quoted\"", ssml)

    def test_tts_forwards_rate_to_azure_v2(self):
        """统一 TTS 入口不能在分发 Azure V2 时丢失 voice_rate。"""
        voice_name = "zh-CN-XiaoxiaoMultilingualNeural-V2-Female"
        with patch.object(vs, "azure_tts_v2", return_value=object()) as mock_tts:
            result = vs.tts(
                text="语速测试",
                voice_name=voice_name,
                voice_rate=1.8,
                voice_file="/tmp/azure-v2-rate.mp3",
            )

        self.assertIsNotNone(result)
        mock_tts.assert_called_once_with(
            "语速测试",
            voice_name,
            "/tmp/azure-v2-rate.mp3",
            voice_rate=1.8,
        )

    def test_gemini_tts_uses_google_genai_and_compatible_submaker_fields(self):
        """
        验证 Gemini TTS 在 edge_tts 7.x 环境下仍会返回项目兼容的字幕结构，
        并且可以被 `subtitle_provider=edge` 的字幕生成链路直接消费，
        避免再次回退 Whisper。同时使用不存在的嵌套输出目录，覆盖 API 或
        CLI 直接调用服务时没有提前创建任务目录的边界情况。
        """

        class _InlineData:
            def __init__(self, data):
                self.data = data

        class _Part:
            def __init__(self, data):
                self.inline_data = _InlineData(data)

        class _Content:
            def __init__(self, data):
                self.parts = [_Part(data)]

        class _Candidate:
            def __init__(self, data):
                self.content = _Content(data)

        class _Response:
            def __init__(self, data):
                self.candidates = [_Candidate(data)]

        captured = {}

        class _FakeModels:
            def generate_content(self, **kwargs):
                captured.update(kwargs)
                tone = (
                    AudioSegment.silent(duration=1800)
                    .set_frame_rate(24000)
                    .set_channels(1)
                    .set_sample_width(2)
                )
                return _Response(tone.raw_data)

        class _FakeClient:
            def __init__(self, **kwargs):
                captured["client_kwargs"] = kwargs
                self.models = _FakeModels()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                captured["closed"] = True

        temp_root = Path(tempfile.mkdtemp(prefix="gemini-tts-output-"))
        self.addCleanup(shutil.rmtree, temp_root, True)
        output_dir = temp_root / "nested" / "audio"
        voice_file = str(output_dir / "tts-gemini-Zephyr.mp3")
        subtitle_file = str(output_dir / "tts-gemini-Zephyr.srt")
        text = "Gemini subtitle generation should work now. Testing multiple lines."

        self.assertFalse(output_dir.exists())

        with patch("google.genai.Client", _FakeClient), patch.object(
            vs.config,
            "app",
            dict(vs.config.app, gemini_api_key="test-key"),
        ):
            sub_maker = vs.gemini_tts(
                text=text,
                voice_name="Zephyr",
                voice_rate=1.0,
                voice_file=voice_file,
            )

        self.assertIsNotNone(sub_maker)
        self.assertTrue(Path(voice_file).is_file())
        self.assertEqual(
            getattr(sub_maker, "subs", []),
            ["Gemini subtitle generation should work now", "Testing multiple lines"],
        )
        self.assertEqual(len(getattr(sub_maker, "offset", [])), 2)
        self.assertEqual(sub_maker.offset[0][0], 0)
        self.assertLess(sub_maker.offset[0][1], sub_maker.offset[1][1])
        self.assertEqual(captured["client_kwargs"], {"api_key": "test-key"})
        self.assertEqual(captured["model"], "gemini-2.5-flash-preview-tts")
        self.assertEqual(captured["contents"], text)
        self.assertEqual(captured["config"].response_modalities, ["AUDIO"])
        voice_config = captured["config"].speech_config.voice_config
        self.assertEqual(
            voice_config.prebuilt_voice_config.voice_name,
            "Zephyr",
        )
        self.assertTrue(captured["closed"])

        vs.create_subtitle(sub_maker=sub_maker, text=text, subtitle_file=subtitle_file)
        subtitle_content = Path(subtitle_file).read_text(encoding="utf-8")
        self.assertIn("Gemini subtitle generation should work now", subtitle_content)
        self.assertIn("Testing multiple lines", subtitle_content)

    def test_mimo_tts_uses_openai_compatible_audio_response(self):
        """
        验证 Xiaomi MiMo TTS 可以消费 OpenAI-compatible 的音频响应结构。

        这里用 fake OpenAI client 和 fake AudioSegment 覆盖真实网络与 ffmpeg，
        确认运行时代码会把待合成文本放到 assistant message，并把返回的
        base64 WAV 音频导出到项目后续流程使用的音频文件。
        """

        class _FakeAudio:
            def __init__(self):
                self.data = base64.b64encode(b"RIFF-fake-wav").decode("utf-8")

        class _FakeMessage:
            def __init__(self):
                self.audio = _FakeAudio()

        class _FakeChoice:
            def __init__(self):
                self.message = _FakeMessage()

        class _FakeCompletion:
            def __init__(self):
                self.choices = [_FakeChoice()]

        class _FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return _FakeCompletion()

        class _FakeAudioSegment:
            def __len__(self):
                return 1800

            def export(self, output_file, format):
                Path(output_file).write_bytes(b"fake-mp3")

        fake_completions = _FakeCompletions()
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake_completions)
        )

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs,
            "OpenAI",
            return_value=fake_client,
        ) as openai_client, patch(
            "pydub.AudioSegment.from_file",
            return_value=_FakeAudioSegment(),
        ), patch.object(
            vs.config,
            "app",
            dict(
                vs.config.app,
                mimo_api_key="mimo-key",
                mimo_base_url="https://api.xiaomimimo.com/v1",
                mimo_tts_model_name="mimo-v2.5-tts",
                mimo_tts_style_prompt="用清晰的中文旁白朗读。",
            ),
        ):
            voice_file = str(Path(tmp_dir) / "mimo-tts.mp3")
            sub_maker = vs.mimo_tts(
                text="小米语音合成测试。第二句话。",
                voice_name="冰糖",
                voice_rate=1.0,
                voice_file=voice_file,
                voice_volume=1.0,
            )
            generated_audio = Path(voice_file).read_bytes()

        openai_client.assert_called_once_with(
            api_key="mimo-key",
            base_url="https://api.xiaomimimo.com/v1",
        )
        self.assertEqual(fake_completions.kwargs["model"], "mimo-v2.5-tts")
        self.assertEqual(
            fake_completions.kwargs["messages"],
            [
                {"role": "user", "content": "用清晰的中文旁白朗读。"},
                {"role": "assistant", "content": "小米语音合成测试。第二句话。"},
            ],
        )
        self.assertEqual(
            fake_completions.kwargs["audio"],
            {"format": "wav", "voice": "冰糖"},
        )
        self.assertEqual(generated_audio, b"fake-mp3")
        self.assertIsNotNone(sub_maker)
        self.assertEqual(getattr(sub_maker, "subs", []), ["小米语音合成测试", "第二句话"])
        self.assertEqual(len(getattr(sub_maker, "offset", [])), 2)

    def test_chatterbox_voice_helpers(self):
        """is_chatterbox_voice / get_chatterbox_voices basics and normalisation."""
        self.assertTrue(vs.is_chatterbox_voice("chatterbox:default-Female"))
        self.assertFalse(vs.is_chatterbox_voice("elevenlabs:abc:Rachel"))
        self.assertFalse(vs.is_chatterbox_voice(""))
        self.assertFalse(vs.is_chatterbox_voice(None))

        # list entries are normalised to the chatterbox:<name> dispatcher format,
        # and entries that are already prefixed are left untouched
        with patch.object(
            vs.config,
            "chatterbox",
            {"voices": ["narrator-Male", "chatterbox:host"]},
        ):
            self.assertEqual(
                vs.get_chatterbox_voices(),
                ["chatterbox:narrator-Male", "chatterbox:host"],
            )

        # a comma-separated string is also accepted (TOML-friendly)
        with patch.object(vs.config, "chatterbox", {"voices": "alpha, beta ,"}):
            self.assertEqual(
                vs.get_chatterbox_voices(),
                ["chatterbox:alpha", "chatterbox:beta"],
            )

        # with nothing configured the dropdown still gets a usable default
        with patch.object(vs.config, "chatterbox", {}):
            self.assertEqual(vs.get_chatterbox_voices(), ["chatterbox:default-Female"])

    def test_chatterbox_tts_posts_to_openai_compatible_endpoint(self):
        """Success path: POST /audio/speech, write audio, return legacy SubMaker."""

        class _FakeResponse:
            status_code = 200
            content = b"RIFF-fake-wav"
            text = ""

        class _FakeClip:
            duration = 3.5

            def close(self):
                pass

        captured = {}

        def _fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _FakeResponse()

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs.config,
            "chatterbox",
            {
                "base_url": "http://localhost:4123/v1/",
                "api_key": "secret",
                "model_id": "chatterbox",
            },
        ), patch.object(
            vs.requests, "post", side_effect=_fake_post
        ) as post, patch.object(
            vs, "AudioFileClip", return_value=_FakeClip()
        ):
            voice_file = str(Path(tmp_dir) / "chatterbox.mp3")
            sub_maker = vs.chatterbox_tts(
                text="Hello world. Second sentence.",
                voice="default",
                voice_file=voice_file,
                voice_rate=1.2,
                voice_volume=1.0,
            )
            generated_audio = Path(voice_file).read_bytes()

        post.assert_called_once()
        # trailing slash on base_url is stripped before appending /audio/speech
        self.assertEqual(captured["url"], "http://localhost:4123/v1/audio/speech")
        self.assertEqual(captured["json"]["model"], "chatterbox")
        self.assertEqual(captured["json"]["voice"], "default")
        self.assertEqual(captured["json"]["input"], "Hello world. Second sentence.")
        self.assertAlmostEqual(captured["json"]["speed"], 1.2)
        # api_key is forwarded as a bearer token
        self.assertEqual(captured["headers"].get("Authorization"), "Bearer secret")
        # volume is intentionally not part of the OpenAI speech payload
        self.assertNotIn("volume", captured["json"])
        self.assertEqual(generated_audio, b"RIFF-fake-wav")
        self.assertIsNotNone(sub_maker)
        self.assertTrue(getattr(sub_maker, "subs", []))

    def test_chatterbox_tts_requires_base_url(self):
        """Missing base_url short-circuits without any network call."""
        with patch.object(
            vs.config, "chatterbox", {"base_url": ""}
        ), patch.object(vs.requests, "post") as post:
            result = vs.chatterbox_tts(
                text="hi", voice="default", voice_file="unused.mp3"
            )
        self.assertIsNone(result)
        post.assert_not_called()

    def test_chatterbox_tts_returns_none_on_http_error(self):
        """A non-200 response is retried up to 3 times, then fails to None."""

        class _FakeResponse:
            status_code = 500
            content = b""
            text = "boom"

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            vs.config, "chatterbox", {"base_url": "http://localhost:4123/v1"}
        ), patch.object(
            vs.requests, "post", return_value=_FakeResponse()
        ) as post:
            voice_file = str(Path(tmp_dir) / "chatterbox.mp3")
            result = vs.chatterbox_tts(
                text="hi", voice="default", voice_file=voice_file
            )
        self.assertIsNone(result)
        self.assertEqual(post.call_count, 3)

    # ---------------------------------------------------------------
    # Piper (offline/local) TTS
    # ---------------------------------------------------------------

    def _install_fake_piper_module(self, load_calls, synth_calls, seconds_per_call=0.2):
        """Install a fake ``piper`` package into sys.modules so piper_tts()
        can be exercised without the real (large, model-loading) library."""
        import types

        class _FakeSynthesisConfig:
            def __init__(self, length_scale=1.0):
                self.length_scale = length_scale

        fake_config_module = types.ModuleType("piper.config")
        fake_config_module.SynthesisConfig = _FakeSynthesisConfig

        class _FakePiperVoiceInstance:
            def synthesize_wav(self, text, wav_file, syn_config=None):
                synth_calls.append((text, syn_config))
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                frame_count = int(16000 * seconds_per_call)
                wav_file.writeframes(b"\x00\x00" * frame_count)

        class _FakePiperVoice:
            @staticmethod
            def load(path):
                load_calls.append(path)
                return _FakePiperVoiceInstance()

        fake_piper_module = types.ModuleType("piper")
        fake_piper_module.PiperVoice = _FakePiperVoice
        fake_piper_module.config = fake_config_module

        return patch.dict(
            sys.modules, {"piper": fake_piper_module, "piper.config": fake_config_module}
        )

    def test_piper_voice_helpers(self):
        self.assertTrue(vs.is_piper_voice("piper:vi_VN-vais1000-medium-Female"))
        self.assertFalse(vs.is_piper_voice("chatterbox:default-Female"))
        self.assertFalse(vs.is_piper_voice(""))
        self.assertFalse(vs.is_piper_voice(None))

        with tempfile.TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, "vi_VN-vais1000-medium.onnx").write_bytes(b"x")
            Path(tmp_dir, "en_US-hfc_male-medium.onnx").write_bytes(b"x")
            Path(tmp_dir, "en_US-hfc_female-medium.onnx").write_bytes(b"x")
            Path(tmp_dir, "readme.txt").write_bytes(b"not a model")

            with patch.object(vs.config, "piper", {"models_dir": tmp_dir}):
                voices = vs.get_all_piper_voices()

        self.assertEqual(
            voices,
            [
                "piper:en_US-hfc_female-medium-Female",
                "piper:en_US-hfc_male-medium-Male",
                "piper:vi_VN-vais1000-medium-Female",
            ],
        )

    def test_get_all_piper_voices_returns_empty_when_unconfigured_or_missing(self):
        with patch.object(vs.config, "piper", {}):
            self.assertEqual(vs.get_all_piper_voices(), [])
        with patch.object(vs.config, "piper", {"models_dir": "/does/not/exist"}):
            self.assertEqual(vs.get_all_piper_voices(), [])

    def test_split_text_with_pauses(self):
        parts = vs.split_text_with_pauses(
            "Xin chào. {{1.0}} Hôm nay chúng ta học bài mới. {{0.5}} Bắt đầu nhé!"
        )
        self.assertEqual(
            parts,
            [
                ("text", "Xin chào."),
                ("pause", 1.0),
                ("text", "Hôm nay chúng ta học bài mới."),
                ("pause", 0.5),
                ("text", "Bắt đầu nhé!"),
            ],
        )

    def test_split_text_with_pauses_handles_no_markers(self):
        self.assertEqual(
            vs.split_text_with_pauses("plain text, no pauses"),
            [("text", "plain text, no pauses")],
        )

    def test_piper_tts_synthesizes_and_returns_legacy_submaker(self):
        load_calls, synth_calls = [], []
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "vi_VN-test-medium.onnx"
            model_path.write_bytes(b"fake-onnx")
            voice_file = str(Path(tmp_dir) / "out.wav")

            with patch.object(
                vs.config, "piper", {"models_dir": tmp_dir}
            ), self._install_fake_piper_module(load_calls, synth_calls):
                sub_maker = vs.piper_tts(
                    text="Xin chào các bạn.",
                    voice="vi_VN-test-medium",
                    voice_file=voice_file,
                    voice_rate=1.0,
                )

            self.assertIsNotNone(sub_maker)
            self.assertTrue(getattr(sub_maker, "subs", []))
            self.assertEqual(load_calls, [str(model_path)])
            self.assertEqual(len(synth_calls), 1)
            self.assertEqual(synth_calls[0][0], "Xin chào các bạn.")
            self.assertTrue(Path(voice_file).is_file())
            self.assertGreater(Path(voice_file).stat().st_size, 0)

    def test_piper_tts_applies_pause_syntax_as_silence(self):
        load_calls, synth_calls = [], []
        with tempfile.TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, "v.onnx").write_bytes(b"fake-onnx")
            voice_file = str(Path(tmp_dir) / "out.wav")

            with patch.object(
                vs.config, "piper", {"models_dir": tmp_dir}
            ), self._install_fake_piper_module(load_calls, synth_calls, seconds_per_call=0.2):
                sub_maker = vs.piper_tts(
                    text="Phần một. {{1.0}} Phần hai.",
                    voice="v",
                    voice_file=voice_file,
                    voice_rate=1.0,
                )

            self.assertIsNotNone(sub_maker)
            # only the two "text" parts are synthesized -- the pause marker
            # itself never reaches the TTS engine as speakable text
            self.assertEqual(
                [call[0] for call in synth_calls], ["Phần một.", "Phần hai."]
            )
            # 0.2s + 0.2s speech + 1.0s silence =~ 1.4s total
            audio = AudioSegment.from_file(voice_file)
            self.assertAlmostEqual(len(audio) / 1000.0, 1.4, delta=0.05)

    def test_piper_tts_inverts_voice_rate_to_piper_length_scale(self):
        load_calls, synth_calls = [], []
        with tempfile.TemporaryDirectory() as tmp_dir:
            Path(tmp_dir, "v.onnx").write_bytes(b"fake-onnx")
            voice_file = str(Path(tmp_dir) / "out.wav")

            with patch.object(
                vs.config, "piper", {"models_dir": tmp_dir}
            ), self._install_fake_piper_module(load_calls, synth_calls):
                vs.piper_tts(
                    text="hello",
                    voice="v",
                    voice_file=voice_file,
                    voice_rate=2.0,  # MPT convention: 2.0 = twice as fast
                )

        # Piper's length_scale is the inverse: larger = slower. 2x faster
        # speech means half the length_scale.
        self.assertAlmostEqual(synth_calls[0][1].length_scale, 0.5)

    def test_piper_tts_requires_models_dir(self):
        with patch.object(vs.config, "piper", {"models_dir": ""}):
            result = vs.piper_tts(text="hi", voice="v", voice_file="unused.wav")
        self.assertIsNone(result)

    def test_piper_tts_missing_model_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(vs.config, "piper", {"models_dir": tmp_dir}):
                result = vs.piper_tts(
                    text="hi", voice="does-not-exist", voice_file="unused.wav"
                )
        self.assertIsNone(result)

    def test_piper_tts_returns_none_when_text_is_empty(self):
        result = vs.piper_tts(text="   ", voice="v", voice_file="unused.wav")
        self.assertIsNone(result)

    def test_piper_tts_returns_none_when_piper_not_installed(self):
        with patch.object(
            vs.config, "piper", {"models_dir": "/whatever"}
        ), patch.dict(sys.modules, {"piper": None}):
            result = vs.piper_tts(text="hi", voice="v", voice_file="unused.wav")
        self.assertIsNone(result)

    def test_piper_voice_cache_evicts_least_recently_used_beyond_max_size(self):
        """
        长驻进程里用过的 Piper 语音不止一个时，缓存不能无限增长（每个模型
        可能几十到上百 MB）。用到超过上限的第 4 个语音后，最早且最久未使用
        的那个应该被换出，需要重新加载。
        """
        load_calls, synth_calls = [], []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for name in ["a", "b", "c", "d"]:
                Path(tmp_dir, f"{name}.onnx").write_bytes(b"fake-onnx")
            voice_file = str(Path(tmp_dir) / "out.wav")

            with patch.object(
                vs.config, "piper", {"models_dir": tmp_dir}
            ), self._install_fake_piper_module(
                load_calls, synth_calls
            ), patch.object(
                # _piper_voice_cache is process-global; isolate this test
                # from whatever other piper tests already cached, otherwise
                # eviction timing depends on test execution order.
                vs,
                "_piper_voice_cache",
                OrderedDict(),
            ):
                for name in ["a", "b", "c", "d"]:
                    vs.piper_tts(
                        text="hi", voice=name, voice_file=voice_file, voice_rate=1.0
                    )
                # "a" was evicted (cache max size is 3) once "d" loaded --
                # using it again must trigger a second real load.
                vs.piper_tts(
                    text="hi", voice="a", voice_file=voice_file, voice_rate=1.0
                )

        model_paths = [str(Path(tmp_dir) / f"{n}.onnx") for n in ["a", "b", "c", "d"]]
        self.assertEqual(load_calls, model_paths + [model_paths[0]])

    # ---------------------------------------------------------------
    # VieNeu-TTS (offline/local Vietnamese) TTS
    # ---------------------------------------------------------------

    def _install_fake_vieneu_module(self, infer_calls, save_calls=None, factory_calls=None):
        """Install a fake ``vieneu`` package into sys.modules so
        vieneu_tts()/get_all_vieneu_voices() can be exercised without the
        real (large, model-downloading) library."""
        import types

        import numpy as np

        save_calls = save_calls if save_calls is not None else []
        factory_calls = factory_calls if factory_calls is not None else []

        class _FakeVieneuClient:
            def list_preset_voices(self):
                return [
                    ("Minh Đức — Nam · Bắc · Phong cách tin tức", "Minh Đức"),
                    ("Trúc Ly — Nữ · Bắc · Phong cách tự nhiên", "Trúc Ly"),
                ]

            def infer(self, text, voice=None, style="tu_nhien", **kwargs):
                infer_calls.append((text, voice))
                return np.zeros(4800, dtype=np.float32)  # 0.1s of silence @ 48kHz

            def save(self, audio, output_path):
                save_calls.append(str(output_path))
                import soundfile as sf

                sf.write(str(output_path), audio, 48000)

        def _fake_factory(mode="v3turbo", **kwargs):
            factory_calls.append(mode)
            return _FakeVieneuClient()

        fake_module = types.ModuleType("vieneu")
        fake_module.Vieneu = _fake_factory
        return patch.dict(sys.modules, {"vieneu": fake_module})

    def test_vieneu_voice_helpers(self):
        self.assertTrue(vs.is_vieneu_voice("vieneu:Minh Đức-Male"))
        self.assertFalse(vs.is_vieneu_voice("piper:vi_VN-vais1000-medium-Female"))
        self.assertFalse(vs.is_vieneu_voice(""))
        self.assertFalse(vs.is_vieneu_voice(None))

    def test_get_all_vieneu_voices_parses_gender_and_formats_names(self):
        infer_calls = []
        with self._install_fake_vieneu_module(infer_calls), patch.object(
            vs, "_vieneu_client", None
        ):
            voices = vs.get_all_vieneu_voices()

        self.assertEqual(voices, ["vieneu:Minh Đức-Male", "vieneu:Trúc Ly-Female"])

    def test_get_all_vieneu_voices_returns_empty_when_not_installed(self):
        with patch.object(vs, "_vieneu_client", None), patch.dict(
            sys.modules, {"vieneu": None}
        ):
            self.assertEqual(vs.get_all_vieneu_voices(), [])

    def test_vieneu_client_is_cached_across_calls(self):
        infer_calls, save_calls, factory_calls = [], [], []
        with self._install_fake_vieneu_module(
            infer_calls, save_calls, factory_calls
        ), patch.object(vs, "_vieneu_client", None):
            vs.get_all_vieneu_voices()
            vs.get_all_vieneu_voices()

        # the fake Vieneu() factory must only be invoked once -- the second
        # call reuses the cached client instead of reloading the model.
        self.assertEqual(len(factory_calls), 1)

    def test_vieneu_tts_synthesizes_and_returns_legacy_submaker(self):
        infer_calls, save_calls = [], []
        with tempfile.TemporaryDirectory() as tmp_dir:
            voice_file = str(Path(tmp_dir) / "out.mp3")

            with self._install_fake_vieneu_module(
                infer_calls, save_calls
            ), patch.object(vs, "_vieneu_client", None):
                sub_maker = vs.vieneu_tts(
                    text="Xin chào các bạn.",
                    voice="Minh Đức",
                    voice_file=voice_file,
                )

            self.assertIsNotNone(sub_maker)
            self.assertTrue(getattr(sub_maker, "subs", []))
            self.assertEqual(infer_calls, [("Xin chào các bạn.", "Minh Đức")])
            self.assertEqual(len(save_calls), 1)
            self.assertTrue(Path(voice_file).is_file())
            self.assertGreater(Path(voice_file).stat().st_size, 0)

    def test_vieneu_tts_returns_none_when_text_is_empty(self):
        result = vs.vieneu_tts(text="   ", voice="Minh Đức", voice_file="unused.mp3")
        self.assertIsNone(result)

    def test_vieneu_tts_returns_none_when_vieneu_not_installed(self):
        with patch.object(vs, "_vieneu_client", None), patch.dict(
            sys.modules, {"vieneu": None}
        ):
            result = vs.vieneu_tts(text="hi", voice="Minh Đức", voice_file="unused.mp3")
        self.assertIsNone(result)

    def test_tts_dispatches_vieneu_voice_to_vieneu_tts(self):
        infer_calls, save_calls = [], []
        with tempfile.TemporaryDirectory() as tmp_dir:
            voice_file = str(Path(tmp_dir) / "out.mp3")
            with self._install_fake_vieneu_module(
                infer_calls, save_calls
            ), patch.object(vs, "_vieneu_client", None):
                sub_maker = vs.tts(
                    text="Xin chào",
                    voice_name="vieneu:Minh Đức-Male",
                    voice_rate=1.0,
                    voice_file=voice_file,
                )

            self.assertIsNotNone(sub_maker)
            self.assertEqual(infer_calls, [("Xin chào", "Minh Đức")])

    def test_generate_subtitle_keeps_edge_provider_for_gemini_legacy_submaker(self):
        """
        验证 Gemini TTS 返回的 legacy 字幕结构在 edge provider 下可以直接产出
        SRT，不会因为匹配失败而回退到 Whisper。
        """
        script = "Gemini subtitle generation should work now. Testing multiple lines."
        sub_maker = vs.populate_legacy_submaker_with_full_text(
            vs.ensure_legacy_submaker_fields(vs.SubMaker()),
            script,
            2.4,
        )

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            task_service.config,
            "app",
            dict(task_service.config.app, subtitle_provider="edge"),
        ), patch("app.services.subtitle.create") as whisper_create, patch(
            "app.utils.utils.task_dir",
            lambda tid="": str(Path(tmp_dir) / tid) if tid else str(Path(tmp_dir)),
        ):
            task_id = "gemini-subtitle-edge-task"
            Path(tmp_dir, task_id).mkdir(parents=True, exist_ok=True)
            subtitle_path, _card_timings = task_service.generate_subtitle(
                task_id=task_id,
                params=type("Params", (), {"subtitle_enabled": True, "video_aspect": "16:9"})(),
                video_script=script,
                sub_maker=sub_maker,
                audio_file="",
            )

            self.assertTrue(subtitle_path.endswith("subtitle.srt"))
            self.assertTrue(Path(subtitle_path).exists())
            self.assertFalse(whisper_create.called)
            subtitle_content = Path(subtitle_path).read_text(encoding="utf-8")
            self.assertIn("Gemini subtitle generation should work now", subtitle_content)
            self.assertIn("Testing multiple lines", subtitle_content)

    def test_generate_subtitle_with_legacy_submaker_and_max_line_length(self):
        """
        复现真实 bug：短视频 (9:16) 的 max_line_length 会按字符数把脚本切得比
        populate_legacy_submaker_with_full_text() 的整句切分更细。两边切分方式
        不一致时，Piper/VieNeu/Gemini 等 legacy submaker 的累计文本永远对不上
        更短的目标行，导致 sub_items 数量和 script_lines 对不上，字幕整个生成
        失败（用户表现为“视频生成成功但完全没有字幕”）。
        """
        script = (
            "Tại sao dòng chữ chỉ còn hai sản phẩm luôn khiến tim bạn đập nhanh hơn. "
            "Não người không xử lý thông tin một cách trung thực. "
            "Nó xử lý theo thứ mà nó sợ mất. "
            "Và điều này nghe hơi lạ nhưng não hoàn toàn có thể tự vẽ ra một sự thiếu "
            "hụt không hề tồn tại."
        )
        sub_maker = vs.populate_legacy_submaker_with_full_text(
            vs.ensure_legacy_submaker_fields(vs.SubMaker()),
            script,
            12.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir, patch.object(
            task_service.config,
            "app",
            dict(task_service.config.app, subtitle_provider="edge"),
        ), patch("app.services.subtitle.create") as whisper_create, patch(
            "app.utils.utils.task_dir",
            lambda tid="": str(Path(tmp_dir) / tid) if tid else str(Path(tmp_dir)),
        ):
            task_id = "legacy-submaker-short-form-task"
            Path(tmp_dir, task_id).mkdir(parents=True, exist_ok=True)
            subtitle_path, _card_timings = task_service.generate_subtitle(
                task_id=task_id,
                params=type(
                    "Params", (), {"subtitle_enabled": True, "video_aspect": "9:16"}
                )(),
                video_script=script,
                sub_maker=sub_maker,
                audio_file="",
            )

            self.assertTrue(subtitle_path.endswith("subtitle.srt"))
            self.assertTrue(Path(subtitle_path).exists())
            self.assertFalse(whisper_create.called)
            subtitle_content = Path(subtitle_path).read_text(encoding="utf-8")
            self.assertIn("Tại sao dòng chữ", subtitle_content)
            self.assertIn("tồn tại", subtitle_content)

    def test_script_split_keeps_thousand_separator_comma(self):
        """
        Edge TTS 会把 "1,000 years" 作为连续文本返回。脚本断句时不能把
        数字中间的英文逗号当成句子边界，否则字幕聚合会出现 issue #894
        里的 sub_items 数量少于 script_lines，并错误回退 Whisper。
        """
        text = (
            "It takes about 1,000 years for a single drop of water to finish "
            "the whole trip!"
        )

        self.assertEqual(
            utils.split_string_by_punctuations(text),
            [
                (
                    "It takes about 1,000 years for a single drop of water to finish "
                    "the whole trip"
                )
            ],
        )

    def test_edge_cue_aggregation_handles_thousand_separator_comma(self):
        """
        复现 issue #894 的关键形态：Edge cues 中最后一句作为连续文本返回，
        包含 `1,000 years`。脚本断句必须与 cues 聚合结果一致，不能把它
        拆成两条字幕。
        """
        text = (
            "The ocean isn't just sitting stil, it moves around the world like a massive "
            "amusement park ride! Cold water at the North and South Poles sinks to the "
            "bottom because it is heavy and salty. At the same time, warm water from the "
            "sunny equator flows along the top to take its place. This creates a giant "
            "underwater conveyor belt that travels all the way around the Earth. It takes "
            "about 1,000 years for a single drop of water to finish the whole trip!"
        )
        script_lines = utils.split_string_by_punctuations(text)
        cues = []
        for index, line in enumerate(script_lines):
            # Edge 的 cue content 经常没有脚本里的空格和标点布局，这里去掉空格
            # 来模拟更严格的匹配场景。
            cues.append(
                SimpleNamespace(
                    content=line.replace(" ", ""),
                    start=timedelta(seconds=index),
                    end=timedelta(seconds=index + 0.8),
                )
            )
        sub_maker = SimpleNamespace(cues=cues)

        sub_items = vs._build_subtitle_items_from_edge_cues(sub_maker, script_lines)

        self.assertEqual(len(sub_items), len(script_lines))
        self.assertIn("1,000 years", sub_items[-1])

    def test_script_split_supports_arabic_punctuation(self):
        """
        阿拉伯语脚本常用 ، ؛ ؟ 作为自然断句标点。断句阶段必须识别这些
        标点，否则 edge-tts cue 的停顿边界和脚本行边界会错位。
        """
        text = "مرحبا بالعالم، كيف حالك؟ هذا اختبار؛ يعمل بشكل جيد."

        self.assertEqual(
            utils.split_string_by_punctuations(text),
            [
                "مرحبا بالعالم",
                "كيف حالك",
                "هذا اختبار",
                "يعمل بشكل جيد",
            ],
        )

    def test_match_script_line_normalizes_arabic_letter_forms(self):
        """
        edge-tts 可能把阿拉伯语中的不同字母形态归一化，或返回带变音符号、
        Tatweel 的 cue 文本。匹配时应容错，但最终字幕仍保留原始脚本文案。
        """
        script_lines = ["أهلاً وسهلاً بك في المدرسة"]

        matched = vs._match_script_line(
            script_lines,
            "اهلا وسهلا بك في المدرسه",
            0,
        )

        self.assertEqual(matched, script_lines[0])

    def test_edge_cue_aggregation_handles_arabic_variant_forms(self):
        """
        复现阿拉伯语字幕失败的核心路径：脚本包含 أ/ة 等字母形态，edge cue
        返回 ا/ه 等归一化形态时，聚合仍应生成完整字幕，避免回退 Whisper。
        """
        text = "أهلاً وسهلاً بك في المدرسة؟ هذا اختبار رائع، شكراً لك."
        script_lines = utils.split_string_by_punctuations(text)
        cue_texts = [
            "اهلا وسهلا بك في المدرسه",
            "هذا اختبار رائع",
            "شكرا لك",
        ]
        sub_maker = SimpleNamespace(
            cues=[
                SimpleNamespace(
                    content=cue_text,
                    start=timedelta(seconds=index),
                    end=timedelta(seconds=index + 0.8),
                )
                for index, cue_text in enumerate(cue_texts)
            ]
        )

        sub_items = vs._build_subtitle_items_from_edge_cues(sub_maker, script_lines)

        self.assertEqual(len(sub_items), len(script_lines))
        self.assertIn("أهلاً وسهلاً بك في المدرسة", sub_items[0])
        self.assertIn("شكراً لك", sub_items[-1])

    def test_create_subtitle_without_max_line_length_matches_current_behavior(self):
        """
        max_line_length 未设置时行为必须与今天完全一致：按标点断句，不按字符数拆分。
        """
        text = "This is a very long sentence that must be wrapped into pieces."
        words = text.rstrip(".").split(" ")
        sub_maker = SimpleNamespace(
            cues=[
                SimpleNamespace(
                    content=word + " ",
                    start=timedelta(seconds=index),
                    end=timedelta(seconds=index + 0.8),
                )
                for index, word in enumerate(words)
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            vs.create_subtitle(
                sub_maker=sub_maker,
                text=text,
                subtitle_file=str(subtitle_file),
            )
            subtitle_content = subtitle_file.read_text(encoding="utf-8")

        self.assertIn(
            "This is a very long sentence that must be wrapped into pieces",
            subtitle_content,
        )

    def test_create_subtitle_with_max_line_length_wraps_long_lines(self):
        """
        max_line_length 设置时，超长句子应在单词边界拆成多条字幕，每条不超过上限。
        """
        text = "This is a very long sentence that must be wrapped into pieces."
        words = text.rstrip(".").split(" ")
        sub_maker = SimpleNamespace(
            cues=[
                SimpleNamespace(
                    content=word + " ",
                    start=timedelta(seconds=index),
                    end=timedelta(seconds=index + 0.8),
                )
                for index, word in enumerate(words)
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            vs.create_subtitle(
                sub_maker=sub_maker,
                text=text,
                subtitle_file=str(subtitle_file),
                max_line_length=20,
            )
            subtitle_items = subtitle_service.file_to_subtitles(str(subtitle_file))

        self.assertGreater(len(subtitle_items), 1)
        for _index, _times, sub_text in subtitle_items:
            self.assertLessEqual(len(sub_text), 20)

    def test_create_subtitle_ignores_markdown_separator_lines(self):
        """
        用户手动脚本可能包含 `---` 这类 Markdown 分隔符。TTS 不会朗读
        这些符号行，字幕聚合也不应把它们当成目标字幕行，否则后续真实
        字幕会卡住并回退到 Whisper。
        """
        text = "第一段\n---\n第二段"
        sub_maker = SimpleNamespace(
            cues=[
                SimpleNamespace(
                    content="第一段",
                    start=timedelta(seconds=0),
                    end=timedelta(seconds=0.8),
                ),
                SimpleNamespace(
                    content="第二段",
                    start=timedelta(seconds=1),
                    end=timedelta(seconds=1.8),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            vs.create_subtitle(
                sub_maker=sub_maker,
                text=text,
                subtitle_file=str(subtitle_file),
            )

            subtitle_content = subtitle_file.read_text(encoding="utf-8")

        self.assertIn("第一段", subtitle_content)
        self.assertIn("第二段", subtitle_content)
        self.assertNotIn("---", subtitle_content)
        self.assertNotIn("00:00:00,000 --> 00:00:00,000", subtitle_content)

    def test_create_subtitle_ignores_markdown_underscore_marks(self):
        """
        `_` 常被用户用作 Markdown 强调标记，但 TTS 返回的 cue 通常不包含
        这些格式符。匹配时应忽略 `_`，避免生成空字幕或回退到 Whisper。
        """
        text = "这是_a_测试。"
        sub_maker = SimpleNamespace(
            cues=[
                SimpleNamespace(
                    content="这是a测试",
                    start=timedelta(seconds=0),
                    end=timedelta(seconds=0.8),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            vs.create_subtitle(
                sub_maker=sub_maker,
                text=text,
                subtitle_file=str(subtitle_file),
            )

            subtitle_content = subtitle_file.read_text(encoding="utf-8")

        self.assertIn("这是a测试", subtitle_content)
        self.assertNotIn("这是_a_测试", subtitle_content)
        self.assertNotIn("00:00:00,000 --> 00:00:00,000", subtitle_content)

    def test_convert_rate_to_percent_signs_zero_rate(self):
        # Rates near but not exactly 1.0 round to 0 percent. edge-tts rejects
        # an unsigned "0%" (ValueError: Invalid rate '0%'), so the helper must
        # emit a sign-prefixed "+0%". Regression test for that crash.
        self.assertEqual(vs.convert_rate_to_percent(1.0), "+0%")
        self.assertEqual(vs.convert_rate_to_percent(1.004), "+0%")
        self.assertEqual(vs.convert_rate_to_percent(0.997), "+0%")
        self.assertEqual(vs.convert_rate_to_percent(1.5), "+50%")
        self.assertEqual(vs.convert_rate_to_percent(0.8), "-20%")

    def test_convert_rate_to_percent_invalid_values_default_to_normal(self):
        # API 和批处理脚本可能把空语速传成 0、None 或空字符串；这些都不应让
        # edge-tts 收到 -100% 或触发异常，而是按正常语速处理。
        self.assertEqual(vs.convert_rate_to_percent(0), "+0%")
        self.assertEqual(vs.convert_rate_to_percent(0.0), "+0%")
        self.assertEqual(vs.convert_rate_to_percent(None), "+0%")
        self.assertEqual(vs.convert_rate_to_percent(""), "+0%")


def test_split_string_by_punctuations_and_length_keeps_short_sentences_whole():
    from app.utils import utils

    result = utils.split_string_by_punctuations_and_length("Hello world.", 50)
    assert result == ["Hello world"]


def test_split_string_by_punctuations_and_length_wraps_long_sentences_at_word_boundaries():
    from app.utils import utils

    text = "This sentence is intentionally long enough that it must be wrapped into multiple pieces."
    result = utils.split_string_by_punctuations_and_length(text, 30)

    assert all(len(line) <= 30 for line in result)
    # Rejoining with single spaces must reproduce the original words in order,
    # proving no word was split or dropped.
    assert " ".join(result) == "This sentence is intentionally long enough that it must be wrapped into multiple pieces"


def test_split_string_by_punctuations_and_length_never_splits_a_single_long_word():
    from app.utils import utils

    result = utils.split_string_by_punctuations_and_length("Supercalifragilisticexpialidocious.", 10)
    assert result == ["Supercalifragilisticexpialidocious"]


class TestElevenLabsVoice(unittest.TestCase):

    def test_is_elevenlabs_voice_true(self):
        self.assertTrue(vs.is_elevenlabs_voice("elevenlabs:pNInz6obpgDQGcFmaJgB:Adam"))

    def test_is_elevenlabs_voice_false_azure(self):
        self.assertFalse(vs.is_elevenlabs_voice("zh-CN-XiaoxiaoNeural-Female"))

    def test_is_elevenlabs_voice_false_siliconflow(self):
        self.assertFalse(vs.is_elevenlabs_voice("siliconflow:model:voice-Male"))

    def test_is_elevenlabs_voice_empty(self):
        self.assertFalse(vs.is_elevenlabs_voice(""))

    def test_is_elevenlabs_voice_none(self):
        self.assertFalse(vs.is_elevenlabs_voice(None))

    def test_get_elevenlabs_voices_empty_api_key(self):
        result = vs.get_elevenlabs_voices("")
        self.assertEqual(result, [])

    @patch("app.services.voice.requests.get")
    def test_get_elevenlabs_voices_success(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "voices": [
                {"voice_id": "abc123", "name": "Adam"},
                {"voice_id": "def456", "name": "Rachel"},
            ]
        }
        result = vs.get_elevenlabs_voices("fake-api-key")
        self.assertEqual(result, [
            "elevenlabs:abc123:Adam",
            "elevenlabs:def456:Rachel",
        ])
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        self.assertIn("xi-api-key", call_kwargs.kwargs.get("headers", {}))

    @patch("app.services.voice.requests.get")
    def test_get_elevenlabs_voices_http_error(self, mock_get):
        mock_get.return_value.status_code = 401
        mock_get.return_value.text = "Unauthorized"
        result = vs.get_elevenlabs_voices("bad-key")
        self.assertEqual(result, [])

    @patch("app.services.voice.requests.get")
    def test_get_elevenlabs_voices_network_error(self, mock_get):
        import requests as req_lib
        mock_get.side_effect = req_lib.exceptions.ConnectionError("timeout")
        result = vs.get_elevenlabs_voices("fake-key")
        self.assertEqual(result, [])

    @patch("app.services.voice.requests.post")
    @patch("app.services.voice.AudioFileClip")
    @patch("app.services.voice.config")
    def test_elevenlabs_tts_success(self, mock_config, mock_clip_cls, mock_post):
        mock_config.elevenlabs.get.return_value = "fake-api-key"
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = b"fake-mp3-bytes"
        mock_clip_cls.return_value.duration = 3.0
        mock_clip_cls.return_value.close = lambda: None

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            out_path = f.name

        try:
            result = vs.elevenlabs_tts("Hello world", "abc123", out_path)
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, "subs"))
            self.assertTrue(hasattr(result, "offset"))
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    @patch("app.services.voice.config")
    def test_elevenlabs_tts_no_api_key(self, mock_config):
        mock_config.elevenlabs.get.return_value = ""
        result = vs.elevenlabs_tts("Hello", "abc123", "/tmp/test.mp3")
        self.assertIsNone(result)

    @patch("app.services.voice.config")
    def test_elevenlabs_tts_empty_text(self, mock_config):
        mock_config.elevenlabs.get.return_value = "fake-key"
        result = vs.elevenlabs_tts("  ", "abc123", "/tmp/test.mp3")
        self.assertIsNone(result)


if __name__ == "__main__":
    # python -m unittest test.services.test_voice.TestVoiceService.test_azure_tts_v1
    # python -m unittest test.services.test_voice.TestVoiceService.test_azure_tts_v2
    unittest.main() 
