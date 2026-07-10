"""Voice gateway — talk to your laptop, it talks back.

    pip install -e '.[voice]'
    make voice

Push-to-talk MVP: press Enter, speak, press Enter again. Your speech runs
through the exact same loop/memory/eval pipeline as typed text — a gateway
only moves words in and out (that's the whole point of the gateway box).

  ears   faster-whisper (local Whisper, ~74MB model downloads on first run)
  voice  macOS `say` with a British voice by default (zero setup), or the
         neural Kokoro voice if installed:  pip install kokoro soundfile
         then set JARVIS_TTS=kokoro  (JARVIS_VOICE=bm_george / bm_fable / ...)

Wake-word mode ("hey <name>, ...") is deliberately v2 — see docs/roadmap:
openWakeWord can train a custom wake word for whatever we name this thing.
"""

from __future__ import annotations

import os
import subprocess
import sys

from jarvis.app import Jarvis

SAMPLE_RATE = 16000


def record_until_enter():
    """Capture mic audio between two Enter presses; returns a float32 array."""
    import numpy as np
    import sounddevice as sd

    frames: list[np.ndarray] = []

    def collect(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=collect):
        input("🎙  recording — press Enter when done… ")
    if not frames:
        return np.zeros(0, dtype="float32")
    return np.concatenate(frames)[:, 0]


class Ears:
    def __init__(self, model_size: str | None = None):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            model_size or os.getenv("JARVIS_WHISPER_MODEL", "base"),
            compute_type="int8",
        )

    def transcribe(self, audio) -> str:
        segments, _ = self.model.transcribe(audio, language=os.getenv("JARVIS_WHISPER_LANG"))
        return " ".join(seg.text.strip() for seg in segments).strip()


class Mouth:
    """TTS with a boring, reliable default (macOS `say`) and a neural upgrade
    (Kokoro-82M, Apache-2.0 — its bm_* voices are the proper British butler)."""

    def __init__(self):
        self.engine = os.getenv("JARVIS_TTS", "say")
        self.voice = os.getenv("JARVIS_VOICE", "")
        if self.engine == "kokoro":
            from kokoro import KPipeline

            self.pipeline = KPipeline(lang_code="b")  # b = British English
            self.voice = self.voice or "bm_george"

    def speak(self, text: str) -> None:
        if not text:
            return
        if self.engine == "kokoro":
            import sounddevice as sd

            for _, _, audio in self.pipeline(text, voice=self.voice):
                sd.play(audio, 24000)
                sd.wait()
        elif sys.platform == "darwin":
            subprocess.run(["say", "-v", self.voice or "Daniel", text], check=False)
        else:
            print("(no TTS engine on this platform — set JARVIS_TTS=kokoro)")


def main() -> None:
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        raise SystemExit("Voice extra not installed: pip install -e '.[voice]'")

    jarvis = Jarvis()
    ears = Ears()
    mouth = Mouth()
    print("Voice Jarvis ready. Press Enter to talk, Ctrl-C to quit.")

    while True:
        try:
            input("\n⏎ press Enter to talk… ")
            audio = record_until_enter()
        except (EOFError, KeyboardInterrupt):
            break
        if audio.size < SAMPLE_RATE // 4:  # under 250ms — probably a slip
            print("(too short, try again)")
            continue

        heard = ears.transcribe(audio)
        if not heard:
            print("(didn't catch that)")
            continue
        print(f"you › {heard}")

        result = jarvis.respond(heard)
        print(f"jarvis › {result.reply}")
        mouth.speak(result.reply)

    print("bye — your memory stays in state.db")


if __name__ == "__main__":
    main()
