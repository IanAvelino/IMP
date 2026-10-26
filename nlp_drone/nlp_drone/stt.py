"""Speech-to-text engines with one interface: transcribe(audio) -> text.

audio: path to an audio file (wav/aiff/mp3...) or a float32 numpy array at 16 kHz mono.
"""
import json
import time
import numpy as np

SAMPLE_RATE = 16000
DOMAIN_PROMPT = ("Drone commands: take off, land, hover, hold, stop, move forward, back, left, right, "
                 "up, down, turn left, turn right, rotate, degrees, metres, centimetres.")

# Words the command grammar needs (Vosk grammar mode restricts recognition to these)
VOCAB = """take off launch lift land landing touch down hold hover stop stay wait freeze pause halt
turn rotate yaw spin around move go fly head travel come forward forwards ahead back backward backwards
reverse left right up upward higher climb ascend rise raise downward lower descend drop clockwise
anticlockwise counter anti meter meters metre metres centimeter centimeters foot feet degree degrees
second seconds and then a an half quarter point to by for please position the drone zero one two three
four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen
nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred""".split()


def load_audio(audio):
    if isinstance(audio, np.ndarray):
        return audio.astype(np.float32)
    from faster_whisper import decode_audio
    return decode_audio(str(audio), sampling_rate=SAMPLE_RATE)


class WhisperSTT:
    def __init__(self, model="base.en", prompt=DOMAIN_PROMPT):
        from faster_whisper import WhisperModel
        self.name = f"whisper-{model}"
        self.model = WhisperModel(model, device="cpu", compute_type="int8")
        self.prompt = prompt

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(load_audio(audio), language="en", beam_size=5,
                                            initial_prompt=self.prompt, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()


class VoskSTT:
    def __init__(self, model="vosk-model-small-en-us-0.15", grammar=True):
        import vosk
        vosk.SetLogLevel(-1)
        self.name = f"vosk-{'grammar' if grammar else 'free'}"
        self.model = vosk.Model(model_name=model)
        self.grammar = json.dumps(VOCAB + ["[unk]"]) if grammar else None

    def transcribe(self, audio):
        import vosk
        pcm = (np.clip(load_audio(audio), -1, 1) * 32767).astype(np.int16).tobytes()
        rec = vosk.KaldiRecognizer(self.model, SAMPLE_RATE, self.grammar) if self.grammar else \
            vosk.KaldiRecognizer(self.model, SAMPLE_RATE)
        for i in range(0, len(pcm), 8000):
            rec.AcceptWaveform(pcm[i:i + 8000])
        text = json.loads(rec.FinalResult()).get("text", "")
        return " ".join(w for w in text.split() if w != "[unk]")


def make(engine):
    """'whisper', 'whisper:tiny.en', 'vosk', 'vosk:free'"""
    kind, _, opt = engine.partition(":")
    if kind == "whisper":
        return WhisperSTT(opt or "base.en")
    if kind == "vosk":
        return VoskSTT(grammar=(opt != "free"))
    raise ValueError(f"unknown engine {engine!r}")


def timed(stt, audio):
    t0 = time.perf_counter()
    text = stt.transcribe(audio)
    return text, time.perf_counter() - t0


def record_push_to_talk():
    """Press Enter to start, Enter to stop. Returns float32 audio at 16 kHz."""
    import sounddevice as sd
    frames = []
    input("  [Enter] to start speaking...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        callback=lambda data, *_: frames.append(data.copy())):
        input("  ...speaking - [Enter] to stop")
    return np.concatenate(frames)[:, 0] if frames else np.zeros(0, np.float32)
