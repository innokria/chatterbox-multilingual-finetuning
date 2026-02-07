from dataclasses import dataclass
from pathlib import Path
import os

import librosa
import torch
import perth
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


# ======================================================
# GLOBAL CPU DEVICE
# ======================================================
DEVICE = torch.device("cpu")

REPO_ID = "ResembleAI/chatterbox"

SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}


def punc_norm(text: str) -> str:
    if not text:
        return "You need to add some text for me to talk."

    if text[0].islower():
        text = text[0].upper() + text[1:]

    text = " ".join(text.split())

    replacements = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
    ]
    for a, b in replacements:
        text = text.replace(a, b)

    if not text.endswith((".", "!", "?", ",")):
        text += "."

    return text


# ======================================================
# CONDITION STRUCT
# ======================================================
@dataclass
class Conditionals:
    t3: T3Cond
    gen: dict

    def to(self, device=DEVICE):
        self.t3 = self.t3.to(device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device)
        return self

    def save(self, fpath: Path):
        torch.save(
            {"t3": self.t3.__dict__, "gen": self.gen},
            fpath,
        )

    @classmethod
    def load(cls, fpath):
        data = torch.load(fpath, map_location="cpu", weights_only=True)
        return cls(T3Cond(**data["t3"]), data["gen"])


# ======================================================
# MAIN TTS CLASS
# ======================================================
class ChatterboxMultilingualTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, t3, s3gen, ve, tokenizer, conds=None):
        self.device = DEVICE
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    # --------------------------------------------------
    # LOADERS
    # --------------------------------------------------
    #@classmethod
    @classmethod
    def from_local(cls, ckpt_dir, device="cpu"):
    # your existing code

    #def from_local(cls, ckpt_dir):
        ckpt_dir = Path(ckpt_dir)

        ve = VoiceEncoder()
        ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", map_location="cpu", weights_only=True))
        ve.to(DEVICE).eval()

        t3 = T3(T3Config.multilingual())

        # CPU-safe attention
        if hasattr(t3.tfmr.config, "attn_implementation"):
            t3.tfmr.config.attn_implementation = "eager"

        t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
        if "model" in t3_state:
            t3_state = t3_state["model"][0]

        t3.load_state_dict(t3_state)
        t3.to(DEVICE).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(ckpt_dir / "s3gen.pt", map_location="cpu", weights_only=True)
        )
        s3gen.to(DEVICE).eval()

        tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))

        # conds = None
        # conds_path = ckpt_dir / "conds.pt"
        # conds = Conditionals.load(conds_path)
        #  # manually move internal tensors if needed
        # conds.t3 = conds.t3.to('cpu')
        # #if conds_path.exists():
        #     #conds = Conditionals.load(conds_path).to(DEVICE)



        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        return cls(t3, s3gen, ve, tokenizer, conds)
        
    @classmethod
    def from_pretrained(cls, device: torch.device) -> "ChatterboxMultilingualTTS":
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=[
                    "ve.pt",
                    "t3_mtl23ls_v2.safetensors",
                    "s3gen.pt",
                    "grapheme_mtl_merged_expanded_v1.json",
                    "conds.pt",
                    "Cangjie5_TC.json",
                ],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device)
    

    # --------------------------------------------------
    # CONDITION PREP
    # --------------------------------------------------
    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        wav, _ = librosa.load(wav_fpath, sr=S3GEN_SR)
        wav_16k = librosa.resample(wav, S3GEN_SR, S3_SR)

        wav = wav[: self.DEC_COND_LEN]

        gen_dict = self.s3gen.embed_ref(wav, S3GEN_SR, device=self.device)

        tokens = None
        if self.t3.hp.speech_cond_prompt_len:
            tokens, _ = self.s3gen.tokenizer.forward(
                [wav_16k[: self.ENC_COND_LEN]],
                max_len=self.t3.hp.speech_cond_prompt_len,
            )
            tokens = torch.atleast_2d(tokens).to(self.device)

        ve_embed = torch.from_numpy(
            self.ve.embeds_from_wavs([wav_16k], sample_rate=S3_SR)
        ).mean(0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(self.device)

        self.conds = Conditionals(t3_cond, gen_dict)

    # --------------------------------------------------
    # GENERATION
    # --------------------------------------------------
    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
    ):
        if language_id not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language_id}")

        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration)
        elif self.conds is None:
            raise RuntimeError("Conditionals not prepared")

        text = punc_norm(text)

        tokens = self.tokenizer.text_to_tokens(text, language_id).to(self.device)
        tokens = torch.cat([tokens, tokens], dim=0)

        tokens = F.pad(tokens, (1, 0), value=self.t3.hp.start_text_token)
        tokens = F.pad(tokens, (0, 1), value=self.t3.hp.stop_text_token)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )[0]

            speech_tokens = drop_invalid_tokens(speech_tokens)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )

        return wav.squeeze(0).cpu()
