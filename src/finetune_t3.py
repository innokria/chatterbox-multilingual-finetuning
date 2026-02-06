import argparse
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
import numpy as np
from transformers import (
    HfArgumentParser,
    EarlyStoppingCallback,
    set_seed,
    Trainer,
    PretrainedConfig,
)
from datasets import load_dataset, DatasetDict, VerificationMode

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import Conditionals, punc_norm, REPO_ID
from chatterbox.models.t3.t3 import T3, T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.s3tokenizer import S3_SR

logger = logging.getLogger(__name__)

# --- Custom Training Arguments ---
@dataclass
class CustomTrainingArguments:
    output_dir: str = field(default="./output")
    seed: int = field(default=42)
    do_train: bool = field(default=True)
    do_eval: bool = field(default=False)
    per_device_train_batch_size: int = field(default=1)  # CPU-friendly
    per_device_eval_batch_size: int = field(default=1)
    logging_steps: int = field(default=50)
    save_steps: int = field(default=500)
    resume_from_checkpoint: Optional[str] = None
    early_stopping_patience: Optional[int] = field(default=None)


# --- Model Arguments ---
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = None
    local_model_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    freeze_voice_encoder: bool = True
    freeze_s3gen: bool = True


# --- Data Arguments ---
@dataclass
class DataArguments:
    language: str = "en"  # default; can change to any supported language
    dataset_dir: Optional[str] = None
    metadata_file: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_split_name: str = "train"
    eval_split_name: str = "validation"
    text_column_name: str = "text"
    audio_column_name: str = "audio"
    max_text_len: int = 256
    max_speech_len: int = 800
    audio_prompt_duration_s: float = 3.0
    eval_split_size: float = 0.0005
    preprocessing_num_workers: Optional[int] = None
    ignore_verifications: bool = False


# --- Dataset Class ---
class SpeechFineTuningDataset(Dataset):
    def __init__(self, data_args: DataArguments, chatterbox_model: ChatterboxMultilingualTTS,
                 t3_config: T3Config, hf_dataset: Union[List[Dict], Any], is_hf_format: bool):
        self.data_args = data_args
        self.chatterbox_model = chatterbox_model
        self.chatterbox_t3_config = t3_config
        self.dataset_source = hf_dataset
        self.is_hf_format = is_hf_format

        self.text_tokenizer = chatterbox_model.tokenizer
        self.speech_tokenizer = chatterbox_model.s3gen.tokenizer
        self.voice_encoder = chatterbox_model.ve
        self.s3_sr = S3_SR
        self.enc_cond_audio_len_samples = int(self.data_args.audio_prompt_duration_s * self.s3_sr)

    def __len__(self):
        return len(self.dataset_source)

    def _load_audio_text_from_item(self, idx):
        item = self.dataset_source[idx]
        if self.is_hf_format:
            text = item[self.data_args.text_column_name]
            audio_data = item[self.data_args.audio_column_name]
            if isinstance(audio_data, str):
                wav_array, sr = librosa.load(audio_data, sr=self.s3_sr)
            elif isinstance(audio_data, dict):
                wav_array = np.array(audio_data["array"])
                sr = audio_data["sampling_rate"]
            else:
                return None, None
        else:
            text = item["text"]
            audio_path = item["audio"]
            wav_array, sr = librosa.load(audio_path, sr=self.s3_sr)
        return wav_array.astype(np.float32), text

    def __getitem__(self, idx):
        wav, text = self._load_audio_text_from_item(idx)
        if wav is None or text is None:
            return None

        # Speaker embedding
        speaker_emb = torch.from_numpy(self.voice_encoder.embeds_from_wavs([wav], sample_rate=self.s3_sr)[0])

        # Text tokens
        text_tokens = self.text_tokenizer.text_to_tokens(text, language_id=self.data_args.language).squeeze(0)
        text_tokens = F.pad(text_tokens, (1, 0), value=self.chatterbox_t3_config.start_text_token)
        text_tokens = F.pad(text_tokens, (0, 1), value=self.chatterbox_t3_config.stop_text_token)
        text_token_len = torch.tensor(len(text_tokens))

        # Speech tokens
        speech_tokens, speech_token_len = self.speech_tokenizer.forward([wav])
        speech_tokens = F.pad(speech_tokens.squeeze(0), (1, 1), value=self.chatterbox_t3_config.start_speech_token)
        speech_token_len = torch.tensor(len(speech_tokens))

        # Conditional prompt tokens
        cond_prompt_tokens, _ = self.speech_tokenizer.forward(
            [wav[:self.enc_cond_audio_len_samples]],
            max_len=self.chatterbox_t3_config.speech_cond_prompt_len
        )
        cond_prompt_tokens = cond_prompt_tokens.squeeze(0)
        return {
            "text_tokens": text_tokens.long(),
            "text_token_lens": text_token_len.long(),
            "speech_tokens": speech_tokens.long(),
            "speech_token_lens": speech_token_len.long(),
            "t3_cond_speaker_emb": speaker_emb.float(),
            "t3_cond_prompt_speech_tokens": cond_prompt_tokens.long(),
            "t3_cond_emotion_adv": torch.tensor(0.5, dtype=torch.float)
        }


# --- Data Collator ---
@dataclass
class SpeechDataCollator:
    t3_config: T3Config
    text_pad_token_id: int
    speech_pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]):
        features = [f for f in features if f is not None]
        if not features:
            return {}
        batch_size = len(features)
        max_text_len = max(len(f["text_tokens"]) for f in features)
        max_speech_len = max(len(f["speech_tokens"]) for f in features)

        padded_text = torch.stack([F.pad(f["text_tokens"], (0, max_text_len - len(f["text_tokens"])),
                                        value=self.text_pad_token_id) for f in features])
        padded_speech = torch.stack([F.pad(f["speech_tokens"], (0, max_speech_len - len(f["speech_tokens"])),
                                          value=self.speech_pad_token_id) for f in features])

        return {
            "text_tokens": padded_text,
            "speech_tokens": padded_speech,
            "t3_cond_speaker_emb": torch.stack([f["t3_cond_speaker_emb"] for f in features]),
            "t3_cond_prompt_speech_tokens": torch.stack([f["t3_cond_prompt_speech_tokens"] for f in features]),
            "t3_cond_emotion_adv": torch.stack([f["t3_cond_emotion_adv"] for f in features]).view(batch_size, 1, 1)
        }


# --- T3 Wrapper ---
class T3ForFineTuning(torch.nn.Module):
    def __init__(self, t3_model: T3, t3_config: T3Config):
        super().__init__()
        self.t3 = t3_model
        self.chatterbox_t3_config = t3_config
        self.config = PretrainedConfig()

    def forward(self, text_tokens, text_token_lens, speech_tokens, speech_token_lens,
                t3_cond_speaker_emb, t3_cond_prompt_speech_tokens, t3_cond_emotion_adv,
                labels_text=None, labels_speech=None):
        t3_cond = T3Cond(
            speaker_emb=t3_cond_speaker_emb,
            cond_prompt_speech_tokens=t3_cond_prompt_speech_tokens,
            cond_prompt_speech_emb=None,
            emotion_adv=t3_cond_emotion_adv
        ).to(device=self.t3.device)

        loss_text, loss_speech, _ = self.t3.loss(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            labels_text=labels_text,
            labels_speech=labels_speech
        )
        return loss_text + loss_speech, None


# --- Main function ---
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )
    set_seed(training_args.seed)

    # --- Load model ---
    if model_args.local_model_dir:
        chatterbox_model = ChatterboxMultilingualTTS.from_pretrained()
    else:
        chatterbox_model = ChatterboxMultilingualTTS.from_pretrained()
    chatterbox_model.ve.to("cpu")
    chatterbox_model.s3gen.to("cpu")
    chatterbox_model.t3.to("cpu")

    if model_args.freeze_voice_encoder:
        for p in chatterbox_model.ve.parameters():
            p.requires_grad = False
    if model_args.freeze_s3gen:
        for p in chatterbox_model.s3gen.parameters():
            p.requires_grad = False
    for p in chatterbox_model.t3.parameters():
        p.requires_grad = True

    # --- Load dataset ---
    if data_args.dataset_name:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=data_args.train_split_name
        )
        train_dataset = SpeechFineTuningDataset(data_args, chatterbox_model,
                                                chatterbox_model.t3.hp,
                                                raw_datasets, True)
    else:
        # Local dataset example
        all_files = []
        if data_args.dataset_dir:
            dataset_path = Path(data_args.dataset_dir)
            for audio_file in dataset_path.rglob("*.wav"):
                text_file = audio_file.with_suffix(".txt")
                if text_file.exists():
                    text = text_file.read_text().strip()
                    all_files.append({"audio": str(audio_file), "text": text})
        train_dataset = SpeechFineTuningDataset(data_args, chatterbox_model,
                                                chatterbox_model.t3.hp,
                                                all_files, False)

    # --- Data collator ---
    data_collator = SpeechDataCollator(chatterbox_model.t3.hp,
                                       chatterbox_model.t3.hp.stop_text_token,
                                       chatterbox_model.t3.hp.stop_speech_token)

    # --- Trainer ---
    trainer = Trainer(
        model=T3ForFineTuning(chatterbox_model.t3, chatterbox_model.t3.hp),
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    # --- Train ---
    if training_args.do_train:
        logger.info("Starting CPU training...")
        trainer.train()


if __name__ == "__main__":
    main()
