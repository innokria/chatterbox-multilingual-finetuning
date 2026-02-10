import argparse
import logging
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    HfArgumentParser,
    EarlyStoppingCallback,
    set_seed,
    TrainerCallback,
    Trainer,
    PretrainedConfig,
)
from transformers import TrainingArguments as HfTrainingArguments
from datasets import load_dataset, DatasetDict, VerificationMode, Audio
import datasets

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS, Conditionals, punc_norm, REPO_ID
from chatterbox.models.t3.t3 import T3, T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.s3tokenizer import S3_SR, SPEECH_VOCAB_SIZE
from chatterbox.models.s3gen import S3GEN_SR

logger = logging.getLogger(__name__)


# --- Custom Training Arguments ---
@dataclass
class CustomTrainingArguments(HfTrainingArguments):
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={
            "help": "Enable early stopping with specified patience. Default: None (disabled)."
        },
    )


# --- Argument Classes (ModelArguments, DataArguments) ---
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    local_model_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to local directory containing ve.safetensors, t3_cfg.safetensors, etc. Overrides model_name_or_path for loading."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    freeze_voice_encoder: bool = field(
        default=True, metadata={"help": "Freeze the Voice Encoder."}
    )
    freeze_s3gen: bool = field(
        default=True,
        metadata={"help": "Freeze the S3Gen model (speech token to waveform)."},
    )


@dataclass
class DataArguments:
    language: Optional[str] = field(
        default="en",
        metadata={"help": "State target language code: 'en' , 'tr' ..."},
    )
    dataset_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the directory containing audio files and text files. Used if dataset_name is not provided."
        },
    )
    metadata_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a metadata file. Used if dataset_name is not provided."
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the Hugging Face datasets library)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the Hugging Face datasets library)."
        },
    )
    train_split_name: str = field(
        default="train", metadata={"help": "The name of the training data set split."}
    )
    eval_split_name: Optional[str] = field(
        default="validation",
        metadata={"help": "The name of the evaluation data set split."},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the text column in the HF dataset."},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the audio column in the HF dataset."},
    )
    max_text_len: int = field(
        default=256,
        metadata={"help": "Maximum length of text tokens (including BOS/EOS)."},
    )
    max_speech_len: int = field(
        default=800,
        metadata={"help": "Maximum length of speech tokens (including BOS/EOS)."},
    )
    audio_prompt_duration_s: float = field(
        default=3.0,
        metadata={
            "help": "Duration of audio (from start) to use for T3 conditioning prompt tokens (in seconds)."
        },
    )
    eval_split_size: float = field(
        default=0.0005,
        metadata={
            "help": "Fraction of data to use for evaluation if splitting manually. Not used if dataset_name provides eval split."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    ignore_verifications: bool = field(
        default=False, metadata={"help": "Set to true to ignore dataset verifications."}
    )


# --- Dataset Class ---
class SpeechFineTuningDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        chatterbox_model: ChatterboxMultilingualTTS,
        t3_config: T3Config,
        hf_dataset: Union[datasets.Dataset, List[Dict[str, str]]],
        is_hf_format: bool,
    ):
        self.data_args = data_args
        self.chatterbox_model = chatterbox_model
        self.chatterbox_t3_config = t3_config
        self.dataset_source = hf_dataset
        self.is_hf_format = is_hf_format

        self.text_tokenizer = chatterbox_model.tokenizer
        self.speech_tokenizer: S3Tokenizer = chatterbox_model.s3gen.tokenizer
        self.voice_encoder = chatterbox_model.ve

        self.s3_sr = S3_SR
        self.enc_cond_audio_len_samples = int(
            data_args.audio_prompt_duration_s * self.s3_sr
        )

    def __len__(self):
        return len(self.dataset_source)

    def _load_audio_text_from_item(self, idx):
        if self.is_hf_format:
            item = self.dataset_source[idx]
            text = item[self.data_args.text_column_name]
            audio_data = item[self.data_args.audio_column_name]

            if isinstance(audio_data, str):
                wav_array, original_sr = librosa.load(audio_data, sr=None, mono=True)
            elif (
                isinstance(audio_data, dict)
                and "array" in audio_data
                and "sampling_rate" in audio_data
            ):
                wav_array = audio_data["array"]
                original_sr = audio_data["sampling_rate"]
            else:
                logger.error(
                    f"Unexpected audio data format for item {idx}: {type(audio_data)}. Skipping."
                )
                return None, None

            if not isinstance(wav_array, np.ndarray):
                logger.error(
                    f"Audio array is not numpy for item {idx}: {type(wav_array)}. Skipping."
                )
                return None, None

            if original_sr != self.s3_sr:
                wav_16k = librosa.resample(
                    wav_array, orig_sr=original_sr, target_sr=self.s3_sr
                )
            else:
                wav_16k = wav_array.copy()

            if wav_16k.ndim > 1:
                wav_16k = librosa.to_mono(wav_16k)
            if wav_16k.dtype != np.float32:
                wav_16k = wav_16k.astype(np.float32)

            return wav_16k, text
        else:
            item = self.dataset_source[idx]
            audio_path = item["audio"]
            text = item["text"]
            try:
                wav_16k, _ = librosa.load(audio_path, sr=self.s3_sr, mono=True)
                return wav_16k, text
            except Exception as e:
                logger.error(f"Error loading audio {audio_path}: {e}")
                return None, None

    def __getitem__(self, idx) -> Optional[Dict[str, Union[torch.Tensor, float]]]:
        wav_16k, text = self._load_audio_text_from_item(idx)
        if wav_16k is None or text is None or len(wav_16k) == 0:
            return None

        # --- Speaker embedding ---
        try:
            speaker_emb_np = self.voice_encoder.embeds_from_wavs(
                [wav_16k], sample_rate=self.s3_sr
            )
            speaker_emb = torch.from_numpy(speaker_emb_np[0])
        except Exception as e:
            logger.error(
                f"Error getting speaker embedding for item {idx}: {e}. Skipping."
            )
            return None

        normalized_text = punc_norm(text)
        raw_text_tokens = self.text_tokenizer.text_to_tokens(
            normalized_text, language_id=self.data_args.language
        ).squeeze(0)
        text_tokens = F.pad(
            raw_text_tokens, (1, 0), value=self.chatterbox_t3_config.start_text_token
        )
        text_tokens = F.pad(
            text_tokens, (0, 1), value=self.chatterbox_t3_config.stop_text_token
        )
        if len(text_tokens) > self.data_args.max_text_len:
            text_tokens = text_tokens[: self.data_args.max_text_len - 1]
            text_tokens = torch.cat(
                [
                    text_tokens,
                    torch.tensor(
                        [self.chatterbox_t3_config.stop_text_token],
                        device=text_tokens.device,
                    ),
                ]
            )
        text_token_len = torch.tensor(len(text_tokens), dtype=torch.long)

        # --- Speech tokens ---
        try:
            raw_speech_tokens_batch, speech_token_lengths_batch = (
                self.speech_tokenizer.forward([wav_16k])
            )
            if raw_speech_tokens_batch is None or speech_token_lengths_batch is None:
                logger.error(f"S3Tokenizer returned None for item {idx}. Skipping.")
                return None
            raw_speech_tokens = raw_speech_tokens_batch.squeeze(0)[
                : speech_token_lengths_batch.squeeze(0).item()
            ]
        except Exception as e:
            logger.error(f"Error getting speech tokens for item {idx}: {e}. Skipping.")
            return None

        speech_tokens = F.pad(
            raw_speech_tokens,
            (1, 0),
            value=self.chatterbox_t3_config.start_speech_token,
        )
        speech_tokens = F.pad(
            speech_tokens, (0, 1), value=self.chatterbox_t3_config.stop_speech_token
        )
        if len(speech_tokens) > self.data_args.max_speech_len:
            speech_tokens = speech_tokens[: self.data_args.max_speech_len - 1]
            speech_tokens = torch.cat(
                [
                    speech_tokens,
                    torch.tensor(
                        [self.chatterbox_t3_config.stop_speech_token],
                        device=speech_tokens.device,
                    ),
                ]
            )
        speech_token_len = torch.tensor(len(speech_tokens), dtype=torch.long)

        # --- Conditional audio prompt ---
        cond_audio_segment = wav_16k[: self.enc_cond_audio_len_samples]
        if len(cond_audio_segment) == 0:
            cond_prompt_speech_tokens = torch.zeros(
                self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long
            )
        else:
            try:
                cond_prompt_tokens_batch, _ = self.speech_tokenizer.forward(
                    [cond_audio_segment],
                    max_len=self.chatterbox_t3_config.speech_cond_prompt_len,
                )
                if cond_prompt_tokens_batch is None:
                    cond_prompt_speech_tokens = torch.zeros(
                        self.chatterbox_t3_config.speech_cond_prompt_len,
                        dtype=torch.long,
                    )
                else:
                    cond_prompt_speech_tokens = cond_prompt_tokens_batch.squeeze(0)
            except Exception as e:
                cond_prompt_speech_tokens = torch.zeros(
                    self.chatterbox_t3_config.speech_cond_prompt_len, dtype=torch.long
                )

        if (
            cond_prompt_speech_tokens.size(0)
            != self.chatterbox_t3_config.speech_cond_prompt_len
        ):
            current_len = cond_prompt_speech_tokens.size(0)
            target_len = self.chatterbox_t3_config.speech_cond_prompt_len
            if current_len > target_len:
                cond_prompt_speech_tokens = cond_prompt_speech_tokens[:target_len]
            else:
                cond_prompt_speech_tokens = F.pad(
                    cond_prompt_speech_tokens, (0, target_len - current_len), value=0
                )

        emotion_adv_scalar_tensor = torch.tensor(0.5, dtype=torch.float)

        return {
            "text_tokens": text_tokens.long(),
            "text_token_lens": text_token_len.long(),
            "speech_tokens": speech_tokens.long(),
            "speech_token_lens": speech_token_len.long(),
            "t3_cond_speaker_emb": speaker_emb.float(),
            "t3_cond_prompt_speech_tokens": cond_prompt_speech_tokens.long(),
            "t3_cond_emotion_adv": emotion_adv_scalar_tensor,
        }


# --- Data Collator ---
@dataclass
class SpeechDataCollator:
    t3_config: T3Config
    text_pad_token_id: int
    speech_pad_token_id: int

    def __call__(self, features: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        valid_features = [f for f in features if f is not None]

        if not valid_features:
            logger.warning(
                "SpeechDataCollator received no valid features. Returning empty batch."
            )
            return {}
        features = valid_features

        batch_size = len(features)
        text_tokens_list = [f["text_tokens"] for f in features]
        speech_tokens_list = [f["speech_tokens"] for f in features]
        max_text_len = max(len(t) for t in text_tokens_list)
        max_speech_len = max(len(s) for s in speech_tokens_list)

        padded_text_tokens = torch.stack(
            [
                F.pad(t, (0, max_text_len - len(t)), value=self.text_pad_token_id)
                for t in text_tokens_list
            ]
        )

        padded_speech_tokens = torch.stack(
            [
                F.pad(s, (0, max_speech_len - len(s)), value=self.speech_pad_token_id)
                for s in speech_tokens_list
            ]
        )

        text_token_lens = torch.stack([f["text_token_lens"] for f in features])
        speech_token_lens = torch.stack([f["speech_token_lens"] for f in features])

        t3_cond_speaker_emb = torch.stack([f["t3_cond_speaker_emb"] for f in features])
        t3_cond_prompt_speech_tokens = torch.stack(
            [f["t3_cond_prompt_speech_tokens"] for f in features]
        )
        emotion_adv_scalars = torch.stack([f["t3_cond_emotion_adv"] for f in features])
        t3_cond_emotion_adv = emotion_adv_scalars.view(batch_size, 1, 1)

        IGNORE_ID = -100
        prompt_len = self.t3_config.speech_cond_prompt_len

        # --- Build labels_text ---
        shifted_text = padded_text_tokens[:, 1:].contiguous()
        T_text = shifted_text.size(1)
        text_lens_minus_one = (text_token_lens - 1).clamp(min=0)
        arange_text = torch.arange(T_text, device=shifted_text.device)
        mask_pad_text = arange_text[None] >= text_lens_minus_one[:, None]
        labels_text = shifted_text.clone()
        labels_text[mask_pad_text] = IGNORE_ID

        # --- Build labels_speech ---
        shifted_speech = padded_speech_tokens[:, 1:].contiguous()
        T_speech = shifted_speech.size(1)
        speech_lens_minus_one = (speech_token_lens - 1).clamp(min=0)
        arange_speech = torch.arange(T_speech, device=shifted_speech.device)
        mask_pad_speech = arange_speech[None] >= speech_lens_minus_one[:, None]
        mask_prompt = (arange_speech[None] < prompt_len).expand(batch_size, T_speech)
        mask_speech_total = mask_pad_speech | mask_prompt
        labels_speech = shifted_speech.clone()
        labels_speech[mask_speech_total] = IGNORE_ID

        return {
            "text_tokens": padded_text_tokens,
            "text_token_lens": text_token_lens,
            "speech_tokens": padded_speech_tokens,
            "speech_token_lens": speech_token_lens,
            "t3_cond_speaker_emb": t3_cond_speaker_emb,
            "t3_cond_prompt_speech_tokens": t3_cond_prompt_speech_tokens,
            "t3_cond_emotion_adv": t3_cond_emotion_adv,
            "labels_text": labels_text,
            "labels_speech": labels_speech,
        }


# --- Model Wrapper ---
class T3ForFineTuning(torch.nn.Module):
    def __init__(self, t3_model: T3, chatterbox_t3_config: T3Config):
        super().__init__()
        self.t3 = t3_model
        self.chatterbox_t3_config = chatterbox_t3_config

        class HFCompatibleConfig(PretrainedConfig):
            model_type = "chatterbox_t3_finetune"

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        hf_config_instance = HFCompatibleConfig()
        hf_config_instance.llama_config_name = chatterbox_t3_config.llama_config_name
        hf_config_instance.text_tokens_dict_size = chatterbox_t3_config.text_tokens_dict_size
        hf_config_instance.speech_tokens_dict_size = chatterbox_t3_config.speech_tokens_dict_size
        hf_config_instance.max_text_tokens = chatterbox_t3_config.max_text_tokens
        hf_config_instance.max_speech_tokens = chatterbox_t3_config.max_speech_tokens
        hf_config_instance.speech_cond_prompt_len = chatterbox_t3_config.speech_cond_prompt_len
        hf_config_instance.start_text_token = chatterbox_t3_config.start_text_token
        hf_config_instance.stop_text_token = chatterbox_t3_config.stop_text_token
        hf_config_instance.start_speech_token = chatterbox_t3_config.start_speech_token
        hf_config_instance.stop_speech_token = chatterbox_t3_config.stop_speech_token
        self.config = hf_config_instance

    def forward(
        self,
        text_tokens,
        text_token_lens,
        speech_tokens,
        speech_token_lens,
        t3_cond_speaker_emb,
        t3_cond_prompt_speech_tokens,
        t3_cond_emotion_adv,
        labels_text=None,
        labels_speech=None,
    ):
        return self.t3(
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            t3_cond_speaker_emb=t3_cond_speaker_emb,
            t3_cond_prompt_speech_tokens=t3_cond_prompt_speech_tokens,
            t3_cond_emotion_adv=t3_cond_emotion_adv,
            labels_text=labels_text,
            labels_speech=labels_speech,
        )


# --- Main function ---
def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    # --- Load HF dataset ---
    if data_args.dataset_name:
        hf_dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        train_hf_dataset = hf_dataset[data_args.train_split_name]
        eval_hf_dataset = (
            hf_dataset[data_args.eval_split_name]
            if data_args.eval_split_name in hf_dataset
            else None
        )

        # ======= FIX: Cast audio column to proper numpy arrays =======
        if data_args.audio_column_name in train_hf_dataset.column_names:
            train_hf_dataset = train_hf_dataset.cast_column(
                data_args.audio_column_name, Audio(sampling_rate=S3_SR)
            )
        if eval_hf_dataset and data_args.audio_column_name in eval_hf_dataset.column_names:
            eval_hf_dataset = eval_hf_dataset.cast_column(
                data_args.audio_column_name, Audio(sampling_rate=S3_SR)
            )
    else:
        raise ValueError("Please provide a HF dataset name or local dataset dir.")

    # --- Load Chatterbox model ---
    

    
    chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    t3_model = chatterbox_model.t3
    t3_config = chatterbox_model.t3
   
    # --- Build datasets ---
    train_dataset = SpeechFineTuningDataset(
        data_args, chatterbox_model, t3_config, train_hf_dataset, is_hf_format=True
    )
    eval_dataset = (
        SpeechFineTuningDataset(
            data_args, chatterbox_model, t3_config, eval_hf_dataset, is_hf_format=True
        )
        if eval_hf_dataset
        else None
    )

    data_collator = SpeechDataCollator(
        t3_config=t3_config,
        text_pad_token_id=t3_config.start_text_token,
        speech_pad_token_id=t3_config.start_speech_token,
    )

    model_for_training = T3ForFineTuning(t3_model, t3_config)

    trainer = Trainer(
        model=model_for_training,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)]
        if training_args.early_stopping_patience
        else None,
    )

    trainer.train()


if __name__ == "__main__":
    main()
