import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import Dataset
from transformers import (
    HfArgumentParser,
    set_seed,
    Trainer,
    EarlyStoppingCallback,
)
from safetensors.torch import save_file

from src.models.t3 import T3
from src.models.t3.modules.t3_config import T3Config
from src.models.s3tokenizer import S3_SR
from src.chatterbox_multilingual import ChatterboxMultilingualTTS
from src.dataset import SpeechFineTuningDataset, SpeechDataCollator
from src.t3_for_finetuning import T3ForFineTuning

logger = logging.getLogger(__name__)
REPO_ID = "rahul7star/chatterbox-v2-exp/pretrained_model_download"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = None
    local_model_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    freeze_voice_encoder: bool = False
    freeze_s3gen: bool = False


@dataclass
class DataArguments:
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    train_split_name: str = "train"
    eval_split_name: Optional[str] = None
    eval_split_size: float = 0.1
    dataset_dir: Optional[str] = None
    metadata_file: Optional[str] = None
    ignore_verifications: bool = False


@dataclass
class CustomTrainingArguments:
    output_dir: str = "./output"
    do_train: bool = True
    do_eval: bool = True
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    early_stopping_patience: Optional[int] = None
    local_rank: int = -1
    label_names: Optional[List[str]] = None


def decode_audio(example):
    """Ensure HF audio columns are converted to numpy arrays"""
    audio = example["audio"]
    if isinstance(audio, dict) and "array" in audio:
        example["audio"] = audio["array"]
    elif hasattr(audio, "array"):
        example["audio"] = audio.array
    return example


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)

    # Load Chatterbox model
    logger.info("Loading ChatterboxTTS model...")
    if model_args.local_model_dir:
        local_dir_path = Path(model_args.local_model_dir)
        chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        original_model_dir_for_copy = local_dir_path
    else:
        repo_to_download = model_args.model_name_or_path or REPO_ID
        logger.info(f"Loading model from Hugging Face Hub: {repo_to_download}")
        download_dir = Path(training_args.output_dir) / "pretrained_model_download"
        download_dir.mkdir(parents=True, exist_ok=True)
        files_to_download = [
            "ve.safetensors",
            "t3_mtl23ls_v2.safetensors",
            "s3gen.safetensors",
            "mtl_tokenizer.json",
        ]
        from huggingface_hub import hf_hub_download as hf_download
        for f in files_to_download:
            try:
                hf_download(
                    repo_id=repo_to_download,
                    filename=f,
                    local_dir=download_dir,
                    local_dir_use_symlinks=False,
                    cache_dir=model_args.cache_dir,
                )
            except Exception as e:
                logger.warning(f"Could not download {f}: {e}.")
        try:
            hf_download(
                repo_id=repo_to_download,
                filename="conds.pt",
                local_dir=download_dir,
                local_dir_use_symlinks=False,
                cache_dir=model_args.cache_dir,
            )
        except:
            logger.info("conds.pt not found or failed to download.")

        chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
        original_model_dir_for_copy = download_dir

    # Freeze models if requested
    t3_model = chatterbox_model.t3
    t3_config_instance = t3_model.hp
    if model_args.freeze_voice_encoder:
        for p in chatterbox_model.ve.parameters():
            p.requires_grad = False
    if model_args.freeze_s3gen:
        for p in chatterbox_model.s3gen.parameters():
            p.requires_grad = False
    for p in t3_model.parameters():
        p.requires_grad = True

    # Load datasets
    raw_train, raw_eval = None, None
    if data_args.dataset_name:
        dataset_loaded = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        raw_train = dataset_loaded[data_args.train_split_name]
        if training_args.do_eval:
            if data_args.eval_split_name and data_args.eval_split_name in dataset_loaded:
                raw_eval = dataset_loaded[data_args.eval_split_name]
            elif "validation" in dataset_loaded:
                raw_eval = dataset_loaded["validation"]
            elif "test" in dataset_loaded:
                raw_eval = dataset_loaded["test"]
            elif data_args.eval_split_size > 0:
                split = raw_train.train_test_split(test_size=data_args.eval_split_size)
                raw_train, raw_eval = split["train"], split["test"]

        # Decode audio to numpy
        raw_train = raw_train.map(decode_audio)
        if raw_eval:
            raw_eval = raw_eval.map(decode_audio)
    else:
        # Local files
        all_files = []
        if data_args.metadata_file:
            metadata_path = Path(data_args.metadata_file)
            dataset_root = metadata_path.parent
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) != 2:
                        parts = line.strip().split("\t")
                    if len(parts) == 2:
                        audio_file, text = parts
                        audio_path = Path(audio_file)
                        if not audio_path.is_absolute():
                            audio_path = dataset_root / audio_file
                        if audio_path.exists():
                            all_files.append({"audio": str(audio_path), "text": text})
        elif data_args.dataset_dir:
            dataset_path = Path(data_args.dataset_dir)
            for audio_file_path in dataset_path.rglob("*.wav"):
                text_file_path = audio_file_path.with_suffix(".txt")
                if text_file_path.exists():
                    with open(text_file_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    all_files.append({"audio": str(audio_file_path), "text": text})
        if not all_files:
            raise ValueError("No data found in local dataset.")
        np.random.shuffle(all_files)
        raw_train = all_files
        if training_args.do_eval and data_args.eval_split_size > 0:
            split_idx = int(len(all_files) * (1 - data_args.eval_split_size))
            raw_train, raw_eval = all_files[:split_idx], all_files[split_idx:]

    # Create datasets for Trainer
    train_dataset = SpeechFineTuningDataset(data_args, chatterbox_model, t3_config_instance, raw_train, isinstance(raw_train, Dataset))
    eval_dataset = None
    if raw_eval and training_args.do_eval:
        eval_dataset = SpeechFineTuningDataset(data_args, chatterbox_model, t3_config_instance, raw_eval, isinstance(raw_eval, Dataset))

    # Data collator
    data_collator = SpeechDataCollator(
        t3_config_instance,
        t3_config_instance.stop_text_token,
        t3_config_instance.stop_speech_token,
    )

    # Model for fine-tuning
    hf_trainable_model = T3ForFineTuning(t3_model, t3_config_instance)

    # Trainer
    callbacks = []
    if training_args.early_stopping_patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))
    trainer_instance = Trainer(
        model=hf_trainable_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
    )

    # Training
    if training_args.do_train:
        logger.info("*** Training T3 model ***")
        train_result = trainer_instance.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer_instance.save_model()

        # Save finetuned weights
        t3_to_save = trainer_instance.model.t3 if hasattr(trainer_instance.model, "t3") else trainer_instance.model.module.t3
        finetuned_t3_state_dict = t3_to_save.state_dict()
        save_file(finetuned_t3_state_dict, Path(training_args.output_dir) / "t3_mtl23ls_v2.safetensors")
        logger.info(f"Finetuned T3 saved to {training_args.output_dir}/t3_mtl23ls_v2.safetensors")

        # Copy other components
        if original_model_dir_for_copy:
            import shutil
            for fname in ["ve.safetensors", "s3gen.safetensors", "mtl_tokenizer.json"]:
                src = original_model_dir_for_copy / fname
                if src.exists():
                    shutil.copy2(src, Path(training_args.output_dir) / fname)
            conds_file = original_model_dir_for_copy / "conds.pt"
            if conds_file.exists():
                shutil.copy2(conds_file, Path(training_args.output_dir) / "conds.pt")
            logger.info("All model components saved.")

    # Evaluation
    if training_args.do_eval and eval_dataset:
        logger.info("*** Evaluating T3 model ***")
        metrics = trainer_instance.evaluate()
        trainer_instance.log_metrics("eval", metrics)
        trainer_instance.save_metrics("eval", metrics)

    logger.info("Finetuning script finished.")


if __name__ == "__main__":
    main()
