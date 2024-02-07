from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class WhisperModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the entire encoder of the seq2seq model."}
    )
    freeze_decoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the entire decoder of the seq2seq model."}
    )
    forced_decoder_ids: List[List[int]] = field(
        default=None,
        metadata={
            "help": (
                "A list of pairs of integers which indicates a mapping from generation indices to token indices "
                "that will be forced before sampling. For example, [[0, 123]] means the first generated token "
                "will always be a token of index 123."
            )
        },
    )
    suppress_tokens: List[int] = field(
        default=None, metadata={"help": "A list of tokens that will be suppressed at generation."}
    )
    apply_spec_augment: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply *SpecAugment* data augmentation to the input features. This is currently only relevant for Wav2Vec2, HuBERT, WavLM and Whisper models."
        },
    )
    patience: int = field(
        default=5,
        metadata={
            "help": "Early stop."
        },
    )
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_path: dict = field(
        default=None, metadata={"help": "Paths to the training datasets"}
    )
    dev_data_path: dict = field(
        default=None, metadata={"help": "Paths to the development datasets."}
    )
    streaming: bool = field(
        default=False, metadata={"help": "Map-style or iterable datasets."}
    )
    
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data. Defaults to 'text'"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Truncate audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    language: str = field(
        default=None,
        metadata={
            "help": (
                "Language for multilingual fine-tuning. This argument should be set for multilingual fine-tuning "
                "only. For English speech recognition, it should be set to `None`."
            )
        },
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "Task, either `transcribe` for speech recognition or `translate` for speech translation."},
    )
    use_vtlp: bool = field(
        default=False, metadata={"help": "use vtlp"}
    )
    vtlp_low: float = field(
        default=0.9, metadata={"help": "use vtlp"}
    )
    vtlp_high: float = field(
        default=1.1, metadata={"help": "use vtlp"}
    )
    use_speed_perturb: bool = field(
        default="False", metadata={"help": "use speed perturbation, 0.9 and 1.1"}
    )
    sp_low: float = field(
        default=0.9, metadata={"help": "use vtlp"}
    )
    sp_high: float = field(
        default=1.1, metadata={"help": "use vtlp"}
    )
    use_pitch_perturb: bool = field(
        default="False", metadata={"help": "use pitch perturbation"}
    )
    pitch_level: float = field(
        default=12, metadata={"help": "12 steps for an octave"}
    )

    use_pif: bool = field(
        default=False, metadata={"help": "use perturb inviarant finetuning"}
    )
    pif_loss_alpha: float = field(
        default=0.1, metadata={"help": "mse loss ratio for pif"}
    )

@dataclass
class PEFTArguments:
    """
    Arguments pertaining to parameter efficient training, only support lora and residual adapter.
    """

    peft_type: str = field(
        default=None, metadata={"help": "peft type, either lora or adapter"}
    )
    lora_dim: int = field(
        default=8, metadata={"help": "The rank for LoRA."}
    )
    lora_alpha: int = field(
        default=128, metadata={"help": "Alpha for scaling factor."}
    )
    dropout: float = field(
        default=0.1, metadata={"help": "Dropout rate in Lora or adapter layers."}
    )
    bottleneck_dim: dict = field(
        default=8, metadata={"help": "bottleneck dimension in residual adapters."}
    )
    peft_encoder_layers: List[int] = field(
        default=None, metadata={"help": "layers to use peft in encoder, 0 is the conv layers 1->n are encoder layers"}
    )
    peft_decoder_layers: List[int] = field(
        default=None, metadata={"help": "layers to use peft in decoder,  1->n are decoder layers"}
    )
    to_encoder: bool = field(
        default=True, metadata={"help": "use peft to encoder."}
    )
    to_decoder: bool = field(
        default=False, metadata={"help": "use peft to decoder."}
    )
    
