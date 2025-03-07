
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    

    output_dir: Optional[str] = field(
            default=None,
            metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
        )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    library_path: Optional[str] = field(
        default="skills.json",
        metadata={"help": "The path to the SQL library file."},
    )


@dataclass
class TrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    test: Optional[int] = field(
        default=5,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )


def __post_init__(self):

    if self.dataset_name is None and self.train_file is None and self.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    if self.val_max_target_length is None:
        self.val_max_target_length = self.max_target_length
