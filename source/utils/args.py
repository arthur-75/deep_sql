
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

    curriculum_model: Optional[str] = field(
        default="llama3.2",
        metadata={"help": "The model checkpoint for the curriculum learning agent."},
    )

    iterative_model: Optional[str] = field(
        default="llama3.2",
        metadata={"help": "The model checkpoint for the iterative prompting agent."},
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

    database_path: Optional[str] = field(
        default='../data/tables/db',
        metadata={"help": "The path to the SQLite database file."},
    )


@dataclass
class TrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    num_iterations: Optional[int] = field(
        default=1,
        metadata={"help": "The number of iterations for the exploration loop."},
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
