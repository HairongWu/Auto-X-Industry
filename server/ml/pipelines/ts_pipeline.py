import pandas as pd  # requires: pip install pandas
import torch
import ast
import random
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import List, Iterator, Optional, Dict

from torch.utils.data import IterableDataset, get_worker_info
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    T5Config,
    Trainer,
    TrainingArguments,
)
import accelerate
import gluonts
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Cyclic, Map, Filter
from gluonts.transform import (
    FilterTransformation,
    TestSplitSampler,
    ValidationSplitSampler,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    MissingValueImputation,
    LeavesMissingValues,
    LastValueImputation,
)

from chronos import ChronosConfig, ChronosPipeline

def is_main_process() -> bool:
    """
    Check if we're on the main process.
    """
    if not dist.is_torchelastic_launched():
        return True
    return int(os.environ["RANK"]) == 0

def load_model(
    model_id="google/t5-efficient-tiny",
    model_type="seq2seq",
    vocab_size=4096,
    random_init=False,
    tie_embeddings=False,
    pad_token_id=0,
    eos_token_id=1,
):
    """
    Load the specified HuggingFace model, adjusting the vocabulary
    size, special token IDs, and initialization options.

    This allows to set a model up for training on a new vocabulary
    of tokens.
    """
    assert model_type in ["seq2seq", "causal"]
    AutoModelClass = (
        AutoModelForSeq2SeqLM if model_type == "seq2seq" else AutoModelForCausalLM
    )
    if random_init:
        log_on_main("Using random initialization", logger)
        config = AutoConfig.from_pretrained(model_id)
        if isinstance(config, T5Config):
            # The default initializer_factor (1.0) in transformers is too large
            config.initializer_factor = 0.05
        config.tie_word_embeddings = tie_embeddings
        model = AutoModelClass.from_config(config)
    else:
        log_on_main(f"Using pretrained initialization from {model_id}", logger)
        model = AutoModelClass.from_pretrained(model_id)

    model.resize_token_embeddings(vocab_size)

    model.config.pad_token_id = model.generation_config.pad_token_id = pad_token_id
    model.config.eos_token_id = model.generation_config.eos_token_id = eos_token_id

    return model


def has_enough_observations(
    entry: dict, min_length: int = 0, max_missing_prop: float = 1.0
) -> bool:
    """
    Check if the given entry has enough observations in the ``"target"`` attribute.

    Parameters
    ----------
    entry
        The data entry (dictionary) to be tested.
    min_length
        The minimum length the ``"target"`` attribute must have.
    max_missing_prop
        The maximum proportion of missing data allowed in the ``"target"``
        attribute.
    """
    if (
        len(entry["target"]) >= min_length
        and np.isnan(entry["target"]).mean() <= max_missing_prop
    ):
        return True
    return False


class PseudoShuffledIterableDataset(IterableDataset):
    """
    Shuffle entries from an iterable by temporarily accumulating them
    in an intermediate buffer.

    Parameters
    ----------
    base_dataset
        The original iterable object, representing the dataset.
    shuffle_buffer_length
        Size of the buffer use to shuffle entries from the base dataset.
    """

    def __init__(self, base_dataset, shuffle_buffer_length: int = 100) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                )
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    """
    Mix-in class that datasets can inherit from to get
    shuffling functionality.
    """

    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)


class ChronosDataset(IterableDataset, ShuffleMixin):
    """
    Dataset wrapper, using a ``ChronosTokenizer`` to turn data from a time series
    into a HuggingFace-compatible set of ``input_ids``, ``attention_mask`` and
    ``labels``.

    Entries from the original datasets are assumed to have a ``"start"`` attribute
    (of type ``pd.Period``), and a ``"target"`` attribute (of type ``np.ndarray``).

    Parameters
    ----------
    datasets
        Datasets containing the original time series data.
    probabilities
        In training mode, data will be sampled from each of the original datasets
        with these probabilities.
    tokenizer
        Tokenizer to be used to turn sequences of real numbers into token IDs.
    context_length
        Samples context will be limited to this length.
    prediction_length
        Samples labels will be limited to this length.
    drop_prob
        In training mode, observations from a sample will be turned into ``np.nan``,
        i.e. turned into missing values, with this probability.
    min_past
        Data samples will be considered only if there's at least ``min_past``-many
        historical observations.
    mode
        One of ``"training"``, ``"validation"``, or ``"test"``.
    np_dtype
        Numpy float data type.
    """

    def __init__(
        self,
        datasets: list,
        probabilities: List[float],
        tokenizer: ChronosTokenizer,
        context_length: int = 512,
        prediction_length: int = 64,
        drop_prob: float = 0.2,
        min_past: Optional[int] = None,
        model_type: str = "seq2seq",
        imputation_method: Optional[MissingValueImputation] = None,
        mode: str = "training",
        np_dtype=np.float32,
    ) -> None:
        super().__init__()

        assert len(probabilities) == len(datasets)
        assert mode in ("training", "validation", "test")
        assert model_type in ("seq2seq", "causal")

        self.datasets = datasets
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.drop_prob = drop_prob if model_type == "seq2seq" else 0.0
        self.min_past = min_past or prediction_length
        self.model_type = model_type
        self.imputation_method = imputation_method or LeavesMissingValues()
        self.mode = mode
        self.np_dtype = np_dtype

    def preprocess_entry(self, entry: dict, mode: str) -> dict:
        entry = {f: entry[f] for f in ["start", "target"]}
        entry["target"] = np.asarray(entry["target"], dtype=self.np_dtype)
        assert entry["target"].ndim == 1, f"got {entry['target'].ndim=}, expected 1"

        if self.model_type == "causal":
            # Causal models do not play nice with missing values, so it is
            # recommended to use an imputation method, e.g., LastValueImputation
            entry["target"] = self.imputation_method(entry["target"])

        if mode == "training" and self.drop_prob > 0:
            target = entry["target"].copy()
            drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
            mask = np.random.choice(
                [True, False], size=len(target), p=[drop_p, 1 - drop_p]
            )
            target[mask] = np.nan
            entry["target"] = target

        return entry

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "test", "validation"]

        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.min_past,
                min_future=self.prediction_length,
            ),
            "test": TestSplitSampler(),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]

        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
        )

    def create_training_data(self, data):
        data = Cyclic(data)
        split_transform = self._create_instance_splitter(
            "training"
        ) + FilterTransformation(
            condition=lambda entry: (~np.isnan(entry["past_target"])).sum() > 0
        )
        data = split_transform.apply(data, is_train=True)
        return data

    def create_test_data(self, data):
        data = self._create_instance_splitter("test").apply(data, is_train=False)
        return data

    def create_validation_data(self, data):
        data = self._create_instance_splitter("validation").apply(data, is_train=False)
        return data

    def to_hf_format(self, entry: dict) -> dict:
        past_target = torch.tensor(entry["past_target"]).unsqueeze(0)
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(
            past_target
        )
        future_target = torch.tensor(entry["future_target"]).unsqueeze(0)
        labels, labels_mask = self.tokenizer.label_input_transform(future_target, scale)
        labels[labels_mask == 0] = -100

        if self.model_type == "causal":
            # The InstanceSplitter pads time series on the left to be equal to the
            # context_length. However, certain models (e.g., GPT2) with absolute
            # position embeddings should not be trained with left padding.
            # The following piece of code moves padding from left to right.

            assert input_ids.shape[-1] == entry["past_is_pad"].shape[0]

            # Find the index where padding starts
            pad_start_idx = np.searchsorted(1 - entry["past_is_pad"], 1)
            padded_input_ids, obs_input_ids = torch.tensor_split(
                input_ids, [pad_start_idx], dim=-1
            )
            padded_attention_mask, obs_attention_mask = torch.tensor_split(
                attention_mask, [pad_start_idx], dim=-1
            )

            # Move padding to the right
            input_ids = torch.cat(
                [
                    obs_input_ids,
                    labels,
                    padded_input_ids,
                ],
                axis=-1,
            )
            attention_mask = torch.cat(
                [
                    obs_attention_mask,
                    labels_mask,
                    padded_attention_mask,
                ],
                axis=-1,
            )

            # labels for causal models are same as the input_ids.
            # Internally transformers shifts the labels by one during training.
            labels = input_ids.clone()
            input_ids[~attention_mask] = self.tokenizer.config.pad_token_id
            labels[~attention_mask] = -100

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    def __iter__(self) -> Iterator:
        preprocessed_datasets = [
            Map(
                partial(self.preprocess_entry, mode=self.mode),
                dataset,
            )
            for dataset in self.datasets
        ]

        if self.mode == "training":
            iterables = [
                self.create_training_data(dataset) for dataset in preprocessed_datasets
            ]
        elif self.mode == "test":
            iterables = [
                self.create_test_data(dataset) for dataset in preprocessed_datasets
            ]
        else:
            iterables = [
                self.create_validation_data(dataset)
                for dataset in preprocessed_datasets
            ]

        worker_info = get_worker_info()
        if worker_info is None:
            probs = list(self.probabilities)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iterables = list(itertools.islice(iterables, worker_id, None, num_workers))
            probs = list(
                itertools.islice(self.probabilities, worker_id, None, num_workers)
            )

        probs = [prob / sum(probs) for prob in probs]

        iterators = list(map(iter, iterables))
        if self.mode == "training":
            while True:
                idx = np.random.choice(range(len(iterators)), p=probs)
                try:
                    yield self.to_hf_format(next(iterators[idx]))
                except StopIteration:
                    probs[idx] = 0
                    if sum(probs) == 0:
                        return
                    probs = [prob / sum(probs) for prob in probs]
        else:
            for entry in itertools.chain(*iterators):
                yield self.to_hf_format(entry)

class TimeSeriesPipeline(): 
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-large",
            device_map=self.device,  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )

    def predict_csv(self, csv_files, col=''):
        forecasts = []
        for c in csv_files:
            df = pd.read_csv(c)

            # context must be either a 1D tensor, a list of 1D tensors,
            # or a left-padded 2D tensor with batch as the first dimension
            # forecast shape: [num_series, num_samples, prediction_length]
            forecast = self.pipeline.predict(
                context=torch.tensor(df[col]),
                prediction_length=12,
                num_samples=20,
            )
            forecasts.append(forecast)

        return forecasts
    
    def train(self, data):
        args = AttrDict()
        args['model_type'] = "seq2seq"
        args['output_dir'] = os.environ.get("MODEL_POOL_DIR") + "time_series"

        seed = random.randint(0, 2**32)

        transformers.set_seed(seed=seed)

        raw_training_config = deepcopy(locals())
        output_dir = Path(output_dir)
        training_data_paths = ast.literal_eval(training_data_paths)
        assert isinstance(training_data_paths, list)

        if isinstance(probability, str):
            probability = ast.literal_eval(probability)
        elif probability is None:
            probability = [1.0 / len(training_data_paths)] * len(training_data_paths)
        assert isinstance(probability, list)

        if isinstance(tokenizer_kwargs, str):
            tokenizer_kwargs = ast.literal_eval(tokenizer_kwargs)
        assert isinstance(tokenizer_kwargs, dict)

        assert args['model_type'] in ["seq2seq", "causal"]

        train_datasets = [
            Filter(
                partial(
                    has_enough_observations,
                    min_length=min_past + prediction_length,
                    max_missing_prop=max_missing_prop,
                ),
                FileDataset(path=Path(data_path), freq="h"),
            )
            for data_path in training_data_paths
        ]

        model = load_model(
            model_id=model_id,
            model_type=model_type,
            vocab_size=n_tokens,
            random_init=random_init,
            tie_embeddings=tie_embeddings,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

        chronos_config = ChronosConfig(
            tokenizer_class=tokenizer_class,
            tokenizer_kwargs=tokenizer_kwargs,
            n_tokens=n_tokens,
            n_special_tokens=n_special_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            use_eos_token=use_eos_token,
            model_type=model_type,
            context_length=context_length,
            prediction_length=prediction_length,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Add extra items to model config so that it's saved in the ckpt
        model.config.chronos_config = chronos_config.__dict__

        shuffled_train_dataset = ChronosDataset(
            datasets=train_datasets,
            probabilities=probability,
            tokenizer=chronos_config.create_tokenizer(),
            context_length=context_length,
            prediction_length=prediction_length,
            min_past=min_past,
            model_type=model_type,
            imputation_method=LastValueImputation() if model_type == "causal" else None,
            mode="training",
        ).shuffle(shuffle_buffer_length=shuffle_buffer_length)

        # Define training args
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            optim=optim,
            logging_dir=str(output_dir / "logs"),
            logging_strategy="steps",
            logging_steps=log_steps,
            save_strategy="steps",
            save_steps=save_steps,
            report_to=["tensorboard"],
            max_steps=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            tf32=tf32,  # remove this if not using Ampere GPUs (e.g., A100)
            torch_compile=torch_compile,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False,
        )

        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=shuffled_train_dataset,
        )

        trainer.train()

        if is_main_process():
            model.save_pretrained(output_dir / "checkpoint-final")
            save_training_info(
                output_dir / "checkpoint-final", training_config=raw_training_config
            )

