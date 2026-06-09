from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from datasets import Dataset, DatasetDict, load_dataset

from .config import PipelineConfig


PromptBuilder = Callable[[dict, str, str], str]
TextNormalizer = Callable[[str], str]


@dataclass(frozen=True)
class DatasetSpec:
    alias: str
    task: str
    hub_name: str
    supported_languages: dict[str, str | None]
    source_column: str
    target_column: str
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    description: str = ""
    prompt_builder: PromptBuilder | None = None
    source_normalizer: TextNormalizer | None = None
    target_normalizer: TextNormalizer | None = None

    def resolve_config_name(self, language: str, override: str | None = None) -> str | None:
        if override is not None:
            return override
        return self.supported_languages.get(language)

    def supports(self, task: str, language: str) -> bool:
        return self.task == task and language in self.supported_languages


def _strip_text(value: str) -> str:
    return str(value).strip()


def _normalize_xlcost_text(value: str) -> str:
    return str(value).replace(" | ", ". ").strip()


def _join_code_tokens(tokens: list[str]) -> str:
    text = " ".join(tokens)
    substitutions = [
        (r"\s+([,.;:!?%\)\]\}])", r"\1"),
        (r"([\(\[\{])\s+", r"\1"),
        (r"\s+\.\s+", "."),
        (r"\s*::\s*", "::"),
        (r"\s*->\s*", "->"),
    ]
    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text)
    return text.strip()


def _normalize_xlcost_code(value: str) -> str:
    tokens = str(value).strip().split()
    indent_level = 0
    current_line: list[str] = []
    lines: list[str] = []

    def flush_line() -> None:
        if not current_line:
            return
        lines.append(("    " * indent_level) + _join_code_tokens(current_line))
        current_line.clear()

    for token in tokens:
        if token == "NEW_LINE":
            flush_line()
        elif token == "INDENT":
            indent_level += 1
        elif token == "DEDENT":
            flush_line()
            indent_level = max(indent_level - 1, 0)
        else:
            current_line.append(token)

    flush_line()
    return "\n".join(line.rstrip() for line in lines).strip()


def _build_default_prompt(task: str, language: str, source_text: str) -> str:
    if task == "summarization":
        return f"Summarize the following {language} method:\n{source_text}"
    return f"Generate a {language} method from the following description:\n{source_text}"


def _build_codexglue_sum_prompt(example: dict, language: str, source_text: str) -> str:
    func_name = example.get("func_name")
    header = f"Summarize the following {language} method."
    if func_name:
        header += f"\nFunction name: {func_name}"
    return f"{header}\nCode:\n{source_text}"


def _build_codexglue_generation_prompt(example: dict, language: str, source_text: str) -> str:
    func_name = example.get("func_name")
    header = f"Generate a {language} method from this docstring."
    if func_name:
        header += f"\nExpected function name: {func_name}"
    return f"{header}\nDocstring:\n{source_text}"


def _build_concode_prompt(_: dict, language: str, source_text: str) -> str:
    return (
        f"Generate a {language} class member method using the following natural language description and class context.\n"
        f"Description and context:\n{source_text}"
    )


def _build_xlcost_prompt(_: dict, language: str, source_text: str) -> str:
    return f"Generate a {language} method or code snippet from the following description:\n{source_text}"


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "codexglue_code_to_text": DatasetSpec(
        alias="codexglue_code_to_text",
        task="summarization",
        hub_name="google/code_x_glue_ct_code_to_text",
        supported_languages={"java": "java", "python": "python"},
        source_column="code",
        target_column="docstring",
        description="CodeXGLUE code-to-text benchmark derived from CodeSearchNet.",
        prompt_builder=_build_codexglue_sum_prompt,
        source_normalizer=_strip_text,
        target_normalizer=_strip_text,
    ),
    "codexglue_docstring_to_code": DatasetSpec(
        alias="codexglue_docstring_to_code",
        task="generation",
        hub_name="google/code_x_glue_ct_code_to_text",
        supported_languages={"java": "java", "python": "python"},
        source_column="docstring",
        target_column="code",
        description="Reversed CodeXGLUE code-to-text for docstring-to-method generation.",
        prompt_builder=_build_codexglue_generation_prompt,
        source_normalizer=_strip_text,
        target_normalizer=_strip_text,
    ),
    "concode_java": DatasetSpec(
        alias="concode_java",
        task="generation",
        hub_name="semeru/Text-Code-concode-Java",
        supported_languages={"java": None},
        source_column="nl",
        target_column="code",
        description="Concode Java member-function generation with programmatic class context.",
        prompt_builder=_build_concode_prompt,
        source_normalizer=_strip_text,
        target_normalizer=_strip_text,
    ),
    "xlcost_text_to_code": DatasetSpec(
        alias="xlcost_text_to_code",
        task="generation",
        hub_name="codeparrot/xlcost-text-to-code",
        supported_languages={"java": "Java-snippet-level", "python": "Python-snippet-level"},
        source_column="text",
        target_column="code",
        description="XLCoST multilingual text-to-code benchmark; defaults to snippet-level subsets.",
        prompt_builder=_build_xlcost_prompt,
        source_normalizer=_normalize_xlcost_text,
        target_normalizer=_normalize_xlcost_code,
    ),
}


@dataclass
class PreparedDatasets:
    spec: DatasetSpec
    text_splits: dict[str, Dataset]


def dataset_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for spec in DATASET_REGISTRY.values():
        rows.append(
            (
                spec.alias,
                spec.task,
                ",".join(sorted(spec.supported_languages)),
                spec.description,
            )
        )
    return rows


def resolve_dataset_spec(config: PipelineConfig) -> DatasetSpec:
    if config.dataset in DATASET_REGISTRY:
        spec = DATASET_REGISTRY[config.dataset]
        if not spec.supports(config.task, config.language):
            raise ValueError(
                f"Dataset '{config.dataset}' does not support task='{config.task}' and language='{config.language}'."
            )
        return spec

    if not config.source_column or not config.target_column:
        raise ValueError(
            "Custom datasets require both --source-column and --target-column. "
            "Alternatively, choose one of the built-in dataset aliases."
        )

    return DatasetSpec(
        alias=config.dataset,
        task=config.task,
        hub_name=config.dataset,
        supported_languages={config.language: config.dataset_config},
        source_column=config.source_column,
        target_column=config.target_column,
        train_split=config.train_split or "train",
        validation_split=config.validation_split or "validation",
        test_split=config.test_split or "test",
        description="User-provided Hugging Face dataset.",
    )


def load_text_datasets(config: PipelineConfig) -> PreparedDatasets:
    spec = resolve_dataset_spec(config)
    dataset_config = spec.resolve_config_name(config.language, config.dataset_config)
    raw_datasets = load_dataset(spec.hub_name, dataset_config, cache_dir=config.cache_dir)

    if not isinstance(raw_datasets, DatasetDict):
        raise ValueError(f"Dataset '{spec.hub_name}' did not return a DatasetDict with named splits.")

    split_mapping = {
        "train": config.train_split or spec.train_split,
        "validation": config.validation_split or spec.validation_split,
        "test": config.test_split or spec.test_split,
    }
    requested_splits = {
        "train": config.do_train,
        "validation": config.do_eval,
        "test": config.do_test,
    }
    sample_limits = {
        "train": config.max_train_samples,
        "validation": config.max_eval_samples,
        "test": config.max_test_samples,
    }
    processed: dict[str, Dataset] = {}

    for split_name, required in requested_splits.items():
        if not required:
            continue
        dataset_split_name = split_mapping[split_name]
        if dataset_split_name not in raw_datasets:
            raise ValueError(
                f"Required split '{dataset_split_name}' for role '{split_name}' was not found in dataset '{spec.hub_name}'. "
                f"Available splits: {list(raw_datasets.keys())}."
            )
        split_dataset = raw_datasets[dataset_split_name]
        limit = sample_limits[split_name]
        if limit is not None:
            split_dataset = split_dataset.select(range(min(limit, len(split_dataset))))
        processed[split_name] = _map_to_text_pairs(split_dataset, spec, config)

    return PreparedDatasets(spec=spec, text_splits=processed)


def _map_to_text_pairs(dataset: Dataset, spec: DatasetSpec, config: PipelineConfig) -> Dataset:
    if spec.source_column not in dataset.column_names or spec.target_column not in dataset.column_names:
        raise ValueError(
            f"Dataset '{spec.alias}' is missing source/target columns '{spec.source_column}' and '{spec.target_column}'. "
            f"Available columns: {dataset.column_names}."
        )

    def convert_example(example: dict) -> dict:
        source_text = str(example[spec.source_column])
        target_text = str(example[spec.target_column])
        if spec.source_normalizer:
            source_text = spec.source_normalizer(source_text)
        if spec.target_normalizer:
            target_text = spec.target_normalizer(target_text)
        if spec.prompt_builder is not None:
            source_prompt = spec.prompt_builder(example, config.language, source_text)
        else:
            source_prompt = _build_default_prompt(config.task, config.language, source_text)
        if config.source_prefix:
            source_prompt = f"{config.source_prefix}{source_prompt}"
        return {"source_text": source_prompt, "target_text": target_text}

    return dataset.map(
        convert_example,
        remove_columns=dataset.column_names,
        desc=f"Formatting {spec.alias}",
    )
