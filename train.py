import os
import sys
from dataclasses import dataclass
from typing import List, cast

from olmo_core.config import Config, DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyDatasetConfig,
    NumpyDatasetType,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_world_size
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    CometCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    SpeedMonitorCallback,
    WandBCallback,
)
from olmo_core.train.common import LoadStrategy
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import LogFilterType, seed_all


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 47


def build_config(
    run_name: str, model: str, tokenizer: str, overrides: List[str]
) -> ExperimentConfig:

    tokenizer_config = get_tokenizer_config(tokenizer)()

    model_config = get_model_config(model)(
        vocab_size=tokenizer_config.padded_vocab_size(),
        use_flash=True,
    )
    dataset_config = NumpyDatasetConfig.glob(
        "...",
        tokenizer=tokenizer_config,
        sequence_length=4096,
        max_target_sequence_length=max(8192, 4096),
        work_dir="...",
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=12 * 2 * get_world_size() * 4096,
        seed=34521,
        num_workers=0,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=12 * 4096,
        max_sequence_length=dataset_config.effective_sequence_length,
        optim=SkipStepAdamWConfig(
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"], opts=dict(weight_decay=0.0)
                )
            ],
            compile=False,
        ),
        scheduler=CosWithWarmup(warmup_steps=2000),
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            num_replicas=get_world_size(),  # NOTE: tune this
        ),
        ac_config=TransformerActivationCheckpointingConfig(
            TransformerActivationCheckpointingMode.full
        ),
        z_loss_multiplier=0,
        max_grad_norm=1.0,
    )

    # If you have 1024 GPUs, you can run slightly faster with a different config.
    if get_world_size() >= 1024:
        train_module_config.rank_microbatch_size //= 2
        train_module_config.ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        )

    trainer_config = (
        TrainerConfig(
            save_folder=f"{"..."}/{run_name}",
            save_overwrite=True,
            metrics_collect_interval=1,
            cancel_check_interval=10,
            load_path=None,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,
                ephemeral_save_interval=500,
                save_async=True,
            ),
        )
        .with_callback(
            "comet",
            CometCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=False,  # NOTE: change to true to enable
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                cancel_check_interval=10,
                enabled=True,  # NOTE: change to true to enable
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "speed_monitor",
            SpeedMonitorCallback(
                device_peak_flops=int(
                    989e12
                )  # Need to provide this, otherwise the pre_train does not set it correctly
            ),
        )
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyDatasetConfig.glob(
                    "...",
                    tokenizer=tokenizer_config,
                    sequence_length=4096,
                    name=NumpyDatasetType.padded_fsl,
                    work_dir="...",
                ),
                eval_interval=1000,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=[
                    # MMLU for backwards compatibility
                    "mmlu_stem_mc_5shot",
                    "mmlu_humanities_mc_5shot",
                    "mmlu_social_sciences_mc_5shot",
                    "mmlu_other_mc_5shot",
                    # MMLU test
                    "mmlu_stem_mc_5shot_test",
                    "mmlu_humanities_mc_5shot_test",
                    "mmlu_social_sciences_mc_5shot_test",
                    "mmlu_other_mc_5shot_test",
                    ## Core 12 tasks for backwards compatibility
                    # "arc_challenge",
                    # "arc_easy",
                    # "basic_arithmetic",
                    # "boolq",
                    # "commonsense_qa",
                    # "copa",
                    # "hellaswag",
                    # "openbook_qa",
                    # "piqa",
                    # "sciq",
                    # "social_iqa",
                    # "winogrande",
                    ## Core 12 tasks 5-shot
                    # "arc_challenge_rc_5shot",
                    # "arc_easy_rc_5shot",
                    ## "basic_arithmetic_rc_5shot",  # doesn't exist
                    ## "boolq_rc_5shot",  # we don't like it
                    # "csqa_rc_5shot",
                    ## "copa_rc_5shot",  # doesn't exist
                    # "hellaswag_rc_5shot",
                    # "openbookqa_rc_5shot",
                    # "piqa_rc_5shot",
                    ## "sciq_rc_5shot",  # doesn't exist
                    # "socialiqa_rc_5shot",
                    # "winogrande_rc_5shot",
                    ## New in-loop evals
                    # "arc_challenge_val_rc_5shot",
                    # "arc_challenge_val_mc_5shot",
                    "arc_challenge_test_rc_5shot",
                    # "arc_challenge_test_mc_5shot",
                    # "arc_easy_val_rc_5shot",
                    # "arc_easy_val_mc_5shot",
                    "arc_easy_test_rc_5shot",
                    # "arc_easy_test_mc_5shot",
                    # "boolq_val_rc_5shot",
                    # "boolq_val_mc_5shot",
                    "csqa_val_rc_5shot",
                    # "csqa_val_mc_5shot",
                    "hellaswag_val_rc_5shot",
                    # "hellaswag_val_mc_5shot",
                    # "openbookqa_val_rc_5shot",
                    # "openbookqa_val_mc_5shot",
                    "openbookqa_test_rc_5shot",
                    # "openbookqa_test_mc_5shot",
                    "piqa_val_rc_5shot",
                    # "piqa_val_mc_5shot",
                    "socialiqa_val_rc_5shot",
                    # "socialiqa_val_mc_5shot",
                    # "winogrande_val_rc_5shot",
                    # "winogrande_val_mc_5shot",
                    # "mmlu_stem_val_rc_5shot",
                    # "mmlu_stem_val_mc_5shot",
                    # "mmlu_humanities_val_rc_5shot",
                    # "mmlu_humanities_val_mc_5shot",
                    # "mmlu_social_sciences_val_rc_5shot",
                    # "mmlu_social_sciences_val_mc_5shot",
                    # "mmlu_other_val_rc_5shot",
                    # "mmlu_other_val_mc_5shot",
                ],
                tokenizer=tokenizer_config,
                eval_interval=1000,
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


def get_model_config(model_name: str):
    method_name = model_name.replace("-", "_")

    if not hasattr(TransformerConfig, method_name):
        raise ValueError(f"Model config '{model_name}' not found in TransformerConfig.")

    return getattr(TransformerConfig, method_name)


def get_tokenizer_config(tokenizer_name: str):
    method_name = tokenizer_name.replace("-", "_")

    if hasattr(TokenizerConfig, method_name):
        return getattr(TokenizerConfig, method_name)

    def fallback():
        try:
            return TokenizerConfig.from_hf(tokenizer_name)
        except Exception as e:
            raise ValueError(
                f"Tokenizer '{tokenizer_name}' not found in TokenizerConfig and failed to load from Hugging Face."
            ) from e

    return fallback


def main(run_name: str, model: str, tokenizer: str, overrides: List[str]):
    config = build_config(run_name, model, tokenizer, overrides)
    # Set RNG states on all devices.
    seed_all(config.init_seed)

    # Build components.
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(
        dataset, dp_process_group=train_module.dp_process_group
    )
    trainer = config.trainer.build(train_module, data_loader)

    # Save config to W&B and each checkpoint dir.
    config_dict = config.as_config_dict()
    cast(CometCallback, trainer.callbacks["comet"]).config = config_dict
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    # Train.
    trainer.fit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: torchrun [OPTS..] {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, model, tokenizer, *overrides = sys.argv[1:]

    prepare_training_environment(log_filter_type=LogFilterType.rank0_only)
    try:
        main(run_name, model, tokenizer, overrides=overrides)
    finally:
        teardown_training_environment()
