import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from shutil import copytree, rmtree
from typing import cast

import torch.nn as nn

import ray
import torch
from mmengine import load
from mmengine.dist import get_rank
from mmengine.runner import set_random_seed
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator
from ray.util.placement_group import placement_group
from typing_extensions import Self

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1._writer import TensorboardWriter
from xtuner.v1.data_proto.rl_data import is_valid_for_training
from xtuner.v1.data_proto.sequence_context import SequenceContext
from xtuner.v1.patch import patch_default_save_plan
from xtuner.v1.ray.base import AcceleratorResourcesConfig, AutoAcceleratorWorkers, AutoCPUWorkers, CPUResourcesConfig
from xtuner.v1.ray.config.worker import RolloutConfig
from xtuner.v1.ray.dataflow import DataFlow, DataFlowConfig, DataFlowProxy, ReplayBufferConfig
from xtuner.v1.ray.environment import SingleTurnEnvironment, SingleTurnEnvironmentProxy
from xtuner.v1.ray.evaluator import Evaluator, EvaluatorConfig
from xtuner.v1.ray.judger import JudgerConfig
from xtuner.v1.rl.base import (
    TrainingController,
    TrainingControllerProxy,
    TrainingWorkerClass,
    TrainingWorkerProxy,
    WorkerConfig,
    WorkerLogItem,
)
from xtuner.v1.rl.base import TrainingWorker as BaseTrainingWorker
from xtuner.v1.train import ResumeConfig
from xtuner.v1.utils import XTUNER_DETERMINISTIC, get_logger, is_hf_model_path, record_git_info, timer, timer_logger
from xtuner.v1.utils.device import get_device, get_torch_device_module
from xtuner.v1.utils.env_check import get_rollout_engine_version

from .trainer import ExpHistory, ExpInfo, GitInfo, LoadCheckpointConfig, XTunerMeta


# TODO: Move DEVICE to `xtuner.utils.device`
PG_READY_TIMEOUT = 30
DEVICE = get_device()
DEVICE_MODULE = get_torch_device_module()


import numpy as np

datas = np.random.rand(10, 10)


def bind_train_rollout(
    train_controller,
    env_controller,
) -> None:
    """Bind the training and rollout workers for update weights."""
    info_dict = ray.get(env_controller.get_rollout_info.remote())  # type: ignore[attr-defined]
    ray.get(train_controller.update_rollout_info.remote(info_dict))
    return


def _flatten_scalar_metrics(prefix: str, value) -> dict[str, float]:
    items: dict[str, float] = {}

    def _visit(name: str, obj) -> None:
        if isinstance(obj, dict):
            for sub_key, sub_value in obj.items():
                _visit(f"{name}/{sub_key}", sub_value)
            return
        if isinstance(obj, (int, float)):
            items[name] = float(obj)

    _visit(prefix, value)
    return items


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default) or default).strip().lower() in ("1", "true", "yes", "y", "on")


def _resolve_verifier_save_dtype(*env_names: str) -> torch.dtype:
    dtype_name = ""
    for env_name in env_names:
        raw_value = str(os.environ.get(env_name, "") or "").strip().lower()
        if raw_value:
            dtype_name = raw_value
            break
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp8", "float8", "float8_e4m3fn"):
        return torch.float8_e4m3fn
    return torch.float32


class RLTrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    load_from: str | Path
    resources: AcceleratorResourcesConfig
    cpu_resources: CPUResourcesConfig | None = None
    rollout_config: RolloutConfig
    dataflow_config: DataFlowConfig
    judger_config: JudgerConfig
    replay_buffer_config: ReplayBufferConfig
    train_worker_config: WorkerConfig
    evaluator_config: EvaluatorConfig | None = None
    tokenizer_path: str | Path
    work_dir: Path | str | None = None
    log_dir: Path | str | None = None
    total_epochs: int
    resume_config: ResumeConfig | None = None
    auto_resume: bool = False
    load_checkpoint_cfg: LoadCheckpointConfig = LoadCheckpointConfig()
    strict_load: bool = True
    checkpoint_interval: int | None = -1
    checkpoint_maxkeep: int | None = -1
    checkpoint_no_save_optimizer: bool = False
    skip_checkpoint_validation: bool = False  # Suggest enabled if fsdp_size is larger than 512
    hf_interval: int | None = None
    hf_max_keep: int | None = None
    seed: int = 42
    debug: bool = False
    debug_rollout: bool = False
    rollout_steps: int | None = None

    @model_validator(mode="after")
    def _convert_work_dir(self):
        if isinstance(self.work_dir, str):
            self.work_dir = Path(self.work_dir)
        elif self.work_dir is None:
            self.work_dir = Path.cwd()
        return self

    @field_serializer("replay_buffer_config")
    def serialize_replay_buffer_cfg(self, replay_buffer_config: ReplayBufferConfig) -> str:
        return replay_buffer_config.model_dump(include={"replay_ratio", "replay_weights"})

    @field_serializer("evaluator_config")
    def serialize_evaluator_cfg(self, evaluator_config: EvaluatorConfig) -> str:
        if evaluator_config:
            return evaluator_config.model_dump(exclude={"tokenizer", "dataset_cfg", "compute_metric_func"})
        else:
            return ""

    @field_serializer("judger_config")
    def serialize_judger_config(self, judger_config: JudgerConfig) -> str:
        return judger_config.model_dump(exclude={"tokenizer", "reward_func"})

def get_train_seq_ctx(
    input_ids: torch.LongTensor, multimodal_train_info: dict | None = None, len_response_ids: int = 0
):
    seq_ctx = SequenceContext.from_input_ids((input_ids,), device="cpu")
    if multimodal_train_info and len(multimodal_train_info) > 0:
        position_ids = multimodal_train_info.get("position_ids")  # (1,n) or (3,1,n)
        if position_ids is not None and len(position_ids.shape) == 3:
            # qwen3vl 需要特殊处理，其余的不需要额外处理
            max_value = position_ids.max(dim=-1).values  # (3,1)
            response_position_ids = max_value.unsqueeze(-1).expand(-1, -1, len_response_ids) + torch.arange(
                1, len_response_ids + 1, device=max_value.device
            )
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
            seq_ctx.position_ids = position_ids  # type: ignore[assignment]
            assert position_ids.size(-1) == input_ids.size(-1)
        seq_ctx.pixel_values = multimodal_train_info.get("pixel_values")
        seq_ctx.image_grid_thw = multimodal_train_info.get("image_grid_thw")
    return seq_ctx


class RLTrainer:
    """Universal Reinforcement Learning Trainer for XTuner.

    A flexible RL training orchestrator that supports multiple RL algorithms
    through pluggable training workers and controllers. Manages the complete
    RL training workflow including rollout generation, policy updates,
    evaluation, and checkpoint management.

    **Training Workflow:**
        1. Initialize distributed workers and rollout environment
        2. Generate experiences using current policy
        3. Update policy using algorithm-specific training logic
        4. Synchronize weights between training and rollout workers
        5. Evaluate model performance and save checkpoints

    Args:
        load_from (str | Path): Path to the base model to load. Should be a HuggingFace
            model path (e.g., "meta-llama/Llama-2-7b-hf") or local model directory.
        resources (AcceleratorResourcesConfig): Configuration for distributed computing
            resources including number of workers, GPU allocation, and placement groups.
        rollout_config (RolloutConfig): Configuration for rollout workers that generate
            experiences by interacting with the environment.
        dataflow_config (DataFlowConfig): Data orchestration configuration controlling
            experience collection, batch formation, and data distribution across workers.
        judger_config (JudgerConfig): Configuration for the reward model or scoring system
            that evaluates generated responses and provides training signals.
        replay_buffer_config (ReplayBufferConfig): Settings for experience replay buffer
            including capacity, sampling strategy, and data retention policies.
        evaluator_config (EvaluatorConfig | None): Evaluation configuration specifying metrics,
            evaluation datasets, and assessment frequency for monitoring training progress. Defaults to None.
        train_worker_cfg (WorkerConfig): Configuration for distributed training workers
            including model architecture, optimizer settings, loss functions, and parallelism.
        tokenizer_path (str | Path): Path to the tokenizer for text preprocessing.
            Should be compatible with the base model specified in load_from.
        work_dir (Path | str | None): Working directory for experiment outputs,
            checkpoints, and logs. Defaults to None.
        log_dir (Path | str | None): Directory for training logs and monitoring outputs.
            Defaults to None.
        total_epochs (int): Total number of training epochs to execute.
        enable_evaluate (bool): Whether to perform periodic evaluation during training.
        resume_config (ResumeConfig | None): Configuration for resuming training from
            a previous checkpoint. Defaults to None.
        auto_resume (bool): Whether to automatically resume training. Defaults to False.
        load_checkpoint_cfg (LoadCheckpointConfig): Configuration for loading checkpoints.
        strict_load (bool): Whether to strictly enforce checkpoint loading compatibility.
            Defaults to True.
        hf_interval (int | None): Interval (in epochs) for saving HuggingFace format
            checkpoints. Defaults to None.
        hf_max_keep (int | None): Maximum number of HuggingFace checkpoints to retain.
            Defaults to None.
        seed (int): Random seed for reproducible training. Defaults to 42.
        debug (bool): Enable debug mode with additional logging. Defaults to False.
        debug_rollout (bool): Enable debug mode for rollout workers. Defaults to False.
        rollout_steps (int | None): Total number of rollout steps to perform.
            If specified, overrides total_epochs. Defaults to None.

    **Examples:**

    Example configuration for GRPO RL training setup::

        trainer = RLTrainer(
            load_from="Qwen3-8B",
            resources=resources_config,
            rollout_config=rollout_cfg,
            dataflow_config=dataflow_cfg,
            judger_config=judger_cfg,
            replay_buffer_config=buffer_cfg,
            evaluator_config=eval_cfg,
            train_worker_cfg=worker_cfg,
            tokenizer_path="Qwen3-8B",
            total_epochs=10,
            enable_evaluate=True
        )
        trainer.fit()
    """

    META_PATH = ".xtuner_grpo"

    _CHECKPOINT_DIR = "checkpoints"
    _SAVE_TRAIN_STATE_PATH = "train_state.json"

    @staticmethod
    def _get_verifier_update_flags(worker_cfg: WorkerConfig) -> tuple[bool, bool]:
        """Return (lora_enabled, update_base_enabled) from worker config."""
        lora_enabled = False
        update_base_enabled = False
        try:
            verifier_lora_cfg = getattr(worker_cfg, "verifier_lora_cfg", None)
            lora_enabled = verifier_lora_cfg is not None and int(getattr(verifier_lora_cfg, "r", 0) or 0) > 0
        except Exception:
            lora_enabled = False
        try:
            update_base_enabled = bool(getattr(worker_cfg, "verifier_update_base", False))
        except Exception:
            update_base_enabled = False
        if lora_enabled and update_base_enabled:
            update_base_enabled = False
        return bool(lora_enabled), bool(update_base_enabled)

    def __init__(
        self,
        *,
        load_from: str | Path,  # Huggingface model path or saved trainer_path
        resources: AcceleratorResourcesConfig,
        cpu_resources: CPUResourcesConfig | None = None,
        rollout_config: RolloutConfig,
        dataflow_config: DataFlowConfig,
        judger_config: JudgerConfig,
        replay_buffer_config: ReplayBufferConfig,
        train_worker_cfg: WorkerConfig,
        evaluator_config: EvaluatorConfig | None = None,
        tokenizer_path: str | Path,
        work_dir: Path | str | None = None,
        log_dir: Path | str | None = None,
        total_epochs: int,
        auto_resume: bool = False,
        load_checkpoint_cfg: LoadCheckpointConfig = LoadCheckpointConfig(),
        strict_load: bool = True,
        checkpoint_interval: int | None = -1,
        checkpoint_maxkeep: int | None = -1,
        checkpoint_no_save_optimizer: bool = False,
        skip_checkpoint_validation: bool = False,  # Suggest enabled if fsdp_size is larger than 512
        hf_interval: int | None = None,
        hf_max_keep: int | None = None,
        seed: int = 42,
        debug: bool = False,
        debug_rollout: bool = False,
        rollout_steps: int | None = None,
        trainer_cfg: RLTrainerConfig | None = None,
    ):
        """Initialize the RL training system."""
        if os.environ.get("XTUNER_USE_FA3", "0") == "1":
            try:
                from xtuner.v1.ops.flash_attn import get_flash_attn_varlen

                get_flash_attn_varlen()
            except RuntimeError as e:
                raise RuntimeError(
                    f"Flash attention v3 runtime error {e}, Please install it first or set XTUNER_USE_FA3=0."
                )
        train_worker_cfg.load_from = load_from

        self._total_epochs = total_epochs
        self._cur_step = 0

        if skip_checkpoint_validation:
            patch_default_save_plan()

        self._rl_trainer_cfg = trainer_cfg
        self._load_from = Path(load_from) if isinstance(load_from, str) else load_from

        is_hf_path, error_info = is_hf_model_path(load_from) if load_from is not None else False, ""
        self._load_from_hf = is_hf_path

        if not self._load_from_hf:
            raise NotImplementedError(error_info)

        self._hf_max_keep = hf_max_keep
        self._hf_interval = hf_interval
        self._checkpoint_interval = checkpoint_interval
        self._checkpoint_maxkeep = checkpoint_maxkeep
        self._checkpoint_no_save_optimizer = checkpoint_no_save_optimizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        self._debug = debug
        self._debug_rollout = debug_rollout
        self._seed = seed
        self._set_deterministic()
        self._set_random_seed(seed)
        self._verifier_last_event_total = 0
        self._verifier_last_batch_count = 0
        self._verifier_last_fit_ran = False

        if work_dir is None:
            work_dir = Path.cwd() / "work_dir"

        if isinstance(work_dir, str):
            work_dir = Path(work_dir)

        if get_rank() == 0:
            work_dir.mkdir(parents=True, exist_ok=True)

        self._work_dir = work_dir
        self._auto_resume = auto_resume
        self._meta = self._init_xtuner_meta(work_dir, self._auto_resume)

        if log_dir is None:
            log_dir = self.exp_dir
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)

        self.logger = self._init_logger(log_dir)

        self._load_checkpoint_cfg = self._resolve_load_checkpoint_cfg(self._auto_resume, load_checkpoint_cfg)

        train_worker_cfg.log_dir = log_dir
        dataflow_config.worker_log_dir = log_dir
        rollout_config.worker_log_dir = log_dir
        self._enable_return_routed_experts = rollout_config.enable_return_routed_experts
        self._enable_evaluate = False
        self._enable_initial_evaluate = False
        if evaluator_config:
            evaluator_config.worker_log_dir = log_dir
            self._enable_evaluate = evaluator_config.enable_evaluate
            self._enable_initial_evaluate = evaluator_config.enable_initial_evaluate
        self._pg = AutoAcceleratorWorkers.build_placement_group(resources)
        if cpu_resources is None:
            # Build a CPU placement group for judgers.
            #
            # NOTE:
            # JudgerController requires enough bundles for sum(num_ray_actors) across reward judgers.
            # Defaulting to 1 bundle can break multi-actor remote judgers (e.g. CompassVerifier).
            required_judger_actors = 1
            try:
                cfgs = getattr(judger_config, "reward_judger_configs", None) or []
                required_judger_actors = int(
                    sum(max(1, int(getattr(c, "num_ray_actors", 1) or 1)) for c in cfgs) or 1
                )
            except Exception:
                required_judger_actors = 1
            bundles = [{"CPU": 1, "memory": 1024**3} for _ in range(max(1, required_judger_actors))]
            self._cpu_pg = placement_group(bundles=bundles, strategy="PACK")
            ray.get(self._cpu_pg.ready(), timeout=PG_READY_TIMEOUT)
        else:
            self._cpu_pg = AutoCPUWorkers.build_placement_group(cpu_resources)
        # We need to build train controller first, and then build rollout dataflow to make
        # inference engines know how much memory they can utilize.
        self._train_controller = self._build_train_controller(train_worker_cfg)

        if self._load_checkpoint_cfg.checkpoint_path is not None:
            rollout_config.skip_load_weights = True
            self.logger.info(
                f"Skip load rollout weights due to resume from checkpoint {self._load_checkpoint_cfg.checkpoint_path}"
            )

            # resume train worker
            ray.get(self._train_controller.resume.remote(self._load_checkpoint_cfg))

            train_state_path = Path(self._load_checkpoint_cfg.checkpoint_path) / self._SAVE_TRAIN_STATE_PATH
            with train_state_path.open("r") as f:
                train_state = json.load(f)
                self._cur_step = train_state["cur_step"]

        self._rollout_env_controller, self._rollout_dataflow = self._build_rollout_dataflow(
            dataflow_cfg=dataflow_config,
            rollout_cfg=rollout_config,
            judger_cfg=judger_config,
            replay_buffer_config=replay_buffer_config,
        )
        self._dataflow_partial_rollout_step = dataflow_config.tail_batch_candidate_steps
        

        if self._load_checkpoint_cfg.checkpoint_path is not None:
            # resume rollout dataflow
            self.logger.info(f"Resume rollout dataflow from checkpoint {self._load_checkpoint_cfg.checkpoint_path}")
            ray.get(self._rollout_dataflow.resume.remote(self._load_checkpoint_cfg.checkpoint_path))

        if self._enable_evaluate and evaluator_config:
            self._evaluator = Evaluator.remote(evaluator_config, self._rollout_env_controller)  # type: ignore[attr-defined]
            self._eval_step = evaluator_config.evaluate_step
        else:
            pass

        self._global_batch_size = dataflow_config.global_batch_size
        self._rollout_steps = (
            ray.get(self._rollout_dataflow.get_train_dataset_length.remote())  # type: ignore[attr-defined]
            // dataflow_config.global_batch_size
            * total_epochs
        )
        if rollout_steps is not None:
            self._rollout_steps = rollout_steps
            self.logger.info(f"Set rollout steps to {self._rollout_steps} according to rollout_steps arg")

        if not rollout_config.skip_load_weights:
            ray.get(self._train_controller.offload.remote(target="all"))

        bind_train_rollout(train_controller=self._train_controller, env_controller=self._rollout_env_controller)
        # update weights if rollout_config.skip_load_weights == True
        if rollout_config.skip_load_weights:
            self.logger.info("Rollout workers skip load weights, update weights from train workers.")
            ray.get(self._train_controller.offload.remote(target="optimizer"))
            ray.get(self._rollout_env_controller.offload.remote())
            ray.get(self._rollout_env_controller.onload_weights.remote())
            ray.get(self._train_controller.update_weights.remote())
            ray.get(self._train_controller.offload.remote(target="model"))
            ray.get(self._rollout_env_controller.onload_kvcache.remote())
            self.logger.info("Rollout workers has updated weights from train workers.")

        self._train_worker_cfg = train_worker_cfg

        # Optional: Co-GRPO verifier rollout routing + weight sync target.
        #
        # If you provide dedicated verifier rollout server URLs, we will:
        #  - route verifier inference requests from rollout workers to those URLs
        #  - allow training workers to sync merged verifier-LoRA weights to those URLs
        verifier_lora_enabled, verifier_update_base_enabled = self._get_verifier_update_flags(self._train_worker_cfg)
        verifier_enabled = bool(verifier_lora_enabled or verifier_update_base_enabled)
        rollout_extra_cfg = (rollout_config.extra_rollout_config or {}) if rollout_config is not None else {}
        verifier_model_cfg = str(
            rollout_extra_cfg.get("cogrpo_verifier_model", rollout_extra_cfg.get("verifier_model", "")) or ""
        ).strip()
        verifier_url_dict_env = os.environ.get(
            "COGRPO_VERIFIER_SERVER_URL_DICT", os.environ.get("VERIFIER_SERVER_URL_DICT", "")
        ).strip()
        verifier_urls_env = os.environ.get(
            "COGRPO_VERIFIER_SERVER_URLS", os.environ.get("VERIFIER_SERVER_URLS", "")
        ).strip()

        strict_verifier_sync = str(os.environ.get("COGRPO_STRICT_VERIFIER_SYNC", "1") or "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        try:
            verifier_lora_sync_freq = int(
                os.environ.get("COGRPO_VERIFIER_LORA_SYNC_FREQ", os.environ.get("VERIFIER_LORA_SYNC_FREQ", "1"))
            )
        except Exception:
            verifier_lora_sync_freq = 1
        verifier_lora_sync_freq = max(0, int(verifier_lora_sync_freq))

        if verifier_lora_enabled and strict_verifier_sync and verifier_lora_sync_freq <= 0:
            raise RuntimeError(
                "[CoGRPO][Verifier] strict sync preflight failed: verifier LoRA is enabled but "
                "COGRPO_VERIFIER_LORA_SYNC_FREQ<=0, which means verifier rollout inference will never receive "
                "online LoRA updates. Set sync freq > 0, or disable strict sync explicitly for debug-only runs."
            )
        if verifier_lora_enabled and (not strict_verifier_sync) and verifier_lora_sync_freq <= 0:
            self.logger.warning(
                "[CoGRPO][Verifier] verifier LoRA sync is disabled (COGRPO_VERIFIER_LORA_SYNC_FREQ<=0). "
                "Verifier LoRA will still be trained, but rollout-side verifier inference will not receive "
                "online LoRA updates in this run."
            )

        if verifier_lora_enabled and strict_verifier_sync and verifier_lora_sync_freq > 0:
            if not (verifier_url_dict_env or verifier_urls_env):
                raise RuntimeError(
                    "[CoGRPO][Verifier] strict sync preflight failed: verifier LoRA sync is enabled "
                    f"(COGRPO_VERIFIER_LORA_SYNC_FREQ={verifier_lora_sync_freq}) but "
                    "COGRPO_VERIFIER_SERVER_URLS / COGRPO_VERIFIER_SERVER_URL_DICT is empty. "
                    "Set sync freq to 0 for debug-only runs, or provide dedicated verifier rollout URLs."
                )

        if verifier_enabled:
            if verifier_url_dict_env or verifier_urls_env:
                try:
                    rollout_info = ray.get(self._rollout_env_controller.get_rollout_info.remote())  # type: ignore[attr-defined]
                    actor_server_url_dict = rollout_info.get("server_url_dict", {}) or {}
                    actor_ranks = sorted(int(k) for k in actor_server_url_dict.keys())

                    verifier_server_url_dict: dict[int, str] = {}
                    if verifier_url_dict_env:
                        parsed = json.loads(verifier_url_dict_env)
                        if isinstance(parsed, dict):
                            for k, v in parsed.items():
                                try:
                                    rk = int(k)
                                except Exception:
                                    continue
                                if isinstance(v, str) and v.strip():
                                    verifier_server_url_dict[rk] = v.strip().rstrip("/")
                    else:
                        urls = [u.strip().rstrip("/") for u in verifier_urls_env.split(",") if u.strip()]
                        if actor_ranks and len(urls) == len(actor_ranks):
                            verifier_server_url_dict = dict(zip(actor_ranks, urls))

                    strict_sync_active = verifier_lora_enabled and strict_verifier_sync and verifier_lora_sync_freq > 0
                    missing_actor_ranks = sorted(set(actor_ranks) - set(verifier_server_url_dict.keys()))
                    if missing_actor_ranks:
                        coverage_msg = (
                            "[CoGRPO][Verifier] verifier rollout URL mapping does not cover all actor ranks. "
                            f"Missing ranks: {missing_actor_ranks}; actor ranks: {actor_ranks}."
                        )
                        if strict_sync_active:
                            raise RuntimeError(coverage_msg)
                        self.logger.warning(coverage_msg)

                    if verifier_server_url_dict:
                        actor_server_url_dict_raw = rollout_info.get("server_url_dict", {}) if isinstance(rollout_info, dict) else {}
                        actor_server_url_dict: dict[int, str] = {}
                        if isinstance(actor_server_url_dict_raw, dict):
                            for k, v in actor_server_url_dict_raw.items():
                                try:
                                    actor_rank = int(k)
                                except Exception:
                                    continue
                                actor_url = str(v or "").strip().rstrip("/")
                                if actor_url:
                                    actor_server_url_dict[actor_rank] = actor_url
                        overlapping_rank_urls = {
                            rank: verifier_url
                            for rank, verifier_url in verifier_server_url_dict.items()
                            if verifier_url and verifier_url == actor_server_url_dict.get(rank)
                        }
                        if overlapping_rank_urls:
                            overlap_msg = (
                                "[CoGRPO][Verifier] verifier rollout URL mapping overlaps actor rollout URLs "
                                f"for ranks {sorted(overlapping_rank_urls.keys())}. "
                                "This means decoupled verifier sync would target actor rollout servers instead of "
                                "dedicated verifier servers."
                            )
                            if strict_sync_active:
                                raise RuntimeError(overlap_msg)
                            self.logger.warning(overlap_msg)
                        # Route verifier inference requests.
                        ray.get(
                            self._rollout_env_controller.set_aux_rollout_server_urls.remote(  # type: ignore[attr-defined]
                                "verifier", verifier_server_url_dict, True
                            )
                        )
                        # Configure training-side verifier rollout target for weight sync.
                        verifier_rollout_info = dict(rollout_info)
                        verifier_rollout_info["server_url_dict"] = verifier_server_url_dict
                        verifier_rollout_info["worker_server_urls_status"] = {
                            url: True for url in verifier_server_url_dict.values()
                        }
                        ray.get(self._train_controller.update_verifier_rollout_info.remote(verifier_rollout_info))
                        self.logger.info(
                            f"[CoGRPO][Verifier] Configured verifier rollout servers for {len(verifier_server_url_dict)} ranks."
                        )
                    else:
                        if strict_sync_active:
                            raise RuntimeError(
                                "[CoGRPO][Verifier] strict sync preflight failed: verifier rollout URL mapping is empty "
                                "after parsing COGRPO_VERIFIER_SERVER_URLS / COGRPO_VERIFIER_SERVER_URL_DICT. "
                                "Provide a valid rank->url mapping (or urls aligned to rollout ranks), or set "
                                "COGRPO_VERIFIER_LORA_SYNC_FREQ=0 for debug-only runs."
                            )
                        if verifier_model_cfg:
                            self.logger.warning(
                                "[CoGRPO][Verifier] verifier enabled, no valid verifier server URLs provided; "
                                f"fallback to actor rollout servers with verifier model override '{verifier_model_cfg}'. "
                                "Verifier LoRA live sync to rollout is disabled without dedicated verifier URLs."
                            )
                        else:
                            self.logger.warning(
                                "[CoGRPO][Verifier] verifier enabled but no valid verifier server URLs provided; "
                                "verifier inference will use actor rollout base model (may cause parser no-decision)."
                            )
                except Exception as e:
                    if strict_verifier_sync and verifier_lora_sync_freq > 0:
                        raise RuntimeError(f"[CoGRPO][Verifier] strict sync preflight failed: {e}") from e
                    self.logger.warning(f"[CoGRPO][Verifier] Failed to configure verifier rollout servers: {e}")
            else:
                if verifier_model_cfg:
                    self.logger.warning(
                        "[CoGRPO][Verifier] verifier enabled but verifier rollout URLs are not set "
                        "(COGRPO_VERIFIER_SERVER_URLS / COGRPO_VERIFIER_SERVER_URL_DICT). "
                        f"Fallback to actor rollout servers with verifier model override '{verifier_model_cfg}'. "
                        "Verifier LoRA live sync to rollout is disabled without dedicated verifier URLs."
                    )
                else:
                    self.logger.warning(
                        "[CoGRPO][Verifier] verifier enabled but verifier rollout URLs are not set "
                        "(COGRPO_VERIFIER_SERVER_URLS / COGRPO_VERIFIER_SERVER_URL_DICT). "
                        "Verifier inference will use actor rollout base model (may cause parser no-decision)."
                    )
        elif verifier_enabled and verifier_update_base_enabled:
            self.logger.info(
                "[CoGRPO][Verifier] verifier update_base mode enabled; "
                "verifier updates reuse actor/base optimizer and rollout weight sync follows actor update_weights."
            )

        if self._rl_trainer_cfg is not None and get_rank() == 0:
            config_path = log_dir / "rl_trainer_config.json"
            with config_path.open("w") as f:
                f.write(self._rl_trainer_cfg.model_dump_json(indent=2))

            env_path = log_dir / "env.json"
            environment_variables = dict(os.environ)
            infer_engine_version = get_rollout_engine_version()
            environment_variables.update(infer_engine_version)
            with env_path.open("w") as f:
                json.dump(environment_variables, f, indent=2)

        self._writer = TensorboardWriter(log_dir / "tb")

    def _resolve_load_checkpoint_cfg(
        self, auto_resume: bool, load_checkpoint_cfg: LoadCheckpointConfig
    ) -> LoadCheckpointConfig:
        # auto_resume优先级高，如果有latest ckp，则说明走auto_resume逻辑
        # 此时，覆盖load checkpoint path
        latest_checkpoint = self.meta.latest_exp.latest_checkpoint
        if latest_checkpoint is not None and auto_resume:
            load_checkpoint_cfg.checkpoint_path = Path(latest_checkpoint)
        return load_checkpoint_cfg

    @classmethod
    def from_config(cls, config: RLTrainerConfig) -> Self:
        """Create a Trainer instance from a TrainerConfig.

        Args:
            config (TrainerConfig): TrainerConfig instance containing all configuration parameters.

        Returns:
            Self: Trainer instance initialized with the provided config.
        """
        self = cls(
            load_from=config.load_from,
            resources=config.resources,
            cpu_resources=config.cpu_resources,
            rollout_config=config.rollout_config,
            dataflow_config=config.dataflow_config,
            judger_config=config.judger_config,
            replay_buffer_config=config.replay_buffer_config,
            train_worker_cfg=config.train_worker_config,
            evaluator_config=config.evaluator_config,
            tokenizer_path=config.tokenizer_path,
            work_dir=config.work_dir,
            log_dir=config.log_dir,
            total_epochs=config.total_epochs,
            auto_resume=config.auto_resume,
            load_checkpoint_cfg=config.load_checkpoint_cfg,
            strict_load=config.strict_load,
            checkpoint_interval=config.checkpoint_interval,
            checkpoint_maxkeep=config.checkpoint_maxkeep,
            checkpoint_no_save_optimizer=config.checkpoint_no_save_optimizer,
            hf_interval=config.hf_interval,
            hf_max_keep=config.hf_max_keep,
            skip_checkpoint_validation=config.skip_checkpoint_validation,
            seed=config.seed,
            debug=config.debug,
            debug_rollout=config.debug_rollout,
            rollout_steps=config.rollout_steps,
            trainer_cfg=config,
        )
        return self

    def _build_rollout_dataflow(
        self,
        dataflow_cfg: DataFlowConfig,
        rollout_cfg: RolloutConfig,
        judger_cfg: JudgerConfig,
        replay_buffer_config: ReplayBufferConfig,
    ) -> tuple[SingleTurnEnvironmentProxy, DataFlowProxy]:
        env = SingleTurnEnvironment.remote("grpo", self._pg, rollout_cfg, self._cpu_pg, judger_cfg)
        flow = DataFlow.remote("grpo", dataflow_cfg, replay_buffer_config, env)
        return env, flow

    def _build_train_controller(self, train_worker_cfg: WorkerConfig) -> TrainingControllerProxy:
        TrainingWorker = cast(
            TrainingWorkerClass,
            ray.remote(
                runtime_env={
                    "env_vars": {
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
                    }
                },
            )(BaseTrainingWorker),
        )
        train_workers: list[TrainingWorkerProxy]
        train_workers, _ = AutoAcceleratorWorkers.from_placement_group(TrainingWorker, train_worker_cfg, self._pg)
        ray.wait([worker.ready.remote() for worker in train_workers])
        train_controller = TrainingController.remote(workers=train_workers)
        return train_controller

    def _should_skip_verifier_live_sync(self) -> tuple[bool, str]:
        if not _env_flag("COGRPO_VERIFIER_SKIP_SYNC_IF_NO_UPDATE", "1"):
            return False, ""
        if self._verifier_last_fit_ran:
            return False, ""
        if int(self._verifier_last_batch_count or 0) <= 0:
            return True, "no_verifier_batches"
        return True, "verifier_fit_not_run"

    def _initial_evaluate(self):
        """Performs an initial evaluation before the training loop starts."""
        if self._debug_rollout:
            return
        if self._enable_initial_evaluate and self._enable_evaluate and self._evaluator:
            ray.get(self._rollout_env_controller.update_active_workers.remote())
            scores, eval_data_groups = ray.get(self._evaluator.run.remote(return_samples=True))
            trajectory_save_path = self.exp_dir / "eval_0_trajectory.jsonl"
            self._save_trajectories(eval_data_groups, trajectory_save_path, 0, is_eval=True)
            self.logger.info(f"Initial rollout evaluate scores {scores} and start training")
            tb_scores = {f"eval/{k}": v for k, v in scores.items()}
            self._writer.add_scalars(
                tag_scalar_dict=tb_scores,
                global_step=0,
            )

    def _rollout_step(self, rollout_idx: int, step_timer_dict: dict):
        """Performs a single rollout step to generate experience."""
        with timer("generation", step_timer_dict):
            ray.get(self._rollout_env_controller.update_active_workers.remote())
            (data_groups, multimodal_train_infos), dataflow_tb_metrics = ray.get(self._rollout_dataflow.run.remote())
            replay_buffer_status = ray.get(self._rollout_dataflow.get_replaybuffer_status.remote())

        self._writer.add_scalar(
            tag="time/generation", scalar_value=step_timer_dict["generation"], global_step=rollout_idx
        )
        self._writer.add_scalars(tag_scalar_dict=dataflow_tb_metrics, global_step=rollout_idx)
        tb_replay_buffer_status = _flatten_scalar_metrics("async", replay_buffer_status)
        self._writer.add_scalars(tag_scalar_dict=tb_replay_buffer_status, global_step=rollout_idx)

        with timer("save_trajectory", step_timer_dict):
            trajectory_save_path = self.exp_dir / f"rollout_idx_{rollout_idx}_trajectory.jsonl"
            self._save_trajectories(data_groups, trajectory_save_path, rollout_idx)
            self.logger.info(f"Rollout_idx {rollout_idx} finished, saved trajectories to {trajectory_save_path}")

        self._writer.add_scalar(
            tag="time/save_trajectory", scalar_value=step_timer_dict["save_trajectory"], global_step=rollout_idx
        )

        if not self._debug_rollout:
            with timer("rollout_offload", step_timer_dict):
                ray.get(self._rollout_dataflow.pause.remote())
                ray.get(self._rollout_env_controller.offload.remote())

            self._writer.add_scalar(
                tag="time/rollout_offload", scalar_value=step_timer_dict["rollout_offload"], global_step=rollout_idx
            )
        return data_groups, multimodal_train_infos

    def _train_step(self, rollout_idx: int, data_groups, multimodal_train_infos, step_timer_dict: dict):
        """Performs a single training step on the generated experience."""
        self._verifier_last_event_total = 0
        self._verifier_last_batch_count = 0
        self._verifier_last_fit_ran = False
        with timer("onload", step_timer_dict):
            ray.get(self._train_controller.onload.remote(target="all"))
            self.logger.info("Training controller loaded")

        with timer("prepare_data", step_timer_dict):
            data_batches, data_info = self._prepare_train_data(
                data_groups,
                self._train_worker_cfg.pack_max_length,
                multimodal_train_infos,
                rollout_idx=rollout_idx,
            )
            self.logger.info(f"Prepared {len(data_batches)} training data batches")
            self._log_data_info(rollout_idx, data_info)

        self._writer.add_scalar(
            tag="time/onload",
            scalar_value=step_timer_dict["onload"],
            global_step=rollout_idx,
        )

        self._writer.add_scalar(
            tag="time/prepare_data",
            scalar_value=step_timer_dict["prepare_data"],
            global_step=rollout_idx,
        )

        # Co-GRPO verifier update (optional): train verifier LoRA on event-level advantages.
        #
        # NOTE: Align with verl behavior: when verifier is LoRA-only (shared base unchanged),
        # update verifier BEFORE actor to keep PPO behavior-policy alignment.
        verifier_lora_enabled, verifier_update_base_enabled = self._get_verifier_update_flags(self._train_worker_cfg)
        verifier_enabled = bool(verifier_lora_enabled or verifier_update_base_enabled)

        if verifier_enabled:
            with timer("prepare_verifier_data", step_timer_dict):
                verifier_batches, verifier_info = self._prepare_verifier_train_data(
                    data_groups, self._train_worker_cfg.pack_max_length
                )
                self._verifier_last_event_total = int(verifier_info.get("co_grpo/verifier_events_total", 0) or 0)
                self._verifier_last_batch_count = len(verifier_batches)
                self.logger.info(
                    "[CoGRPO][Verifier] verifier batch prep "
                    f"batch_size={int(verifier_info.get('batch_size', 0) or 0)} "
                    f"events_total={int(verifier_info.get('co_grpo/verifier_events_total', 0) or 0)} "
                    f"cf_events_total={int(verifier_info.get('co_grpo/cf_verifier_events_total', 0) or 0)} "
                    f"cf_adv_used={int(verifier_info.get('co_grpo/verifier_events_cf_advantage_used', 0) or 0)} "
                    f"cf_adv_reconstructed={int(verifier_info.get('co_grpo/verifier_events_cf_advantage_reconstructed', 0) or 0)} "
                    f"equal_share_used={int(verifier_info.get('co_grpo/verifier_events_equal_share_used', 0) or 0)} "
                    f"adv_abs_max={float(verifier_info.get('advantages/abs_max', 0.0) or 0.0):.4f} "
                    f"adv_zero_ratio={float(verifier_info.get('advantages/zero_ratio', 0.0) or 0.0):.4f} "
                    f"adv_neg_ratio={float(verifier_info.get('advantages/neg_ratio', 0.0) or 0.0):.4f} "
                    f"delta_abs_max={float(verifier_info.get('co_grpo/cf_delta_abs_max_verifier_events_used', 0.0) or 0.0):.4f} "
                    f"delta_zero_ratio={float(verifier_info.get('co_grpo/cf_delta_zero_ratio_verifier_events_used', 0.0) or 0.0):.4f} "
                    f"trunc_skip={int(verifier_info.get('co_grpo/cf_trunc_event_skip_count', 0) or 0)} "
                    f"skip_missing_adv={int(verifier_info.get('co_grpo/verifier_events_skip_missing_advantage', 0) or 0)} "
                    f"skip_missing_parent_reward={int(verifier_info.get('co_grpo/verifier_events_skip_missing_parent_reward', 0) or 0)} "
                    f"skip_missing_parent_count={int(verifier_info.get('co_grpo/verifier_events_skip_missing_parent_count', 0) or 0)}"
                )
                if verifier_batches:
                    self.logger.info(f"Prepared {len(verifier_batches)} verifier training data batches")

                # Log verifier rollout stats and verl-aligned cf_branch diagnostics.
                verifier_rollout_tb = {
                    f"verifier_rollout/{k}": v
                    for k, v in verifier_info.items()
                    if (not str(k).startswith("co_grpo/")) and isinstance(v, (int, float))
                }
                if verifier_rollout_tb:
                    self._writer.add_scalars(tag_scalar_dict=verifier_rollout_tb, global_step=rollout_idx)

                cogrpo_tb = {
                    k: v
                    for k, v in verifier_info.items()
                    if str(k).startswith("co_grpo/") and isinstance(v, (int, float))
                }
                if cogrpo_tb:
                    self._writer.add_scalars(tag_scalar_dict=cogrpo_tb, global_step=rollout_idx)

            self._writer.add_scalar(
                tag="time/prepare_verifier_data",
                scalar_value=step_timer_dict.get("prepare_verifier_data", 0.0),
                global_step=rollout_idx,
            )

            if verifier_batches:
                with timer("training_verifier", step_timer_dict):
                    verifier_log_items: list[WorkerLogItem] = ray.get(
                        self._train_controller.fit_verifier.remote(
                            verifier_batches,
                            pack_max_length=self._train_worker_cfg.pack_max_length,
                            rollout_idx=rollout_idx,
                        )
                    )
                self._verifier_last_fit_ran = True
                self._writer.add_scalar(
                    tag="time/training_verifier",
                    scalar_value=step_timer_dict.get("training_verifier", 0.0),
                    global_step=rollout_idx,
                )

                verifier_rank0 = verifier_log_items[0]
                if verifier_rank0.get("train_entropy") is not None:
                    self._writer.add_scalar(
                        tag="verifier_entropy/train",
                        scalar_value=float(verifier_rank0.get("train_entropy", 0.0) or 0.0),
                        global_step=rollout_idx,
                    )

        with timer("training", step_timer_dict):
            workers_log_item: list[WorkerLogItem] = ray.get(
                self._train_controller.fit.remote(
                    data_batches, pack_max_length=self._train_worker_cfg.pack_max_length, rollout_idx=rollout_idx
                )
            )
        self._writer.add_scalar(tag="time/training", scalar_value=step_timer_dict["training"], global_step=rollout_idx)

        rank0_log_item = workers_log_item[0]
        # These metrics are already aggregated across distributed workers and logging only the metrics from rank 0.
        rank0_rollout_is_metrics = rank0_log_item.get("rollout_is_metrics")
        rank0_mismatch_metrics = rank0_log_item.get("mismatch_metrics")
        rank0_rollout_entropy = rank0_log_item.get("rollout_entropy")
        if rank0_rollout_is_metrics is not None:
            tb_rollout_is_metrics = {f"rollout_is/{k}": v for k, v in rank0_rollout_is_metrics.items()}
            self._writer.add_scalars(tag_scalar_dict=tb_rollout_is_metrics, global_step=rollout_idx)
        if rank0_mismatch_metrics is not None:
            tb_mismatch_metrics = {f"{k}": v for k, v in rank0_mismatch_metrics.items()}
            self._writer.add_scalars(tag_scalar_dict=tb_mismatch_metrics, global_step=rollout_idx)
        if rank0_rollout_entropy is not None:
            tb_rollout_entropy = {"entropy/rollout": rank0_rollout_entropy}
            self._writer.add_scalars(tag_scalar_dict=tb_rollout_entropy, global_step=rollout_idx)
        tb_entropy = {"entropy/train": rank0_log_item["train_entropy"]}
        self._writer.add_scalars(tag_scalar_dict=tb_entropy, global_step=rollout_idx)

        for worker_idx, log_item in enumerate(workers_log_item):
            if worker_idx == 0:
                mini_batch_metrics: dict[str, list[float]] = {}
                for mini_batch_log in log_item["train_metrics"]:
                    rl_worker_log = {**mini_batch_log["loss_log"], **mini_batch_log["rl_other_log"]}
                    # Aggregate logs for the mini-batch
                    for k, v in rl_worker_log.items():
                        mini_batch_metrics.setdefault(k, []).append(cast(float, v))

                for key, value in mini_batch_metrics.items():
                    for i, v in enumerate(value):
                        global_step = (rollout_idx - 1) * len(value) + i + 1
                        self._writer.add_scalar(
                            tag=f"train_metrics/worker_{worker_idx}/{key}",
                            scalar_value=v,
                            global_step=global_step,
                        )

    def _sync_weights_and_save(self, rollout_idx: int, step_timer_dict: dict):
        """Synchronizes weights and saves checkpoints."""
        with timer("train_offload_optimizer", step_timer_dict):
            ray.get(self._train_controller.offload.remote(target="optimizer"))
        with timer("save_ckpt", step_timer_dict):
            self._maybe_save_hf()
            self._maybe_save_checkpoint()

        with timer("bind", step_timer_dict):
            bind_train_rollout(train_controller=self._train_controller, env_controller=self._rollout_env_controller)
        with timer("rollout_onload_weights", step_timer_dict):
            ray.get(self._rollout_env_controller.onload_weights.remote())
        with timer("sync_weight", step_timer_dict):
            ray.get(self._train_controller.update_weights.remote())
        # Co-GRPO verifier LoRA: optionally sync merged verifier weights to dedicated verifier rollout servers.
        verifier_lora_enabled, _ = self._get_verifier_update_flags(self._train_worker_cfg)
        if verifier_lora_enabled:
            strict_verifier_sync = str(os.environ.get("COGRPO_STRICT_VERIFIER_SYNC", "1") or "1").strip().lower() in (
                "1",
                "true",
                "yes",
                "y",
                "on",
            )
            try:
                sync_freq = int(
                    os.environ.get("COGRPO_VERIFIER_LORA_SYNC_FREQ", os.environ.get("VERIFIER_LORA_SYNC_FREQ", "1"))
                )
            except Exception:
                sync_freq = 1
            verifier_sync_dir = None
            verifier_live_save_success = False
            live_sync_dtype = _resolve_verifier_save_dtype(
                "COGRPO_VERIFIER_LIVE_SAVE_DTYPE",
                "COGRPO_VERIFIER_SAVE_DTYPE",
            )
            archive_save_dtype = _resolve_verifier_save_dtype(
                "COGRPO_VERIFIER_ARCHIVE_SAVE_DTYPE",
                "COGRPO_VERIFIER_SAVE_DTYPE",
            )
            if sync_freq > 0 and (rollout_idx % sync_freq == 0):
                verifier_sync_dir = str(self.exp_dir / "verifier_lora_live" / f"checkpoint-{rollout_idx}")
                skip_live_sync, skip_reason = self._should_skip_verifier_live_sync()
                self._writer.add_scalar(
                    tag="co_grpo/verifier_live_sync_skipped_no_update",
                    scalar_value=1.0 if skip_live_sync else 0.0,
                    global_step=rollout_idx,
                )
                if skip_live_sync:
                    self.logger.info(
                        "[CoGRPO][Verifier] skip live verifier sync because no verifier update happened in this rollout "
                        f"(reason={skip_reason}, batches={int(self._verifier_last_batch_count or 0)}, "
                        f"events_total={int(self._verifier_last_event_total or 0)}, rollout_idx={rollout_idx})"
                    )
                else:
                    verifier_sync_success = False
                    verifier_save_success = False
                    with timer("save_verifier_lora", step_timer_dict):
                        verifier_save_success = bool(
                            ray.get(
                                self._train_controller.save_verifier_lora.remote(
                                    verifier_sync_dir,
                                    live_sync_dtype,
                                )
                            )
                        )
                    self._writer.add_scalar(
                        tag="co_grpo/verifier_save_success",
                        scalar_value=1.0 if verifier_save_success else 0.0,
                        global_step=rollout_idx,
                    )
                    if not verifier_save_success:
                        message = (
                            "[CoGRPO][Verifier] controller-side save_verifier_lora returned False; "
                            f"skip verifier live-sync. dir={verifier_sync_dir}"
                        )
                        if strict_verifier_sync:
                            raise RuntimeError(message)
                        self.logger.warning(message)
                    adapter_file = Path(verifier_sync_dir) / "adapter_model.safetensors"
                    adapter_cfg = Path(verifier_sync_dir) / "adapter_config.json"
                    if verifier_save_success and (not adapter_file.exists() or not adapter_cfg.exists()):
                        message = (
                            "[CoGRPO][Verifier] verifier live adapter files missing after save_verifier_lora; "
                            f"adapter_file={adapter_file.exists()} adapter_cfg={adapter_cfg.exists()} dir={verifier_sync_dir}"
                        )
                        if strict_verifier_sync:
                            raise RuntimeError(message)
                        self.logger.warning(message)
                        verifier_save_success = False
                    verifier_live_save_success = verifier_save_success
                    with timer("sync_verifier_weight", step_timer_dict):
                        if verifier_save_success:
                            sync_status = ray.get(self._train_controller.update_verifier_weights.remote(verifier_sync_dir))
                            verifier_sync_success = bool(sync_status)
                    self._writer.add_scalar(
                        tag="co_grpo/verifier_sync_success",
                        scalar_value=1.0 if verifier_sync_success else 0.0,
                        global_step=rollout_idx,
                    )
                    if not verifier_sync_success:
                        if strict_verifier_sync:
                            raise RuntimeError(
                                "[CoGRPO][Verifier] strict verifier sync failed: update_verifier_weights returned False."
                            )
                        self.logger.warning(
                            "[CoGRPO][Verifier] update_verifier_weights executed but no rollout-side verifier sync happened."
                        )

            try:
                save_freq = int(
                    os.environ.get("COGRPO_VERIFIER_LORA_SAVE_FREQ", os.environ.get("VERIFIER_LORA_SAVE_FREQ", "0"))
                )
            except Exception:
                save_freq = 0
            if save_freq > 0 and (rollout_idx % save_freq == 0):
                save_dir = str(self.exp_dir / "verifier_lora" / f"checkpoint-{rollout_idx}")
                archive_reused_live_sync = False
                if save_dir != verifier_sync_dir:
                    reuse_live_sync = (
                        _env_flag("COGRPO_VERIFIER_ARCHIVE_FROM_LIVE", "0")
                        and verifier_sync_dir is not None
                        and verifier_live_save_success
                        and live_sync_dtype == archive_save_dtype
                    )
                    with timer("save_verifier_lora", step_timer_dict):
                        if reuse_live_sync:
                            archive_dir = Path(save_dir)
                            if archive_dir.exists():
                                rmtree(archive_dir)
                            copytree(verifier_sync_dir, save_dir)
                            archive_reused_live_sync = True
                        else:
                            ray.get(self._train_controller.save_verifier_lora.remote(save_dir, archive_save_dtype))
                    if archive_reused_live_sync:
                        self.logger.info(
                            "[CoGRPO][Verifier] reused live verifier adapter for archive checkpoint "
                            f"source={verifier_sync_dir} target={save_dir}"
                        )
                    self._writer.add_scalar(
                        tag="co_grpo/verifier_archive_reused_live_sync",
                        scalar_value=1.0 if archive_reused_live_sync else 0.0,
                        global_step=rollout_idx,
                    )
        with timer("train_offload_model", step_timer_dict):
            ray.get(self._train_controller.offload.remote(target="model"))
        with timer("rollout_onload_cache", step_timer_dict):
            ray.get(self._rollout_env_controller.onload_kvcache.remote())

        self._writer.add_scalar(
            tag="time/train_offload_optimizer",
            scalar_value=step_timer_dict["train_offload_optimizer"],
            global_step=rollout_idx,
        )
        self._writer.add_scalar(
            tag="time/save_ckpt",
            scalar_value=step_timer_dict["save_ckpt"],
            global_step=rollout_idx,
        )
        self._writer.add_scalar(
            tag="time/rollout_onload_weights",
            scalar_value=step_timer_dict["rollout_onload_weights"],
            global_step=rollout_idx,
        )
        self._writer.add_scalar(
            tag="time/sync_weight",
            scalar_value=step_timer_dict["sync_weight"],
            global_step=rollout_idx,
        )
        if "sync_verifier_weight" in step_timer_dict:
            self._writer.add_scalar(
                tag="time/sync_verifier_weight",
                scalar_value=step_timer_dict["sync_verifier_weight"],
                global_step=rollout_idx,
            )
        if "save_verifier_lora" in step_timer_dict:
            self._writer.add_scalar(
                tag="time/save_verifier_lora",
                scalar_value=step_timer_dict["save_verifier_lora"],
                global_step=rollout_idx,
            )
        self._writer.add_scalar(
            tag="time/train_offload_model",
            scalar_value=step_timer_dict["train_offload_model"],
            global_step=rollout_idx,
        )
        self._writer.add_scalar(
            tag="time/rollout_onload_cache",
            scalar_value=step_timer_dict["rollout_onload_cache"],
            global_step=rollout_idx,
        )
    def _evaluate_step(self, rollout_idx: int, step_timer_dict: dict):
        """Performs an evaluation step."""
        if self._enable_evaluate and self._evaluator and rollout_idx % self._eval_step == 0:
            with timer("evaluation", step_timer_dict):
                scores, eval_data_groups = ray.get(self._evaluator.run.remote(return_samples=True))
                trajectory_save_path = self.exp_dir / f"eval_{rollout_idx}_trajectory.jsonl"
                self._save_trajectories(eval_data_groups, trajectory_save_path, rollout_idx, is_eval=True)
                self.logger.info(f"Evaluate idx {rollout_idx} scores {scores}")
            tb_scores = {f"eval/{k}": v for k, v in scores.items()}
            self._writer.add_scalars(
                tag_scalar_dict=tb_scores,
                global_step=rollout_idx,
            )

    def fit(self):
        """Run the RL training loop.

        This method executes the main rl training loop, iterating generating through the dataset and performing
        training steps. It handles rollout, prepare training data, update policy , synchronize model weights, and
        evaluation.
        """
        self.logger.info("Start RL training")
        if self._cur_step >= self._rollout_steps:
            self.logger.info(f"Rollout steps {self._rollout_steps} reached, stop training")
            return

        self._initial_evaluate()

        for rollout_idx in range(self._cur_step + 1, self._rollout_steps + 1):
            self.logger.info(f"Rollout {rollout_idx}/{self._rollout_steps} start")
            step_timer_dict = {}
            with timer("step", step_timer_dict):
                # 1. Rollout to generate experience
                data_groups, multimodal_train_infos = self._rollout_step(rollout_idx, step_timer_dict)

                if not self._debug_rollout:
                    # 2. Train on the generated experience
                    self._train_step(rollout_idx, data_groups, multimodal_train_infos, step_timer_dict)

                    # 3. Synchronize weights and save checkpoints
                    self._sync_weights_and_save(rollout_idx, step_timer_dict)

                    # 4. Evaluate model performance
                    self._evaluate_step(rollout_idx, step_timer_dict)

            # 5. Log timing information
            self._writer.add_scalar(
                tag="time/step",
                scalar_value=step_timer_dict["step"],
                global_step=rollout_idx,
            )
            timer_log_str = f"Rollout {rollout_idx} training finished and timing listed: \n"
            timer_log_str += timer_logger(step_timer_dict)
            self.logger.info(timer_log_str)
            self._cur_step = rollout_idx

    def _log_data_info(self, rollout_idx: int, data_info: dict):
        """Formats and logs the data statistics dictionary."""
        log_lines = [f"Rollout {rollout_idx} data statistics:"]
        for key, value in data_info.items():
            if isinstance(value, float):
                log_lines.append(f"  - {key:<20}: {value:.5f}")
            else:
                log_lines.append(f"  - {key:<20}: {value}")
        self.logger.info("\n".join(log_lines))

        tb_scalars: dict[str, float] = {}
        for k, v in data_info.items():
            if not isinstance(v, (int, float)):
                continue
            if k.startswith("advantages/"):
                tb_scalars[f"rollout/{k}"] = float(v)
            elif k.startswith("co_grpo/"):
                tb_scalars[str(k)] = float(v)
        if tb_scalars:
            self._writer.add_scalars(tag_scalar_dict=tb_scalars, global_step=rollout_idx)

    def rotate(self, inputs: torch.Tensor)-> torch.Tensor:
        return torch.cat(-inputs[..., len(inputs.shape[-1]//2):], inputs[..., :len(inputs.shape[-1]//2)], dim=-1)
    def cal_apply(self, x: torch.Tensor, y: torch.Tensor, base: float = 1e6, dim: int = 512, seq_len: int= 32*1024)-> torch.Tensor:
        pre_cla_val = 1.0 / (base ** range(0, dim, 2)[:dim//2]).float() / dim
        all_seq_val = torch.outer(torch.arange(seq_len), pre_cla_val)
        sin_pre_val = torch.cat(torch.sin(all_seq_val), torch.sin(all_seq_val), dim=-1)
        cos_pre_val = torch.cat(torch.cos(all_seq_val), torch.cos(all_seq_val), dim=-1) #  (seq_len, dim)
        x = cos_pre_val.unsqueeze(0) * x + sin_pre_val.unsqueeze(0) * self.rotate(x)
        y = 

    
    # TODO: advantage 是在 DataFlow 里算好，还是在 train controller 里算？
    # 因为可能有根据 advantage 来判断数据能否进 rl 训练的需求。暂时先放在这
    def _prepare_train_data(self, data_groups, pack_max_length, multimodal_train_infos=None, *, rollout_idx: int | None = None):
        all_input_ids = []
        all_response_ids = []
        all_multimodal_train_infos = []
        all_routed_experts = []
        all_shifted_labels = []
        all_advantages = []
        all_rollout_logprobs = []

        rewards_list = []
        advantages_list = []
        prompt_len_list = []
        response_len_list = []

        data_batches = []
        is_multimodal = False
        if multimodal_train_infos and len(multimodal_train_infos) > 0:
            assert len(multimodal_train_infos) == len(data_groups), (
                f"{len(multimodal_train_infos)} vs {len(data_groups)}"
            )
            is_multimodal = True

        # Advantage estimator
        #
        # Align with verl semantics:
        # - ADV_ESTIMATOR in {"grpo","co_grpo"} => GRPO group normalization
        # - fallback => historical RLOO (leave-one-out) baseline
        adv_estimator = str(os.environ.get("ADV_ESTIMATOR", "grpo") or "grpo").lower().strip()
        use_grpo = adv_estimator in ("grpo", "co_grpo", "co-grpo", "cogrpo", "cogrpo_v2", "co_grpo_v2")
        use_co_grpo = adv_estimator in ("co_grpo", "co-grpo", "cogrpo", "cogrpo_v2", "co_grpo_v2")

        actor_update_streams = str(os.environ.get("ACTOR_UPDATE_STREAMS", "exp") or "exp").lower().strip()
        update_both_streams = actor_update_streams in ("both", "control+exp", "control_exp", "dual", "control-exp")

        # Co-GRPO curriculum weighting (optional; mirrors verl compute_co_grpo_advantage).
        use_curriculum_weighting = str(os.environ.get("USE_CURRICULUM_WEIGHTING", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        try:
            control_group_weight = float(os.environ.get("CONTROL_GROUP_WEIGHT", "0.5"))
        except Exception:
            control_group_weight = 0.5
        try:
            curriculum_start_weight = float(os.environ.get("CURRICULUM_START_WEIGHT", "0.3"))
        except Exception:
            curriculum_start_weight = 0.3
        try:
            curriculum_end_weight = float(os.environ.get("CURRICULUM_END_WEIGHT", "0.7"))
        except Exception:
            curriculum_end_weight = 0.7

        training_progress = 0.0
        try:
            if rollout_idx is not None and int(self._rollout_steps) > 0:
                training_progress = float(rollout_idx) / float(int(self._rollout_steps))
        except Exception:
            training_progress = 0.0
        training_progress = max(0.0, min(1.0, float(training_progress)))

        effective_control_weight = float(control_group_weight)
        if use_curriculum_weighting:
            effective_control_weight = float(curriculum_start_weight) + float(curriculum_end_weight - curriculum_start_weight) * float(training_progress)
        effective_control_weight = max(0.0, min(1.0, float(effective_control_weight)))

        control_rewards_list: list[float] = []
        exp_rewards_list: list[float] = []
        relative_rewards_list: list[float] = []
        actor_reward_values: list[float] = []
        actor_response_len_values: list[float] = []
        actor_hint_len_values: list[float] = []
        actor_num_interventions_values: list[float] = []
        actor_intervened_reward_values: list[float] = []
        actor_non_intervened_reward_values: list[float] = []
        first_step_tokens_len_values: list[float] = []
        actor_step_count_values: list[float] = []
        actor_stop_sequence_hits_values: list[float] = []

        finish_reason_counts = {
            "eos": 0,
            "gen_budget_exhausted": 0,
            "context_exhausted": 0,
            "max_steps_exhausted": 0,
            "other": 0,
        }

        verifier_outputs_total = 0
        verifier_go_total = 0
        verifier_wait_total = 0
        verifier_request_failed_total = 0
        verifier_request_timeout_total = 0
        verifier_no_valid_decision_total = 0
        verifier_no_valid_decision_final_like_total = 0
        controller_timeout_total = 0
        verifier_wait_conf_count = 0
        verifier_wait_conf_sum = 0.0
        verifier_wait_conf_min = float("inf")
        verifier_wait_conf_max = float("-inf")
        verifier_wait_conf_missing = 0
        verifier_wait_conf_invalid = 0
        verifier_wait_blocked_low_conf = 0
        verifier_policy_blocked_total = 0
        verifier_intervention_attempt_total = 0
        verifier_hint_inserted_total = 0
        verifier_hint_not_inserted_total = 0
        verifier_skipped_no_insert_anchor_total = 0
        hint_skipped_late_stage_total = 0

        cf_event_total = 0
        cf_event_with_delta = 0
        cf_event_truncated = 0
        cf_delta_values: list[float] = []
        cf_diff_values: list[float] = []
        cf_cost_values: list[float] = []
        cf_delta_values_untrunc: list[float] = []
        cf_event_response_len_values: list[float] = []
        cf_event_hint_len_values: list[float] = []
        control_sample_count = 0
        exp_sample_count = 0
        stream_unknown_count = 0

        strict_dual_stream = str(os.environ.get("COGRPO_STRICT_DUAL_STREAM", "1") or "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        strict_dual_stream_mismatch = str(
            os.environ.get("COGRPO_STRICT_DUAL_STREAM_MISMATCH", "0") or "0"
        ).strip().lower() in ("1", "true", "yes", "y", "on")

        def _env_int(name: str, fallback: int = 0) -> int:
            raw = os.environ.get(name, "")
            if raw is None:
                return int(fallback)
            txt = str(raw).strip()
            if txt == "":
                return int(fallback)
            try:
                return int(txt)
            except Exception:
                return int(fallback)

        control_k_cfg = max(0, _env_int("COGRPO_CONTROL_K", _env_int("CO_GRPO_CONTROL_K", 0)))
        exp_k_cfg_raw = _env_int("COGRPO_EXP_K", _env_int("CO_GRPO_EXP_K", 0))
        exp_k_cfg = max(0, exp_k_cfg_raw)
        dual_stream_cfg_enabled = bool(control_k_cfg > 0 or exp_k_cfg > 0)

        def _normalize_stream_tag(raw_tag: object) -> str:
            try:
                tag = str(raw_tag or "").strip().lower()
            except Exception:
                tag = ""
            if tag in ("control", "exp"):
                return tag
            return ""

        def _resolve_cogrpo_stream(data_item) -> tuple[str, str, str]:
            """Resolve CoGRPO stream tag with rollout-extra priority.

            In some pipelines, `data.data.extra_info["cogrpo_stream"]` can be stale
            while rollout-side `env.rollout.extra_info["cogrpo_stream"]` reflects the
            actual stream used by by_step intervention. Prefer rollout info first.
            """
            rollout_tag = ""
            data_tag = ""
            try:
                rollout_extra = getattr(getattr(data_item.env, "rollout", None), "extra_info", None)
                if isinstance(rollout_extra, dict):
                    rollout_tag = _normalize_stream_tag(
                        rollout_extra.get("cogrpo_stream_rollout", rollout_extra.get("cogrpo_stream", ""))
                    )
            except Exception:
                pass
            try:
                data_extra = getattr(data_item.data, "extra_info", None)
                if isinstance(data_extra, dict):
                    data_tag = _normalize_stream_tag(data_extra.get("cogrpo_stream_data", data_extra.get("cogrpo_stream", "")))
            except Exception:
                pass

            if rollout_tag and data_tag and rollout_tag != data_tag:
                msg = (
                    "[CoGRPO][DualStream] stream tag mismatch between rollout/data "
                    f"(rollout={rollout_tag}, data={data_tag}) action_id={getattr(data_item.uid, 'action_id', '<unknown>')}"
                )
                repaired = False
                try:
                    data_extra = getattr(data_item.data, "extra_info", None)
                    if isinstance(data_extra, dict):
                        data_extra["cogrpo_stream_data"] = rollout_tag
                        data_extra["cogrpo_stream"] = rollout_tag
                        repaired = True
                except Exception:
                    repaired = False
                if strict_dual_stream and strict_dual_stream_mismatch and (not repaired):
                    raise RuntimeError(msg)
                if repaired:
                    self.logger.warning(f"{msg}; repaired_by=rollout_tag")
                    data_tag = rollout_tag
                else:
                    self.logger.warning(msg)

            resolved = rollout_tag or data_tag
            if (not resolved) and dual_stream_cfg_enabled and strict_dual_stream:
                raise RuntimeError(
                    "[CoGRPO][DualStream] Missing stream tag in rollout/data while dual-stream is enabled. "
                    f"action_id={getattr(data_item.uid, 'action_id', '<unknown>')}"
                )
            return resolved, rollout_tag, data_tag

        for j, group in enumerate(data_groups):
            if not is_valid_for_training(group):
                self.logger.error(f"Skip one data group {group} due to rollout failed or empty response.")
                continue

            multimodal_train_info = multimodal_train_infos[j] if is_multimodal else None
            prompt_ids = group[0].data.extra_info["train_prompt_ids"]

            rewards_all = [float(data.env.judger.reward["score"]) for data in group]
            group_k = int(len(group))

            # Detect Co-GRPO dual streams (CONTROL vs EXP) tagged by DataFlow.
            control_indices: list[int] = []
            exp_indices: list[int] = []
            stream_info_by_index: list[tuple[str, str, str]] = []
            for idx, data in enumerate(group):
                stream_info = _resolve_cogrpo_stream(data)
                stream, _, _ = stream_info
                stream_info_by_index.append(stream_info)
                if stream == "control":
                    control_indices.append(int(idx))
                    control_sample_count += 1
                elif stream == "exp":
                    exp_indices.append(int(idx))
                    exp_sample_count += 1
                else:
                    stream_unknown_count += 1
                    exp_indices.append(int(idx))

            if dual_stream_cfg_enabled:
                expected_control_k = min(int(control_k_cfg), int(group_k))
                expected_exp_k = int(exp_k_cfg) if int(exp_k_cfg) > 0 else int(group_k - expected_control_k)
                if expected_control_k + expected_exp_k != int(group_k):
                    msg = (
                        "[CoGRPO][DualStream] Configured split does not match group size: "
                        f"control_k={expected_control_k}, exp_k={expected_exp_k}, group_k={group_k}."
                    )
                    if strict_dual_stream:
                        raise RuntimeError(msg)
                    self.logger.warning(msg)
                if strict_dual_stream and (
                    len(control_indices) != expected_control_k or len(exp_indices) != expected_exp_k
                ):
                    raise RuntimeError(
                        "[CoGRPO][DualStream] Observed split mismatch under strict mode: "
                        f"observed(control={len(control_indices)}, exp={len(exp_indices)}) vs "
                        f"expected(control={expected_control_k}, exp={expected_exp_k}) "
                        f"action_id={group[0].uid.action_id}"
                    )

            if control_indices:
                control_rewards_list.extend([float(rewards_all[i]) for i in control_indices])
            if exp_indices:
                exp_rewards_list.extend([float(rewards_all[i]) for i in exp_indices])
            if control_indices and exp_indices and len(control_indices) == len(exp_indices):
                relative_rewards_list.extend(
                    [float(rewards_all[exp_indices[i]] - rewards_all[control_indices[i]]) for i in range(len(exp_indices))]
                )

            def _compute_adv(reward_vec: torch.Tensor) -> torch.Tensor:
                k = int(reward_vec.numel())
                if k <= 0:
                    return reward_vec
                if use_grpo:
                    if k > 1:
                        mean = reward_vec.mean(0)
                        try:
                            std = reward_vec.std(0, unbiased=True)
                        except TypeError:
                            try:
                                std = reward_vec.std(0, correction=1)
                            except TypeError:
                                std = reward_vec.std(0)
                        return (reward_vec - mean) / (std + 1e-6)
                    return reward_vec

                # RLOO baseline (historical behavior)
                if k > 1:
                    baseline = (reward_vec.sum(0) - reward_vec) / (k - 1)
                    return reward_vec - baseline
                return reward_vec

            selected_indices: list[int] = list(range(group_k))
            selected_advantages: torch.Tensor = torch.tensor([], dtype=torch.float32)
            adv_by_index: list[float | None] = [None] * group_k

            if use_co_grpo and control_indices and exp_indices:
                if update_both_streams:
                    rewards_t = torch.tensor(rewards_all, dtype=torch.float32)
                    advantages_t = _compute_adv(rewards_t)
                    selected_indices = list(range(group_k))
                    selected_advantages = advantages_t
                    for idx in selected_indices:
                        adv_by_index[idx] = float(advantages_t[idx].item())
                else:
                    if len(control_indices) != len(exp_indices):
                        raise ValueError(
                            f"[CoGRPO][DualStream] control_k != exp_k for action_id={group[0].uid.action_id}: "
                            f"control_k={len(control_indices)} exp_k={len(exp_indices)}. "
                            "Please set COGRPO_CONTROL_K=COGRPO_EXP_K."
                        )
                    control_rewards_t = torch.tensor([rewards_all[i] for i in control_indices], dtype=torch.float32)
                    exp_rewards_t = torch.tensor([rewards_all[i] for i in exp_indices], dtype=torch.float32)
                    control_adv_t = _compute_adv(control_rewards_t)
                    exp_adv_t = _compute_adv(exp_rewards_t)
                    policy_adv_t = (1.0 - float(effective_control_weight)) * exp_adv_t + float(effective_control_weight) * control_adv_t
                    selected_indices = list(exp_indices)
                    selected_advantages = policy_adv_t
                    for pos, idx in enumerate(selected_indices):
                        adv_by_index[idx] = float(policy_adv_t[pos].item())
            else:
                rewards_t = torch.tensor(rewards_all, dtype=torch.float32)
                advantages_t = _compute_adv(rewards_t)
                selected_indices = list(range(group_k))
                selected_advantages = advantages_t
                for idx in selected_indices:
                    adv_by_index[idx] = float(advantages_t[idx].item())

            # Log reward stats only for samples used in actor update.
            rewards_list.extend([float(rewards_all[i]) for i in selected_indices])

            # downsample for overlong filter
            # overlong_mask = []
            # for d in group:
            #     if d.env.rollout.finish_reason == "stop":
            #         overlong_mask.append(1.0)
            #     else:
            #         overlong_mask.append(1.0 if random.random() < 0.5 else 0.0)
            # overlong_mask = torch.tensor(overlong_mask, dtype=torch.float32)

            # # overlong filter RLOO
            # masked_rewards = rewards * overlong_mask
            # valid_k = overlong_mask.sum()
            # baseline = (masked_rewards.sum() - masked_rewards) / torch.clamp(valid_k - 1, min=1.0)
            # advantages = (rewards - baseline) * overlong_mask

            # ## OPO
            # lengths = torch.tensor([len(d.env.rollout.response_ids) for d in group], dtype=torch.float32)
            # baseline = (rewards * lengths).sum() / lengths.sum()
            # advantages = rewards - baseline

            entropy_control_cfg = self._rl_trainer_cfg.train_worker_config.loss_cfg.entropy_control_cfg or {}
            if entropy_control_cfg.get("control_level") == "group" and selected_indices:
                sum_entropy = None
                total_tokens = 0
                for idx in selected_indices:
                    raw_logprobs = group[idx].env.rollout.logprobs
                    logprobs = (
                        raw_logprobs
                        if isinstance(raw_logprobs, torch.Tensor)
                        else torch.tensor(raw_logprobs, dtype=torch.float32)
                    )
                    entropy = -(logprobs).sum()
                    sum_entropy = entropy if sum_entropy is None else sum_entropy + entropy
                    try:
                        response_ids = group[idx].env.rollout.response_ids
                        if isinstance(response_ids, torch.Tensor):
                            total_tokens += int(response_ids.numel())
                        else:
                            total_tokens += int(len(cast(list, response_ids)))
                    except Exception:
                        pass
                avg_entropy = sum_entropy / max(total_tokens, 1)
                entropy_upper_bound = entropy_control_cfg.get("upper_bound", 0.75)
                entropy_lower_bound = entropy_control_cfg.get("lower_bound", 0)
                tau_upper = entropy_control_cfg.get("tau_upper", 0)
                tau_lower = entropy_control_cfg.get("tau_lower", 0)  # 越大scale下降的越慢, 置0则不控制平滑
                upper_scale = entropy_control_cfg.get("upper_scale", 0.2)  # 熵高分支的最小缩放
                lower_scale = entropy_control_cfg.get("lower_scale", 0.5)  # 熵低分支的最小缩放
                if avg_entropy > entropy_upper_bound:
                    # 熵高：减弱负优势，保留正优势
                    delta = (avg_entropy - entropy_upper_bound) / entropy_upper_bound
                    tau_upper = tau_upper if tau_upper != 0 else 1e-8
                    s = torch.sigmoid(torch.tensor(-delta / tau_upper, device=selected_advantages.device)).item()
                    entropy_coeff = upper_scale + (1 - upper_scale) * s / 0.5
                    selected_advantages = torch.where(
                        selected_advantages < 0, selected_advantages * entropy_coeff, selected_advantages
                    )
                elif avg_entropy < entropy_lower_bound:
                    # 熵低：减弱正优势，保留负优势
                    delta = (entropy_lower_bound - avg_entropy) / entropy_lower_bound
                    tau_lower = tau_lower if tau_lower != 0 else 1e-8
                    s = torch.sigmoid(torch.tensor(-delta / tau_lower, device=selected_advantages.device)).item()
                    entropy_coeff = lower_scale + (1 - lower_scale) * s / 0.5
                    selected_advantages = torch.where(
                        selected_advantages > 0, selected_advantages * entropy_coeff, selected_advantages
                    )

                for pos, idx in enumerate(selected_indices):
                    adv_by_index[idx] = float(selected_advantages[pos].item())

            # # pass@k
            # def calc_passk_adv(val, k=4):
            #     from scipy.special import comb

            #     c = len(np.where(val==1)[0])
            #     # print(c)
            #     n = len(val)
            #     rho = 1 - comb(n-c, k) / comb(n, k)
            #     # print(rho)
            #     sigma = np.sqrt(rho * (1 - rho))
            #     adv_p = (1 - rho) / (sigma + 1e-6)
            #     # print(adv_p)
            #     adv_n = (1 - rho - comb(n-c-1, k-1)/comb(n-1,k-1)) / (sigma + 1e-6)
            #     new_val = np.where(val==1, adv_p, val)
            #     new_val = np.where(new_val==0, adv_n, new_val)
            #     return new_val
            # val = rewards.numpy()
            # new_reward = np.zeros_like(val, dtype=np.float32)
            # new_reward[val>0] = 1
            # advantages = torch.tensor(calc_passk_adv(new_reward, k=4))

            for idx in selected_indices:
                if adv_by_index[idx] is None:
                    raise RuntimeError(
                        f"[CoGRPO] Missing advantage for action_id={group[0].uid.action_id} idx={idx}. "
                        "This is a bug: selected sample must have computed advantage."
                    )
                adv_scalar = float(adv_by_index[idx] or 0.0)

                item = group[idx].env.rollout.response
                logprobs = None
                if group[idx].env.rollout.response_ids is not None:
                    response_ids = group[idx].env.rollout.response_ids
                    if isinstance(response_ids, torch.Tensor):
                        response_ids = response_ids.flatten().tolist()
                    logprobs = group[idx].env.rollout.logprobs
                    assert len(logprobs) == len(response_ids), f"{len(logprobs)} vs {len(response_ids)}"
                    # 只有 response 部分有 logprobs, 需要前面追加
                    logprobs = [0] * (len(prompt_ids) - 1) + logprobs
                else:
                    response_ids = self.tokenizer(item, return_tensors="pt")["input_ids"].flatten().tolist()
                # 返回的 routed_experts 不包括 eos 的值，实际上也不需要，需要减一
                input_ids = prompt_ids + response_ids[:-1]

                prompt_len_list.append(len(prompt_ids))
                response_len_list.append(len(response_ids))
                advantages_list.extend([adv_scalar] * len(response_ids))

                extra_rollout = group[idx].env.rollout.extra_info or {}
                if 0 <= int(idx) < len(stream_info_by_index):
                    stream_tag = stream_info_by_index[int(idx)][0]
                else:
                    stream_tag, _, _ = _resolve_cogrpo_stream(group[idx])
                is_exp_stream = stream_tag != "control"

                if is_exp_stream:
                    actor_reward_values.append(float(rewards_all[idx]))
                    actor_response_len_values.append(float(len(response_ids)))

                try:
                    hint_len_val = float(extra_rollout.get("cogrpo_hint_len", 0.0) or 0.0)
                except Exception:
                    hint_len_val = 0.0
                if is_exp_stream:
                    actor_hint_len_values.append(float(hint_len_val))

                try:
                    num_int_val = float(extra_rollout.get("cogrpo_num_interventions", 0.0) or 0.0)
                except Exception:
                    num_int_val = 0.0
                if is_exp_stream:
                    actor_num_interventions_values.append(float(num_int_val))
                    if float(num_int_val) > 0.0:
                        actor_intervened_reward_values.append(float(rewards_all[idx]))
                    else:
                        actor_non_intervened_reward_values.append(float(rewards_all[idx]))

                first_step_raw = extra_rollout.get("cogrpo_first_step_tokens_len", None)
                if is_exp_stream and first_step_raw is not None:
                    try:
                        first_step_tokens_len_values.append(float(first_step_raw))
                    except Exception:
                        pass
                step_count_raw = extra_rollout.get("cogrpo_step_count", None)
                if is_exp_stream and step_count_raw is not None:
                    try:
                        actor_step_count_values.append(float(step_count_raw))
                    except Exception:
                        pass
                stop_sequence_hits_raw = extra_rollout.get("cogrpo_stop_sequence_hits", None)
                if is_exp_stream and stop_sequence_hits_raw is not None:
                    try:
                        actor_stop_sequence_hits_values.append(float(stop_sequence_hits_raw))
                    except Exception:
                        pass

                finish_reason = str(extra_rollout.get("cogrpo_last_finish_reason") or "")
                context_exhausted = bool(extra_rollout.get("cogrpo_context_exhausted", False))
                is_context_finish = context_exhausted or finish_reason in ("context_exhausted", "context_budget_exhausted")
                if is_exp_stream:
                    if finish_reason == "eos":
                        finish_reason_counts["eos"] += 1
                    elif finish_reason == "gen_budget_exhausted":
                        finish_reason_counts["gen_budget_exhausted"] += 1
                    elif is_context_finish:
                        finish_reason_counts["context_exhausted"] += 1
                    elif finish_reason == "max_steps_exhausted":
                        finish_reason_counts["max_steps_exhausted"] += 1
                    else:
                        finish_reason_counts["other"] += 1

                def _extra_int(key: str, default: int = 0) -> int:
                    try:
                        return int(extra_rollout.get(key, default) or default)
                    except Exception:
                        return int(default)

                def _extra_float(key: str, default: float = 0.0) -> float:
                    try:
                        return float(extra_rollout.get(key, default) or default)
                    except Exception:
                        return float(default)

                verifier_outputs_total += _extra_int("cogrpo_verifier_outputs")
                verifier_go_total += _extra_int("cogrpo_verifier_go_total")
                verifier_wait_total += _extra_int("cogrpo_verifier_wait_total")
                verifier_request_failed_total += _extra_int("cogrpo_verifier_request_failed_total")
                verifier_request_timeout_total += _extra_int("cogrpo_verifier_request_timeout_total")
                try:
                    if bool(group[idx].env.rollout.extra_info.get("controller_timeout", False)):
                        controller_timeout_total += 1
                except Exception:
                    pass
                verifier_no_valid_decision_total += _extra_int("cogrpo_verifier_no_valid_decision")
                verifier_no_valid_decision_final_like_total += _extra_int("cogrpo_verifier_no_valid_decision_final_like")

                wait_conf_count_sample = _extra_int("cogrpo_verifier_wait_conf_count")
                wait_conf_mean_sample = _extra_float("cogrpo_verifier_wait_conf_mean")
                verifier_wait_conf_count += int(wait_conf_count_sample)
                verifier_wait_conf_sum += float(wait_conf_mean_sample) * float(wait_conf_count_sample)
                if int(wait_conf_count_sample) > 0:
                    wait_conf_min = extra_rollout.get("cogrpo_verifier_wait_conf_min", None)
                    if wait_conf_min is not None:
                        try:
                            v = float(wait_conf_min)
                            verifier_wait_conf_min = min(verifier_wait_conf_min, v)
                        except Exception:
                            pass
                    wait_conf_max = extra_rollout.get("cogrpo_verifier_wait_conf_max", None)
                    if wait_conf_max is not None:
                        try:
                            v = float(wait_conf_max)
                            verifier_wait_conf_max = max(verifier_wait_conf_max, v)
                        except Exception:
                            pass
                verifier_wait_conf_missing += _extra_int("cogrpo_verifier_wait_conf_missing")
                verifier_wait_conf_invalid += _extra_int("cogrpo_verifier_wait_conf_invalid")
                verifier_wait_blocked_low_conf += _extra_int("cogrpo_verifier_wait_blocked_low_conf")
                verifier_policy_blocked_total += _extra_int("cogrpo_verifier_policy_blocked_total")
                verifier_intervention_attempt_total += _extra_int("cogrpo_verifier_intervention_attempt_total")
                verifier_hint_inserted_total += _extra_int("cogrpo_verifier_hint_inserted_total")
                verifier_hint_not_inserted_total += _extra_int("cogrpo_verifier_hint_not_inserted_total")
                verifier_skipped_no_insert_anchor_total += _extra_int("cogrpo_verifier_skipped_no_insert_anchor")
                hint_skipped_late_stage_total += _extra_int("cogrpo_hint_skipped_late_stage")

                events = extra_rollout.get("cogrpo_verifier_events") or []
                if isinstance(events, list) and events:
                    is_truncated_sample = is_context_finish or finish_reason in (
                        "gen_budget_exhausted",
                        "max_steps_exhausted",
                    )
                    for ev in events:
                        if not isinstance(ev, dict):
                            continue
                        cf_event_total += 1
                        if is_truncated_sample:
                            cf_event_truncated += 1
                        step_reward = ev.get("step_reward", None)
                        if step_reward is None:
                            continue
                        try:
                            delta = float(step_reward)
                        except Exception:
                            continue
                        cf_event_with_delta += 1
                        cf_delta_values.append(float(delta))
                        cf_event_response_len_values.append(float(len(response_ids)))
                        try:
                            cf_event_hint_len_values.append(float(ev.get("hint_token_count") or 0.0))
                        except Exception:
                            cf_event_hint_len_values.append(0.0)
                        if not is_truncated_sample:
                            cf_delta_values_untrunc.append(float(delta))
                        try:
                            if ev.get("step_diff", None) is not None:
                                cf_diff_values.append(float(ev.get("step_diff")))
                        except Exception:
                            pass
                        try:
                            if ev.get("step_cost", None) is not None:
                                cf_cost_values.append(float(ev.get("step_cost")))
                        except Exception:
                            pass

                # Prefer token-level labels from rollout (e.g., Co-GRPO hints => -100)
                response_labels = None
                if group[idx].env.rollout.labels is not None:
                    response_labels = group[idx].env.rollout.labels
                    if isinstance(response_labels, torch.Tensor):
                        response_labels = response_labels.flatten().tolist()
                    else:
                        response_labels = list(response_labels)
                    if len(response_labels) != len(response_ids):
                        self.logger.warning(
                            f"Found rollout.labels with mismatched length: labels={len(response_labels)} vs "
                            f"response_ids={len(response_ids)}. Fallback to response_ids."
                        )
                        response_labels = None
                if response_labels is None:
                    response_labels = response_ids

                shifted_labels = [-100] * (len(prompt_ids) - 1) + response_labels
                assert len(input_ids) <= pack_max_length, (
                    f"{len(input_ids)} vs {pack_max_length}, input_ids: {len(input_ids)}, response_ids: {len(response_ids)}"
                )
                input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)
                shifted_labels = torch.tensor(shifted_labels, dtype=torch.int64).unsqueeze(0)

                all_input_ids.append(input_ids)
                all_response_ids.append(response_ids)
                all_shifted_labels.append(shifted_labels)
                all_advantages.append(adv_scalar)
                all_multimodal_train_infos.append(multimodal_train_info)

                if logprobs is not None:
                    rollout_logprobs = torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0)
                    assert rollout_logprobs.size() == shifted_labels.size(), (
                        f"{rollout_logprobs.size()} vs {shifted_labels.size()}"
                    )
                    all_rollout_logprobs.append(rollout_logprobs)
                else:
                    rollout_logprobs = None
                    all_rollout_logprobs.append(None)

                if "routed_experts" in group[idx].env.rollout.extra_info:
                    routed_experts = group[idx].env.rollout.extra_info.pop("routed_experts")  # n,layer*expert
                    all_routed_experts.append(routed_experts)  # n,layer,expert
                else:
                    assert not self._enable_return_routed_experts, (
                        "enable_return_routed_experts is True, but no routed_experts found in rollout extra_info."
                    )
                    all_routed_experts.append(None)

        num_samples = len(all_input_ids)
        indices = list(range(num_samples))
        random.shuffle(indices)

        data_batches = []
        for i in indices:
            seq_ctx = get_train_seq_ctx(all_input_ids[i], all_multimodal_train_infos[i], len(all_response_ids[i]) - 1)
            data_dict = {
                "seq_ctx": seq_ctx,
                "shifted_labels": all_shifted_labels[i],
                "advantages": all_advantages[i],
                "rollout_logprobs": all_rollout_logprobs[i],
            }
            if all_routed_experts[i] is not None:
                seq_ctx.rollout_routed_experts = all_routed_experts[i]

            data_batches.append(data_dict)

        rewards_arr = torch.tensor(rewards_list).float() if rewards_list else torch.tensor([0.0]).float()
        advantages_arr = torch.tensor(advantages_list).float() if advantages_list else torch.tensor([0.0]).float()
        prompt_len_arr = torch.tensor(prompt_len_list).float() if prompt_len_list else torch.tensor([0.0]).float()
        response_len_arr = (
            torch.tensor(response_len_list).float() if response_len_list else torch.tensor([0.0]).float()
        )

        info_dict = {
            "batch_size": len(rewards_list),
            "rewards/mean": rewards_arr.mean().item(),
            "rewards/min": rewards_arr.min().item(),
            "rewards/max": rewards_arr.max().item(),
            "advantages/mean": advantages_arr.mean().item(),
            "advantages/min": advantages_arr.min().item(),
            "advantages/max": advantages_arr.max().item(),
            "advantages/std": advantages_arr.std(unbiased=False).item(),
            "advantages/pos_ratio": (advantages_arr > 0).float().mean().item(),
            "response_len/mean": response_len_arr.mean().item(),
            "response_len/min": response_len_arr.min().item(),
            "response_len/max": response_len_arr.max().item(),
            "response_len/std": response_len_arr.std(unbiased=False).item(),
            "prompt_len/mean": prompt_len_arr.mean().item(),
            "prompt_len/min": prompt_len_arr.min().item(),
            "prompt_len/max": prompt_len_arr.max().item(),
        }

        # Co-GRPO dual-stream diagnostics (verl-aligned naming).
        #
        # NOTE:
        # - `rewards/*` above are for samples used in actor update (exp-only by default).
        # - Here we additionally log CONTROL/EXP/RELATIVE reward stats for monitoring.
        def _safe_std(x: torch.Tensor) -> float:
            if int(x.numel()) <= 1:
                return 0.0
            try:
                return float(x.std(unbiased=True).item())
            except TypeError:
                try:
                    return float(x.std(correction=1).item())
                except TypeError:
                    return float(x.std().item())

        def _safe_mean(vals: list[float]) -> float:
            if not vals:
                return 0.0
            valid_vals = []
            for v in vals:
                try:
                    fv = float(v)
                except Exception:
                    continue
                if math.isfinite(fv):
                    valid_vals.append(fv)
            if not valid_vals:
                return 0.0
            return float(sum(valid_vals) / max(1, len(valid_vals)))

        def _safe_ratio(numer: float, denom: float) -> float:
            if float(denom) <= 0:
                return 0.0
            return float(numer) / float(denom)

        def _safe_corr(x_vals: list[float], y_vals: list[float]) -> float:
            if len(x_vals) < 2 or len(y_vals) < 2 or len(x_vals) != len(y_vals):
                return 0.0
            valid_pairs: list[tuple[float, float]] = []
            for xv, yv in zip(x_vals, y_vals):
                try:
                    fx = float(xv)
                    fy = float(yv)
                except Exception:
                    continue
                if math.isfinite(fx) and math.isfinite(fy):
                    valid_pairs.append((fx, fy))
            if len(valid_pairs) < 2:
                return 0.0
            x = torch.tensor([p[0] for p in valid_pairs], dtype=torch.float32)
            y = torch.tensor([p[1] for p in valid_pairs], dtype=torch.float32)
            xm = x.mean()
            ym = y.mean()
            vx = ((x - xm) ** 2).mean()
            vy = ((y - ym) ** 2).mean()
            denom = torch.sqrt(vx * vy)
            if float(denom.item()) <= 1e-12:
                return 0.0
            corr = ((x - xm) * (y - ym)).mean() / (denom + 1e-12)
            return float(torch.clamp(corr, min=-1.0, max=1.0).item())

        info_dict["co_grpo/effective_control_weight"] = float(effective_control_weight)
        info_dict["co_grpo/effective_exp_weight"] = float(1.0 - float(effective_control_weight))
        info_dict["co_grpo/training_progress"] = float(training_progress)
        info_dict["co_grpo/control_sample_count"] = float(control_sample_count)
        info_dict["co_grpo/exp_sample_count"] = float(exp_sample_count)
        info_dict["co_grpo/stream_unknown_count"] = float(stream_unknown_count)
        info_dict["co_grpo/actor_update_sample_count"] = float(len(rewards_list))

        if control_rewards_list:
            control_arr = torch.tensor(control_rewards_list, dtype=torch.float32)
            info_dict["co_grpo/control_reward_mean"] = float(control_arr.mean().item())
            info_dict["co_grpo/control_reward_std"] = _safe_std(control_arr)
        if exp_rewards_list:
            exp_arr = torch.tensor(exp_rewards_list, dtype=torch.float32)
            info_dict["co_grpo/exp_reward_mean"] = float(exp_arr.mean().item())
            info_dict["co_grpo/exp_reward_std"] = _safe_std(exp_arr)
        if relative_rewards_list:
            rel_arr = torch.tensor(relative_rewards_list, dtype=torch.float32)
            info_dict["co_grpo/relative_reward_mean"] = float(rel_arr.mean().item())
            info_dict["co_grpo/relative_reward_std"] = _safe_std(rel_arr)
            info_dict["co_grpo/relative_reward_positive_ratio"] = float((rel_arr > 0).float().mean().item())

        actor_sample_count = int(len(actor_reward_values))
        if actor_sample_count > 0:
            info_dict["co_grpo/verifier_help_rate"] = float(
                sum(1.0 for v in actor_num_interventions_values if float(v) > 0.0) / float(actor_sample_count)
            )
            info_dict["co_grpo/exp_num_interventions_mean"] = _safe_mean(actor_num_interventions_values)
            info_dict["co_grpo/exp_num_interventions_max"] = float(max(actor_num_interventions_values))
            info_dict["co_grpo/exp_hint_len_mean"] = _safe_mean(actor_hint_len_values)
            info_dict["co_grpo/exp_finish_eos_rate"] = _safe_ratio(finish_reason_counts["eos"], actor_sample_count)
            info_dict["co_grpo/exp_finish_gen_budget_exhausted_rate"] = _safe_ratio(
                finish_reason_counts["gen_budget_exhausted"], actor_sample_count
            )
            info_dict["co_grpo/exp_finish_context_exhausted_rate"] = _safe_ratio(
                finish_reason_counts["context_exhausted"], actor_sample_count
            )
            info_dict["co_grpo/exp_finish_max_steps_exhausted_rate"] = _safe_ratio(
                finish_reason_counts["max_steps_exhausted"], actor_sample_count
            )
            info_dict["co_grpo/exp_finish_other_rate"] = _safe_ratio(finish_reason_counts["other"], actor_sample_count)
            info_dict["co_grpo/corr_reward_response_len"] = _safe_corr(actor_reward_values, actor_response_len_values)
            info_dict["co_grpo/corr_reward_hint_len"] = _safe_corr(actor_reward_values, actor_hint_len_values)
            info_dict["co_grpo/corr_reward_num_interventions"] = _safe_corr(
                actor_reward_values, actor_num_interventions_values
            )
            if first_step_tokens_len_values:
                info_dict["co_grpo/exp_first_step_tokens_len_mean"] = _safe_mean(first_step_tokens_len_values)
            if actor_step_count_values:
                info_dict["co_grpo/exp_avg_steps"] = _safe_mean(actor_step_count_values)
            if actor_stop_sequence_hits_values:
                info_dict["co_grpo/exp_stop_sequence_hits_mean"] = _safe_mean(actor_stop_sequence_hits_values)
            if actor_step_count_values and actor_stop_sequence_hits_values:
                total_steps = sum(float(v) for v in actor_step_count_values if math.isfinite(float(v)))
                total_stop_hits = sum(float(v) for v in actor_stop_sequence_hits_values if math.isfinite(float(v)))
                info_dict["co_grpo/exp_stop_sequence_hit_rate"] = _safe_ratio(total_stop_hits, total_steps)
            if actor_intervened_reward_values:
                info_dict["co_grpo/hint_effective_rate"] = _safe_ratio(
                    sum(1.0 for v in actor_intervened_reward_values if float(v) > 0.0),
                    len(actor_intervened_reward_values),
                )
                info_dict["co_grpo/hint_intervened_reward_mean"] = _safe_mean(actor_intervened_reward_values)
            if actor_intervened_reward_values and actor_non_intervened_reward_values:
                info_dict["co_grpo/hint_reward_delta_intervened_vs_nohint"] = float(
                    _safe_mean(actor_intervened_reward_values) - _safe_mean(actor_non_intervened_reward_values)
                )

        if verifier_outputs_total > 0:
            info_dict["co_grpo/no_decision_rate"] = _safe_ratio(verifier_no_valid_decision_total, verifier_outputs_total)
            info_dict["co_grpo/no_decision_final_like_rate"] = _safe_ratio(
                verifier_no_valid_decision_final_like_total, verifier_outputs_total
            )
            info_dict["co_grpo/verifier_wait_rate"] = _safe_ratio(verifier_wait_total, verifier_outputs_total)
            info_dict["co_grpo/verifier_go_rate"] = _safe_ratio(verifier_go_total, verifier_outputs_total)
        else:
            info_dict["co_grpo/no_decision_rate"] = 0.0
            info_dict["co_grpo/no_decision_final_like_rate"] = 0.0
            info_dict["co_grpo/verifier_wait_rate"] = 0.0
            info_dict["co_grpo/verifier_go_rate"] = 0.0
        info_dict["co_grpo/verifier_outputs"] = float(verifier_outputs_total)
        info_dict["co_grpo/verifier_go_total"] = float(verifier_go_total)
        info_dict["co_grpo/verifier_wait_total"] = float(verifier_wait_total)
        info_dict["co_grpo/verifier_request_failed_total"] = float(verifier_request_failed_total)
        info_dict["co_grpo/verifier_request_timeout_total"] = float(verifier_request_timeout_total)
        info_dict["co_grpo/verifier_request_failed_rate"] = _safe_ratio(
            verifier_request_failed_total, verifier_outputs_total + verifier_request_failed_total
        )
        info_dict["co_grpo/verifier_request_timeout_rate"] = _safe_ratio(
            verifier_request_timeout_total, verifier_outputs_total + verifier_request_failed_total
        )
        info_dict["co_grpo/controller_timeout_total"] = float(controller_timeout_total)
        info_dict["co_grpo/controller_timeout_rate"] = _safe_ratio(controller_timeout_total, actor_sample_count)
        info_dict["co_grpo/verifier_no_valid_decision"] = float(verifier_no_valid_decision_total)
        info_dict["co_grpo/verifier_no_valid_decision_final_like"] = float(verifier_no_valid_decision_final_like_total)
        info_dict["co_grpo/verifier_wait_conf_coverage"] = _safe_ratio(verifier_wait_conf_count, verifier_wait_total)
        info_dict["co_grpo/verifier_wait_blocked_low_conf"] = float(verifier_wait_blocked_low_conf)
        info_dict["co_grpo/verifier_wait_blocked_low_conf_rate"] = _safe_ratio(
            verifier_wait_blocked_low_conf, verifier_wait_total
        )
        info_dict["co_grpo/verifier_policy_blocked_total"] = float(verifier_policy_blocked_total)
        info_dict["co_grpo/verifier_policy_blocked_rate"] = _safe_ratio(
            verifier_policy_blocked_total, verifier_outputs_total
        )
        info_dict["co_grpo/verifier_intervention_attempt_total"] = float(verifier_intervention_attempt_total)
        info_dict["co_grpo/verifier_hint_inserted_total"] = float(verifier_hint_inserted_total)
        info_dict["co_grpo/verifier_hint_not_inserted_total"] = float(verifier_hint_not_inserted_total)
        info_dict["co_grpo/hint_insert_success_rate"] = _safe_ratio(
            verifier_hint_inserted_total, verifier_intervention_attempt_total
        )
        info_dict["co_grpo/hint_insert_not_applied_rate"] = _safe_ratio(
            verifier_hint_not_inserted_total, verifier_intervention_attempt_total
        )
        info_dict["co_grpo/verifier_skipped_no_insert_anchor"] = float(verifier_skipped_no_insert_anchor_total)
        verifier_anchor_denominator = verifier_outputs_total + verifier_skipped_no_insert_anchor_total
        info_dict["co_grpo/verifier_skipped_no_insert_anchor_rate"] = _safe_ratio(
            verifier_skipped_no_insert_anchor_total, verifier_anchor_denominator
        )
        info_dict["co_grpo/hint_skipped_late_stage_rate"] = _safe_ratio(
            hint_skipped_late_stage_total, verifier_outputs_total
        )
        info_dict["co_grpo/hint_skipped_late_stage"] = float(hint_skipped_late_stage_total)
        if verifier_wait_conf_count > 0:
            info_dict["co_grpo/verifier_wait_conf_mean"] = float(verifier_wait_conf_sum / float(verifier_wait_conf_count))
            if verifier_wait_conf_min != float("inf"):
                info_dict["co_grpo/verifier_wait_conf_min"] = float(verifier_wait_conf_min)
            if verifier_wait_conf_max != float("-inf"):
                info_dict["co_grpo/verifier_wait_conf_max"] = float(verifier_wait_conf_max)
        else:
            info_dict["co_grpo/verifier_wait_conf_mean"] = 0.0
            info_dict["co_grpo/verifier_wait_conf_min"] = 0.0
            info_dict["co_grpo/verifier_wait_conf_max"] = 0.0
        info_dict["co_grpo/verifier_wait_conf_missing"] = float(verifier_wait_conf_missing)
        info_dict["co_grpo/verifier_wait_conf_invalid"] = float(verifier_wait_conf_invalid)

        info_dict["co_grpo/cf_event_total"] = float(cf_event_total)
        info_dict["co_grpo/cf_event_with_delta"] = float(cf_event_with_delta)
        info_dict["co_grpo/cf_event_truncated"] = float(cf_event_truncated)
        if cf_event_total > 0:
            info_dict["co_grpo/cf_missing_ratio"] = _safe_ratio(cf_event_total - cf_event_with_delta, cf_event_total)
            info_dict["co_grpo/cf_trunc_event_ratio"] = _safe_ratio(cf_event_truncated, cf_event_total)
        if cf_delta_values:
            cf_delta_arr = torch.tensor(cf_delta_values, dtype=torch.float32)
            info_dict["co_grpo/cf_delta_mean"] = float(cf_delta_arr.mean().item())
            info_dict["co_grpo/cf_delta_std"] = float(cf_delta_arr.std(unbiased=False).item())
            info_dict["co_grpo/cf_delta_pos_ratio"] = float((cf_delta_arr > 0).float().mean().item())
            info_dict["co_grpo/cf_delta_zero_ratio"] = float((cf_delta_arr.abs() <= 1e-6).float().mean().item())
            info_dict["co_grpo/cf_delta_neg_ratio"] = float((cf_delta_arr < -1e-6).float().mean().item())
            if cf_delta_values_untrunc:
                cf_delta_untr_arr = torch.tensor(cf_delta_values_untrunc, dtype=torch.float32)
                info_dict["co_grpo/cf_delta_mean_untrunc"] = float(cf_delta_untr_arr.mean().item())
                info_dict["co_grpo/cf_delta_pos_ratio_untrunc"] = float((cf_delta_untr_arr > 0).float().mean().item())
                info_dict["co_grpo/cf_delta_zero_ratio_untrunc"] = float(
                    (cf_delta_untr_arr.abs() <= 1e-6).float().mean().item()
                )
                info_dict["co_grpo/cf_delta_neg_ratio_untrunc"] = float((cf_delta_untr_arr < -1e-6).float().mean().item())
            info_dict["co_grpo/corr_cf_delta_response_len"] = _safe_corr(cf_delta_values, cf_event_response_len_values)
            info_dict["co_grpo/corr_cf_delta_hint_len"] = _safe_corr(cf_delta_values, cf_event_hint_len_values)
        if cf_diff_values:
            cf_diff_arr = torch.tensor(cf_diff_values, dtype=torch.float32)
            info_dict["co_grpo/cf_diff_mean"] = float(cf_diff_arr.mean().item())
            info_dict["co_grpo/cf_diff_std"] = float(cf_diff_arr.std(unbiased=False).item())
            info_dict["co_grpo/cf_diff_pos_ratio"] = float((cf_diff_arr > 0).float().mean().item())
            info_dict["co_grpo/cf_diff_zero_ratio"] = float((cf_diff_arr.abs() <= 1e-6).float().mean().item())
            info_dict["co_grpo/cf_diff_neg_ratio"] = float((cf_diff_arr < -1e-6).float().mean().item())
        if cf_cost_values:
            info_dict["co_grpo/cf_cost_mean"] = _safe_mean(cf_cost_values)

        return data_batches, info_dict

    def _prepare_verifier_train_data(self, data_groups, pack_max_length: int):
        """Prepare Co-GRPO verifier event batches for training.

        The rollout side records verifier trajectories per intervention event into
        `env.rollout.extra_info["cogrpo_verifier_events"]`, and the environment attaches
        per-event `advantage` (cf_branch uplift) for training.
        """
        all_input_ids = []
        all_shifted_labels = []
        all_advantages = []
        all_rollout_logprobs = []

        advantages_list = []
        prompt_len_list = []
        response_len_list = []

        skip_truncated_verifier_env = os.environ.get("COGRPO_VERIFIER_SKIP_TRUNCATED", "0")
        skip_truncated_verifier = str(skip_truncated_verifier_env).strip().lower() in ("1", "true", "yes", "y", "on")
        trunc_finish_reasons = {
            "gen_budget_exhausted",
            "context_exhausted",
            "context_budget_exhausted",
            "max_steps_exhausted",
        }
        total_cf_events = 0
        trunc_event_count = 0
        trunc_skip_count = 0
        delta_values_all: list[float] = []
        delta_values_used: list[float] = []
        diff_values_used: list[float] = []
        cost_values_used: list[float] = []
        verifier_event_total = 0
        verifier_event_cf_advantage_used = 0
        verifier_event_equal_share_used = 0
        verifier_event_credit_mode_cf = 0
        verifier_event_credit_mode_equal = 0
        verifier_event_skip_missing_advantage = 0
        verifier_event_skip_missing_parent_reward = 0
        verifier_event_skip_missing_parent_count = 0
        verifier_event_cf_advantage_reconstructed = 0
        verifier_event_old_logprobs_used = 0
        verifier_event_old_logprobs_missing = 0
        verifier_adv_delta_sign_conflict = 0
        verifier_adv_delta_sign_count = 0
        cf_adv_reconstructed_by_event_uid: dict[str, float] = {}

        def _rebuild_cf_group_index(ev: dict, parent_action_id_fallback: str = "") -> str:
            try:
                step_idx = int(ev.get("step_idx") or 0)
            except Exception:
                step_idx = -1
            try:
                prefix_bucket = int(ev.get("prefix_len") or 0) // 2048
            except Exception:
                prefix_bucket = -1
            conf_bucket = -1
            try:
                wait_confidence = float(ev.get("wait_confidence"))
                if not math.isnan(wait_confidence):
                    conf_bucket = int(wait_confidence * 20)
            except Exception:
                conf_bucket = -1
            try:
                hash_bucket = int(ev.get("state_hash_bucket"))
            except Exception:
                hash_bucket = -1
            parent_action_id = str(ev.get("parent_action_id") or parent_action_id_fallback or "")
            return f"{parent_action_id}:{step_idx}:{prefix_bucket}:{conf_bucket}:{hash_bucket}"

        cf_delta_by_group_index: dict[str, list[tuple[str, float]]] = {}
        for group in data_groups:
            if not is_valid_for_training(group):
                continue
            for data in group:
                extra = data.env.rollout.extra_info or {}
                credit_mode_default = str(extra.get("cogrpo_verifier_credit_assignment") or "").strip().lower()
                parent_action_id_fallback = str(getattr(data.uid, "action_id", "") or "")
                events = extra.get("cogrpo_verifier_events") or []
                if not isinstance(events, list):
                    continue
                for ev in events:
                    if not isinstance(ev, dict) or ev.get("advantage", None) is not None:
                        continue
                    credit_mode = str(
                        ev.get("verifier_credit_assignment")
                        or ev.get("credit_assignment")
                        or credit_mode_default
                    ).strip().lower()
                    if credit_mode not in ("cf_branch", "cf"):
                        continue
                    event_uid = str(ev.get("event_uid") or "")
                    if not event_uid:
                        continue
                    try:
                        delta = float(ev.get("step_reward"))
                    except Exception:
                        continue
                    group_index = str(ev.get("group_index") or "").strip()
                    if not group_index:
                        group_index = _rebuild_cf_group_index(ev, parent_action_id_fallback)
                    if not group_index:
                        continue
                    cf_delta_by_group_index.setdefault(group_index, []).append((event_uid, delta))

        for group_index, event_items in cf_delta_by_group_index.items():
            deltas = [float(delta) for _, delta in event_items]
            if len(deltas) == 1:
                group_mean = 0.0
                group_std = 1.0
            else:
                group_mean = float(sum(deltas) / float(len(deltas)))
                group_var = float(sum((x - group_mean) ** 2 for x in deltas) / float(len(deltas) - 1))
                group_std = float(math.sqrt(max(group_var, 0.0)))
            for event_uid, delta in event_items:
                cf_adv_reconstructed_by_event_uid[event_uid] = (float(delta) - float(group_mean)) / (
                    float(group_std) + 1e-6
                )

        for group in data_groups:
            if not is_valid_for_training(group):
                continue

            sample_reward_by_observation_id: dict[str, float] = {}
            event_count_by_observation_id: dict[str, int] = {}
            credit_mode_by_observation_id: dict[str, str] = {}

            # First pass: build parent reward and fallback denominator mappings.
            for data in group:
                obs_id = str(getattr(data.uid, "observation_id", "") or "")
                try:
                    sample_reward = float((data.env.judger.reward or {}).get("score", 0.0))
                except Exception:
                    sample_reward = 0.0
                if obs_id:
                    sample_reward_by_observation_id[obs_id] = float(sample_reward)

                extra_for_mode = data.env.rollout.extra_info or {}
                credit_mode_default = str(extra_for_mode.get("cogrpo_verifier_credit_assignment") or "").strip().lower()
                if obs_id:
                    credit_mode_by_observation_id[obs_id] = credit_mode_default

                finish_reason_for_count = str(extra_for_mode.get("cogrpo_last_finish_reason") or "")
                context_exhausted_for_count = bool(extra_for_mode.get("cogrpo_context_exhausted", False))
                is_truncated_for_count = context_exhausted_for_count or (finish_reason_for_count in trunc_finish_reasons)

                events_for_count = extra_for_mode.get("cogrpo_verifier_events") or []
                if not isinstance(events_for_count, list):
                    continue
                for ev in events_for_count:
                    if not isinstance(ev, dict):
                        continue
                    parent_obs = str(ev.get("parent_observation_id") or "")
                    if not parent_obs:
                        parent_obs = obs_id
                    if not parent_obs:
                        event_uid = str(ev.get("event_uid") or "")
                        if ":" in event_uid:
                            parent_obs = event_uid.split(":", 1)[0]
                    credit_mode = str(
                        ev.get("verifier_credit_assignment")
                        or ev.get("credit_assignment")
                        or credit_mode_by_observation_id.get(parent_obs, "")
                        or credit_mode_default
                    ).strip().lower()
                    if credit_mode in ("cf_branch", "cf"):
                        continue
                    if is_truncated_for_count and skip_truncated_verifier:
                        continue

                    prompt_ids_raw = ev.get("prompt_token_ids") or []
                    response_ids_raw = ev.get("response_ids") or []
                    if not isinstance(prompt_ids_raw, list) or not isinstance(response_ids_raw, list):
                        continue
                    if len(prompt_ids_raw) <= 0 or len(response_ids_raw) <= 0:
                        continue
                    try:
                        prompt_ids_tmp = [int(x) for x in prompt_ids_raw]
                        response_ids_tmp = [int(x) for x in response_ids_raw]
                    except Exception:
                        continue
                    input_ids_len = len(prompt_ids_tmp) + max(0, len(response_ids_tmp) - 1)
                    shifted_labels_len = max(0, len(prompt_ids_tmp) - 1) + len(response_ids_tmp)
                    if input_ids_len != shifted_labels_len:
                        continue
                    if input_ids_len > int(pack_max_length):
                        continue
                    if parent_obs:
                        event_count_by_observation_id[parent_obs] = int(event_count_by_observation_id.get(parent_obs, 0)) + 1

            for data in group:
                extra = data.env.rollout.extra_info or {}
                finish_reason = str(extra.get("cogrpo_last_finish_reason") or "")
                context_exhausted = bool(extra.get("cogrpo_context_exhausted", False))
                is_truncated = context_exhausted or (finish_reason in trunc_finish_reasons)
                obs_id = str(getattr(data.uid, "observation_id", "") or "")
                credit_mode_default = str(extra.get("cogrpo_verifier_credit_assignment") or "").strip().lower()

                events = extra.get("cogrpo_verifier_events") or []
                if not isinstance(events, list):
                    continue
                for ev in events:
                    if not isinstance(ev, dict):
                        continue
                    verifier_event_total += 1
                    total_cf_events += 1
                    parent_obs = str(ev.get("parent_observation_id") or "")
                    if not parent_obs:
                        parent_obs = obs_id
                    if not parent_obs:
                        event_uid_for_parent = str(ev.get("event_uid") or "")
                        if ":" in event_uid_for_parent:
                            parent_obs = event_uid_for_parent.split(":", 1)[0]

                    credit_mode = str(
                        ev.get("verifier_credit_assignment")
                        or ev.get("credit_assignment")
                        or credit_mode_by_observation_id.get(parent_obs, "")
                        or credit_mode_default
                    ).strip().lower()
                    if credit_mode in ("cf_branch", "cf"):
                        verifier_event_credit_mode_cf += 1
                    else:
                        verifier_event_credit_mode_equal += 1

                    try:
                        if ev.get("step_reward", None) is not None:
                            delta_values_all.append(float(ev.get("step_reward")))
                    except Exception:
                        pass
                    if is_truncated:
                        trunc_event_count += 1
                        if skip_truncated_verifier:
                            trunc_skip_count += 1
                            continue

                    adv = None
                    adv_from_cf = False
                    adv_from_equal_share = False
                    try:
                        if ev.get("advantage", None) is not None:
                            adv = float(ev.get("advantage"))
                            adv_from_cf = True
                    except Exception:
                        adv = None
                    if adv is None and credit_mode in ("cf_branch", "cf"):
                        reconstructed_adv = cf_adv_reconstructed_by_event_uid.get(str(ev.get("event_uid") or ""))
                        if reconstructed_adv is not None:
                            adv = float(reconstructed_adv)
                            adv_from_cf = True
                            verifier_event_cf_advantage_reconstructed += 1

                    # VERL parity: for non-cf credit assignment, still build verifier train
                    # events by sharing parent sample reward across interventions.
                    if adv is None:
                        if credit_mode in ("cf_branch", "cf"):
                            verifier_event_skip_missing_advantage += 1
                            continue
                        parent_reward = sample_reward_by_observation_id.get(parent_obs, None)
                        if parent_reward is None:
                            verifier_event_skip_missing_parent_reward += 1
                            continue
                        parent_event_count = int(event_count_by_observation_id.get(parent_obs, 0))
                        if parent_event_count <= 0:
                            verifier_event_skip_missing_parent_count += 1
                            continue
                        adv = float(parent_reward) / float(parent_event_count)
                        adv_from_equal_share = True

                    try:
                        loss_weight = float(
                            os.environ.get(
                                "COGRPO_VERIFIER_LOSS_WEIGHT",
                                os.environ.get("VERIFIER_LOSS_WEIGHT", "1.0"),
                            )
                        )
                        adv = float(adv) * float(loss_weight)
                    except Exception:
                        continue
                    prompt_ids = ev.get("prompt_token_ids") or []
                    response_ids = ev.get("response_ids") or []
                    old_logprobs = ev.get("old_logprobs", None)
                    if not isinstance(prompt_ids, list) or not isinstance(response_ids, list):
                        continue
                    if len(prompt_ids) <= 0 or len(response_ids) <= 0:
                        continue
                    try:
                        prompt_ids = [int(x) for x in prompt_ids]
                        response_ids = [int(x) for x in response_ids]
                    except Exception:
                        continue

                    # Input/label shift follows the same convention as actor training:
                    # input_ids = prompt + response[:-1], shifted_labels = [-100]*(len(prompt)-1) + response
                    input_ids = prompt_ids + response_ids[:-1]
                    shifted_labels = [-100] * (len(prompt_ids) - 1) + list(response_ids)

                    if len(input_ids) != len(shifted_labels):
                        continue
                    if len(input_ids) > int(pack_max_length):
                        continue

                    rollout_logprobs = None
                    if old_logprobs is not None:
                        try:
                            old_logprobs = list(old_logprobs)
                            if len(old_logprobs) == len(response_ids):
                                full_logprobs = [0.0] * (len(prompt_ids) - 1) + [float(x) for x in old_logprobs]
                                if len(full_logprobs) == len(shifted_labels):
                                    rollout_logprobs = torch.tensor(full_logprobs, dtype=torch.float32).unsqueeze(0)
                        except Exception:
                            rollout_logprobs = None
                    if rollout_logprobs is None:
                        verifier_event_old_logprobs_missing += 1
                    else:
                        verifier_event_old_logprobs_used += 1

                    input_ids_tensor = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0)
                    shifted_labels_tensor = torch.tensor(shifted_labels, dtype=torch.int64).unsqueeze(0)

                    all_input_ids.append(input_ids_tensor)
                    all_shifted_labels.append(shifted_labels_tensor)
                    all_advantages.append(float(adv))
                    all_rollout_logprobs.append(rollout_logprobs)

                    advantages_list.extend([float(adv)] * len(response_ids))
                    prompt_len_list.append(len(prompt_ids))
                    response_len_list.append(len(response_ids))

                    if adv_from_cf:
                        verifier_event_cf_advantage_used += 1
                    elif adv_from_equal_share:
                        verifier_event_equal_share_used += 1
                    try:
                        if ev.get("step_reward", None) is not None:
                            step_reward_value = float(ev.get("step_reward"))
                            delta_values_used.append(step_reward_value)
                            if abs(float(adv)) > 1e-6 or abs(step_reward_value) > 1e-6:
                                verifier_adv_delta_sign_count += 1
                                if float(adv) * float(step_reward_value) < 0.0:
                                    verifier_adv_delta_sign_conflict += 1
                    except Exception:
                        pass
                    try:
                        if ev.get("step_diff", None) is not None:
                            diff_values_used.append(float(ev.get("step_diff")))
                    except Exception:
                        pass
                    try:
                        if ev.get("step_cost", None) is not None:
                            cost_values_used.append(float(ev.get("step_cost")))
                    except Exception:
                        pass

        num_samples = len(all_input_ids)
        indices = list(range(num_samples))
        random.shuffle(indices)

        data_batches = []
        for i in indices:
            seq_ctx = SequenceContext.from_input_ids((all_input_ids[i],), device="cpu")
            data_batches.append(
                {
                    "seq_ctx": seq_ctx,
                    "shifted_labels": all_shifted_labels[i],
                    "advantages": all_advantages[i],
                    "rollout_logprobs": all_rollout_logprobs[i],
                }
            )

        advantages_arr = torch.tensor(advantages_list).float() if advantages_list else torch.tensor([0.0]).float()
        prompt_len_arr = torch.tensor(prompt_len_list).float() if prompt_len_list else torch.tensor([0.0]).float()
        response_len_arr = (
            torch.tensor(response_len_list).float() if response_len_list else torch.tensor([0.0]).float()
        )

        def _safe_tensor_std(x: torch.Tensor) -> float:
            if int(x.numel()) <= 1:
                return 0.0
            return float(x.std(unbiased=False).item())

        info_dict = {
            "batch_size": int(num_samples),
            "advantages/mean": advantages_arr.mean().item(),
            "advantages/min": advantages_arr.min().item(),
            "advantages/max": advantages_arr.max().item(),
            "advantages/std": _safe_tensor_std(advantages_arr),
            "advantages/abs_max": advantages_arr.abs().max().item(),
            "advantages/pos_ratio": (advantages_arr > 0).float().mean().item(),
            "advantages/zero_ratio": (advantages_arr.abs() <= 1e-6).float().mean().item(),
            "advantages/neg_ratio": (advantages_arr < -1e-6).float().mean().item(),
            "response_len/mean": response_len_arr.mean().item(),
            "response_len/min": response_len_arr.min().item(),
            "response_len/max": response_len_arr.max().item(),
            "response_len/std": _safe_tensor_std(response_len_arr),
            "prompt_len/mean": prompt_len_arr.mean().item(),
            "prompt_len/min": prompt_len_arr.min().item(),
            "prompt_len/max": prompt_len_arr.max().item(),
            # Align verl metrics naming for cf_branch truncation diagnostics.
            "co_grpo/cf_trunc_event_ratio": float(trunc_event_count / max(1, total_cf_events)),
            "co_grpo/cf_trunc_event_count": float(trunc_event_count),
            "co_grpo/cf_trunc_event_skip_ratio": float(trunc_skip_count / max(1, total_cf_events)),
            "co_grpo/cf_trunc_event_skip_count": float(trunc_skip_count),
            "co_grpo/cf_verifier_events_total": float(total_cf_events),
            "co_grpo/cf_verifier_events_used": float(num_samples),
            "co_grpo/verifier_events_total": float(verifier_event_total),
            "co_grpo/verifier_events_cf_advantage_used": float(verifier_event_cf_advantage_used),
            "co_grpo/verifier_events_cf_advantage_reconstructed": float(verifier_event_cf_advantage_reconstructed),
            "co_grpo/verifier_events_equal_share_used": float(verifier_event_equal_share_used),
            "co_grpo/verifier_events_credit_mode_cf": float(verifier_event_credit_mode_cf),
            "co_grpo/verifier_events_credit_mode_equal": float(verifier_event_credit_mode_equal),
            "co_grpo/verifier_events_skip_missing_advantage": float(verifier_event_skip_missing_advantage),
            "co_grpo/verifier_events_skip_missing_parent_reward": float(verifier_event_skip_missing_parent_reward),
            "co_grpo/verifier_events_skip_missing_parent_count": float(verifier_event_skip_missing_parent_count),
            "co_grpo/verifier_events_old_logprobs_used": float(verifier_event_old_logprobs_used),
            "co_grpo/verifier_events_old_logprobs_missing": float(verifier_event_old_logprobs_missing),
            "co_grpo/verifier_events_old_logprobs_missing_ratio": float(
                verifier_event_old_logprobs_missing / max(1, verifier_event_old_logprobs_used + verifier_event_old_logprobs_missing)
            ),
            "co_grpo/adv_delta_sign_conflict_count": float(verifier_adv_delta_sign_conflict),
            "co_grpo/adv_delta_sign_conflict_total": float(verifier_adv_delta_sign_count),
            "co_grpo/adv_delta_sign_conflict_rate": float(
                verifier_adv_delta_sign_conflict / max(1, verifier_adv_delta_sign_count)
            ),
        }
        if delta_values_all:
            delta_all_arr = torch.tensor(delta_values_all, dtype=torch.float32)
            info_dict["co_grpo/cf_delta_mean_verifier_events_all"] = float(delta_all_arr.mean().item())
            info_dict["co_grpo/cf_delta_pos_ratio_verifier_events_all"] = float((delta_all_arr > 0).float().mean().item())
            info_dict["co_grpo/cf_delta_neg_ratio_verifier_events_all"] = float((delta_all_arr < -1e-6).float().mean().item())
        if delta_values_used:
            delta_used_arr = torch.tensor(delta_values_used, dtype=torch.float32)
            info_dict["co_grpo/cf_delta_mean_verifier_events_used"] = float(delta_used_arr.mean().item())
            info_dict["co_grpo/cf_delta_abs_max_verifier_events_used"] = float(delta_used_arr.abs().max().item())
            info_dict["co_grpo/cf_delta_pos_ratio_verifier_events_used"] = float((delta_used_arr > 0).float().mean().item())
            info_dict["co_grpo/cf_delta_zero_ratio_verifier_events_used"] = float(
                (delta_used_arr.abs() <= 1e-6).float().mean().item()
            )
            info_dict["co_grpo/cf_delta_neg_ratio_verifier_events_used"] = float(
                (delta_used_arr < -1e-6).float().mean().item()
            )
        if diff_values_used:
            diff_used_arr = torch.tensor(diff_values_used, dtype=torch.float32)
            info_dict["co_grpo/cf_diff_zero_ratio_verifier_events_used"] = float(
                (diff_used_arr.abs() <= 1e-6).float().mean().item()
            )
        if cost_values_used:
            info_dict["co_grpo/cf_cost_mean_verifier_events_used"] = float(
                sum(cost_values_used) / max(1, len(cost_values_used))
            )
        return data_batches, info_dict

    def _save_trajectories(self, data_groups, save_path, rollout_idx, is_eval: bool = False):
        rewards = []

        rollout_response_len_list = []
        version_dict = {i: 0 for i in range(self._dataflow_partial_rollout_step + 1)}
        dump_cogrpo_extra = str(os.environ.get("COGRPO_DUMP_TRAJECTORY_EXTRA", "0") or "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        print_cogrpo_debug = str(os.environ.get("COGRPO_PRINT_TRAJECTORY_DEBUG", "0") or "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        try:
            dump_cogrpo_max_samples = int(os.environ.get("COGRPO_DUMP_TRAJECTORY_MAX_SAMPLES", "0") or 0)
        except Exception:
            dump_cogrpo_max_samples = 0
        try:
            print_cogrpo_max_samples = int(os.environ.get("COGRPO_PRINT_TRAJECTORY_MAX_SAMPLES", "0") or 0)
        except Exception:
            print_cogrpo_max_samples = 0
        if print_cogrpo_max_samples <= 0:
            print_cogrpo_max_samples = dump_cogrpo_max_samples

        pretty_jsonl = str(os.environ.get("XTUNER_PRETTY_TRAJECTORY_JSONL", "0") or "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        dump_verl_compat_fields = str(os.environ.get("COGRPO_DUMP_VERL_COMPAT_FIELDS", "0") or "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )

        def _write_jsonl(fp, obj: dict):
            if pretty_jsonl:
                json.dump(obj, fp, ensure_ascii=False, indent=2)
                fp.write("\n")
            else:
                fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

        def _trim_cogrpo_extra_info(extra_info: object) -> dict | None:
            if not isinstance(extra_info, dict):
                return None

            out: dict = {}

            def _compact_cogrpo_event(event: object) -> dict | None:
                if not isinstance(event, dict):
                    return None
                compact = {
                    "event_uid": event.get("event_uid"),
                    "step_idx": event.get("step_idx"),
                    "prefix_len": event.get("prefix_len"),
                    "wait_confidence": event.get("wait_confidence"),
                    "wait_avg_logprob": event.get("wait_avg_logprob"),
                    "hint_token_count": event.get("hint_token_count"),
                    "prethink_anchor": event.get("prethink_anchor"),
                    "state_hash_bucket": event.get("state_hash_bucket"),
                }
                optional_scalar_keys = (
                    "parent_root_id",
                    "parent_action_id",
                    "parent_observation_id",
                    "cf_r_main",
                    "cf_r0",
                    "step_cost",
                    "step_cost_applied",
                    "step_diff",
                    "step_gap",
                    "step_positive_gap",
                    "step_reward_before_penalty",
                    "step_headroom",
                    "step_reward",
                    "group_index",
                    "advantage",
                )
                for key in optional_scalar_keys:
                    value = event.get(key)
                    if value is None or isinstance(value, (int, float, bool, str)):
                        if value is not None:
                            compact[key] = value
                return compact

            # Small, high-signal scalar fields.
            scalar_keys = (
                "cogrpo_stream",
                "cogrpo_stream_data",
                "cogrpo_stream_rollout",
                "cogrpo_stream_resolved",
                "cogrpo_num_interventions",
                "cogrpo_hint_len",
                "cogrpo_gen_len",
                "cogrpo_response_len",
                "cogrpo_prompt_len",
                "cogrpo_first_step_tokens_len",
                "cogrpo_step_count",
                "cogrpo_stop_sequence_hits",
                "cogrpo_stop_sequence_hit_rate",
                "cogrpo_last_finish_reason",
                "cogrpo_context_exhausted",
                "cogrpo_confidence_threshold",
                "cogrpo_verifier_credit_assignment",
                "cogrpo_intervention_penalty_freq_coef",
                "cogrpo_intervention_penalty_len_coef",
                "cogrpo_verifier_reward_mode",
                "cogrpo_verifier_reward_headroom_min",
                "cogrpo_verifier_reward_improve_coef",
                "cogrpo_verifier_outputs",
                "cogrpo_verifier_go_total",
                "cogrpo_verifier_wait_total",
                "cogrpo_verifier_no_valid_decision",
                "cogrpo_verifier_no_valid_decision_final_like",
                "cogrpo_verifier_wait_conf_count",
                "cogrpo_verifier_wait_conf_mean",
                "cogrpo_verifier_wait_conf_min",
                "cogrpo_verifier_wait_conf_max",
                "cogrpo_verifier_wait_conf_missing",
                "cogrpo_verifier_wait_conf_invalid",
                "cogrpo_verifier_wait_blocked_low_conf",
                "cogrpo_verifier_policy_blocked_total",
                "cogrpo_verifier_intervention_attempt_total",
                "cogrpo_verifier_hint_inserted_total",
                "cogrpo_verifier_hint_not_inserted_total",
                "cogrpo_verifier_skipped_no_insert_anchor",
                "cogrpo_hint_skipped_late_stage",
                "cogrpo_verifier_debug_enabled",
                "cogrpo_verifier_debug_logged",
                "cogrpo_verifier_debug_dump_enabled",
                "cogrpo_verifier_debug_dumped",
                "cogrpo_verifier_parser_mode",
                "cogrpo_verifier_parser_source",
                "cogrpo_verifier_debug_dump_path",
                "cogrpo_verifier_event_count",
                "cogrpo_partial_resume",
                "cogrpo_partial_resume_fallback",
                "cogrpo_partial_resume_initial_response_len",
                "cogrpo_cf_k",
                "cogrpo_cf_reward_tail_tokens",
                "cogrpo_cf_baseline_count",
                "cogrpo_cf_baseline_original_len_mean",
                "cogrpo_cf_baseline_original_len_max",
                "cogrpo_cf_baseline_tail_len_mean",
                "cogrpo_cf_baseline_tail_len_max",
            )
            for k in scalar_keys:
                v = extra_info.get(k)
                if v is None or isinstance(v, (int, float, bool, str)):
                    if v is not None:
                        out[k] = v

            # Hints and intervention metadata (small; no token arrays).
            hints = extra_info.get("cogrpo_hints")
            if isinstance(hints, list) and all(isinstance(x, str) for x in hints):
                out["cogrpo_hints"] = list(hints)
            critiques = extra_info.get("cogrpo_critiques")
            if isinstance(critiques, list) and all(isinstance(x, str) for x in critiques):
                out["cogrpo_critiques"] = list(critiques)

            interventions = extra_info.get("cogrpo_interventions")
            if isinstance(interventions, list):
                cleaned = []
                for it in interventions:
                    if not isinstance(it, dict):
                        continue
                    # Keep JSON-safe subset.
                    cleaned.append(
                        {
                            "event_uid": it.get("event_uid"),
                            "step_idx": it.get("step_idx"),
                            "prefix_len": it.get("prefix_len"),
                            "insert_pos": it.get("insert_pos"),
                            "think_insert_pos": it.get("think_insert_pos"),
                            "inserted_before_think": it.get("inserted_before_think"),
                            "hint_token_count": it.get("hint_token_count"),
                            "prethink_anchor": it.get("prethink_anchor"),
                            "state_hash_bucket": it.get("state_hash_bucket"),
                            "wait_confidence": it.get("wait_confidence"),
                            "wait_avg_logprob": it.get("wait_avg_logprob"),
                            # Optional debug fields (guarded by env on rollout side).
                            "hint_preview": it.get("hint_preview"),
                            "hint_formatted_preview": it.get("hint_formatted_preview"),
                            "context_before": it.get("context_before"),
                            "context_after": it.get("context_after"),
                            "context_tokens": it.get("context_tokens"),
                        }
                    )
                if cleaned:
                    out["cogrpo_interventions"] = cleaned

            verifier_events_cleaned = []
            verifier_events = extra_info.get("cogrpo_verifier_events")
            if isinstance(verifier_events, list):
                for ev in verifier_events:
                    compact = _compact_cogrpo_event(ev)
                    if compact is not None:
                        verifier_events_cleaned.append(compact)
                if verifier_events_cleaned:
                    out["cogrpo_verifier_events"] = verifier_events_cleaned
                    out["cogrpo_verifier_event_count"] = int(len(verifier_events_cleaned))

            # Keep compact cf_branch diagnostics if present (after env may pop heavy objects).
            for k, v in extra_info.items():
                if not isinstance(k, str):
                    continue
                if k.startswith("cogrpo_cf_") and (v is None or isinstance(v, (int, float, bool, str))):
                    if v is not None:
                        out[k] = v

            # Summarize optional heavy lists when explicitly kept.
            cf_events = extra_info.get("cogrpo_cf_events")
            cleaned = []
            if isinstance(cf_events, list):
                for ev in cf_events:
                    compact = _compact_cogrpo_event(ev)
                    if compact is not None:
                        cleaned.append(compact)
            elif verifier_events_cleaned:
                cleaned = [
                    {
                        key: ev.get(key)
                        for key in (
                            "event_uid",
                            "step_idx",
                            "prefix_len",
                            "wait_confidence",
                            "wait_avg_logprob",
                            "hint_token_count",
                            "prethink_anchor",
                            "state_hash_bucket",
                        )
                    }
                    for ev in verifier_events_cleaned
                ]
            if cleaned:
                out["cogrpo_cf_events"] = cleaned
                out.setdefault("cogrpo_cf_event_count", int(len(cleaned)))

            cf_rollouts = extra_info.get("cogrpo_cf_rollouts")
            if isinstance(cf_rollouts, list):
                cleaned = []
                for r in cf_rollouts:
                    if not isinstance(r, dict):
                        continue
                    cleaned.append(
                        {
                            "event_uid": r.get("event_uid"),
                            "original_len": r.get("original_len"),
                            "tail_len": r.get("tail_len"),
                        }
                    )
                if cleaned:
                    out["cogrpo_cf_rollouts"] = cleaned

            return out or None

        strict_dual_stream = str(os.environ.get("COGRPO_STRICT_DUAL_STREAM", "1") or "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        strict_dual_stream_mismatch = str(
            os.environ.get("COGRPO_STRICT_DUAL_STREAM_MISMATCH", "0") or "0"
        ).strip().lower() in ("1", "true", "yes", "y", "on")

        def _normalize_stream_tag(raw_tag: object) -> str:
            try:
                tag = str(raw_tag or "").strip().lower()
            except Exception:
                tag = ""
            if tag in ("control", "exp"):
                return tag
            return ""

        def _resolve_cogrpo_stream(data_item) -> tuple[str, str, str]:
            """Resolve CoGRPO stream tag with rollout-extra priority.

            Returns:
                tuple(resolved, rollout_tag, data_tag)
            """
            rollout_tag = ""
            data_tag = ""
            try:
                rollout_extra = getattr(getattr(data_item.env, "rollout", None), "extra_info", None)
                if isinstance(rollout_extra, dict):
                    rollout_tag = _normalize_stream_tag(
                        rollout_extra.get("cogrpo_stream_rollout", rollout_extra.get("cogrpo_stream", ""))
                    )
            except Exception:
                pass
            try:
                data_extra = getattr(data_item.data, "extra_info", None)
                if isinstance(data_extra, dict):
                    data_tag = _normalize_stream_tag(data_extra.get("cogrpo_stream_data", data_extra.get("cogrpo_stream", "")))
            except Exception:
                pass

            if rollout_tag and data_tag and rollout_tag != data_tag:
                msg = (
                    "[CoGRPO][DualStream] dump stream mismatch between rollout/data: "
                    f"rollout={rollout_tag}, data={data_tag}, action_id={getattr(data_item.uid, 'action_id', '<unknown>')}"
                )
                repaired = False
                try:
                    data_extra = getattr(data_item.data, "extra_info", None)
                    if isinstance(data_extra, dict):
                        data_extra["cogrpo_stream_data"] = rollout_tag
                        data_extra["cogrpo_stream"] = rollout_tag
                        repaired = True
                except Exception:
                    repaired = False
                if strict_dual_stream and strict_dual_stream_mismatch and (not repaired):
                    raise RuntimeError(msg)
                if repaired:
                    self.logger.warning(f"{msg}; repaired_by=rollout_tag")
                    data_tag = rollout_tag
                else:
                    self.logger.warning(msg)
            resolved = rollout_tag or data_tag
            return resolved, rollout_tag, data_tag

        # NOTE: Since we currently default to token-in token-out, the code for checking whether response_ids have Retokenization Drift is commented out.
        # If you need to debug, you can uncomment it.
        # mismatch_token_ids_count = 0
        # response_len_list = []
        stream_control_count = 0
        stream_exp_count = 0
        stream_unknown_count = 0
        invalid_group_count = 0
        invalid_item_count = 0
        invalid_state_counts: dict[str, int] = {}
        for group in data_groups:
            group_valid = is_valid_for_training(group)
            if not group_valid:
                self.logger.error(f"Skip one data group {group} due to rollout failed or empty response.")
                invalid_group_count += 1
            for data in group:
                if group_valid:
                    rewards.append(data.env.judger.reward["score"])
                    if data.env.rollout.response_ids is not None:
                        if isinstance(data.env.rollout.response_ids, torch.Tensor):
                            response_ids = data.env.rollout.response_ids.flatten().tolist()
                        else:
                            response_ids = data.env.rollout.response_ids
                        rollout_response_len_list.append(len(response_ids))
                        # response_str = self.tokenizer.decode(response_ids, skip_special_tokens=False)
                        # revert_encode_response_ids = self.tokenizer.encode(response_str, add_special_tokens=False)

                        # response_str_to_ids = self.tokenizer.encode(data.env.rollout.response, add_special_tokens=False)
                        # response_len_list.append(len(response_str_to_ids))

                        # if response_ids != revert_encode_response_ids or response_ids != response_str_to_ids:
                        #     mismatch_token_ids_count += 1
                    else:
                        response_ids = self.tokenizer.encode(data.env.rollout.response, add_special_tokens=False)
                        rollout_response_len_list.append(len(response_ids))

                    version = data.uid.version
                    if version not in version_dict:
                        version_dict[version] = 0
                    version_dict[version] += 1
                else:
                    invalid_item_count += 1
                    rollout_state = getattr(data.env.rollout, "state", None)
                    state_name = str(rollout_state.value if hasattr(rollout_state, "value") else rollout_state or "unknown")
                    invalid_state_counts[state_name] = int(invalid_state_counts.get(state_name, 0) + 1)

                stream_resolved, _, _ = _resolve_cogrpo_stream(data)
                if stream_resolved == "control":
                    stream_control_count += 1
                elif stream_resolved == "exp":
                    stream_exp_count += 1
                else:
                    stream_unknown_count += 1

        rewards_tensor = torch.tensor(rewards).float()
        rollout_response_lens: torch.Tensor = torch.tensor([0.0]).float()
        if len(rollout_response_len_list) > 0:
            rollout_response_lens = torch.tensor(rollout_response_len_list).float()

        
        _count = 0
        
        
        with open(save_path, "w", encoding="utf-8") as f:
            item = {
                "reward_mean": rewards_tensor.mean().item(),
                "reward_std": rewards_tensor.std().item(),
                "reward_max": rewards_tensor.max().item(),
                "reward_min": rewards_tensor.min().item(),
                "response_len_mean": rollout_response_lens.mean().item(),
                "response_len_std": rollout_response_lens.std().item(),
                "response_len_max": rollout_response_lens.max().item(),
                "response_len_min": rollout_response_lens.min().item(),
                "total_len": len(rewards),
                "versions": version_dict,
                "cogrpo_dump_extra_enabled": bool(dump_cogrpo_extra),
                "cogrpo_dump_max_samples": int(dump_cogrpo_max_samples),
                "cogrpo_dump_cap_active": bool(dump_cogrpo_extra and dump_cogrpo_max_samples > 0),
                "cogrpo_stream_control_count": int(stream_control_count),
                "cogrpo_stream_exp_count": int(stream_exp_count),
                "cogrpo_stream_unknown_count": int(stream_unknown_count),
                "cogrpo_invalid_group_count": int(invalid_group_count),
                "cogrpo_invalid_item_count": int(invalid_item_count),
                "cogrpo_invalid_state_counts": invalid_state_counts,
                "cogrpo_dump_cap_skipped_samples": int(
                    max(0, len(rewards) - int(dump_cogrpo_max_samples))
                    if (dump_cogrpo_extra and dump_cogrpo_max_samples > 0)
                    else 0
                ),
                # "mismatch_token_ids_count": mismatch_token_ids_count,
            }
            self.logger.info(f"versions distribution: {version_dict}")
            _write_jsonl(f, item)
            tb_prefix = "eval" if is_eval else "response"
            # tb_item = {f"{tb_prefix}/{k}": v for k, v in item.items()}
            tb_item = {}
            for k, v in item.items():
                if k == "versions":
                    continue
                if isinstance(v, dict):
                    tb_item.update(_flatten_scalar_metrics(f"{tb_prefix}/{k}", v))
                elif isinstance(v, (int, float)):
                    tb_item[f"{tb_prefix}/{k}"] = float(v)
            tb_version_dict = {f"{tb_prefix}/version_{k}": float(v) for k, v in version_dict.items()}
            self._writer.add_scalars(
                tag_scalar_dict=tb_item,
                global_step=rollout_idx,
            )
            self._writer.add_scalars(
                tag_scalar_dict=tb_version_dict,
                global_step=rollout_idx,
            )
            for group in data_groups:
                group_valid = is_valid_for_training(group)
                if not group_valid:
                    self.logger.error(f"Dump invalid data group {group} for audit only; excluded from training statistics.")
                for data in group:
                    logprobs = data.env.rollout.logprobs
                    if logprobs is not None:
                        logprobs_t = logprobs if isinstance(logprobs, torch.Tensor) else torch.tensor(logprobs, dtype=torch.float32)
                        entropy = -logprobs_t.mean().item()
                    else:
                        entropy = None
                    judger_reward = getattr(getattr(data.env, "judger", None), "reward", None)
                    reward_score = None
                    if isinstance(judger_reward, dict):
                        try:
                            reward_score = float(judger_reward.get("score", 0.0))
                        except Exception:
                            reward_score = 0.0
                    if data.env.rollout.response_ids is not None:
                        if isinstance(data.env.rollout.response_ids, torch.Tensor):
                            response_ids = data.env.rollout.response_ids.flatten().tolist()
                        else:
                            response_ids = data.env.rollout.response_ids
                        row_response_len = len(response_ids)
                    else:
                        response_text = data.env.rollout.response or ""
                        row_response_len = len(self.tokenizer.encode(response_text, add_special_tokens=False))
                    rollout_state = getattr(data.env.rollout, "state", None)
                    rollout_state_name = str(rollout_state.value if hasattr(rollout_state, "value") else rollout_state or "")
                    item = {
                        "action_id": data.uid.action_id,
                        "prompt": data.data.extra_info["raw_prompt"],
                        "response": data.env.rollout.response,
                        "versioned_response": data.env.rollout.versioned_response,
                        # "response_ids": str(data.env.rollout.response_ids),
                        # "versioned_response_ids": str(data.env.rollout.versioned_response_ids),
                        "response_len": row_response_len,
                        "origin_data_source": data.data.extra_info.get("origin_data_source", "Unknown"),
                        "versioned_response_len": data.env.rollout.versioned_num_return_tokens,
                        "label": data.data.reward_model.get("ground_truth", "None"),
                        "reward": reward_score,
                        "version": data.uid.version,
                        "finish_reason": data.env.rollout.finish_reason,
                        "entropy": entropy,
                        "cogrpo_training_valid": bool(group_valid),
                        "rollout_state": rollout_state_name,
                    }
                    if not group_valid:
                        item["cogrpo_dump_only_reason"] = "invalid_for_training"
                    if dump_verl_compat_fields:
                        item["question"] = item["prompt"]
                        item["student_response_full"] = item["response"]
                        item["student_response_policy"] = item["response"]
                        item["ground_truth"] = item.get("label", None)
                        item["hints"] = ""
                        item["critiques"] = ""
                    stream_resolved, stream_rollout, stream_data = _resolve_cogrpo_stream(data)
                    item["cogrpo_stream_resolved"] = stream_resolved
                    item["cogrpo_stream_rollout"] = stream_rollout
                    item["cogrpo_stream_data"] = stream_data
                    if stream_resolved:
                        item["cogrpo_stream"] = stream_resolved
                        if dump_verl_compat_fields:
                            item["stream_type"] = stream_resolved
                    if dump_cogrpo_extra and (dump_cogrpo_max_samples <= 0 or _count < dump_cogrpo_max_samples):
                        try:
                            extra = getattr(data.env.rollout, "extra_info", None)
                        except Exception:
                            extra = None
                        trimmed = _trim_cogrpo_extra_info(extra)
                        if trimmed is not None:
                            item["cogrpo"] = trimmed
                            if dump_verl_compat_fields:
                                hints = trimmed.get("cogrpo_hints")
                                critiques = trimmed.get("cogrpo_critiques")
                                if isinstance(hints, list):
                                    item["hints"] = "\n".join(str(x) for x in hints)
                                if isinstance(critiques, list):
                                    item["critiques"] = "\n---\n".join(str(x) for x in critiques)
                            if print_cogrpo_debug and (print_cogrpo_max_samples <= 0 or _count < print_cogrpo_max_samples):
                                try:
                                    hints = trimmed.get("cogrpo_hints") or []
                                    interventions = trimmed.get("cogrpo_interventions") or []
                                    self.logger.info(
                                        f"[CoGRPO][Debug][rollout={rollout_idx}] action_id={data.uid.action_id} "
                                        f"ver={data.uid.version} reward={item['reward']} "
                                        f"finish={item['finish_reason']} resp_len={item['response_len']} "
                                        f"num_interventions={trimmed.get('cogrpo_num_interventions', None)} "
                                        f"last_finish_reason={trimmed.get('cogrpo_last_finish_reason', None)} "
                                        f"context_exhausted={trimmed.get('cogrpo_context_exhausted', None)}"
                                    )
                                    if isinstance(hints, list) and hints:
                                        for hi, h in enumerate(hints[:8]):
                                            self.logger.info(f"[CoGRPO][Debug][hint#{hi}] {str(h)[:240]}")
                                    if isinstance(interventions, list) and interventions:
                                        for ii, it in enumerate(interventions[:8]):
                                            if not isinstance(it, dict):
                                                continue
                                            self.logger.info(
                                                "[CoGRPO][Debug][intervention#{idx}] "
                                                "step={step} prefix_len={prefix_len} insert_pos={insert_pos} "
                                                "think_insert_pos={think_insert_pos} hint_tokens={hint_tokens} "
                                                "prethink_anchor={prethink_anchor} wait_conf={wait_conf} "
                                                "ctx_before={ctx_before} ctx_after={ctx_after}".format(
                                                    idx=ii,
                                                    step=it.get("step_idx"),
                                                    prefix_len=it.get("prefix_len"),
                                                    insert_pos=it.get("insert_pos"),
                                                    think_insert_pos=it.get("think_insert_pos"),
                                                    hint_tokens=it.get("hint_token_count"),
                                                    prethink_anchor=it.get("prethink_anchor"),
                                                    wait_conf=it.get("wait_confidence"),
                                                    ctx_before=str(it.get("context_before") or "")[:240],
                                                    ctx_after=str(it.get("context_after") or "")[:240],
                                                )
                                            )
                                except Exception:
                                    pass
                    elif dump_cogrpo_extra and dump_cogrpo_max_samples > 0 and _count >= dump_cogrpo_max_samples:
                        item["cogrpo_dump_skipped_reason"] = "max_samples_cap"
                        item["cogrpo_dump_max_samples"] = int(dump_cogrpo_max_samples)
                    _write_jsonl(f, item)
                    if group_valid:
                        _count += 1

def silu(x):
    return x * torch.sigmoid(x)

class fn(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.up_proj = nn.Linear(input_dim, output_dim * 4)
        self.down_proj = nn.Linear(output_dim * 4, output_dim)
        self.gated_act = silu()
        self.gated_proj = nn.Linear(input_dim, output_dim * 4)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        up = self.up_proj(x)
        gate = self.gated_proj(x)
        gated_up = up * self.gated_act(gate)
        dropped = self.dropout(gated_up)
        output = self.down_proj(dropped)
        return output


class moefn(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_experts: int = 4, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.experts = nn.ModuleList([fn(input_dim, output_dim, dropout) for _ in range(num_experts)])
        self.router = nn.Linear(input_dim, num_experts)
        self.top_k = top_k
        self.aux_loss_coef = 0.01
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        router_logits = self.router(x)
        router_probs = torch.softmax(router_logits, dim=-1) # [batch_size, seq_len, num_experts]
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1) # (bsz, seq_len, top_k)
        expert_outputs = torch.stack([self.experts[i](x) for i in range(len(self.experts))], dim=2) # (batch_size, seq_len, num_experts, output_dim)
        # index: (batch_size, seq_len, num_experts, 1) -> (batch_size, seq_len, num_experts, output_dim) -> (batch_size, seq_len, top_k, output_dim)
        topk_expert_outputs = torch.gather(expert_outputs, dim=2, index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, expert_outputs.shape[-1]))
        print(topk_expert_outputs.shape)
        # weight: (batch_size, seq_len, top_k, output_dim) -> (batch_size, seq_len, output_dim)
        weighted_output = (topk_probs.unsqueeze(-1) * topk_expert_outputs).sum(dim=2)
        aux_loss = (router_probs * router_probs).mean() * self.aux_loss_coef
        return weighted_output, aux_loss
        
        
        
        
        


class CoPPOTrainer:
    def __init__(self, lam=0.95, gamma=0.99, clip=0.2, beta=0.01):
        self.lam = lam
        self.gamma = gamma
        self.clip = clip
        self.beta = beta
    
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor):
        seq_len = rewards.shape[0]
        token_adv = torch.zeros_like(rewards)
        gae = 0.0
        
        for t in range(seq_len - 1, -1, -1):
            delta = rewards[:, t] + self.gamma * (values[:, t + 1] if t < seq_len - 1 else 0.0) - values[:, t]
            gae = delta + self.gamma * self.lam * gae
            token_adv[:, t] = gae
        returns = token_adv + values
        return token_adv, returns

    def get_loss(self, logprobs, rewards, values, old_logprobs):
        advantages, returns = self.compute_gae(rewards, values)
        ratio = torch.exp(logprobs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
        surrogate_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
        loss = -torch.mean(surrogate_loss + self.beta * (logprobs - old_logprobs))
        critic_loss = self.critic_loss(values, returns)
        total_loss = loss + critic_loss
        return loss

    def critic_loss(self, values, returns):
        return torch.mean((returns - values) ** 2)
        
        
                        
    def get_grpo_loss(self, logprobs, rewards, beta=0.01, epsilon=0.2):
        # logprobs: [batch_size, seq_len]
        # For simplicity, we use the same logprobs as both current and old policy, which makes the importance sampling ratio always 1.
        # In practice, you should use the logprobs from the old policy for the baseline and the logprobs from the current policy for the numerator.
        new_logprobs, ref_logprobs, old_logprobs = logprobs # Placeholder; replace with actual logprobs from current, reference, and old policies.
        ratio = torch.exp(new_logprobs - old_logprobs)  # This will be all ones due to the placeholder, but kept for clarity.
        advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # Normalize rewards to have mean 0 and std 1.
        clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        surrogate_loss = torch.min(ratio * advantage, clipped_ratio * advantage)
        kl_divergence = new_logprobs - ref_logprobs  # This will be zero due to the placeholder, but replace with actual KL in practice.
        # kl_divergence_v3 = torch.exp(new_logprobs - ref_logprobs ) - (new_logprobs - ref_logprobs ) - 1 
        loss = -torch.mean(surrogate_loss - beta * kl_divergence)
        return loss

    def get_dpo_loss(self, logprobs, beta=0.1):
        accept_new_logprobs, accept_ref_logprobs, reject_new_logprobs, reject_ref_logprobs = logprobs # Placeholder; replace with actual logprobs for accepted and rejected samples under current and reference policies.
        accept_logit_diff = accept_new_logprobs - accept_ref_logprobs
        reject_logit_diff = reject_new_logprobs - reject_ref_logprobs        
        return -torch.mean(torch.log_sigmoid(beta * (accept_logit_diff - reject_logit_diff)))

    def _load_trajectories(self, save_path):
        data_groups = []
        with open(save_path) as f:
            for line in f:
                item = json.loads(line)
                messages = item["messages"]
                responses = item["response"]
                rewards = item["reward"]
                group = []
                for response, reward in zip(responses, rewards):
                    group.append(
                        {
                            "messages": messages,
                            "response_str": response,
                            "reward": reward,
                        }
                    )
                data_groups.append(group)
        return data_groups

    def drop(self, data_groups, drop_ratio=0.1):
        drop_point = torch.randn(data_groups.shape, device=data_groups.device) > drop_ratio
        dropped_data_groups = (data_groups * drop_point) / (1 - drop_ratio)
        return dropped_data_groups
    
    def ce_loss(outputs, targets):
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        loss = -torch.mean(torch.gathers(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1))
        return loss
    # y ** 3 - x = 0    y - (y ** 3 - x)/(3 * y ** 2)
    def cal_val(a, max_diff = 1e-5):
        x = a
        while True:
            y = x ** 3
            diff = abs(y - a)
            if diff < max_diff:
                break
            x = y - (y ** 3 - x)/(3 * y ** 2)
        return x
    
        


# \mathcal{L}_{GRPO} = -\mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min\left(\rho_i \hat{A}_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \hat{A}_i\right) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) \right) \right]

        
 # E_


class CurMHA(nn.Module):
    def __init__(self, hidden_dim, num_q_heads, num_kv_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_q_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(hidden_dim, self.head_dim * num_q_heads)
        self.k_proj = nn.Linear(hidden_dim, self.head_dim * num_kv_heads)
        self.v_proj = nn.Linear(hidden_dim, self.head_dim * num_kv_heads)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.atten_dropout = nn.Dropout(dropout)
    
    def forward(self, x, kv_cache=None):
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"] = k
            kv_cache["v"] = v
        kv_seq_len = k.size(2)
        k = k.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
        casual_mask = torch.triu(torch.full((kv_seq_len, kv_seq_len), float('-inf'), device=x.device), diagonal=1)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights + casual_mask, dim=-1)
        attn_weights = self.atten_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        output = self.o_proj(attn_output)
        return output, kv_cache
        