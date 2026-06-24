#!/usr/bin/env python3
"""Local DiffusionGemma SFT launcher with PEFT optimizer checkpoint disabled.

NeMo AutoModel currently forces PEFT checkpoints to safetensors. That works for
LoRA adapter weights, but optimizer state is a nested dict and safetensors
rejects it. For this experiment we only need adapter checkpoints for later eval,
so this wrapper keeps model saving intact and skips optimizer saving for PEFT.
"""

from __future__ import annotations

import logging

import torch.distributed as dist
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.dllm.train_ft import DiffusionGemmaSFTRecipe, DiffusionLMSFTRecipe


_ORIGINAL_SAVE_OPTIMIZER = Checkpointer.save_optimizer


def _rank0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _save_optimizer_skip_peft(self, optimizer, model, weights_path: str, scheduler=None):
    if getattr(self.config, "is_peft", False):
        if _rank0():
            logging.warning(
                "Skipping PEFT optimizer checkpoint at %s/optim because the current "
                "NeMo safetensors PEFT path cannot serialize nested optimizer state. "
                "Adapter model checkpoints are still saved.",
                weights_path,
            )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return None
    return _ORIGINAL_SAVE_OPTIMIZER(self, optimizer, model, weights_path, scheduler)


Checkpointer.save_optimizer = _save_optimizer_skip_peft


def main(default_config_path: str = "examples/dllm_sft/mdlm_sft.yaml") -> None:
    cfg = parse_args_and_load_config(default_config_path)
    recipe_name = cfg.get("recipe", "DiffusionLMSFTRecipe")
    recipe_cls = DiffusionGemmaSFTRecipe if recipe_name == "DiffusionGemmaSFTRecipe" else DiffusionLMSFTRecipe
    recipe = recipe_cls(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
