#!/usr/bin/env python3
"""Patch Axolotl MoE expert quantization for 3D fused expert tensors.

Axolotl's dequant path flattens 3D+ expert tensors to 2D before calling
bitsandbytes, but the 8-bit quant path in the installed dev build can pass the
3D tensor directly. GLM-5.2 fused expert weights are 3D, and this can trigger a
CUDA illegal memory access while loading weights. This local patch makes the
quant path match the dequant path: flatten to row-wise 2D, quantize contiguous
fp16 rows, then reshape int8 storage back to the original expert tensor shape.

The 4-bit path has the same issue: bitsandbytes' raw `quantize_4bit` can reject
large 3D tensors. For 4-bit we flatten to 2D for quantization and set the
resulting QuantState shape back to the original 3D tensor shape for dequant.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path


NEW_8BIT_FUNC = '''def replace_parameter_8bit(module, param_name):
    """Replace a module parameter with an 8-bit quantized version using parametrization."""
    original_param = getattr(module, param_name)
    orig_shape = original_param.data.shape
    quant_input = original_param.data
    if quant_input.ndim > 2:
        quant_input = quant_input.reshape(-1, orig_shape[-1])

    # GLM fused expert tensors are large enough that one bitsandbytes quant
    # kernel can cross CUDA indexing limits on H100/H200. Quantize row chunks
    # and then rebuild the same row-wise int8 representation expected by the
    # dequant parametrization.
    max_elems = int(os.environ.get("AXOLOTL_MOE_8BIT_QUANT_MAX_ELEMS", "67108864"))
    chunk_rows = max(1, max_elems // quant_input.shape[-1])
    int8_chunks = []
    row_stats_chunks = []
    for start in range(0, quant_input.shape[0], chunk_rows):
        chunk = quant_input[start : start + chunk_rows].to(torch.float16).contiguous()
        int8_chunk, row_stats_chunk, _ = bnb.functional.int8_vectorwise_quant(chunk)
        int8_chunks.append(int8_chunk)
        row_stats_chunks.append(row_stats_chunk)
        del chunk

    int8_data = torch.cat(int8_chunks, dim=0)
    row_stats = torch.cat(row_stats_chunks, dim=0)
    if len(orig_shape) > 2:
        int8_data = int8_data.reshape(orig_shape)

    setattr(module, param_name, torch.nn.Parameter(int8_data, requires_grad=False))
    del original_param
'''

NEW_4BIT_FUNC = '''def replace_parameter_4bit(
    module,
    param_name,
    compress_statistics=False,
    quant_type="nf4",
    blocksize=None,
):
    """Replace a 3D+ MoE expert parameter with 4-bit storage using a 2D quant input."""
    from bitsandbytes.nn.parametrize import Bnb4bitParametrization, _register_parametrization_hooks
    import bitsandbytes.functional as F

    original_param = getattr(module, param_name)
    orig_shape = original_param.data.shape
    quant_input = original_param.data
    if quant_input.ndim > 2:
        quant_input = quant_input.reshape(-1, orig_shape[-1])

    quantized_data, quant_state = F.quantize_4bit(
        quant_input.contiguous(),
        blocksize=blocksize,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
    )
    if len(orig_shape) > 2:
        quant_state.shape = orig_shape

    setattr(module, param_name, torch.nn.Parameter(quantized_data, requires_grad=False))
    del original_param

    P.register_parametrization(module, param_name, Bnb4bitParametrization(quant_state), unsafe=True)
    _register_parametrization_hooks(module, param_name)
'''


def find_target(venv: Path) -> Path:
    matches = glob.glob(
        str(venv / "lib" / "python*" / "site-packages" / "axolotl" / "monkeypatch" / "moe_quant.py")
    )
    if not matches:
        raise FileNotFoundError(f"could not find axolotl monkeypatch/moe_quant.py under {venv}")
    if len(matches) > 1:
        matches.sort()
    return Path(matches[-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("venv", type=Path, help="Axolotl virtualenv path")
    args = parser.parse_args()

    target = find_target(args.venv)
    text = target.read_text()
    has_8bit_patch = "AXOLOTL_MOE_8BIT_QUANT_MAX_ELEMS" in text
    has_4bit_patch = "Replace a 3D+ MoE expert parameter with 4-bit storage" in text
    if has_8bit_patch and has_4bit_patch and "Use local replace_parameter_4bit" in text:
        print(f"already patched: {target}")
        return 0
    if "import os\n" not in text:
        text = text.replace("import bitsandbytes as bnb\n", "import os\n\nimport bitsandbytes as bnb\n")
    start_8 = text.find("def replace_parameter_8bit(module, param_name):")
    end_8 = text.find("\n\ndef patch_moe_quantization_on_load", start_8)
    if start_8 == -1 or end_8 == -1:
        raise RuntimeError(f"expected 8-bit quant function not found in {target}")
    replacement = NEW_8BIT_FUNC + "\n\n" + NEW_4BIT_FUNC
    text = text[:start_8] + replacement + text[end_8:]
    text = text.replace(
        "        from bitsandbytes.nn.parametrize import replace_parameter_4bit\n",
        "        # Use local replace_parameter_4bit patched for GLM fused 3D expert tensors.\n",
    )
    target.write_text(text)
    print(f"patched: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
