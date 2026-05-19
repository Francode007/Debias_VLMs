"""
test_pipeline_sanity.py — Sanity check for the Debias_VLMs pipeline architecture.

Validates that:
  1. All module imports resolve correctly.
  2. All subpackages are importable.
  3. Key classes and functions are accessible.
  4. The subprocess commands used by run_modal.py can at least parse --help
     (confirms the scripts exist and are importable without runtime data).
  5. The Modal app definition loads without errors.

Run from project root:
    cd Debias_VLMs
    source debias_env/bin/activate
    python src/test_pipeline_sanity.py
"""

import sys
import os
import subprocess
import importlib
import traceback

# Ensure src/ is on the path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
os.chdir(SRC_DIR)

PYTHON = sys.executable

# ─── Test Results Tracking ────────────────────────────────────────────────────
passed = []
failed = []


def check(name: str, fn):
    """Run a check function; record pass/fail."""
    try:
        fn()
        passed.append(name)
        print(f"  ✓ {name}")
    except Exception as e:
        failed.append((name, str(e)))
        print(f"  ✗ {name}: {e}")


# ─── 1. Module Import Checks ─────────────────────────────────────────────────
print("\n[1/5] Module Import Checks")
print("-" * 50)

IMPORT_CHECKS = [
    # Root modules package
    ("modules", "modules"),
    ("modules.__init__ exports", "modules"),
    # Utils subpackage
    ("modules.utils", "modules.utils"),
    ("modules.utils.config", "modules.utils.config"),
    ("modules.utils.device_manager", "modules.utils.device_manager"),
    ("modules.utils.model_loader", "modules.utils.model_loader"),
    ("modules.utils.model_architecture", "modules.utils.model_architecture"),
    ("modules.utils.dataset_builder", "modules.utils.dataset_builder"),
    ("modules.utils.data_collator", "modules.utils.data_collator"),
    ("modules.utils.reward_trainer", "modules.utils.reward_trainer"),
    # RL Components subpackage
    ("modules.rl_components", "modules.rl_components"),
    ("modules.rl_components.fast_rl", "modules.rl_components.fast_rl"),
    ("modules.rl_components.caa_feedback", "modules.rl_components.caa_feedback"),
    ("modules.rl_components.score_head", "modules.rl_components.score_head"),
    ("modules.rl_components.rl_data_collator", "modules.rl_components.rl_data_collator"),
    # Data subpackage
    ("modules.data", "modules.data"),
    ("modules.data.registry", "modules.data.registry"),
    ("modules.data.model_registry", "modules.data.model_registry"),
    ("modules.data.load_pope", "modules.data.load_pope"),
    ("modules.data.load_sb_bench", "modules.data.load_sb_bench"),
    # Embeddings subpackage
    ("modules.embeddings", "modules.embeddings"),
    ("modules.embeddings.generate_drm_heads", "modules.embeddings.generate_drm_heads"),
    # Inference subpackage
    ("modules.inference", "modules.inference"),
    # Evaluation subpackage
    ("modules.evaluation", "modules.evaluation"),
    ("modules.evaluation.registry", "modules.evaluation.registry"),
    # Training subpackage
    ("modules.training", "modules.training"),
    ("modules.training.args", "modules.training.args"),
    ("modules.training.checkpoint", "modules.training.checkpoint"),
    ("modules.training.drm_loader", "modules.training.drm_loader"),
    ("modules.training.ppo_loop", "modules.training.ppo_loop"),
]

for name, mod in IMPORT_CHECKS:
    check(f"import {name}", lambda m=mod: importlib.import_module(m))


# ─── 2. Key Class/Function Accessibility ─────────────────────────────────────
print("\n[2/5] Key Class/Function Accessibility")
print("-" * 50)

CLASS_CHECKS = [
    ("ScriptArguments from utils", lambda: getattr(importlib.import_module("modules.utils"), "ScriptArguments")),
    ("DeviceManager from utils", lambda: getattr(importlib.import_module("modules.utils"), "DeviceManager")),
    ("ModelLoader from utils", lambda: getattr(importlib.import_module("modules.utils"), "ModelLoader")),
    ("DatasetBuilder from utils", lambda: getattr(importlib.import_module("modules.utils"), "DatasetBuilder")),
    ("create_custom_forward from utils", lambda: getattr(importlib.import_module("modules.utils"), "create_custom_forward")),
    ("RewardDataCollatorWithPadding from utils", lambda: getattr(importlib.import_module("modules.utils"), "RewardDataCollatorWithPadding")),
    ("RewardVisualizer from utils", lambda: getattr(importlib.import_module("modules.utils"), "RewardVisualizer")),
    ("FastRLNode from rl_components", lambda: getattr(importlib.import_module("modules.rl_components"), "FastRLNode")),
    ("compute_caa_weight from rl_components", lambda: getattr(importlib.import_module("modules.rl_components"), "compute_caa_weight")),
    ("PPOVLMController from rl_components", lambda: getattr(importlib.import_module("modules.rl_components"), "PPOVLMController")),
    ("MultipleHead from rl_components", lambda: getattr(importlib.import_module("modules.rl_components"), "MultipleHead")),
    ("RLDataCollatorWithPadding from rl_components", lambda: getattr(importlib.import_module("modules.rl_components"), "RLDataCollatorWithPadding")),
    ("get_dataset from data.registry", lambda: getattr(importlib.import_module("modules.data.registry"), "get_dataset")),
    ("get_model_wrapper from data.model_registry", lambda: getattr(importlib.import_module("modules.data.model_registry"), "get_model_wrapper")),
    ("get_evaluator from evaluation", lambda: getattr(importlib.import_module("modules.evaluation"), "get_evaluator")),
    ("generate_orthogonal_heads from embeddings", lambda: getattr(importlib.import_module("modules.embeddings"), "generate_orthogonal_heads")),
    ("parse_training_args from training", lambda: getattr(importlib.import_module("modules.training"), "parse_training_args")),
    ("run_ppo_loop from training", lambda: getattr(importlib.import_module("modules.training"), "run_ppo_loop")),
    # Top-level re-exports
    ("MultipleHead from modules (root)", lambda: getattr(importlib.import_module("modules"), "MultipleHead")),
    ("FastRLNode from modules (root)", lambda: getattr(importlib.import_module("modules"), "FastRLNode")),
]

for name, fn in CLASS_CHECKS:
    check(name, fn)


# ─── 3. Data Registry Instantiation ──────────────────────────────────────────
print("\n[3/5] Data & Model Registry Instantiation")
print("-" * 50)

def _check_sb_bench_adapter():
    from modules.data.registry import get_dataset
    adapter = get_dataset("sb_bench")
    assert hasattr(adapter, "extract_image_bytes")
    assert hasattr(adapter, "get_chosen_rejected")
    assert hasattr(adapter, "get_prompt_text")

def _check_pope_adapter():
    from modules.data.registry import get_dataset
    adapter = get_dataset("pope")
    assert hasattr(adapter, "extract_image_bytes")
    assert hasattr(adapter, "get_prompt_text")

def _check_qwen_wrapper():
    from modules.data.model_registry import get_model_wrapper
    wrapper = get_model_wrapper("qwen")
    assert hasattr(wrapper, "find_token_for_gating")
    assert hasattr(wrapper, "extract_pixel_values")

def _check_evaluator_pope():
    from modules.evaluation import get_evaluator
    ev = get_evaluator("pope")
    assert hasattr(ev, "evaluate")

def _check_evaluator_sb_bench():
    from modules.evaluation import get_evaluator
    ev = get_evaluator("sb_bench")
    assert hasattr(ev, "evaluate")

check("SB-Bench dataset adapter", _check_sb_bench_adapter)
check("POPE dataset adapter", _check_pope_adapter)
check("Qwen model wrapper", _check_qwen_wrapper)
check("POPE evaluator", _check_evaluator_pope)
check("SB-Bench evaluator", _check_evaluator_sb_bench)


# ─── 4. Subprocess Entrypoint Checks (--help) ────────────────────────────────
print("\n[4/5] Script Entrypoint Checks (argparse --help)")
print("-" * 50)

# These are the exact commands run_modal.py uses (minus data args).
# We just check they can parse --help (proves imports work at script level).
SCRIPT_COMMANDS = [
    ("modules.embeddings.extract --help", [PYTHON, "-m", "modules.embeddings.extract", "--help"]),
    ("modules.embeddings.generate_drm_heads --help", [PYTHON, "-m", "modules.embeddings.generate_drm_heads", "--help"]),
    ("modules.evaluation.eval_pope --help", [PYTHON, "-m", "modules.evaluation.eval_pope", "--help"]),
    ("modules.evaluation.eval_sb_bench --help", [PYTHON, "-m", "modules.evaluation.eval_sb_bench", "--help"]),
    ("modules.evaluation.evaluate_drm_heads --help", [PYTHON, "-m", "modules.evaluation.evaluate_drm_heads", "--help"]),
    ("modules.inference.generate_answers --help", [PYTHON, "-m", "modules.inference.generate_answers", "--help"]),
    ("modules.inference.generate_sb_bench_answers --help", [PYTHON, "-m", "modules.inference.generate_sb_bench_answers", "--help"]),
    ("modules.training.train_rl --help", [PYTHON, "-m", "modules.training.train_rl", "--help"]),
    ("modules.data.load_sb_bench (import only)", [PYTHON, "-c", "from modules.data.load_sb_bench import load_and_save_sb_bench"]),
    ("modules.data.load_pope (import only)", [PYTHON, "-c", "from modules.data.load_pope import load_and_save_pope"]),
]

for name, cmd in SCRIPT_COMMANDS:
    def _run(c=cmd):
        result = subprocess.run(c, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"Exit code {result.returncode}\nSTDERR: {result.stderr[-500:]}")
    check(name, _run)


# ─── 5. Modal App Definition Check ───────────────────────────────────────────
print("\n[5/5] Modal App Definition Check")
print("-" * 50)

def _check_modal_app():
    """Verify run_modal.py can be loaded and the Modal app is defined."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_modal", os.path.join(SRC_DIR, "run_modal.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "app"), "Modal 'app' not defined"
    assert hasattr(mod, "run_setup"), "run_setup function missing"
    assert hasattr(mod, "run_preprocess"), "run_preprocess function missing"
    assert hasattr(mod, "run_inference"), "run_inference function missing"
    assert hasattr(mod, "run_drm_generation"), "run_drm_generation function missing"
    assert hasattr(mod, "run_training"), "run_training function missing"
    assert hasattr(mod, "run_generation"), "run_generation function missing"
    assert hasattr(mod, "run_evaluation"), "run_evaluation function missing"
    assert hasattr(mod, "main"), "main entrypoint missing"

check("Modal app loads and all functions defined", _check_modal_app)


# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RESULTS: {len(passed)} passed, {len(failed)} failed")
print("=" * 60)

if failed:
    print("\nFailed checks:")
    for name, err in failed:
        print(f"  ✗ {name}")
        print(f"    → {err[:200]}")
    sys.exit(1)
else:
    print("\n✓ All sanity checks passed. Pipeline architecture is intact.")
    sys.exit(0)
