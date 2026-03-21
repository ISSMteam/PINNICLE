import os
import runpy
import shutil
from pathlib import Path
from deepxde.backend import backend_name
import pytest
import pinnicle

def _link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)  # fast, no repo pollution, no big-file copy
    except OSError:
        shutil.copy2(src, dst)  # fallback (e.g., symlink not allowed)

def _mirror_example_inputs(example_dir: Path, workdir: Path):
    """
    Make workdir contain the same *input* files as example_dir so that
    relative file paths inside the example script still work, but outputs
    go to workdir (not the repo).
    """
    for p in example_dir.iterdir():
        # Skip python sources and typical junk
        if p.name in {"__pycache__", ".pytest_cache"}:
            continue
        if p.suffix == ".py":
            continue
        _link_or_copy(p, workdir / p.name)

@pytest.mark.skipif(backend_name in ["jax"], reason="skip testing examples with JAX")
@pytest.mark.parametrize("expfolder, expname", [
    ("example1_Helheim_inverse", "example1.py"),
    ("example2_PIG", "example2.py"),
    ("example3_Helheim_Transient", "example3.py"),
    ("example4_Coupled_Physics", "Coupled_Physics.py"),
])
def test_example1_runs_with_10_epochs(monkeypatch, tmp_path, expfolder, expname):
    # Path to your script
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / expfolder / expname
    assert script.exists()
    example_dir = script.parent

    # Make the temp working directory look like the script directory for inputs
    _mirror_example_inputs(example_dir, tmp_path)

    # Optional: run in a temp working directory so outputs don't pollute your repo
    monkeypatch.chdir(tmp_path)

    # Keep the original constructor
    RealPINN = pinnicle.PINN

    called = {"ok": False, "epochs_seen": None}

    def PINN_wrapper(hp, *args, **kwargs):
        # Force epochs to 10, without changing the script
        hp["epochs"] = 5
        called["ok"] = True
        called["epochs_seen"] = hp["epochs"]
        return RealPINN(hp, *args, **kwargs)

    # Patch pinnicle.PINN so the script uses our wrapper
    monkeypatch.setattr(pinnicle, "PINN", PINN_wrapper)

    # Execute the script as __main__ (it will do: compile(); train();)
    globals_after = runpy.run_path(str(script), run_name="__main__")

    # Assert our patch was actually used
    assert called["ok"], f"{expname} did not call pinnicle.PINN as expected"
    assert called["epochs_seen"] == 5

    # Optional: if the script leaves `experiment` in globals, you can sanity-check it
    # (only if PINN exposes something like hp / config)
    if "experiment" in globals_after:
        exp = globals_after["experiment"]
        # Try a couple of common patterns; adjust if you know the exact attribute
        if hasattr(exp, "hp"):
            assert exp.hp.get("epochs") == 5

