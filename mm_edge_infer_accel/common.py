from __future__ import annotations

def quantization_plan(config: dict) -> dict:
    return {
        "kind": "quantization",
        "status": "planned",
        "config": config,
    }


def profile_command(name: str, config_path: str, tool: str) -> dict:
    if tool not in {"nsys", "ncu"}:
        raise ValueError("tool must be 'nsys' or 'ncu'")
    if tool == "nsys":
        command = (
            "nsys profile --sample=none --trace=cuda,nvtx,osrt --stats=true "
            "--force-overwrite=true "
            f"-o profiling/{name}_nsys "
            f"python -m mm_edge_infer_accel.cli benchmark --config {config_path} --run"
        )
    else:
        command = (
            "ncu --set full "
            f"-o profiling/{name}_ncu "
            f"python -m mm_edge_infer_accel.cli benchmark --config {config_path} --run"
        )
    return {
        "kind": "profile",
        "status": "planned",
        "tool": tool,
        "command": command,
    }
