"""Export LIBERO dataset episodes to a FlashRT-friendly NPZ for offline eval."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mm_edge_infer_accel.pi05_runtime import set_pi05_cache_env


def scalar_int(value) -> int:
    return int(value.item() if hasattr(value, "item") else value)


def to_hwc_uint8(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)
        value = tensor.numpy()

    array = np.asarray(value)
    if array.dtype != np.uint8:
        if array.size and float(array.max()) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LIBERO frames to a FlashRT-friendly npz.")
    parser.add_argument("--dataset-id", default="HuggingFaceVLA/libero")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--episodes",
        help="Comma-separated episode ids. Overrides --episode when set.",
    )
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--output", default="outputs/libero_episode0_100frames.npz")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_pi05_cache_env()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = (
        [int(value) for value in args.episodes.split(",") if value.strip()]
        if args.episodes
        else [args.episode]
    )

    images = []
    wrist_images = []
    states = []
    tasks = []
    frame_indices = []
    episode_indices = []
    dataset_indices = []
    reference_actions = []

    for episode in episodes:
        dataset = LeRobotDataset(repo_id=args.dataset_id, episodes=[episode])
        count = min(args.sample_count, len(dataset))
        for index in range(count):
            raw = dataset[index]
            images.append(to_hwc_uint8(raw["observation.images.image"]))
            wrist_images.append(to_hwc_uint8(raw["observation.images.image2"]))
            states.append(np.asarray(raw["observation.state"].detach().cpu(), dtype=np.float32))
            tasks.append(str(raw["task"]))
            frame_indices.append(scalar_int(raw["frame_index"]))
            episode_indices.append(scalar_int(raw["episode_index"]))
            dataset_indices.append(scalar_int(raw["index"]))
            action = raw["action"]
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            reference_actions.append(np.asarray(action, dtype=np.float32))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        dataset_id=np.asarray(args.dataset_id),
        episodes=np.asarray(episodes, dtype=np.int64),
        sample_count_per_episode=np.asarray(args.sample_count, dtype=np.int64),
        images=np.stack(images),
        wrist_images=np.stack(wrist_images),
        states=np.stack(states).astype(np.float32),
        tasks=np.asarray(tasks),
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        episode_indices=np.asarray(episode_indices, dtype=np.int64),
        dataset_indices=np.asarray(dataset_indices, dtype=np.int64),
        reference_actions=np.stack(reference_actions).astype(np.float32),
    )

    print(f"wrote {output}")
    print("images", np.stack(images).shape, np.stack(images).dtype)
    print("wrist_images", np.stack(wrist_images).shape, np.stack(wrist_images).dtype)
    print("states", np.stack(states).shape, np.stack(states).dtype)
    print("episodes", episodes)
    print("tasks", len(tasks), tasks[0] if tasks else "")
    print("reference_actions", np.stack(reference_actions).shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
