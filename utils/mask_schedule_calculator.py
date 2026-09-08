# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import argparse

# train samples
AVSB_SPLIT_TO_SAMPLE_COUNT = {"s4": 3452, "ms3": 296, "ss": (4750 + 3452 + 296)}
AVSB_SPLIT_TO_SAMPLE_COUNT_PF = {
    "s4": 3452,
    "ms3": 296 * 5,
    "ss": (4750 * 10 + 3452 + 296 * 5),
}


def get_args():
    parser = argparse.ArgumentParser(description="Mask Schedule Calculator")
    parser.add_argument(
        "--max_epochs", type=int, default=12, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training"
    )
    parser.add_argument(
        "--device_count",
        type=int,
        default=1,
        help="Number of devices to use for training",
    )
    parser.add_argument(
        "--data_split", type=str, default="s4", help="AVSBench data split"
    )
    parser.add_argument(
        "--anneal_start", type=int, default=2, help="Epoch to start annealing"
    )
    parser.add_argument(
        "--anneal_length", type=int, default=2, help="Length of a annealing period"
    )
    parser.add_argument(
        "--anneal_periods", type=int, default=4, help="Number of annealing periods"
    )
    parser.add_argument(
        "--per_frame", action="store_true", help="Whether to use per-frame data loading"
    )
    return parser.parse_args()


def calculate_schedule(
    max_epochs: int,
    batch_size: int,
    device_count: int,
    data_split: str,
    anneal_start: int,
    anneal_length: int,
    anneal_periods: int,
    per_frame: bool = False,
):
    assert anneal_start > 0, "Annealing start epoch must be greater than 0"
    assert anneal_length > 0, "Annealing length must be greater than 0"
    assert anneal_periods > 0, "Annealing periods must be greater than 0"
    assert max_epochs > 0, "Maximum epochs must be greater than 0"
    if per_frame:
        avsb_split_to_sample_count = AVSB_SPLIT_TO_SAMPLE_COUNT_PF
    else:
        avsb_split_to_sample_count = AVSB_SPLIT_TO_SAMPLE_COUNT
    assert data_split in avsb_split_to_sample_count, f"Unknown data split: {data_split}"
    assert (
        anneal_start + anneal_length * anneal_periods <= max_epochs
    ), "Total epochs for annealing exceeds maximum epochs"

    bs = batch_size * device_count
    total_samples = avsb_split_to_sample_count[data_split]
    steps_per_epoch = total_samples // bs
    print(f"Total samples: {total_samples}")
    print(f"Steps per epoch: {steps_per_epoch}")

    start_steps, end_steps = [], []
    for i in range(anneal_periods):
        start_step = (anneal_start + i * anneal_length) * steps_per_epoch
        end_step = start_step + anneal_length * steps_per_epoch
        print(f"Annealing period {i + 1}:")
        print(f"  Start step: {start_step}")
        print(f"  End step: {end_step}")
        start_steps.append(start_step)
        end_steps.append(end_step)

    print(f"Start steps: {start_steps}")
    print(f"End steps: {end_steps}")


if __name__ == "__main__":
    args = get_args()
    calculate_schedule(
        args.max_epochs,
        args.batch_size,
        args.device_count,
        args.data_split,
        args.anneal_start,
        args.anneal_length,
        args.anneal_periods,
        args.per_frame,
    )
