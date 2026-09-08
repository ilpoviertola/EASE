# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import sys

sys.path.append(".")

import csv
import random
import typing as tp
from pathlib import Path
from argparse import ArgumentParser

from tqdm import tqdm

from datasets.constants import (
    AVSB_LABELS2CATEGORY,
    OFFSCREEN_AUDIO_PROB,
    SILENT_AUDIO_PROB,
    NOISE_AUDIO_PROB,
    NEG_META_HEADER,
)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to the output CSV file to save the metadata.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to the AVSBench metadata CSV file.",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Path to the AVSBench dataset root directory.",
    )
    parser.add_argument(
        "--path_to_v1s_neg_samples",
        type=str,
        required=True,
        help="Path to the directory containing v1s VGGish embeddings (for S4).",
    )
    parser.add_argument(
        "--path_to_v1m_neg_samples",
        type=str,
        required=True,
        help="Path to the directory containing v1m VGGish embeddings (for MS3).",
    )
    parser.add_argument(
        "--path_to_v2_neg_samples",
        type=str,
        required=True,
        help="Path to the directory containing v2 VGGish embeddings (for SS).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--neg_audio_p",
        type=float,
        default=0.2,
        help="Probability of using negative audio (random, silent, offscreen).",
    )
    return parser.parse_args()


def get_offscreen_audio(
    current_cls: str, metadata: tp.List[tp.Dict[str, str]]
) -> tp.Tuple[tp.Optional[str], tp.Optional[str]]:
    # current_cls can have multiple labels divided by "_"
    current_cls = current_cls.split("_")
    # get categories for all of the labels
    current_ctg = set(
        [
            AVSB_LABELS2CATEGORY[label]
            for label in current_cls
            if label not in ["background", "off-the-screen"]
        ]
    )
    sample = random.sample(metadata, 1)[0]
    # get categories for the sample
    sample_ctg = set(
        [
            AVSB_LABELS2CATEGORY[label]
            for label in sample["a_obj"].split("_")
            if label not in ["background", "off-the-screen"]
        ]
    )
    retries = 0
    while sample_ctg.intersection(current_ctg):
        sample = random.sample(metadata, 1)[0]
        sample_ctg = set(
            [
                AVSB_LABELS2CATEGORY[label]
                for label in sample["a_obj"].split("_")
                if label not in ["background", "off-the-screen"]
            ]
        )
        retries += 1
        if retries > 50:
            return None, None
    return sample["uid"], sample["a_obj"]


def get_silent_audio(path_to_nvggish: Path) -> tp.Tuple[str, str]:
    return (path_to_nvggish / "silent_audio").as_posix(), "background"


def get_noise_audio(path_to_nvggish: Path) -> tp.Tuple[str, str]:
    d = random.choice(list(path_to_nvggish.glob("random_audio_*")))
    return (d).as_posix(), "background"


def generate_neg_audio_meta(
    avsb_task: str,
    neg_audio_p: float,
    metadata_path: Path,
    output_csv_path: Path,
    v1s_neg_samples: Path,
    v1m_neg_samples: Path,
    v2_neg_samples: Path,
    datapath: Path,
):
    weights = [OFFSCREEN_AUDIO_PROB, SILENT_AUDIO_PROB, NOISE_AUDIO_PROB]
    elements = ["offscreen", "silent", "noise"]

    with metadata_path.open("r") as f:
        reader = csv.DictReader(f)
        metadata = list(reader)
        metadata = [
            item
            for item in metadata
            if item["split"] in ["val", "test"] and item["label"] == avsb_task
        ]

    header_written = False
    with output_csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NEG_META_HEADER)
        if not header_written:
            writer.writeheader()
            header_written = True

        for item in tqdm(
            metadata, desc=f"Processing {avsb_task} metadata", total=len(metadata)
        ):
            if random.random() > neg_audio_p:
                continue

            na_type = random.choices(elements, weights=weights, k=1)[0]

            if na_type == "offscreen":
                na_path, na_label = get_offscreen_audio(item["a_obj"], metadata)
                if na_path is not None:
                    row = {
                        "vid": item["vid"],
                        "uid": item["uid"],
                        "orig_a_obj": item["a_obj"],
                        "s_min": item["s_min"],
                        "s_sec": item["s_sec"],
                        "split": item["split"],
                        "label": item["label"],
                        "neg_a_obj": na_label,
                        "use_noise": False,
                        "use_silent": False,
                        "use_offscreen": True,
                        "neg_vggish_emb_dir": (datapath / na_path).as_posix(),
                    }
                    writer.writerow(row)
            elif na_type == "silent":
                if task == "v1s":
                    path_to_nvggish = Path(v1s_neg_samples)
                elif task == "v1m":
                    path_to_nvggish = Path(v1m_neg_samples)
                else:  # ss
                    path_to_nvggish = Path(v2_neg_samples)
                na_path, na_label = get_silent_audio(path_to_nvggish)
                row = {
                    "vid": item["vid"],
                    "uid": item["uid"],
                    "orig_a_obj": item["a_obj"],
                    "s_min": item["s_min"],
                    "s_sec": item["s_sec"],
                    "split": item["split"],
                    "label": item["label"],
                    "neg_a_obj": na_label,
                    "use_noise": False,
                    "use_silent": True,
                    "use_offscreen": False,
                    "neg_vggish_emb_dir": na_path,
                }
                writer.writerow(row)
            elif na_type == "noise":
                if task == "v1s":
                    path_to_nvggish = Path(v1s_neg_samples)
                elif task == "v1m":
                    path_to_nvggish = Path(v1m_neg_samples)
                else:  # ss
                    path_to_nvggish = Path(v2_neg_samples)
                na_path, na_label = get_noise_audio(path_to_nvggish)
                row = {
                    "vid": item["vid"],
                    "uid": item["uid"],
                    "orig_a_obj": item["a_obj"],
                    "s_min": item["s_min"],
                    "s_sec": item["s_sec"],
                    "split": item["split"],
                    "label": item["label"],
                    "neg_a_obj": na_label,
                    "use_noise": True,
                    "use_silent": False,
                    "use_offscreen": False,
                    "neg_vggish_emb_dir": na_path,
                }
                writer.writerow(row)
            else:
                raise ValueError(f"Unknown negative audio type: {na_type}")


if __name__ == "__main__":
    args = get_args()

    if not (0.0 <= args.neg_audio_p <= 1.0):
        raise ValueError("neg_audio_p must be between 0 and 1.")
    if not Path(args.path_to_v1m_neg_samples).is_dir():
        raise ValueError(
            f"Path {args.path_to_v1m_neg_samples} is not a valid directory."
        )
    if not Path(args.path_to_v1s_neg_samples).is_dir():
        raise ValueError(
            f"Path {args.path_to_v1s_neg_samples} is not a valid directory."
        )
    if not Path(args.path_to_v2_neg_samples).is_dir():
        raise ValueError(
            f"Path {args.path_to_v2_neg_samples} is not a valid directory."
        )
    if not Path(args.datapath).is_dir():
        raise ValueError(f"Path {args.datapath} is not a valid directory.")
    if not Path(args.metadata).is_file():
        raise ValueError(f"Metadata file {args.metadata} does not exist.")
    if Path(args.output_csv).exists():
        Path(args.output_csv).unlink()

    random.seed(args.random_seed)
    output_csv_path = Path(args.output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata)
    path_to_v1s_vggish = Path(args.path_to_v1s_neg_samples)
    path_to_v1m_vggish = Path(args.path_to_v1m_neg_samples)
    path_to_v2_vggish = Path(args.path_to_v2_neg_samples)
    dataroot = Path(args.datapath)
    neg_audio_p = args.neg_audio_p

    tasks = ["v1s", "v1m", "v2"]
    for task in tasks:
        print(f"Generating negative audio metadata for task: {task}")
        generate_neg_audio_meta(
            avsb_task=task,
            neg_audio_p=neg_audio_p,
            metadata_path=metadata_path,
            output_csv_path=output_csv_path,
            v1s_neg_samples=path_to_v1s_vggish,
            v1m_neg_samples=path_to_v1m_vggish,
            v2_neg_samples=path_to_v2_vggish,
            datapath=dataroot / task,
        )
        print(f"Saved negative audio metadata to {output_csv_path}")
