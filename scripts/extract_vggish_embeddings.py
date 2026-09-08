# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import sys

sys.path.append(".")
import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio import load
import pytorch_lightning as pl

from models.vggish.vggish import VGGish


torch.set_float32_matmul_precision("high")


VGGISH_URLS = {
    "vggish": "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth",
    "pca": "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish_pca_params-970ea276.pth",
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract VGGish embeddings from audio files."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Path to the directory containing audio files.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for DataLoader.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the embeddings. If not provided, embeddings will be saved in the same directory as the audio files.",
    )
    parser.add_argument(
        "--vggish_ckpt_path",
        type=str,
        default=None,
        help="Path to a local VGGish checkpoint file. If not provided, the pretrained weights will be downloaded.",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Set this if the VGGish ckpt is trained with DenseAV strategy.",
    )
    parser.add_argument(
        "--neg_audio_len",
        type=float,
        default=5.0,
        help="Length of negative audio samples in seconds (only used if --negative is set).",
    )
    parser.add_argument(
        "--negative",
        action="store_true",
        help="If set, use NegativeAudioDataset instead of AudioDataset.",
    )
    return parser.parse_args()


class AudioDataset(Dataset):
    def __init__(self, datapath: str, dense: bool = False):
        super().__init__()
        self.datapath = datapath
        self.data = self._load_data()
        self.dense = dense

    def _load_data(self):
        data = Path(self.datapath).rglob("*.wav")
        return list(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path: Path = self.data[index]
        waveform, sample_rate = load(
            file_path.as_posix(),
            normalize=True,
            channels_first=False,
            format=f"{file_path.suffix[1:]}",  # e.g., "wav"
        )
        if self.dense:
            waveform -= waveform.mean()
        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "filename": Path(file_path).parent.stem,
        }


class NegativeAudioDataset(Dataset):
    def __init__(
        self,
        datapath: str,
        sample_count: int = 500,
        sample_rate: int = 16000,
        a_len: float = 5.0,
        random_seed: int = 42,
        dense: bool = False,
    ):
        super().__init__()
        self.datapath = datapath
        self.sample_count = sample_count
        self.sample_rate = sample_rate
        self.a_len = a_len
        self.dense = dense
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        if index == 0:
            audio = torch.zeros(
                int(self.sample_rate * self.a_len), 1, dtype=torch.float32
            )
            fn = "silent_audio"
        else:
            audio = torch.randn(
                int(self.sample_rate * self.a_len), 1, dtype=torch.float32
            )
            if self.dense:
                audio -= audio.mean()
            fn = f"random_audio_{index:03d}"

        return {
            "waveform": audio,
            "sample_rate": self.sample_rate,
            "filename": fn,
        }


class AudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datapath: str,
        num_workers: int = 4,
        dense: bool = False,
        negative: bool = False,
        neg_audio_len: float = 5.0,
    ):
        super().__init__()
        self.datapath = datapath
        self.batch_size = 1
        self.num_workers = num_workers
        self.dense = dense
        self.negative = negative
        self.neg_audio_len = neg_audio_len

    def setup(self, stage=None):
        if self.negative:
            self.dataset = NegativeAudioDataset(
                datapath=self.datapath, a_len=self.neg_audio_len, dense=self.dense
            )
        else:
            self.dataset = AudioDataset(datapath=self.datapath, dense=self.dense)

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class VGGIshModel(pl.LightningModule):
    def __init__(
        self,
        save_path: str,
        urls: dict = VGGISH_URLS,
        preprocess: bool = True,
        postprocess: bool = False,
        vggish_ckpt_path: Optional[str] = None,
        dense: bool = False,
    ):
        super().__init__()
        self.save_path = Path(save_path)
        self.vggish = VGGish(
            urls=urls,
            preprocess=preprocess,
            postprocess=postprocess,
            vggish_ckpt_path=vggish_ckpt_path,
            dense=dense,
        )
        self.preprocess = preprocess
        if dense:
            self.fn = "dense_vggish_emb.pt"
        else:
            self.fn = "vggish_emb.pt"

    def forward(self, x, fs):
        return self.vggish(x, fs)

    def predict_step(self, batch, batch_idx):
        waveform = batch["waveform"]
        fs = batch["sample_rate"][0]
        if self.preprocess:
            # If we need to preprocess the waveform we are using BS=1
            waveform = waveform[0].cpu().numpy()
            fs = fs.item() if isinstance(fs, torch.Tensor) else fs
        embeddings = self.forward(waveform, fs)
        return embeddings

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs.dim() == 2:
            outputs = outputs.unsqueeze(0)  # Ensure outputs are 3D for batch processing

        for i, output in enumerate(outputs):
            filename = batch["filename"][i]
            embedding = output.cpu()
            self.save_embedding(embedding, filename)

    def save_embedding(self, embedding: torch.Tensor, filename: str):
        if not (self.save_path / filename).exists():
            (self.save_path / filename).mkdir(parents=True, exist_ok=False)
        save_path = self.save_path / filename / self.fn
        torch.save(embedding.cpu(), save_path)


def main(args: argparse.Namespace):
    data_module = AudioDataModule(
        datapath=args.datapath,
        num_workers=args.num_workers,
        dense=args.dense,
        negative=args.negative,
        neg_audio_len=args.neg_audio_len,
    )
    data_module.setup()

    save_path = args.save_path if args.save_path else args.datapath
    model = VGGIshModel(
        save_path=save_path,
        vggish_ckpt_path=args.vggish_ckpt_path,
        dense=args.dense,
    )
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_progress_bar=True,
        enable_checkpointing=False,
    )
    trainer.predict(model, data_module)


if __name__ == "__main__":
    args = get_args()
    main(args)
