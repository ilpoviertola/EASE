# EASE: Encoder-only Audio-Visual Segmentation

[Ilpo Viertola](https://scholar.google.com/citations?user=gGWNg4EAAAAJ&hl=en), [Vladimir Iashin](https://scholar.google.com/citations?user=rh8_sSkAAAAJ&hl=en), and [Esa Rahtu](https://scholar.google.com/citations?user=SmGZwHYAAAAJ&hl=en)

## Installation

We support using Conda/Miniconda for environment management.
This code has been tested on the following OS:

- Ubuntu 22.04 and 24.04 LTS
- RHEL 8.10
- SLES 15.6

```bash
git clone git@github.com:ilpoviertola/EASE.git
cd EASE
```

### Basic environment setup

Check the PyTorch to match your system requirements.

```bash
conda create --name ease python=3.12 -y
conda activate ease
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

### Roihu environment setup

If you are using the Roihu cluster, please follow the instructions below to set up the environment.

```bash
# Connect to GPU login node
sinteractive
module load tykky
mkdir env
conda-containerize new --prefix env roihu_env/env.yml
conda-containerize update env --post-install roihu_env/post-install.txt
```

## Dataset

We use the AVSBench dataset for training and evaluation. Please refer to the [AVSBench repository](https://github.com/OpenNLPLab/AVSBench/tree/main) for instructions on downloading the dataset.

### Preparing the dataset

Extract the VGGIsh audio embeddings for the S4/MS3/AVSS subsets. The script should be executed separately for each subset (v1s, v1m, v2). The extracted embeddings will be stored in the same directory as the original audio files.

```bash
python scripts/extract_vggish_embeddings.py --dataset_path /path/to/avsbench/dataset/[v1s, v1m, v2] --vggish_ckpt_path /path/to/vggish/ckpt
```

## Training

You can find the different configuration files for training in the `config/avsbench` directory. Below is an example command for training the PVTv2-S4 model. To train other models, simply replace the configuration file with the corresponding one.

NOTE: PVTv2 does not support compiling, so the `--compile_disabled` flag is required for training. For other models, you can choose to enable the compilation.

### Example: PVTv2 - S4 - 224x224

```bash
python main.py fit -c config/avsbench/pvtv2/s4/exps-ease_b5_224/clustering-enhance-71_av-aligner_fuse-all-steps+lf.yaml --data.path /path/to/avsbench/dataset --trainer.devices num_gpus --compile_disabled
```

## Testing

To evaluate the trained model, use the following command. Make sure to replace the configuration file and checkpoint path with the appropriate ones.

### Example: DINOv3 - MS3 - 224x224

```bash
python main.py test -c config/avsbench/dinov3/ms3/exps-ease_base_224/clustering-enhance-71_av-aligner_fuse-all-steps+lf.yaml --data.path /path/to/avsbench/dataset --trainer.devices num_gpus --model.ckpt_path /path/to/trained/model/checkpoint.ckpt
```

## Acknowledgements

Thanks for the teams of:

- [EoMT](https://github.com/tue-mps/eomt) (MIT License)
- [DDESeg](https://github.com/YenanLiu/DDESeg/tree/main?tab=readme-ov-file) (Apache-2.0 License)

## TODOs

- [ ] Add project page and ArXiv link.
