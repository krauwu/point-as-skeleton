# DWM / PAS Generation Stack

This directory contains the generation-side implementation of **Point as Skeleton (PAS)**. It is adapted from [OpenDWM](https://github.com/SenseTime-FVG/OpenDWM) and adds point-cloud skeleton conditioning for nuScenes and nuPlan generation.

## Environment

Start from the [OpenDWM](https://github.com/SenseTime-FVG/OpenDWM) environment setup. This repository follows the same general training stack.

The main version difference used in our experiments is:

```text
diffusers==0.37.0
```

The generation-side code does not rely on custom CUDA operators. Users may try newer versions of `diffusers`, `torch`, and CUDA-compatible packages if they are compatible with their checkpoints and hardware.

Before running training or preview code from the repository root, expose the source tree:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

The external packages as FVD could also export as [OpenDWM](https://github.com/SenseTime-FVG/OpenDWM) mentioned.

## Public initialization checkpoints

The default initialization uses:

| Component | Link |
| --- | --- |
| Base diffusion model | [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) |
| OpenDWM initialization checkpoint | [ctsd_35_df16_tirda_bm_nwao_40k.pth](https://huggingface.co/wzhgba/opendwm-models/resolve/main/ctsd_35_df16_tirda_bm_nwao_40k.pth?download=true) |

## PAS checkpoints and assets

We provide or plan to provide the following resources separately from the code release.

| Resource | Link | Notes |
| --- | --- | --- |
| nuScenes foreground/background point assets | [Download](https://pan.baidu.com/s/1ZS4iYvlA9_99peCs_iNssA?pwd=4930) | Precomputed nuScenes PAS point-cloud assets. |
| nuPlan background point cloud for `2021.05.12.22.28.35_veh-35_00620_01164` | [Download](https://pan.baidu.com/s/13dkOwreppg0u8mNyV6POZg?pwd=4930) | Background asset for the default nuPlan scene. |
| nuPlan foreground actor assets | [Download](https://pan.baidu.com/s/1p45vjTZ2HnbpI-lScuZCew?pwd=4930) | Actor tracks and category templates used by the nuPlan interface. |
| nuPlan SFT checkpoint before PAS conditioning | [Download](https://pan.baidu.com/s/108mctkWJDfNqz6BbOeJsJQ?pwd=4930) | Use this as the initialization for nuPlan PAS training. |
| nuPlan PAS checkpoint | [Download](https://pan.baidu.com/s/168k-Jv2YgvjiaOIHOA_y7g?pwd=4930) | Use this checkpoint with the nuPlan closed-loop interface. |
| nuScenes PAS checkpoint | [Download](https://pan.baidu.com/s/14AK4gyR2spohirET7yw9mA?pwd=4930) | The nuScenes checkpoint can be used with the long-rollout config. |

## Directory structure

```text
configs/pas/
├── nus/
│   ├── pas_train.json       nuScenes PAS training config
│   └── pas_long.json        nuScenes long-horizon preview / evaluation config
└── nuplan/
    ├── nuplan-dwm.json      nuPlan SFT config initialized from OpenDWM
    └── nuplan-pas.json      nuPlan PAS config initialized from nuPlan SFT

src/dwm/
├── datasets/                nuScenes, nuPlan, and other dataset readers
├── models/                  PAS-related model components
├── pipelines/               DWM / PAS diffusion pipelines
├── tools/data_tools/        point-cloud asset construction tools
├── train.py                 training entry point
└── preview.py               preview / generation entry point
```

## Point-cloud asset construction

Point-cloud skeletons can be built with the utilities under:

```text
src/dwm/tools/data_tools/
```

The most relevant scripts are:

| Script | Purpose |
| --- | --- |
| `bk_nus.py` | Build nuScenes background point-cloud assets. |
| `track_actor_gen_nus.py` | Build nuScenes foreground actor assets. |
| `bk_nuplan.py` | Build nuPlan background point-cloud assets. |
| `track_actor_gen.py` | Build nuPlan foreground actor assets. |
| `actor_template_nuplan.py` | Build category-level nuPlan actor templates. |
| `pts_down.py` | Downsample point-cloud assets. |
| `pts_cut.py` | Crop or filter point-cloud assets. |

These tools use dataset calibration, 3D boxes, track IDs, and camera images to construct foreground/background assets. Some scripts use MMCV-style 3D box utilities. If your main training environment does not include MMCV, you can create a separate preprocessing environment for asset construction.

A typical asset workflow is:

```text
raw dataset lidar + camera + boxes
  -> background accumulation
  -> foreground actor extraction
  -> optional actor template construction
  -> optional downsampling / cropping
  -> projected PAS conditions during training or simulation
```

If you do not want to build assets from scratch, use the precomputed asset links above and update the config paths accordingly.

## Dataset preparation

Data preparation can follow the dataset implementations in `src/dwm/datasets`,

## Training

### nuScenes PAS

nuScenes PAS can be initialized directly from the OpenDWM checkpoint.

```bash
torchrun --nproc_per_node=<num_gpus> src/dwm/train.py \
  --config-path configs/pas/nus/pas_train.json \
  --output-path outputs/pas_nus
```

### nuPlan SFT

Before PAS point-cloud training on nuPlan, first adapt the OpenDWM checkpoint to the nuPlan data distribution.

```bash
torchrun --nproc_per_node=<num_gpus> src/dwm/train.py \
  --config-path configs/pas/nuplan/nuplan-dwm.json \
  --output-path outputs/nuplan_sft
```

### nuPlan PAS

After nuPlan SFT, train PAS with point-cloud conditioning.

```bash
torchrun --nproc_per_node=<num_gpus> src/dwm/train.py \
  --config-path configs/pas/nuplan/nuplan-pas.json \
  --output-path outputs/nuplan_pas
```

Make sure `model_checkpoint_path` in `nuplan-pas.json` points to the nuPlan SFT checkpoint.

## Preview and inference

### nuScenes long-horizon generation

Use the long-rollout config with a trained nuScenes PAS checkpoint:

```bash
python src/dwm/preview.py \
  --config-path configs/pas/nus/pas_long.json \
  --output-path outputs/nus_pas_long
```

### nuPlan generation

A trained nuPlan PAS checkpoint can be used in two ways:

- standard preview through DWM configs;
- closed-loop scene generation through the nuPlan interface.

For closed-loop simulation, see:

```text
nuplan/README.md
```

