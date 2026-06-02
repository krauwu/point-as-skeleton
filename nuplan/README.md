# nuPlan Scene Interface

This module connects a trained PAS generator with nuPlan scene rollout. It is intended as a lightweight interface for closed-loop generative simulation and website-side demos.

The interface starts from a selected nuPlan DB scene. It can either replay the original future trajectory from the database or switch to a user-controlled ego trajectory after warm-up. At every step, the simulator state is converted into DWM/PAS conditions, point-cloud skeletons are projected into camera views, and the generator renders the next observation.

## What this module provides

| Mode | Description |
| --- | --- |
| `db` | Replays the original ego trajectory from the nuPlan DB. This is useful for sanity checks and log-aligned generation. |
| `trajectory` | Uses DB frames for warm-up and then follows a preset or user-provided local ego trajectory. The trajectory interface can be extended to accept commands from an ego planner. |

The default scene is:

```text
2021.05.12.22.28.35_veh-35_00620_01164
```

## Directory structure

```text
nuplan/
├── launch.py                         
├── nuplan_text.json                  text annotation of nuplan
├── sim_tools/
│   ├── config_paths.yaml             user-editable path config
│   ├── configs/nuplan_site.json      PAS generation config for scene rollout
│   ├── nuplan_sim_db.py              DB replay runner
│   ├── nuplan_sim_trajectory.py      user-trajectory runner
│   ├── prepare_cond.py               condition construction utilities
│   ├── prepare_cond_gpu.py           GPU condition construction utilities
│   ├── optimized_projection.py       projected point-cloud utilities
│   └── generator.py                  streaming generator client
├── assets/
│   ├── trajectories/                 example user trajectory files
│   └── pointcloud/                   scene and actor point-cloud assets
├── checkpoints/                      local model checkpoints
├── data/                             local nuPlan data
└── outputs/                          generated outputs
```

## Environment

Use the DWM / PAS generation environment described in:

```text
../src/dwm/README.md
```

In addition, install the official [nuPlan devkit](https://github.com/motional/nuplan-devkit) and prepare the nuPlan dataset, maps, DB files, and sensor blobs.

When running from this directory, expose both the PAS source tree and this module:

```bash
export PYTHONPATH=$PWD/../src:$PWD:$PYTHONPATH
```

## Required checkpoints

Place the base model and the trained nuPlan PAS checkpoint under:

```text
nuplan/checkpoints/
├── stable-diffusion-3.5-medium/
└── nuplan_pas_checkpoint.pth
```

The base diffusion model is:

```text
https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
```

For nuPlan closed-loop generation, use the nuPlan PAS checkpoint trained through:

```text
OpenDWM checkpoint
  -> nuPlan SFT
  -> nuPlan PAS training
```

> [!NOTE]
> A released nuPlan PAS checkpoint is available at [Download](https://pan.baidu.com/s/168k-Jv2YgvjiaOIHOA_y7g?pwd=4930).

## Required data and assets

For the default scene, prepare:

```text
nuplan/data/nuplan/nuplan-v1.1/splits/mini/2021.05.12.22.28.35_veh-35_00620_01164.db
nuplan/data/nuplan/nuplan-v1.1/sensor_blobs
nuplan/data/nuplan/maps
nuplan/assets/pointcloud/bg2021.05.12.22.28.35_veh-35_00620_01164_vox0.01.npy
nuplan/assets/pointcloud/actors/track_actor
nuplan/assets/pointcloud/actors/bike.pkl
nuplan/assets/pointcloud/actors/ped.pkl
nuplan/assets/pointcloud/actors/pickup.pkl
nuplan/assets/pointcloud/actors/sedan.pkl
nuplan/assets/pointcloud/actors/suv.pkl
nuplan/nuplan_text.json
```

Precomputed nuPlan point-cloud assets:

| Asset | Link |
| --- | --- |
| Default-scene background point cloud | [Download](https://pan.baidu.com/s/13dkOwreppg0u8mNyV6POZg?pwd=4930) |
| Foreground actor assets and templates | [Download](https://pan.baidu.com/s/1p45vjTZ2HnbpI-lScuZCew?pwd=4930) |

If you use another scene, provide the corresponding DB file and background point cloud, or pass explicit paths through the launcher.

## Run

All scene generation uses the same entry point:

```bash
cd nuplan
python launch.py --mode trajectory --scene-name 2021.05.12.22.28.35_veh-35_00620_01164 --gpu-id 0
```

Replay the original nuPlan DB trajectory:

```bash
python launch.py --mode db --scene-name 2021.05.12.22.28.35_veh-35_00620_01164 --gpu-id 0
```

Use a preset user trajectory:

```bash
python launch.py \
  --mode trajectory \
  --scene-name 2021.05.12.22.28.35_veh-35_00620_01164 \
  --trajectory-type right_then_back \
  --gpu-id 0
```

Use a custom local trajectory file:

```bash
python launch.py \
  --mode trajectory \
  --scene-name 2021.05.12.22.28.35_veh-35_00620_01164 \
  --trajectory-file assets/trajectories/example_local_xy.json \
  --gpu-id 0
```

Override asset paths if your files do not follow the default layout:

```bash
python launch.py \
  --mode trajectory \
  --scene-name <scene-name> \
  --db-file <path-to-scene-db> \
  --clr-scene-file <path-to-background-pointcloud.npy> \
  --output-dir <path-to-output-dir> \
  --gpu-id 0
```

## User trajectory format

A JSON trajectory file should contain local ego-frame `(x, y)` points:

```json
{
  "points": [[0.0, 0.0], [4.0, 0.0], [8.0, -0.6], [12.0, 0.0]]
}
```

The launcher also supports:

```text
.csv / .txt files with two columns
.npy arrays with shape [N, 2]
```

The points are interpreted in the local ego frame and resampled to the future horizon used by the rollout runner.

## Config files

Two config files control most local settings:

```text
sim_tools/config_paths.yaml
sim_tools/configs/nuplan_site.json
```

`config_paths.yaml` stores local paths such as nuPlan data roots, maps, sensor blobs, point-cloud assets, and output folders.

`nuplan_site.json` stores the DWM/PAS generation config used by the scene interface.

## Runtime overrides

Some config values are placeholders for compatibility with the training config. The launcher overwrites them at runtime.

| Field | Runtime source |
| --- | --- |
| `NUPLAN_DB_FILES` | `--db-file` or `--scene-name` |
| `STAGE3_OUTPUT_DIR` | `--output-dir` |
| `CUDA_VISIBLE_DEVICES` | `--gpu-id` |
| `TARGET_CLR_SCENE_FILE` | `--clr-scene-file` or `--scene-name` |
| `MAX_SIM_STEPS` | `--max-steps` |
| `TRAJECTORY_TYPE` | `--trajectory-type` |
| `TRAJECTORY_FILE` | `--trajectory-file` |
| `cfg.scenario_builder.db_files` | selected scene DB |
| `cfg.scenario_builder.map_root` | `config_paths.yaml` |
| `cfg.scenario_builder.map_version` | `config_paths.yaml` |
| `cfg.scenario_filter.limit_total_scenarios` | launcher argument |
| `projected_pc_settings.color_scene_by_location` | selected background point cloud |
| `projected_pc_settings.actor_root` | `config_paths.yaml` |
| `projected_pc_settings.actor_template_root` | `config_paths.yaml` |
| `pipeline.common_config.distribution_framework` | set for single-process inference |
| `pipeline.common_config.ddp_wrapper_settings` | removed during single-process inference |
| `pipeline.common_config.t5_fsdp_wrapper_settings` | removed during single-process inference |
| `pipeline.metrics` | removed during scene inference |

If a field appears both in JSON and in launcher arguments, the launcher argument is authoritative.

