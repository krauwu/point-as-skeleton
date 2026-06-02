## Dataset Preparation

PAS uses a local file-system layout. The original OpenDWM TGZ-to-ZIP conversion step is not required for this release.

The dataset behavior is different for nuScenes and nuPlan:

* **nuScenes** is loaded directly by `dwm.datasets.nuscenes.MotionDataset` from local JSON tables.
* **nuPlan** is loaded by `dwm.datasets.nuplan.NuPlanDataset` from pre-generated PKL info files.
* Point-cloud skeleton assets are prepared separately and then referenced by the config.

---

## nuScenes

1. Download and extract the official [nuScenes](https://www.nuscenes.org/download) dataset to `{NUSCENES_ROOT}`.

   A typical local structure is:

   ```text
   {NUSCENES_ROOT}/
   ├── samples/
   ├── sweeps/
   ├── maps/
   └── v1.0-trainval/
   ```

2. Download and prepare the 12Hz nuScenes metadata.

   We provide the prepared 12Hz metadata [here](https://pan.baidu.com/s/1LfQ0xoROOZUOAqvL25ZqQg?pwd=4930):

   After extraction, place it under `{NUSCENES_ROOT}`. The expected structure is:

   ```text
   {NUSCENES_ROOT}/
   ├── samples/
   ├── sweeps/
   ├── maps/
   └── interp_12Hz_trainval/
       ├── calibrated_sensor.json
       ├── category.json
       ├── ego_pose.json
       ├── instance.json
       ├── log.json
       ├── map.json
       ├── sample.json
       ├── sample_annotation.json
       ├── sample_data.json
       ├── scene.json
       └── sensor.json
   ```

3. Prepare the nuScenes map expansion files.

   Download `nuScenes-map-expansion-v1.3.zip` from the nuScenes website and extract it into:

   ```text
   {NUSCENES_ROOT}/maps/
   ```

   The dataset loader reads HD map files through the same local file system root.

4. Prepare text annotations if text prompts are used.

   The config expects:

   ```text
   ./data/nuscenes/captions/nuscenes_caption_train_or_val.json
   ./data/nuscenes/captions/nuscenes_caption_times_train_or_val.json
   ```

   Update these paths if your caption files are stored elsewhere.

5. Prepare or download the nuScenes point-cloud skeleton assets.

   The PAS config expects foreground and background assets such as:

   ```text
   ./data/nuscenes/pointcloud/boston-seaport_point_cloud.npy
   ./data/nuscenes/pointcloud/singapore-onenorth_point_cloud.npy
   ./data/nuscenes/pointcloud/singapore-queenstown_point_cloud.npy
   ./data/nuscenes/pointcloud/singapore-hollandvillage_point_cloud.npy
   ./data/nuscenes/actors/track_actor/
   ./data/nuscenes/actors/templates/
   ```

   If you do not use the default layout, update `projected_pc_settings` in the config.

6. Update the nuScenes config files:

   ```text
   configs/pas/nus/pas_train.json
   configs/pas/nus/pas_long.json
   ```

   The main fields to check are:

   ```text
   global_state.nuscenes_fs.path
   training_dataset.base_dataset.datasets[0].dataset_name
   validation_dataset.base_dataset.datasets[0].dataset_name
   cache_root
   image_description_settings.path
   image_description_settings.time_list_dict_path
   projected_pc_settings.color_scene_by_location
   projected_pc_settings.actor_root
   projected_pc_settings.actor_template_root
   ```

   For the provided 12Hz metadata, the dataset name should be:

   ```text
   interp_12Hz_trainval
   ```

---

## nuPlan

nuPlan does not use the nuScenes-style JSON table loader. It first needs PKL info files generated from the nuPlan DB logs.

1. Prepare the official nuPlan data.

   A typical local structure is:

   ```text
   {NUPLAN_ROOT}/
   ├── nuplan-v1.1/
   │   ├── splits/
   │   │   └── mini/
   │   │       ├── <log_name>.db
   │   │       └── ...
   │   └── sensor_blobs/
   └── maps/
   ```

   The config usually points to:

   ```text
   ./data/nuplan/nuplan-v1.1/splits/mini
   ./data/nuplan/nuplan-v1.1/sensor_blobs
   ./data/nuplan/maps
   ```

2. Generate nuPlan PKL info files.

   The helper script is:

   ```text
   src/dwm/tools/nuplan_info.py
   ```

   Before running it, update the paths at the bottom of the script according to your local dataset layout:

   ```python
   sensor_blobs_root = "./data/nuplan/nuplan-v1.1/sensor_blobs"
   dataset_root = "./data/nuplan/nuplan-v1.1"
   data_root = "<your-nuplan-db-root>"
   map_root = "<your-nuplan-map-root>"
   out_path = "./data/cache/nuplan_cache"
   ```

   Then run:

   ```bash
   export PYTHONPATH=$PWD/src:$PYTHONPATH
   python src/dwm/tools/nuplan_info.py
   ```

   The script writes PKL files in the form:

   ```text
   {version}_infos_train.pkl
   {version}_infos_val.pkl
   ```

   Move or rename them to match the paths used by your config, or directly update `pkl_path` in the config.

3. Update the nuPlan SFT config:

   ```text
   configs/pas/nuplan/nuplan-dwm.json
   ```

   The key fields are:

   ```text
   sensor_root
   pkl_path
   cache_root
   dataset_root
   map_root
   image_description_settings.path
   ```

   The default config uses:

   ```text
   ./data/cache/nuplan_cache/mini_full_infos_train.pkl
   ./data/cache/nuplan_cache/mini_full_infos_val.pkl
   ```

4. Update the nuPlan PAS config:

   ```text
   configs/pas/nuplan/nuplan-pas.json
   ```

   In addition to the fields above, PAS also requires point-cloud skeleton assets:

   ```text
   projected_pc_settings.color_scene_by_location
   projected_pc_settings.actor_root
   projected_pc_settings.actor_template_root
   ```

   The default config expects paths such as:

   ```text
   ./data/nuplan/pointcloud/bg<log_name>_vox0.01.npy
   ./data/nuplan/actors/track_actor/
   ./data/nuplan/actors/
   ```

   The default PKL paths are:

   ```text
   ./data/cache/nuplan_cache/mini_infos_train_pas.pkl
   ./data/cache/nuplan_cache/mini_infos_val_pas.pkl
   ```

5. Prepare nuPlan text annotations.

   The repository provides:

   ```text
   nuplan/nuplan_text.json
   ```

   Make sure `image_description_settings.path` points to this file or to your own text annotation file.

6. Prepare or download the nuPlan point-cloud skeleton assets.

   For PAS training or closed-loop rollout, the config needs:

   ```text
   background point clouds
   foreground actor assets
   category-level actor templates
   ```

   These paths are controlled by:

   ```text
   projected_pc_settings.color_scene_by_location
   projected_pc_settings.actor_root
   projected_pc_settings.actor_template_root
   ```

---

## Notes

* nuScenes metadata is read as local JSON tables through `DirFileSystem`.
* nuPlan metadata is read from PKL files generated by `src/dwm/tools/nuplan_info.py`.
* `cache_root` stores rendered conditions such as 3D boxes, HD maps, and projected point-cloud conditions.
* If any path is changed, update both training and validation sections in the corresponding config.
