import argparse
import torch
import debugpy
from torch.utils.data import DataLoader
from torchvision import transforms as T
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from PIL import Image

from tqdm import tqdm
from dwm.datasets.nuscenes_0118 import MotionDataset
import debugpy

DEFAULT_NUSC_ROOT = "./data/nuscenes"

DEFAULT_IMG_DESC_PATH_TRAIN = (
    "./data/nuscenes/"
    "nuscenes_v1.0-trainval_caption_v2_train.json"
)
DEFAULT_IMG_DESC_TIMES_TRAIN = (
    "./data/nuscenes/"
    "nuscenes_v1.0-trainval_caption_v2_times_train.json"
)

def _infer_val_path(train_path: str) -> str:
    return train_path.replace("_train.json", "_val.json")


resize_to_tensor = T.Compose([T.Resize((256, 448)), T.ToTensor()])


def _apply_nested_images(pil_nested):
    # nested: [T][V] of PIL -> [T][V] of Tensor(C,H,W)
    if pil_nested is None:
        return None
    out = []
    for row in pil_nested:
        row_t = []
        for im in row:
            row_t.append(resize_to_tensor(im) if isinstance(im, Image.Image) else im)
        out.append(row_t)
    return out


class SimpleDatasetAdapter(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]

        if "images" in sample:
            sample["vae_images"] = _apply_nested_images(sample["images"])
        if "3dbox_images" in sample:
            sample["3dbox_images"] = _apply_nested_images(sample["3dbox_images"])
        if "hdmap_images" in sample:
            sample["hdmap_images"] = _apply_nested_images(sample["hdmap_images"])

        if "image_description" in sample:
            sample["clip_text"] = sample["image_description"]

        for k in ("images", "lidar_points", "image_description"):
            sample.pop(k, None)

        return sample


def collate_ignore_clip_text(batch):
    return batch[0]


def make_base_ds(args):
    fs = DirFileSystem(path=args.nusc_root, fs=LocalFileSystem())

    img_desc_path = args.img_desc_path
    img_desc_times = args.img_desc_times

    return MotionDataset(
        fs=fs,
        dataset_name="interp_12Hz_trainval",
        split=args.split,
        sequence_length=30,
        fps_stride_tuples=[(0, 30)],
        sensor_channels=[
            "LIDAR_TOP",
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
        ],
        keyframe_only=True,
        enable_synchronization_check=False,
        enable_camera_transforms=True,
        enable_ego_transforms=True,
        cache_root="./data/cache/nuscenes_cache",
        _3dbox_image_settings={},
        hdmap_image_settings={},
        image_description_settings={
            "path": img_desc_path,
            "time_list_dict_path": img_desc_times,
            "align_keys": ["time", "weather"],
        },
        stub_key_data_dict={
            "crossview_mask": [
                "content",
                torch.tensor(
                    [
                        [1, 1, 0, 0, 0, 1],
                        [1, 1, 1, 0, 0, 0],
                        [0, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 0],
                        [0, 0, 0, 1, 1, 1],
                        [1, 0, 0, 0, 1, 1],
                    ],
                    dtype=torch.bool,
                ),
            ]
        }
        # projected_pc_settings=dict(
        #     color_scene_by_location={
        #         "boston-seaport": "./data/nuscenes/boston-seaport_point_cloud.npy",
        #         "singapore-onenorth": "./data/nuscenes/singapore-onenorth_point_cloud.npy",
        #         "singapore-queenstown": "./data/nuscenes/singapore-queenstown_point_cloud.npy",
        #         "singapore-hollandvillage": "./data/nuscenes/singapore-hollandvillage_point_cloud.npy",
        #     },
        #     radius=100.0,
        #     ori_hw=(900, 1600),
        #     final_hw=(256, 448),
        #     invalid_depth=-300.0,
        #     depth_bin_mode="linear",
        #     log_gamma=1.0,
        #     lidar_channel="LIDAR_TOP",
        #     actor_root="./data/nuscenes/nuscenes_actor",
        #     actor_template_root="./data/nuscenes/actor/merged_output",
        #     do_cache=True,
        #     splat=[(15.0, 2), (35.0, 1), (1e9, 0)],
        #     depth_bins=256,
        #     data_type='clr',
        # ),
    )


def parse_args():
    p = argparse.ArgumentParser("Cache nuScenes projected point cloud")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--nusc-root", default=DEFAULT_NUSC_ROOT)

    p.add_argument("--img-desc-path", default=None)
    p.add_argument("--img-desc-times", default=None)

    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--no-persistent-workers", action="store_true")
    return p.parse_args()


def main():
    debugpy.listen(("0.0.0.0", 9876))
    print("[debugpy] listening on, waiting for VS Code to attach...")
    debugpy.wait_for_client()        
    print("attached")

    args = parse_args()

    if args.img_desc_path is None:
        args.img_desc_path = (
            DEFAULT_IMG_DESC_PATH_TRAIN
            if args.split == "train"
            else _infer_val_path(DEFAULT_IMG_DESC_PATH_TRAIN)
        )
    if args.img_desc_times is None:
        args.img_desc_times = (
            DEFAULT_IMG_DESC_TIMES_TRAIN
            if args.split == "train"
            else _infer_val_path(DEFAULT_IMG_DESC_TIMES_TRAIN)
        )

    base_ds = make_base_ds(args)
    ds = SimpleDatasetAdapter(base_ds)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle=False,
        persistent_workers=not args.no_persistent_workers,
        collate_fn=collate_ignore_clip_text,
        pin_memory=False,
    )

    pbar = tqdm(loader, total=len(ds), desc=f"scan nus ({args.split})", dynamic_ncols=True)
    for step, batch in enumerate(pbar):
        pass

if __name__ == "__main__":
    main()
