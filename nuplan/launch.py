import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

import yaml

DEFAULT_SCENE = "2021.05.12.22.28.35_veh-35_00620_01164"
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "sim_tools" / "config_paths.yaml"


def read_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def default_db_file(cfg: dict, scene_name: str) -> Path:
    root = cfg.get("nuplan_db_root")
    if root:
        return resolve_path(root) / f"{scene_name}.db"
    return resolve_path(cfg["nuplan_db_files"])


def default_pointcloud_file(cfg: dict, scene_name: str) -> Path:
    root = cfg.get("clr_scene_root")
    if root:
        return resolve_path(root) / f"bg{scene_name}_vox0.01.npy"
    return resolve_path(cfg["clr_scene_file"])


def collect_required_paths(cfg: dict, db_file: Path, pointcloud_file: Path, trajectory_file: str) -> list[tuple[str, Path]]:
    actor_template_root = resolve_path(cfg["actor_template_root"])
    resources = [
        ("nuPlan scene database", db_file),
        ("nuPlan sensor blobs", resolve_path(cfg["blob_path"])),
        ("nuPlan maps", resolve_path(cfg["nuplan_maps_root"])),
        ("DWM source tree", resolve_path(cfg["dwm_path"])),
        ("generation config", resolve_path(cfg["gen_cfg"])),
        ("scene point cloud", pointcloud_file),
        ("actor track assets", resolve_path(cfg["actor_root"])),
        ("actor template directory", actor_template_root),
    ]

    for name in ["bike.pkl", "ped.pkl", "pickup.pkl", "sedan.pkl", "suv.pkl"]:
        resources.append((f"actor template: {name}", actor_template_root / name))

    site_cfg = resolve_path(cfg["gen_cfg"])
    if site_cfg.exists():
        import json
        with open(site_cfg, "r", encoding="utf-8") as f:
            gen_cfg = json.load(f)
        pipe = gen_cfg.get("pipeline", {})
        for key in ["pretrained_model_name_or_path", "model_checkpoint_path"]:
            value = pipe.get(key)
            if value:
                resources.append((key, resolve_path(value)))
        text_path = (
            gen_cfg.get("validation_dataset", {})
            .get("base_dataset", {})
            .get("datasets", [{}])[0]
            .get("image_description_settings", {})
            .get("path")
        )
        if text_path:
            resources.append(("nuPlan text annotation", resolve_path(text_path)))
    if trajectory_file:
        resources.append(("user trajectory file", resolve_path(trajectory_file)))
    return resources


def print_missing(resources: list[tuple[str, Path]]) -> None:
    missing = [(name, path) for name, path in resources if not path.exists()]
    if not missing:
        return
    print("Missing required resources:")
    for name, path in missing:
        print(f"  - {name}: {path}")
    raise SystemExit(1)


def build_env(args, cfg: dict, db_file: Path, pointcloud_file: Path) -> dict:
    env = os.environ.copy()
    env["RUN_TAG"] = args.run_tag or f"{args.mode}_{args.scene_name}"
    env["NUPLAN_DB_FILES"] = str(db_file)
    env["TARGET_CLR_SCENE_FILE"] = str(pointcloud_file)
    env["STAGE3_OUTPUT_DIR"] = str(resolve_path(args.output_dir))
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["MAX_SIM_STEPS"] = str(args.max_steps)
    env["LIMIT_TOTAL_SCENARIOS"] = str(args.limit_total_scenarios)
    env["TRAJECTORY_TYPE"] = args.trajectory_type
    env["TRAJ_STRAIGHT_PLAN_STEPS"] = str(args.trajectory_straight_steps)
    if args.trajectory_file:
        env["TRAJECTORY_FILE"] = str(resolve_path(args.trajectory_file))
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a nuPlan scene with DB replay or user trajectory control.")
    parser.add_argument("--mode", choices=["db", "trajectory"], default="trajectory")
    parser.add_argument("--scene-name", default=DEFAULT_SCENE)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--limit-total-scenarios", type=int, default=1)
    parser.add_argument("--output-dir", default="./outputs/default")
    parser.add_argument("--db-file", default="")
    parser.add_argument("--clr-scene-file", default="")
    parser.add_argument("--trajectory-type", choices=["straight", "right_then_back", "cosine"], default="right_then_back")
    parser.add_argument("--trajectory-straight-steps", type=int, default=6)
    parser.add_argument("--trajectory-file", default="")
    parser.add_argument("--run-tag", default="")
    args = parser.parse_args()

    cfg = read_config()
    db_file = resolve_path(args.db_file) if args.db_file else default_db_file(cfg, args.scene_name)
    pointcloud_file = resolve_path(args.clr_scene_file) if args.clr_scene_file else default_pointcloud_file(cfg, args.scene_name)
    required_paths = collect_required_paths(cfg, db_file, pointcloud_file, args.trajectory_file)
    print_missing(required_paths)

    main_py = ROOT / "sim_tools" / ("nuplan_sim_db.py" if args.mode == "db" else "nuplan_sim_trajectory.py")
    env = build_env(args, cfg, db_file, pointcloud_file)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("nuPlan scene interface")
    print(f"  mode: {args.mode}")
    print(f"  scene: {args.scene_name}")
    print(f"  db: {db_file}")
    print(f"  pointcloud: {pointcloud_file}")
    print(f"  output: {output_dir}")
    if args.mode == "trajectory":
        print(f"  trajectory_type: {args.trajectory_type}")
        if args.trajectory_file:
            print(f"  trajectory_file: {resolve_path(args.trajectory_file)}")

    proc = subprocess.Popen([sys.executable, str(main_py)], cwd=str(ROOT), env=env, start_new_session=True)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
