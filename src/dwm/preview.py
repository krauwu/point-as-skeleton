import argparse
import dwm.common
import json
import os
import torch
import debugpy

def customize_text(clip_text, preview_config):

    # text
    if preview_config["text"] is not None:
        text_config = preview_config["text"]

        if text_config["type"] == "add":
            new_clip_text = \
                [
                    [
                        [
                            text_config["prompt"] + k
                            for k in j
                        ]
                        for j in i
                    ]
                    for i in clip_text
                ]

        elif text_config["type"] == "replace":
            new_clip_text = \
                [
                    [
                        [
                            text_config["prompt"]
                            for k in j
                        ]
                        for j in i
                    ]
                    for i in clip_text
                ]

        elif text_config["type"] == "template":
            time = text_config["time"]
            weather = text_config["weather"]
            new_clip_text = \
                [
                    [
                        [
                            text_config["template"][time][weather][idx][0]
                            for idx, k in enumerate(j)
                        ]
                        for j in i
                    ]
                    for i in clip_text
                ]

        else:
            raise NotImplementedError(
                f"{text_config['type']}has not been implemented yet.")

        return new_clip_text

    else:

        return clip_text


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to finetune a stable diffusion model to the "
        "driving dataset.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The config to load the train model and dataset.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="The path to save checkpoint files.")
    parser.add_argument(
        "-pc", "--preview-config-path", default=None, type=str,
        help="The config for preview setting")
    parser.add_argument(
        "-eic", "--export-item-config", default=False, type=bool,
        help="The flag to export the item config as JSON")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-port", type=int, default=9876)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if args.debug:
        debugpy.listen(("0.0.0.0", args.debug_port))
        print(f"[debugpy] listening on 0.0.0.0:{args.debug_port}, waiting for VS Code to attach...")
        debugpy.wait_for_client()
        print("[debugpy] attached")
    
    # bad = set(range(3, 151, 3))
    bad = []

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.preview_config_path is not None:
        with open(args.preview_config_path, "r", encoding="utf-8") as f:
            preview_config = json.load(f)
    else:
        preview_config = None

    # set distributed training (if enabled), log, random number generator, and
    # load the checkpoint (if required).
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(config["device"], local_rank)
        if config["device"] == "cuda":
            torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(backend=config["ddp_backend"])
    else:
        device = torch.device(config["device"])

    # setup the global state
    if "global_state" in config:
        for key, value in config["global_state"].items():
            dwm.common.global_state[key] = \
                dwm.common.create_instance_from_config(value)

    should_log = (ddp and local_rank == 0) or not ddp
    should_save = not torch.distributed.is_initialized() or \
        torch.distributed.get_rank() == 0

    # load the pipeline including the models
    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], output_path=args.output_path, config=config,
        device=device)
    if should_log:
        print("The pipeline is loaded.")

    validation_dataset = dwm.common.create_instance_from_config(config["validation_dataset"])
    # good = [k for k in range(len(validation_dataset)) if k not in bad]
    good = [k for k in range(7, len(validation_dataset)) if k not in bad]
    # preview_dataloader = torch.utils.data\
    #     .DataLoader(
    #         torch.utils.data.Subset(validation_dataset, good),
    #         **dwm.common.instantiate_config(config["preview_dataloader"])) if \
    #     "preview_dataloader" in config else None
        
    if ddp:
        bs = config["preview_dataloader"]["batch_size"]
        world = torch.distributed.get_world_size()
        total_bs = world * bs

        L = (len(good) // total_bs) * total_bs
        subset = torch.utils.data.Subset(validation_dataset, good[:L])
    else:
        subset = torch.utils.data.Subset(validation_dataset, good)
        
    sampler = None
    if ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            subset, shuffle=False, drop_last=False
        )
        sampler.set_epoch(0)

    preview_dataloader = torch.utils.data.DataLoader(
        subset,
        sampler=sampler,
        **dwm.common.instantiate_config(config["preview_dataloader"])
    )

    if should_log:
        print("The validation dataset is loaded with {} items.".format(
            len(validation_dataset)))

    export_batch_except = ["vae_images"]
    output_path = args.output_path
    
    rank = torch.distributed.get_rank() if ddp else 0
    world = torch.distributed.get_world_size() if ddp else 1

    print('----------------- len ----------------', len(preview_dataloader), '----------------- len ----------------')
    
    for i, batch in enumerate(preview_dataloader):
        
        if ddp and sampler is not None:
            sampler.set_epoch(i)   
        global_step = i * world + rank

        if preview_config is not None:
            new_clip_text = customize_text(batch["clip_text"], preview_config)
            batch["clip_text"] = new_clip_text

        pipeline.preview_pipeline(batch, output_path, global_step)

        if args.export_item_config:
            with open(
                os.path.join(
                    output_path, "preview",
                    "{}.json".format(global_step)),
                "w", encoding="utf-8"
            ) as f:
                json.dump({
                    k: v.tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                    if k not in export_batch_except
                }, f, indent=4)

        global_step += 1
        if should_log:
            print(f"preview: {global_step}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
