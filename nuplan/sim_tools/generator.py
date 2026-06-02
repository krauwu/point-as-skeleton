# sim_tools/streaming_ar_client.py
import numpy as np
import torch
from PIL import Image as PILImage
import torch.nn.functional as F


def test_fn(L, S, T, C, start_timestep, stop_timestep, take_time, frozen_ref_count):
    base = start_timestep - take_time * S
    j = torch.arange(L)
    j_eff = torch.clamp(j - C, min=0)
    inner = torch.clamp(base - j_eff * S, min=0)
    idx = torch.minimum(inner, torch.full_like(inner, base))
    max_idx = T - 1
    idx = torch.clamp(idx, 0, max_idx)
    if frozen_ref_count > 0:
        clean_idx = max_idx
        idx[:frozen_ref_count] = clean_idx
    print("start_t", idx)

    base = stop_timestep - take_time * S
    j = torch.arange(L)
    j_eff = torch.clamp(j - C, min=0)
    inner = torch.clamp(base - j_eff * S, min=0)
    idx = torch.minimum(inner, torch.full_like(inner, base))
    max_idx = T - 1
    idx = torch.clamp(idx, 0, max_idx)
    if frozen_ref_count > 0:
        clean_idx = max_idx
        idx[:frozen_ref_count] = clean_idx
    print("end_t", idx)


class StreamingARClient:
    """
    """

    def __init__(self, pipeline, device, history_len=9, pack_hz=5):
        self.pipeline = pipeline
        self.device = device
        self.pack_hz = int(pack_hz)

        self.history_len = int(self.pipeline.inference_config.get("reference_frame_count", history_len))
        self.window_len = int(self.pipeline.inference_config.get("sequence_length_per_iteration", 10))

        seed = self.pipeline.config.get("generator_seed", None) if hasattr(self.pipeline, "config") else None
        if seed is not None:
            self.pipeline.generator.manual_seed(int(seed))


        self.img_h = None
        self.img_w = None
        self._vae_buf = None

        self.image_latents = None
        self.dataset_ref_left = 0
        self.latent_shape = None

        self.pipeline.model.eval()

        for m in [self.pipeline.vae, self.pipeline.model]:
            if m is None:
                continue
            m.requires_grad_(False)

        self.state = None
            

    def reset(self):
        self.image_latents = None
        self.dataset_ref_left = 0
        self.latent_shape = None
        self.state = None

    def _nested_to_tensor(self, x, name="cond"):
        if torch.is_tensor(x):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        if isinstance(x, PILImage.Image):
            arr = np.asarray(x.convert("RGB"), dtype=np.uint8)
            return torch.from_numpy(arr)
        if isinstance(x, list):
            if len(x) == 0:
                raise ValueError(f"{name} is empty list")
            if isinstance(x[0], list):
                stacked = [self._nested_to_tensor(e, name=name) for e in x]
                return torch.stack(stacked, dim=0)
            stacked = []
            for e in x:
                if torch.is_tensor(e):
                    stacked.append(e)
                elif isinstance(e, np.ndarray):
                    stacked.append(torch.from_numpy(e))
                elif isinstance(e, PILImage.Image):
                    arr = np.asarray(e.convert("RGB"), dtype=np.uint8)
                    stacked.append(torch.from_numpy(arr))
                else:
                    raise TypeError(f"{name} element type unsupported: {type(e)}")
            return torch.stack(stacked, dim=0)
        raise TypeError(f"{name} type unsupported: {type(x)}")

    def _ensure_chw(self, t, name="cond"):
        if t.ndim < 3:
            raise ValueError(f"{name} ndim too small: {t.ndim}, shape={tuple(t.shape)}")

        if t.dtype == torch.uint8:
            t = t.float() / 255.0
        else:
            t = t.float()
            if t.max() > 1.5:
                t = t / 255.0

        if t.shape[-1] == 3 and t.shape[-3] != 3:
            perm = list(range(t.ndim))
            perm = perm[:-3] + [perm[-1], perm[-3], perm[-2]]
            t = t.permute(*perm).contiguous()

        if t.shape[-3] != 3:
            raise ValueError(f"{name} channel dim != 3, got shape={tuple(t.shape)}")
        return t

    def _as_btvc3hw(self, x, B, T, V, H, W, name="cond_img"):
        """
        """
        t = self._nested_to_tensor(x, name=name)
        t = self._ensure_chw(t, name=name)

        if t.ndim == 3:
            t = t.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif t.ndim == 4:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 5:
            t = t.unsqueeze(0)
        elif t.ndim == 6:
            pass
        else:
            raise ValueError(f"{name} ndim invalid: {t.ndim}, shape={tuple(t.shape)}")

        if t.shape[0] != B:
            if t.shape[0] == 1 and B == 1:
                pass
            else:
                raise ValueError(f"{name} B mismatch: got {t.shape[0]} expect {B}")

        if t.shape[1] < T:
            pad = t[:, -1:].repeat(1, T - t.shape[1], 1, 1, 1, 1)
            t = torch.cat([t, pad], dim=1)
        elif t.shape[1] > T:
            t = t[:, :T]

        if t.shape[2] < V:
            pad = t[:, :, -1:].repeat(1, 1, V - t.shape[2], 1, 1, 1)
            t = torch.cat([t, pad], dim=2)
        elif t.shape[2] > V:
            t = t[:, :, :V]

        if t.shape[-2] != H or t.shape[-1] != W:
            flat = t.flatten(0, 2)
            flat = F.interpolate(flat, size=(H, W), mode="bilinear", align_corners=False)
            t = flat.unflatten(0, (B, T, V))

        return t.to(device=self.device, dtype=torch.float32)

    def _pil8_to_tensor_vchw(self, cam8_pil, out_h, out_w):
        out = []
        for im in cam8_pil:
            if im.size != (out_w, out_h):
                im = im.resize((out_w, out_h), PILImage.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8)
            arr = arr.astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            out.append(arr)
        x = np.stack(out, axis=0)
        return torch.from_numpy(x)

    def _tensor_vchw_to_pil8(self, x_vchw):
        if isinstance(x_vchw, torch.Tensor):
            x = x_vchw.detach().cpu()
            if x.dtype != torch.uint8:
                x = (x.clamp(0, 1) * 255.0).to(torch.uint8)
            x = x.numpy()
        else:
            x = x_vchw

        pil8 = []
        for i in range(8):
            chw = x[i]
            hwc = np.transpose(chw, (1, 2, 0))
            pil8.append(PILImage.fromarray(hwc, mode="RGB"))
        return pil8

    def _infer_hw_from_cond(self, cond_pack):
        for k in ["3dbox_images", "hdmap_images"]:
            v = cond_pack.get(k, None)
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list) and len(v[0]) > 0:
                im0 = v[0][0]
                if isinstance(im0, PILImage.Image):
                    w, h = im0.size
                    return int(h), int(w)

        for k in ["3dbox_images", "hdmap_images"]:
            v = cond_pack.get(k, None)
            if torch.is_tensor(v) and v.ndim >= 6:
                H = int(v.shape[-2])
                W = int(v.shape[-1])
                return H, W

        return 448, 896

    def _prepare_cond_cache(self, cond_pack):
        B = 1
        V = 8

        T_total = int(cond_pack["camera_intrinsics"].shape[0])

        if self.img_h is None or self.img_w is None:
            self.img_h, self.img_w = self._infer_hw_from_cond(cond_pack)
        H, W = int(self.img_h), int(self.img_w)

        # -------- numeric: CPU ( action_id needed) --------
        camK_cpu = torch.as_tensor(cond_pack["camera_intrinsics"], dtype=torch.float32).unsqueeze(0)
        camT_cpu = torch.as_tensor(cond_pack["camera_transforms"],  dtype=torch.float32).unsqueeze(0)
        egoT_cpu = torch.as_tensor(cond_pack["ego_transforms"],     dtype=torch.float32).unsqueeze(0)

        fps_cpu = torch.tensor([float(self.pack_hz)], dtype=torch.float32)               # CPU
        pts_full_cpu = torch.arange(T_total, dtype=torch.float32).unsqueeze(0)           # CPU [1,T_total]

        image_size_cpu = torch.zeros((B, T_total, V, 2), dtype=torch.long)               # CPU
        image_size_cpu[..., 0] = 1920
        image_size_cpu[..., 1] = 1080

        box_img_gpu = self._as_btvc3hw(cond_pack["3dbox_images"], B, T_total, V, H, W, name="3dbox_images")
        hd_img_gpu  = self._as_btvc3hw(cond_pack["hdmap_images"],  B, T_total, V, H, W, name="hdmap_images")
        if "proj_clr" in cond_pack:
            proj_clr_gpu  = cond_pack["proj_clr"].unsqueeze(0).to(device=self.device, dtype=torch.float32)
            proj_depth_gpu  = cond_pack["proj_depth"].unsqueeze(0).to(device=self.device, dtype=torch.float32)
        else:
            proj_clr_gpu = None
            proj_depth_gpu = None

        I4 = torch.eye(4, dtype=torch.float32, device=self.device)
        lidarT_gpu = I4.view(1, 1, 1, 4, 4).repeat(B, T_total, 1, 1, 1)

        crossview_mask_gpu = None
        if "crossview_mask" in cond_pack and cond_pack["crossview_mask"] is not None:
            m = torch.as_tensor(cond_pack["crossview_mask"], dtype=torch.bool)
            if m.ndim == 2:
                m = m.unsqueeze(0)
            crossview_mask_gpu = m.to(self.device)

        clip_text = cond_pack.get("clip_text", None)
        if clip_text is None:
            clip_text = cond_pack.get("image_description", None)

        if clip_text is None:
            clip_text_TV_full = [["This is a nuplan video clip"] * V for _ in range(T_total)]
        elif isinstance(clip_text, str):
            clip_text_TV_full = [[clip_text] * V for _ in range(T_total)]
        elif isinstance(clip_text, list) and len(clip_text) == T_total and isinstance(clip_text[0], str):
            clip_text_TV_full = [[clip_text[t]] * V for t in range(T_total)]
        elif isinstance(clip_text, list) and len(clip_text) == T_total and isinstance(clip_text[0], list):
            clip_text_TV_full = clip_text
        else:
            clip_text_TV_full = [["This is a nuplan video clip"] * V for _ in range(T_total)]

        return {
            "T_total": T_total,
            "H": H,
            "W": W,
            "V": V,

            # CPU numeric
            "camera_intrinsics_cpu": camK_cpu,
            "camera_transforms_cpu": camT_cpu,
            "ego_transforms_cpu": egoT_cpu,
            "fps_cpu": fps_cpu,
            "pts_full_cpu": pts_full_cpu,
            "image_size_cpu": image_size_cpu,

            # GPU heavy
            "3dbox_images_gpu": box_img_gpu,
            "hdmap_images_gpu": hd_img_gpu,
            "proj_clr_gpu": proj_clr_gpu,
            "proj_depth_gpu": proj_depth_gpu,
            "lidar_transforms_gpu": lidarT_gpu,
            "crossview_mask_gpu": crossview_mask_gpu,

            # text
            "clip_text_TV_full": clip_text_TV_full,
        }


    def _build_vae_images_once(self, hist_cam8_list, H, W):
        """
        """
        B = 1
        T = int(self.window_len)
        V = 8
        shape = (B, T, V, 3, H, W)
        if self._vae_buf is None or self._vae_buf.shape != shape or self._vae_buf.device != self.device:
            self._vae_buf = torch.zeros(shape, dtype=torch.float32, device=self.device)
        else:
            self._vae_buf.zero_()
        vae = self._vae_buf

        for t in range(self.history_len):
            vae[0, t] = self._pil8_to_tensor_vchw(hist_cam8_list[t], H, W).to(self.device)
        return vae

    def _build_batch_from_cache(self, cache, vae_images_win, start_idx):
        T = int(self.window_len)
        T_total = int(cache["T_total"])

        max_start = max(0, T_total - T)
        s = int(min(max(0, start_idx), max_start))

        # CPU numeric
        camK = cache["camera_intrinsics_cpu"].narrow(1, s, T)
        camT = cache["camera_transforms_cpu"].narrow(1, s, T)
        egoT = cache["ego_transforms_cpu"].narrow(1, s, T)
        fps = cache["fps_cpu"]
        pts = cache["pts_full_cpu"].narrow(1, s, T)
        image_size = cache["image_size_cpu"].narrow(1, s, T)

        # GPU images
        box_img = cache["3dbox_images_gpu"].narrow(1, s, T)
        hd_img  = cache["hdmap_images_gpu"].narrow(1, s, T)
        if cache["proj_clr_gpu"] != None:
            proj_clr = cache["proj_clr_gpu"].narrow(1, s, T)
            proj_depth  = cache["proj_depth_gpu"].narrow(1, s, T)
        else:
            proj_clr = None
            proj_depth = None

        lidarT  = cache["lidar_transforms_gpu"].narrow(1, s, T)

        clip_text_TV = cache["clip_text_TV_full"][s:s + T]

        batch = {
            # GPU
            "vae_images": vae_images_win,
            "3dbox_images": box_img,
            "hdmap_images": hd_img,
            "proj_clr": proj_clr,
            "proj_depth": proj_depth,
            "lidar_transforms": lidarT,

            "fps": fps,
            "pts": pts,
            "camera_intrinsics": camK,
            "camera_transforms": camT,
            "ego_transforms": egoT,
            "image_size": image_size,

            # text
            "clip_text": [clip_text_TV],
        }

        if cache["crossview_mask_gpu"] is not None:
            m = cache["crossview_mask_gpu"]
            if torch.is_tensor(m) and m.ndim >= 2 and m.shape[1] == T_total:
                batch["crossview_mask"] = m.narrow(1, s, T)
            else:
                batch["crossview_mask"] = m

        return batch

    def _init_latents_if_needed(self, batch):
        if self.image_latents is not None:
            return

        self.pipeline.test_scheduler.set_timesteps(
            self.pipeline.inference_config["inference_steps"], self.device
        )

        B, T, V = batch["vae_images"].shape[:3]

        down = 2 ** (len(self.pipeline.vae.config.down_block_types) - 1)
        latent_h = batch["vae_images"].shape[-2] // down
        latent_w = batch["vae_images"].shape[-1] // down

        latent_T = self.pipeline.get_latent_sequence_length(T)
        self.latent_shape = (
            B, latent_T, V,
            self.pipeline.vae.config.latent_channels,
            latent_h, latent_w
        )

        ref = batch["vae_images"][:, :self.history_len]  # [1,9,8,3,H,W]
        img_tensor = self.pipeline.image_processor.preprocess(
            ref.flatten(0, 2).to(self.device)
        )

        shift_factor = self.pipeline.vae.config.shift_factor or 0
        lat = self.pipeline.vae.encode(img_tensor.to(self.pipeline.vae.dtype)).latent_dist.mode()
        lat = (lat - shift_factor) * self.pipeline.vae.config.scaling_factor
        lat = lat.unflatten(0, (B, self.history_len, V))  # [1,9,8,c,h,w]

        self.dataset_ref_left = lat.shape[1]

        if self.dataset_ref_left < latent_T:
            tail = torch.randn(
                (B, latent_T - self.dataset_ref_left) + lat.shape[2:],
                generator=self.pipeline.generator
            ).to(self.device) * getattr(self.pipeline.test_scheduler, "init_noise_sigma", 1)
            self.image_latents = torch.cat([lat, tail], dim=1)
        else:
            self.image_latents = lat


    @torch.no_grad()
    def generate_next_cam8(self, cond_pack, hist_cam8_list, gen_state=None):

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                assert len(hist_cam8_list) == self.history_len

                T_steps = int(self.pipeline.inference_config["inference_steps"])
                clear_ref = int(self.pipeline.inference_config.get("clear_reference_frame_count", 0))
                V = 8
                cache = self._prepare_cond_cache(cond_pack)

                H, W = int(cache["H"]), int(cache["W"])
                vae_images_win = self._build_vae_images_once(hist_cam8_list, H, W)

                batch0 = self._build_batch_from_cache(cache, vae_images_win, start_idx=0)
                
                if gen_state is not None:
                    lat = gen_state['latent'][:, 1:] 
                    self.dataset_ref_left = gen_state['frozen_ref_count']
                    tail = torch.randn(
                            (lat.shape[0], 1) + lat.shape[2:],
                            generator=self.pipeline.generator
                        ).to(self.device) * getattr(self.pipeline.test_scheduler, "init_noise_sigma", 1)
                    self.image_latents = torch.cat([lat, tail], dim=1)

                self._init_latents_if_needed(batch0)

                latent_T = int(self.latent_shape[1])
                steps_per_inf = T_steps // (latent_T - clear_ref)
                num_roll = latent_T - clear_ref

                win_len = int(self.window_len)
                big_len = int(win_len * 2)

                T_total = int(cache["T_total"])
                base0 = max(0, T_total - big_len)
                max_start_abs = max(0, T_total - win_len)

                out = None
                for roll_i in range(num_roll):
                    start_idx = base0 + roll_i
                    if start_idx > max_start_abs:
                        start_idx = max_start_abs

                    batch = self._build_batch_from_cache(cache, vae_images_win, start_idx=start_idx)

                    # test_fn(
                    #     L=latent_T,
                    #     S=steps_per_inf,
                    #     T=T_steps,
                    #     C=clear_ref,
                    #     start_timestep=(T_steps - steps_per_inf),
                    #     stop_timestep=T_steps,
                    #     take_time=0,
                    #     frozen_ref_count=int(self.dataset_ref_left),
                    # )

                    out = self.pipeline.inference_pipeline(
                        self.latent_shape,
                        batch,
                        output_type="pt",
                        image_latents=self.image_latents,
                        start_timestep=(T_steps - steps_per_inf),
                        stop_timestep=T_steps,
                        frozen_ref_count=self.dataset_ref_left,
                    )

                    if self.dataset_ref_left > clear_ref:
                        self.dataset_ref_left -= 1

                    if roll_i == 0:
                        self.state = {
                            'latent': out['latents'].clone(),
                            'frozen_ref_count': self.dataset_ref_left
                        }

                    self.image_latents = out["latents"].detach()

                    noise = torch.randn(
                        (self.latent_shape[0], 1) + self.image_latents.shape[2:],
                        generator=self.pipeline.generator
                    ).to(self.device) * getattr(self.pipeline.test_scheduler, "init_noise_sigma", 1)

                    # self.image_latents = torch.cat([self.image_latents[:, 1:], noise], dim=1)
                    self.image_latents[:, :-1].copy_(self.image_latents[:, 1:].clone())
                    self.image_latents[:, -1:].copy_(noise)


                out_images = out["images"] if out is not None else None
                if torch.is_tensor(out_images) and out_images.ndim == 4 and out_images.shape[0] >= V:
                    if out_images.shape[0] == V:
                        return self._tensor_vchw_to_pil8(out_images), self.state
                    return self._tensor_vchw_to_pil8(out_images[:V]), self.state

                return None
