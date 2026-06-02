"""
"""
import numpy as np
import torch


class OptimizedPointProjector:
    """

    """

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.gpu_available = self.device.type == 'cuda'

    def project_depth(
        self, pts_xyz: np.ndarray, l2i: np.ndarray, ori_hw: tuple, invalid_depth=-300.0
    ) -> np.ndarray:
        """

        """
        H, W = int(ori_hw[0]), int(ori_hw[1])

        if isinstance(pts_xyz, torch.Tensor):
            xyz = pts_xyz
        else:
            xyz = torch.from_numpy(pts_xyz).float().to(self.device)

        if isinstance(l2i, torch.Tensor):
            l2i_t = l2i
        else:
            l2i_t = torch.from_numpy(l2i).float().to(self.device)

        if xyz.shape[0] == 0:
            return np.full((H, W), invalid_depth, np.float32)

        N = xyz.shape[0]
        xyz1 = torch.cat([xyz, torch.ones(N, 1, device=self.device)], dim=1)
        p = torch.matmul(xyz1, l2i_t.T)

        z = p[:, 2]
        valid = z > 1e-5

        if not valid.any():
            return np.full((H, W), invalid_depth, np.float32)

        p = p[valid]
        z = z[valid]

        u = torch.div(p[:, 0], z, rounding_mode='trunc').long()
        v = torch.div(p[:, 1], z, rounding_mode='trunc').long()

        bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        if not bounds.any():
            return np.full((H, W), invalid_depth, np.float32)

        u = u[bounds]
        v = v[bounds]
        z = z[bounds]

        indices = v * W + u

        depth = torch.full((H * W,), float('inf'), device=self.device)
        depth.scatter_reduce_(0, indices, z, reduce='amin', include_self=False)

        depth = torch.where(torch.isinf(depth), torch.tensor(invalid_depth, device=self.device), depth)

        return depth.reshape(H, W).cpu().numpy()

    def project_clr_simple(
        self, clr_xyz: np.ndarray, clr_rgb: np.ndarray, l2i: np.ndarray, ori_hw: tuple
    ) -> np.ndarray:
        """

        """
        H, W = int(ori_hw[0]), int(ori_hw[1])

        if isinstance(clr_xyz, torch.Tensor):
            xyz = clr_xyz
        else:
            xyz = torch.from_numpy(clr_xyz).float().to(self.device)

        if isinstance(clr_rgb, torch.Tensor):
            rgb = clr_rgb
        else:
            rgb = torch.from_numpy(clr_rgb).float().to(self.device)

        if isinstance(l2i, torch.Tensor):
            l2i_t = l2i
        else:
            l2i_t = torch.from_numpy(l2i).float().to(self.device)

        if xyz.shape[0] == 0:
            return np.zeros((H, W, 3), np.uint8)

        N = xyz.shape[0]
        xyz1 = torch.cat([xyz, torch.ones(N, 1, device=self.device)], dim=1)
        p = torch.matmul(xyz1, l2i_t.T)

        z = p[:, 2]
        valid = z > 1e-5

        if not valid.any():
            return np.zeros((H, W, 3), np.uint8)

        p = p[valid]
        z = z[valid]
        rgb = rgb[valid]

        u = torch.div(p[:, 0], z, rounding_mode='trunc').long()
        v = torch.div(p[:, 1], z, rounding_mode='trunc').long()

        bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        if not bounds.any():
            return np.zeros((H, W, 3), np.uint8)

        u = u[bounds]
        v = v[bounds]
        rgb = rgb[bounds]

        indices = v * W + u

        scatter_indices = torch.flip(indices, [0])
        scatter_rgb = torch.flip(rgb, [0])

        rgb_uint8 = torch.clamp(scatter_rgb, 0, 255).byte()

        clr = torch.zeros((H * W, 3), dtype=torch.uint8, device=self.device)
        clr.scatter_(0, scatter_indices.unsqueeze(1).expand(-1, 3), rgb_uint8)

        return clr.reshape(H, W, 3).cpu().numpy()

    def project_clr_sortbased(
        self, clr_xyz: np.ndarray, clr_rgb: np.ndarray, l2i: np.ndarray, ori_hw: tuple
    ) -> np.ndarray:
        """

        """
        H, W = int(ori_hw[0]), int(ori_hw[1])

        xyz = torch.from_numpy(clr_xyz).float().to(self.device)
        rgb = torch.from_numpy(clr_rgb).float().to(self.device)
        l2i_t = torch.from_numpy(l2i).float().to(self.device)

        if xyz.shape[0] == 0:
            return np.zeros((H, W, 3), np.uint8)

        N = xyz.shape[0]
        xyz1 = torch.cat([xyz, torch.ones(N, 1, device=self.device)], dim=1)
        p = torch.matmul(xyz1, l2i_t.T)

        z = p[:, 2]
        valid = z > 1e-5

        if not valid.any():
            return np.zeros((H, W, 3), np.uint8)

        p = p[valid]
        z = z[valid]
        rgb = rgb[valid]

        u = torch.div(p[:, 0], z, rounding_mode='trunc').long()
        v = torch.div(p[:, 1], z, rounding_mode='trunc').long()

        bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        if not bounds.any():
            return np.zeros((H, W, 3), np.uint8)

        u = u[bounds]
        v = v[bounds]
        z = z[bounds]
        rgb = rgb[bounds]
        indices = v * W + u

        sorted_indices = torch.argsort(z, descending=True)

        clr = torch.zeros((H * W, 3), device=self.device, dtype=torch.uint8)
        rgb_clamped = torch.clamp(rgb, 0, 255).to(torch.uint8)

        clr[indices[sorted_indices]] = rgb_clamped[sorted_indices]

        return clr.reshape(H, W, 3).cpu().numpy()

    def project_sem_simple(
        self, pts_xyz: np.ndarray, pts_sem: np.ndarray,
        l2i: np.ndarray, ori_hw: tuple, n_actor_classes=3
    ) -> np.ndarray:
        """


        """
        H, W = int(ori_hw[0]), int(ori_hw[1])

        if isinstance(pts_xyz, torch.Tensor):
            xyz = pts_xyz
        else:
            xyz = torch.from_numpy(pts_xyz).float().to(self.device)

        if isinstance(pts_sem, torch.Tensor):
            sem = pts_sem
        else:
            sem = torch.from_numpy(pts_sem).long().to(self.device)

        if isinstance(l2i, torch.Tensor):
            l2i_t = l2i
        else:
            l2i_t = torch.from_numpy(l2i).float().to(self.device)

        if xyz.shape[0] == 0:
            return np.zeros((H, W, n_actor_classes), np.uint8)

        N = xyz.shape[0]
        xyz1 = torch.cat([xyz, torch.ones(N, 1, device=self.device)], dim=1)
        p = torch.matmul(xyz1, l2i_t.T)

        z = p[:, 2]
        valid = z > 1e-5

        if not valid.any():
            return np.zeros((H, W, n_actor_classes), np.uint8)

        p = p[valid]
        z = z[valid]
        sem = sem[valid]

        u = torch.div(p[:, 0], z, rounding_mode='trunc').long()
        v = torch.div(p[:, 1], z, rounding_mode='trunc').long()

        bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        if not bounds.any():
            return np.zeros((H, W, n_actor_classes), np.uint8)

        u = u[bounds]
        v = v[bounds]
        z = z[bounds]
        sem = sem[bounds]

        indices = v * W + u

        min_depth = torch.full((H * W,), float('inf'), device=self.device)
        argmin_depth = torch.full((H * W,), -1, dtype=torch.long, device=self.device)

        min_depth.scatter_reduce_(0, indices, z, reduce='amin', include_self=False)

        min_depth_at_indices = min_depth[indices]
        min_mask = (z == min_depth_at_indices) & (min_depth_at_indices != float('inf'))

        valid_min_indices = indices[min_mask]
        valid_min_sem = sem[min_mask]

        valid_min_sem = torch.clamp(valid_min_sem, 0, n_actor_classes - 1)

        sem_map = torch.zeros((H * W, n_actor_classes), dtype=torch.uint8, device=self.device)

        sem_map.scatter_(0, valid_min_indices.unsqueeze(1).expand(-1, n_actor_classes),
                         torch.eye(n_actor_classes, device=self.device, dtype=torch.uint8)[valid_min_sem])

        return sem_map.reshape(H, W, n_actor_classes).cpu().numpy()

    def project_all_fast(
        self, pts_xyz: np.ndarray, pts_sem: np.ndarray,
        clr_xyz: np.ndarray, clr_rgb: np.ndarray,
        l2i: np.ndarray, ori_hw: tuple, invalid_depth=-300.0, n_actor_classes=3
    ) -> tuple:
        """

        """
        depth = self.project_depth(pts_xyz, l2i, ori_hw, invalid_depth)
        clr = self.project_clr_simple(clr_xyz, clr_rgb, l2i, ori_hw)
        sem = self.project_sem_simple(pts_xyz, pts_sem, l2i, ori_hw, n_actor_classes)

        return depth, sem, clr

    def project_all_unified(
        self, pts_xyz: np.ndarray, pts_sem: np.ndarray,
        clr_xyz: np.ndarray, clr_rgb: np.ndarray,
        l2i: np.ndarray, ori_hw: tuple, invalid_depth=-300.0, n_actor_classes=3
    ) -> tuple:
        """


        """
        H, W = int(ori_hw[0]), int(ori_hw[1])

        if isinstance(pts_xyz, torch.Tensor):
            xyz = pts_xyz
        else:
            xyz = torch.from_numpy(pts_xyz).float().to(self.device)

        if isinstance(pts_sem, torch.Tensor):
            sem = pts_sem
        else:
            sem = torch.from_numpy(pts_sem).long().to(self.device)

        if isinstance(clr_xyz, torch.Tensor):
            xyz_clr = clr_xyz
        else:
            xyz_clr = torch.from_numpy(clr_xyz).float().to(self.device)

        if isinstance(clr_rgb, torch.Tensor):
            rgb = clr_rgb
        else:
            rgb = torch.from_numpy(clr_rgb).float().to(self.device)

        if isinstance(l2i, torch.Tensor):
            l2i_t = l2i
        else:
            l2i_t = torch.from_numpy(l2i).float().to(self.device)

        if xyz.shape[0] == 0:
            return np.full((H, W), invalid_depth, np.float32), \
                   np.zeros((H, W, n_actor_classes), np.uint8), \
                   np.zeros((H, W, 3), np.uint8)

        N = xyz.shape[0]
        xyz1 = torch.cat([xyz, torch.ones(N, 1, device=self.device)], dim=1)
        p = torch.matmul(xyz1, l2i_t.T)

        z = p[:, 2]
        valid = z > 1e-5

        if not valid.any():
            return np.full((H, W), invalid_depth, np.float32), \
                   np.zeros((H, W, n_actor_classes), np.uint8), \
                   np.zeros((H, W, 3), np.uint8)

        p, z = p[valid], z[valid]
        sem = sem[valid]

        u = torch.div(p[:, 0], z, rounding_mode='trunc').long()
        v = torch.div(p[:, 1], z, rounding_mode='trunc').long()
        bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        if not bounds.any():
            return np.full((H, W), invalid_depth, np.float32), \
                   np.zeros((H, W, n_actor_classes), np.uint8), \
                   np.zeros((H, W, 3), np.uint8)

        u, v = u[bounds], v[bounds]
        z, sem = z[bounds], sem[bounds]

        indices = v * W + u

        indices_clr = None
        rgb_sel = None

        if xyz_clr.shape[0] > 0:
            N_clr = xyz_clr.shape[0]
            xyz1_clr = torch.cat([xyz_clr, torch.ones(N_clr, 1, device=self.device)], dim=1)
            p_clr = torch.matmul(xyz1_clr, l2i_t.T)

            z_clr = p_clr[:, 2]
            valid_clr = z_clr > 1e-5

            if valid_clr.any():
                p_clr_sel = p_clr[valid_clr]
                z_clr_sel = z_clr[valid_clr]
                rgb_sel = rgb[valid_clr]

                u_clr = torch.div(p_clr_sel[:, 0], z_clr_sel, rounding_mode='trunc').long()
                v_clr = torch.div(p_clr_sel[:, 1], z_clr_sel, rounding_mode='trunc').long()
                bounds_clr = (u_clr >= 0) & (u_clr < W) & (v_clr >= 0) & (v_clr < H)

                if bounds_clr.any():
                    u_clr = u_clr[bounds_clr]
                    v_clr = v_clr[bounds_clr]
                    rgb_sel = rgb_sel[bounds_clr]
                    indices_clr = v_clr * W + u_clr

        # ========== Depth: scatter_reduce_amin ==========
        depth = torch.full((H * W,), float('inf'), device=self.device)
        depth.scatter_reduce_(0, indices, z, reduce='amin', include_self=False)
        depth = torch.where(torch.isinf(depth), torch.tensor(invalid_depth, device=self.device), depth)
        depth_np = depth.reshape(H, W).cpu().numpy()

        min_depth_at_indices = depth[indices]
        min_mask = (z == min_depth_at_indices) & (min_depth_at_indices != float('inf'))

        valid_min_indices = indices[min_mask]
        valid_min_sem = sem[min_mask]
        valid_min_sem = torch.clamp(valid_min_sem, 0, n_actor_classes - 1)

        sem_map = torch.zeros((H * W, n_actor_classes), dtype=torch.uint8, device=self.device)
        sem_map.scatter_(0, valid_min_indices.unsqueeze(1).expand(-1, n_actor_classes),
                         torch.eye(n_actor_classes, device=self.device, dtype=torch.uint8)[valid_min_sem])
        sem_np = sem_map.reshape(H, W, n_actor_classes).cpu().numpy()

        if indices_clr is not None and rgb_sel is not None:
            scatter_indices = torch.flip(indices_clr, [0])
            scatter_rgb = torch.flip(rgb_sel, [0])
            rgb_uint8 = torch.clamp(scatter_rgb, 0, 255).byte()

            clr = torch.zeros((H * W, 3), dtype=torch.uint8, device=self.device)
            clr.scatter_(0, scatter_indices.unsqueeze(1).expand(-1, 3), rgb_uint8)
            clr_np = clr.reshape(H, W, 3).cpu().numpy()
        else:
            clr_np = np.zeros((H, W, 3), np.uint8)

        return depth_np, sem_np, clr_np


def get_optimized_projector(device='cuda'):
    return OptimizedPointProjector(device)


def _generate_splat_offsets(radius: int, H: int, W: int, device):
    """


    kh = kw = 2*r + 1
    """
    if radius <= 0:
        return None

    d = torch.arange(-radius, radius + 1, device=device)
    dx, dy = torch.meshgrid(d, d, indexing='xy')
    offsets_h = dy.reshape(-1)
    offsets_w = dx.reshape(-1)
    k = offsets_h.shape[0]

    return offsets_h, offsets_w, k


class OptimizedPointProjectorWithSplat(OptimizedPointProjector):
    """

    """

    def project_depth_with_splat(
        self, pts_xyz: np.ndarray, l2i: np.ndarray, ori_hw: tuple,
        invalid_depth=-300.0, splat=None
    ) -> np.ndarray:
        """

        """
        H, W = int(ori_hw[0]), int(ori_hw[1])

        if splat is None:
            splat = [(1e9, 0)]

        if isinstance(pts_xyz, torch.Tensor):
            xyz = pts_xyz
        else:
            xyz = torch.from_numpy(pts_xyz).float().to(self.device)

        if isinstance(l2i, torch.Tensor):
            l2i_t = l2i
        else:
            l2i_t = torch.from_numpy(l2i).float().to(self.device)

        if xyz.shape[0] == 0:
            return np.full((H, W), invalid_depth, np.float32)

        N = xyz.shape[0]
        xyz1 = torch.cat([xyz, torch.ones(N, 1, device=self.device)], dim=1)
        p = torch.matmul(xyz1, l2i_t.T)

        z = p[:, 2]
        valid = z > 1e-5

        if not valid.any():
            return np.full((H, W), invalid_depth, np.float32)

        p, z = p[valid], z[valid]
        u = torch.div(p[:, 0], z, rounding_mode='trunc').long()
        v = torch.div(p[:, 1], z, rounding_mode='trunc').long()

        depth = torch.full((H * W,), float('inf'), device=self.device)

        z_remaining, u_remaining, v_remaining = z.clone(), u.clone(), v.clone()

        for zmax, radius in splat:
            mask = z_remaining <= zmax
            if not mask.any():
                continue

            z_layer, u_layer, v_layer = z_remaining[mask], u_remaining[mask], v_remaining[mask]

            if radius <= 0:
                bounds = (u_layer >= 0) & (u_layer < W) & (v_layer >= 0) & (v_layer < H)
                if bounds.any():
                    u_valid, v_valid, z_valid = u_layer[bounds], v_layer[bounds], z_layer[bounds]
                    indices = v_valid * W + u_valid
                    depth.scatter_reduce_(0, indices, z_valid, reduce='amin', include_self=False)
            else:
                offsets_h, offsets_w, k = _generate_splat_offsets(radius, H, W, self.device)

                u_expanded = u_layer.unsqueeze(1) + offsets_w.unsqueeze(0)  # [N, k]
                v_expanded = v_layer.unsqueeze(1) + offsets_h.unsqueeze(0)
                z_expanded = z_layer.unsqueeze(1).expand(-1, k)

                u_flat = u_expanded.reshape(-1)
                v_flat = v_expanded.reshape(-1)
                z_flat = z_expanded.reshape(-1)

                bounds = (u_flat >= 0) & (u_flat < W) & (v_flat >= 0) & (v_flat < H)
                if bounds.any():
                    u_valid, v_valid, z_valid = u_flat[bounds], v_flat[bounds], z_flat[bounds]
                    indices = v_valid * W + u_valid
                    depth.scatter_reduce_(0, indices, z_valid, reduce='amin', include_self=False)

            z_remaining = z_remaining[~mask]
            u_remaining = u_remaining[~mask]
            v_remaining = v_remaining[~mask]

            if z_remaining.shape[0] == 0:
                break

        depth = torch.where(torch.isinf(depth), torch.tensor(invalid_depth, device=self.device), depth)
        return depth.reshape(H, W).cpu().numpy()

    def project_sem_with_splat(
        self, pts_xyz: np.ndarray, pts_sem: np.ndarray,
        l2i: np.ndarray, ori_hw: tuple, n_actor_classes=3, splat=None
    ) -> np.ndarray:
        """
        """
        H, W = int(ori_hw[0]), int(ori_hw[1])

        if splat is None:
            splat = [(1e9, 0)]

        if isinstance(pts_xyz, torch.Tensor):
            xyz = pts_xyz
        else:
            xyz = torch.from_numpy(pts_xyz).float().to(self.device)

        if isinstance(pts_sem, torch.Tensor):
            sem = pts_sem
        else:
            sem = torch.from_numpy(pts_sem).long().to(self.device)

        if sem.ndim > 1:
            sem = sem.view(-1)

        if isinstance(l2i, torch.Tensor):
            l2i_t = l2i
        else:
            l2i_t = torch.from_numpy(l2i).float().to(self.device)

        if xyz.shape[0] == 0:
            return np.zeros((H, W, n_actor_classes), np.uint8)

        N = xyz.shape[0]
        xyz1 = torch.cat([xyz, torch.ones(N, 1, device=self.device)], dim=1)
        p = torch.matmul(xyz1, l2i_t.T)

        z = p[:, 2]
        valid = z > 1e-5

        if not valid.any():
            return np.zeros((H, W, n_actor_classes), np.uint8)

        p, z, sem = p[valid], z[valid], sem[valid]

        sem_valid = (sem >= 1) & (sem <= n_actor_classes)
        if not sem_valid.any():
            return np.zeros((H, W, n_actor_classes), np.uint8)

        p, z, sem = p[sem_valid], z[sem_valid], sem[sem_valid]
        u = torch.div(p[:, 0], z, rounding_mode='trunc').long()
        v = torch.div(p[:, 1], z, rounding_mode='trunc').long()

        all_indices, all_z, all_sem = [], [], []

        z_remaining, u_remaining, v_remaining, sem_remaining = z.clone(), u.clone(), v.clone(), sem.clone()

        for zmax, radius in splat:
            mask = z_remaining <= zmax
            if not mask.any():
                continue

            z_layer, u_layer, v_layer, sem_layer = z_remaining[mask], u_remaining[mask], v_remaining[mask], sem_remaining[mask]

            if radius <= 0:
                bounds = (u_layer >= 0) & (u_layer < W) & (v_layer >= 0) & (v_layer < H)
                if bounds.any():
                    u_valid, v_valid, z_valid, sem_valid = u_layer[bounds], v_layer[bounds], z_layer[bounds], sem_layer[bounds]
                    idx = v_valid * W + u_valid
                    all_indices.append(idx)
                    all_z.append(z_valid)
                    all_sem.append(sem_valid)
            else:
                offsets_h, offsets_w, k = _generate_splat_offsets(radius, H, W, self.device)

                u_expanded = u_layer.unsqueeze(1) + offsets_w.unsqueeze(0)
                v_expanded = v_layer.unsqueeze(1) + offsets_h.unsqueeze(0)
                z_expanded = z_layer.unsqueeze(1).expand(-1, k)
                sem_expanded = sem_layer.unsqueeze(1).expand(-1, k)

                u_flat, v_flat, z_flat, sem_flat = u_expanded.reshape(-1), v_expanded.reshape(-1), z_expanded.reshape(-1), sem_expanded.reshape(-1)
                bounds = (u_flat >= 0) & (u_flat < W) & (v_flat >= 0) & (v_flat < H)
                if bounds.any():
                    u_valid, v_valid, z_valid, sem_valid = u_flat[bounds], v_flat[bounds], z_flat[bounds], sem_flat[bounds]
                    idx = v_valid * W + u_valid
                    all_indices.append(idx)
                    all_z.append(z_valid)
                    all_sem.append(sem_valid)

            z_remaining = z_remaining[~mask]
            u_remaining = u_remaining[~mask]
            v_remaining = v_remaining[~mask]
            sem_remaining = sem_remaining[~mask]

            if z_remaining.shape[0] == 0:
                break

        if not all_indices:
            return np.zeros((H, W, n_actor_classes), np.uint8)

        all_indices = torch.cat(all_indices)
        all_z = torch.cat(all_z)
        all_sem = torch.cat(all_sem) - 1

        min_depth = torch.full((H * W,), float('inf'), device=self.device)
        min_depth.scatter_reduce_(0, all_indices, all_z, reduce='amin', include_self=False)

        min_depth_at_indices = min_depth[all_indices]
        min_mask = (all_z == min_depth_at_indices) & (min_depth_at_indices != float('inf'))

        valid_indices = all_indices[min_mask]
        valid_sem = torch.clamp(all_sem[min_mask], 0, n_actor_classes - 1)

        sem_map = torch.zeros((H * W, n_actor_classes), dtype=torch.uint8, device=self.device)
        sem_map.scatter_(0, valid_indices.unsqueeze(1).expand(-1, n_actor_classes),
                         torch.eye(n_actor_classes, device=self.device, dtype=torch.uint8)[valid_sem])

        return sem_map.reshape(H, W, n_actor_classes).cpu().numpy()

    def project_clr_with_splat(
        self, clr_xyz: np.ndarray, clr_rgb: np.ndarray,
        l2i: np.ndarray, ori_hw: tuple, splat=None
    ) -> np.ndarray:
        """
        """
        H, W = int(ori_hw[0]), int(ori_hw[1])

        if splat is None:
            splat = [(1e9, 0)]

        if isinstance(clr_xyz, torch.Tensor):
            xyz = clr_xyz
        else:
            xyz = torch.from_numpy(clr_xyz).float().to(self.device)

        if isinstance(clr_rgb, torch.Tensor):
            rgb = clr_rgb
        else:
            rgb = torch.from_numpy(clr_rgb).float().to(self.device)

        if isinstance(l2i, torch.Tensor):
            l2i_t = l2i
        else:
            l2i_t = torch.from_numpy(l2i).float().to(self.device)

        if xyz.shape[0] == 0:
            return np.zeros((H, W, 3), np.uint8)

        N = xyz.shape[0]
        xyz1 = torch.cat([xyz, torch.ones(N, 1, device=self.device)], dim=1)
        p = torch.matmul(xyz1, l2i_t.T)

        z = p[:, 2]
        valid = z > 1e-5

        if not valid.any():
            return np.zeros((H, W, 3), np.uint8)

        p, z, rgb = p[valid], z[valid], rgb[valid]
        u = torch.div(p[:, 0], z, rounding_mode='trunc').long()
        v = torch.div(p[:, 1], z, rounding_mode='trunc').long()

        all_indices, all_z, all_rgb = [], [], []

        z_remaining, u_remaining, v_remaining, rgb_remaining = z.clone(), u.clone(), v.clone(), rgb.clone()

        for zmax, radius in splat:
            mask = z_remaining <= zmax
            if not mask.any():
                continue

            z_layer, u_layer, v_layer, rgb_layer = z_remaining[mask], u_remaining[mask], v_remaining[mask], rgb_remaining[mask]

            if radius <= 0:
                bounds = (u_layer >= 0) & (u_layer < W) & (v_layer >= 0) & (v_layer < H)
                if bounds.any():
                    u_valid, v_valid, z_valid, rgb_valid = u_layer[bounds], v_layer[bounds], z_layer[bounds], rgb_layer[bounds]
                    idx = v_valid * W + u_valid
                    all_indices.append(idx)
                    all_z.append(z_valid)
                    all_rgb.append(rgb_valid)
            else:
                offsets_h, offsets_w, k = _generate_splat_offsets(radius, H, W, self.device)

                u_expanded = u_layer.unsqueeze(1) + offsets_w.unsqueeze(0)
                v_expanded = v_layer.unsqueeze(1) + offsets_h.unsqueeze(0)
                z_expanded = z_layer.unsqueeze(1).expand(-1, k)
                rgb_expanded = rgb_layer.unsqueeze(1).expand(-1, k)

                u_flat, v_flat, z_flat, rgb_flat = u_expanded.reshape(-1), v_expanded.reshape(-1), z_expanded.reshape(-1), rgb_expanded.reshape(-1)
                bounds = (u_flat >= 0) & (u_flat < W) & (v_flat >= 0) & (v_flat < H)
                if bounds.any():
                    u_valid, v_valid, z_valid, rgb_valid = u_flat[bounds], v_flat[bounds], z_flat[bounds], rgb_flat[bounds]
                    idx = v_valid * W + u_valid
                    all_indices.append(idx)
                    all_z.append(z_valid)
                    all_rgb.append(rgb_valid)

            z_remaining = z_remaining[~mask]
            u_remaining = u_remaining[~mask]
            v_remaining = v_remaining[~mask]
            rgb_remaining = rgb_remaining[~mask]

            if z_remaining.shape[0] == 0:
                break

        if not all_indices:
            return np.zeros((H, W, 3), np.uint8)

        all_indices = torch.cat(all_indices)
        all_z = torch.cat(all_z)
        all_rgb = torch.cat(all_rgb)

        min_depth = torch.full((H * W,), float('inf'), device=self.device)
        min_depth.scatter_reduce_(0, all_indices, all_z, reduce='amin', include_self=False)

        min_depth_at_indices = min_depth[all_indices]
        min_mask = (all_z == min_depth_at_indices) & (min_depth_at_indices != float('inf'))

        valid_indices = all_indices[min_mask]
        valid_rgb = all_rgb[min_mask]

        clr = torch.zeros((H * W, 3), dtype=torch.uint8, device=self.device)
        rgb_uint8 = torch.clamp(valid_rgb, 0, 255).byte()
        clr.scatter_(0, valid_indices.unsqueeze(1).expand(-1, 3), rgb_uint8)

        return clr.reshape(H, W, 3).cpu().numpy()

    def project_all_with_splat(
        self, pts_xyz: np.ndarray, pts_sem: np.ndarray,
        clr_xyz: np.ndarray, clr_rgb: np.ndarray,
        l2i: np.ndarray, ori_hw: tuple, invalid_depth=-300.0, n_actor_classes=3,
        splat=None
    ) -> tuple:
        """


        """
        H, W = int(ori_hw[0]), int(ori_hw[1])

        if splat is None:
            splat = [(1e9, 0)]

        if isinstance(pts_xyz, torch.Tensor):
            xyz = pts_xyz
        else:
            xyz = torch.from_numpy(pts_xyz).float().to(self.device)

        if isinstance(pts_sem, torch.Tensor):
            sem = pts_sem
        else:
            sem = torch.from_numpy(pts_sem).long().to(self.device)

        if isinstance(clr_xyz, torch.Tensor):
            xyz_clr = clr_xyz
        else:
            xyz_clr = torch.from_numpy(clr_xyz).float().to(self.device)

        if isinstance(clr_rgb, torch.Tensor):
            rgb = clr_rgb
        else:
            rgb = torch.from_numpy(clr_rgb).float().to(self.device)

        if rgb.ndim > 2:
            rgb = rgb.reshape(-1, 3)
        elif rgb.ndim == 1:
            rgb = rgb.unsqueeze(1)

        if isinstance(l2i, torch.Tensor):
            l2i_t = l2i
        else:
            l2i_t = torch.from_numpy(l2i).float().to(self.device)

        if xyz.shape[0] == 0:
            return np.full((H, W), invalid_depth, np.float32), \
                   np.zeros((H, W, n_actor_classes), np.uint8), \
                   np.zeros((H, W, 3), np.uint8)
                

        N = xyz.shape[0]
        xyz1 = torch.cat([xyz, torch.ones(N, 1, device=self.device)], dim=1)
        p = torch.matmul(xyz1, l2i_t.T)

        z_pts = p[:, 2]
        valid = z_pts > 1e-5

        if not valid.any():
            return self._process_clr_only(xyz_clr, rgb, l2i_t, splat, H, W, invalid_depth)

        p, z_pts, sem = p[valid], z_pts[valid], sem[valid]
        u_pts = torch.div(p[:, 0], z_pts, rounding_mode='trunc').long()
        v_pts = torch.div(p[:, 1], z_pts, rounding_mode='trunc').long()

        clr_data = self._compute_clr_info(xyz_clr, rgb, l2i_t, H, W)

        same_cloud = False
        if clr_data is not None:
            if xyz.shape[0] == clr_data['z'].shape[0]:
                same_cloud = torch.allclose(xyz, xyz_clr, rtol=1e-6)

        depth = torch.full((H * W,), float('inf'), device=self.device)
        all_sem_indices, all_sem_z, all_sem_val = [], [], []
        all_clr_indices, all_clr_z, all_clr_rgb = [], [], []

        z_d_rem, u_d_rem, v_d_rem = z_pts.clone(), u_pts.clone(), v_pts.clone()

        if same_cloud:
            sem_valid_mask = (sem >= 1) & (sem <= n_actor_classes)
            if sem_valid_mask.any():
                z_sc_rem = z_pts[sem_valid_mask].clone()
                u_sc_rem = u_pts[sem_valid_mask].clone()
                v_sc_rem = v_pts[sem_valid_mask].clone()
                sem_vals = sem[sem_valid_mask].clone()
                if sem_vals.ndim > 1:
                    sem_vals = sem_vals.view(-1)
                rgb_vals = rgb[sem_valid_mask].clone()
            else:
                z_sc_rem = None
        else:
            sem_valid_mask = (sem >= 1) & (sem <= n_actor_classes)
            if sem_valid_mask.any():
                z_s_rem = z_pts[sem_valid_mask].clone()
                u_s_rem = u_pts[sem_valid_mask].clone()
                v_s_rem = v_pts[sem_valid_mask].clone()
                sem_vals = sem[sem_valid_mask].clone()
                if sem_vals.ndim > 1:
                    sem_vals = sem_vals.view(-1)
            else:
                z_s_rem = None

            if clr_data is not None:
                z_c_rem, u_c_rem, v_c_rem, rgb_rem = (
                    clr_data['z'].clone(),
                    clr_data['u'].clone(),
                    clr_data['v'].clone(),
                    clr_data['rgb'].clone()
                )
            else:
                z_c_rem = None

        splat_sorted = sorted(splat, key=lambda x: x[0])
        n_layers = len(splat_sorted)

        for i in range(n_layers - 1, -1, -1):
            zmax, radius = splat_sorted[i]
            z_prev = splat_sorted[i - 1][0] if i > 0 else float('-inf')

            layer_mask = (z_d_rem > z_prev) & (z_d_rem <= zmax)

            if layer_mask.any():
                z_layer, u_layer, v_layer = z_d_rem[layer_mask], u_d_rem[layer_mask], v_d_rem[layer_mask]
                self._splat_to_depth_direct(
                    z_layer, u_layer, v_layer,
                    radius, depth, H, W, self.device
                )

            if same_cloud:
                if z_sc_rem is not None:
                    sc_layer_mask = (z_sc_rem > z_prev) & (z_sc_rem <= zmax)
                    if sc_layer_mask.any():
                        u_layer = u_sc_rem[sc_layer_mask]
                        v_layer = v_sc_rem[sc_layer_mask]
                        z_layer = z_sc_rem[sc_layer_mask]
                        sem_layer = sem_vals[sc_layer_mask]
                        rgb_layer = rgb_vals[sc_layer_mask]

                        idx, z_out, sem_out, rgb_out = self._splat_collect_direct(
                            u_layer, v_layer, z_layer, sem_layer, rgb_layer,
                            radius, H, W, self.device
                        )
                        if idx is not None:
                            all_sem_indices.append(idx)
                            all_sem_z.append(z_out)
                            all_sem_val.append(sem_out)
                            all_clr_indices.append(idx)
                            all_clr_z.append(z_out)
                            all_clr_rgb.append(rgb_out)
            else:
                if z_s_rem is not None:
                    s_layer_mask = (z_s_rem > z_prev) & (z_s_rem <= zmax)
                    if s_layer_mask.any():
                        u_layer = u_s_rem[s_layer_mask]
                        v_layer = v_s_rem[s_layer_mask]
                        z_layer = z_s_rem[s_layer_mask]
                        sem_layer = sem_vals[s_layer_mask]

                        idx, z_out, sem_out, _ = self._splat_collect_direct(
                            u_layer, v_layer, z_layer, sem_layer, None,
                            radius, H, W, self.device
                        )
                        if idx is not None:
                            all_sem_indices.append(idx)
                            all_sem_z.append(z_out)
                            all_sem_val.append(sem_out)

                if z_c_rem is not None:
                    c_layer_mask = (z_c_rem > z_prev) & (z_c_rem <= zmax)
                    if c_layer_mask.any():
                        u_layer = u_c_rem[c_layer_mask]
                        v_layer = v_c_rem[c_layer_mask]
                        z_layer = z_c_rem[c_layer_mask]
                        rgb_layer = rgb_rem[c_layer_mask]

                        idx, z_out, _, rgb_out = self._splat_collect_direct(
                            u_layer, v_layer, z_layer, None, rgb_layer,
                            radius, H, W, self.device
                        )
                        if idx is not None:
                            all_clr_indices.append(idx)
                            all_clr_z.append(z_out)
                            all_clr_rgb.append(rgb_out)

        depth = torch.where(torch.isinf(depth), torch.tensor(invalid_depth, device=self.device), depth)

        sem_np = self._zbuffer_process(
            all_sem_indices, all_sem_z, all_sem_val, depth,
            H, W, n_actor_classes, self.device, is_sem=True
        )

        clr_np = self._zbuffer_process(
            all_clr_indices, all_clr_z, all_clr_rgb, None,
            H, W, n_actor_classes, self.device, is_sem=False
        )

        return depth.reshape(H, W).cpu().numpy(), sem_np, clr_np


    def _process_clr_only(self, xyz_clr, rgb, l2i_t, splat, H, W, invalid_depth):
        clr_data = self._compute_clr_info(xyz_clr, rgb, l2i_t, H, W)
        if clr_data is None:
            return np.full((H, W), invalid_depth, np.float32), \
                   np.zeros((H, W, 3), np.uint8), \
                   np.zeros((H, W, 3), np.uint8)

        depth = torch.full((H * W,), float('inf'), device=self.device)
        all_clr_indices, all_clr_z, all_clr_rgb = [], [], []

        z_c, u_c, v_c, rgb_c = clr_data['z'].clone(), clr_data['u'].clone(), clr_data['v'].clone(), clr_data['rgb'].clone()

        for zmax, radius in splat:
            mask = z_c <= zmax
            if not mask.any():
                continue
            idx, z_layer, rgb_layer = self._splat_collect(
                u_c, v_c, z_c, rgb_c, mask, radius, H, W, self.device, is_rgb=True
            )
            if idx is not None:
                all_clr_indices.append(idx)
                all_clr_z.append(z_layer)
                all_clr_rgb.append(rgb_layer)
            z_c, u_c, v_c, rgb_c = z_c[~mask], u_c[~mask], v_c[~mask], rgb_c[~mask]
            if z_c.shape[0] == 0:
                break

        depth = torch.where(torch.isinf(depth), torch.tensor(invalid_depth, device=self.device), depth)
        clr_np = self._zbuffer_process(all_clr_indices, all_clr_z, all_clr_rgb, None, H, W, 3, self.device, is_sem=False)
        return depth.reshape(H, W).cpu().numpy(), np.zeros((H, W, 3), np.uint8), clr_np

    def _compute_clr_info(self, xyz_clr, rgb, l2i_t, H, W):
        if xyz_clr.shape[0] == 0:
            return None
        N = xyz_clr.shape[0]
        xyz1 = torch.cat([xyz_clr, torch.ones(N, 1, device=self.device)], dim=1)
        p = torch.matmul(xyz1, l2i_t.T)
        z = p[:, 2]
        valid = z > 1e-5
        if not valid.any():
            return None
        p, z, rgb = p[valid], z[valid], rgb[valid]
        u = torch.div(p[:, 0], z, rounding_mode='trunc').long()
        v = torch.div(p[:, 1], z, rounding_mode='trunc').long()
        if rgb.ndim > 2:
            rgb = rgb.reshape(-1, 3)
        elif rgb.ndim == 1:
            rgb = rgb.unsqueeze(1)
        if rgb.shape[-1] != 3:
            rgb = rgb.view(-1, 3)
        return {'z': z, 'u': u, 'v': v, 'rgb': rgb}

    def _splat_to_depth(self, z, u, v, mask, radius, depth, H, W, device):
        z_l, u_l, v_l = z[mask], u[mask], v[mask]
        if radius <= 0:
            bounds = (u_l >= 0) & (u_l < W) & (v_l >= 0) & (v_l < H)
            if bounds.any():
                idx = v_l[bounds] * W + u_l[bounds]
                depth.scatter_reduce_(0, idx, z_l[bounds], reduce='amin', include_self=False)
        else:
            offsets_h, offsets_w, k = _generate_splat_offsets(radius, H, W, device)
            u_exp = u_l.unsqueeze(1) + offsets_w.unsqueeze(0)
            v_exp = v_l.unsqueeze(1) + offsets_h.unsqueeze(0)
            z_exp = z_l.unsqueeze(1).expand(-1, k)
            u_flat, v_flat, z_flat = u_exp.reshape(-1), v_exp.reshape(-1), z_exp.reshape(-1)
            bounds = (u_flat >= 0) & (u_flat < W) & (v_flat >= 0) & (v_flat < H)
            if bounds.any():
                idx = v_flat[bounds] * W + u_flat[bounds]
                depth.scatter_reduce_(0, idx, z_flat[bounds], reduce='amin', include_self=False)

    def _splat_collect(self, u, v, z, data, mask, radius, H, W, device, is_rgb):
        u_l, v_l, z_l, d_l = u[mask], v[mask], z[mask], data[mask]
        if radius <= 0:
            bounds = (u_l >= 0) & (u_l < W) & (v_l >= 0) & (v_l < H)
            if bounds.any():
                idx = v_l[bounds] * W + u_l[bounds]
                return idx, z_l[bounds], d_l[bounds]
            return None, None, None
        else:
            offsets_h, offsets_w, k = _generate_splat_offsets(radius, H, W, device)
            u_exp = u_l.unsqueeze(1) + offsets_w.unsqueeze(0)
            v_exp = v_l.unsqueeze(1) + offsets_h.unsqueeze(0)
            z_exp = z_l.unsqueeze(1).expand(-1, k)

            if is_rgb:
                # rgb: [N, 3] -> [N, k, 3]
                d_exp = d_l.unsqueeze(1).expand(-1, k, -1)
                u_flat, v_flat, z_flat = u_exp.reshape(-1), v_exp.reshape(-1), z_exp.reshape(-1)
                d_flat = d_exp.reshape(-1, 3)
            else:
                # sem: [N] -> [N, k]
                d_exp = d_l.unsqueeze(1).expand(-1, k)
                u_flat, v_flat, z_flat, d_flat = u_exp.reshape(-1), v_exp.reshape(-1), z_exp.reshape(-1), d_exp.reshape(-1)

            bounds = (u_flat >= 0) & (u_flat < W) & (v_flat >= 0) & (v_flat < H)
            if bounds.any():
                idx = v_flat[bounds] * W + u_flat[bounds]
                return idx, z_flat[bounds], d_flat[bounds]
            return None, None, None

    def _splat_collect_combined(self, u, v, z, sem_data, rgb_data, mask, radius, H, W, device):
        u_l, v_l, z_l, sem_l, rgb_l = u[mask], v[mask], z[mask], sem_data[mask], rgb_data[mask]
        if radius <= 0:
            bounds = (u_l >= 0) & (u_l < W) & (v_l >= 0) & (v_l < H)
            if bounds.any():
                idx = v_l[bounds] * W + u_l[bounds]
                return idx, z_l[bounds], sem_l[bounds], rgb_l[bounds]
            return None, None, None, None
        else:
            offsets_h, offsets_w, k = _generate_splat_offsets(radius, H, W, device)
            u_exp = u_l.unsqueeze(1) + offsets_w.unsqueeze(0)
            v_exp = v_l.unsqueeze(1) + offsets_h.unsqueeze(0)
            z_exp = z_l.unsqueeze(1).expand(-1, k)

            sem_exp = sem_l.unsqueeze(1).expand(-1, k)
            rgb_exp = rgb_l.unsqueeze(1).expand(-1, k, -1)

            u_flat, v_flat, z_flat = u_exp.reshape(-1), v_exp.reshape(-1), z_exp.reshape(-1)
            sem_flat = sem_exp.reshape(-1)
            rgb_flat = rgb_exp.reshape(-1, 3)

            bounds = (u_flat >= 0) & (u_flat < W) & (v_flat >= 0) & (v_flat < H)
            if bounds.any():
                idx = v_flat[bounds] * W + u_flat[bounds]
                return idx, z_flat[bounds], sem_flat[bounds], rgb_flat[bounds]
            return None, None, None, None

    def _splat_to_depth_direct(self, z, u, v, radius, depth, H, W, device):
        if radius <= 0:
            bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if bounds.any():
                idx = v[bounds] * W + u[bounds]
                depth.scatter_reduce_(0, idx, z[bounds], reduce='amin', include_self=False)
        else:
            offsets_h, offsets_w, k = _generate_splat_offsets(radius, H, W, device)
            u_exp = u.unsqueeze(1) + offsets_w.unsqueeze(0)
            v_exp = v.unsqueeze(1) + offsets_h.unsqueeze(0)
            z_exp = z.unsqueeze(1).expand(-1, k)
            u_flat, v_flat, z_flat = u_exp.reshape(-1), v_exp.reshape(-1), z_exp.reshape(-1)
            bounds = (u_flat >= 0) & (u_flat < W) & (v_flat >= 0) & (v_flat < H)
            if bounds.any():
                idx = v_flat[bounds] * W + u_flat[bounds]
                depth.scatter_reduce_(0, idx, z_flat[bounds], reduce='amin', include_self=False)

    def _splat_collect_direct(self, u, v, z, sem_data, rgb_data, radius, H, W, device):
        """
        """
        if radius <= 0:
            bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if bounds.any():
                idx = v[bounds] * W + u[bounds]
                z_out = z[bounds]
                sem_out = sem_data[bounds] if sem_data is not None else None
                rgb_out = rgb_data[bounds] if rgb_data is not None else None
                return idx, z_out, sem_out, rgb_out
            return None, None, None, None
        else:
            offsets_h, offsets_w, k = _generate_splat_offsets(radius, H, W, device)
            u_exp = u.unsqueeze(1) + offsets_w.unsqueeze(0)
            v_exp = v.unsqueeze(1) + offsets_h.unsqueeze(0)
            z_exp = z.unsqueeze(1).expand(-1, k)

            if sem_data is not None:
                sem_exp = sem_data.unsqueeze(1).expand(-1, k)
                sem_flat = sem_exp.reshape(-1)
            else:
                sem_flat = None

            if rgb_data is not None:
                rgb_exp = rgb_data.unsqueeze(1).expand(-1, k, -1)
                rgb_flat = rgb_exp.reshape(-1, 3)
            else:
                rgb_flat = None

            u_flat, v_flat, z_flat = u_exp.reshape(-1), v_exp.reshape(-1), z_exp.reshape(-1)
            bounds = (u_flat >= 0) & (u_flat < W) & (v_flat >= 0) & (v_flat < H)
            if bounds.any():
                idx = v_flat[bounds] * W + u_flat[bounds]
                z_out = z_flat[bounds]
                sem_out = sem_flat[bounds] if sem_flat is not None else None
                rgb_out = rgb_flat[bounds] if rgb_flat is not None else None
                return idx, z_out, sem_out, rgb_out
            return None, None, None, None

    def _zbuffer_process(self, all_indices, all_z, all_data, depth, H, W, n_actor_classes, device, is_sem):
        if not all_indices or all_indices[0] is None:
            return np.zeros((H, W, n_actor_classes), np.uint8) if is_sem else np.zeros((H, W, 3), np.uint8)

        all_indices = torch.cat(all_indices)
        all_z = torch.cat(all_z)
        all_data = torch.cat(all_data)

        if is_sem:
            all_data = all_data - 1
            if all_data.ndim > 1:
                all_data = all_data.view(-1)
            # one-hot scatter
            min_depth_at = depth[all_indices]
            min_mask = (all_z == min_depth_at) & (min_depth_at != float('inf'))
            valid_idx = all_indices[min_mask]
            valid_data = torch.clamp(all_data[min_mask], 0, n_actor_classes - 1)

            output = torch.zeros((H * W, n_actor_classes), dtype=torch.uint8, device=device)
            output.scatter_(0, valid_idx.unsqueeze(1).expand(-1, n_actor_classes),
                           torch.eye(n_actor_classes, device=device, dtype=torch.uint8)[valid_data])
            return output.reshape(H, W, n_actor_classes).cpu().numpy()
        else:
            # rgb scatter
            # min_depth_at = depth[all_indices]
            # min_mask = (all_z == min_depth_at) & (min_depth_at != float('inf'))
            # valid_idx = all_indices[min_mask]
            # valid_rgb = all_data[min_mask]

            if depth is None:
                min_depth = torch.full((H * W,), float('inf'), device=device)
                min_depth.scatter_reduce_(0, all_indices, all_z, reduce='amin', include_self=False)
                min_depth_at = min_depth[all_indices]
            else:
                min_depth_at = depth[all_indices]

            min_mask = (all_z == min_depth_at) & (min_depth_at != float('inf'))
            valid_idx = all_indices[min_mask]
            valid_rgb = all_data[min_mask]

            output = torch.zeros((H * W, 3), dtype=torch.uint8, device=device)
            rgb_uint8 = torch.clamp(valid_rgb, 0, 255).byte()
            output.scatter_(0, valid_idx.unsqueeze(1).expand(-1, 3), rgb_uint8)
            return output.reshape(H, W, 3).cpu().numpy()


def get_optimized_projector_with_splat(device='cuda'):
    return OptimizedPointProjectorWithSplat(device)
