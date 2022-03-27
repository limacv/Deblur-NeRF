import torch

# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

HALF_PIX = 0.5


class ToneMapping(nn.Module):
    def __init__(self, map_type: str):
        super(ToneMapping, self).__init__()
        assert map_type in ['none', 'gamma', 'learn', 'ycbcr']
        self.map_type = map_type
        if map_type == 'learn':
            self.linear = nn.Sequential(
                nn.Linear(1, 16), nn.ReLU(),
                nn.Linear(16, 16), nn.ReLU(),
                nn.Linear(16, 16), nn.ReLU(),
                nn.Linear(16, 1)
            )

    def forward(self, x):
        if self.map_type == 'none':
            return x
        elif self.map_type == 'learn':
            ori_shape = x.shape
            x_in = x.reshape(-1, 1)
            res_x = self.linear(x_in) * 0.1
            x_out = torch.sigmoid(res_x + x_in)
            return x_out.reshape(ori_shape)
        elif self.map_type == 'gamma':
            return x ** (1. / 2.2)
        else:
            assert RuntimeError("map_type not recognized")


def visualize_crf2d(crf: nn.Module, min_=0, max_=1, mine=-3, maxe=3, islog=False, reverse=False):
    i = torch.linspace(min_, max_, 256)
    e = torch.linspace(mine, maxe, 256)
    x, y = torch.meshgrid(i, e)
    with torch.no_grad():
        out = crf(-x if reverse else x, y)
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x.cpu().numpy(), y.cpu().numpy(), out.cpu().numpy(), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def visualize_crf(crf: nn.Module, min_=0, max_=1, e=0, islog=False, reverse=False):
    i = torch.linspace(min_, max_, 256)
    with torch.no_grad():
        out = crf(-i if reverse else i, torch.ones_like(i) * e)
    import matplotlib.pyplot as plt
    plt.plot(i.cpu().numpy(), out.detach().cpu().numpy())
    plt.show()


@torch.no_grad()
def visualize_kernel(H, W, K, nerf: nn.Module, img_idx=1, x=1, y=1, depth=0.5, color=False, weight=False):
    if isinstance(nerf, nn.DataParallel):
        nerf = nerf.module

    nerf.cuda()
    ray_info = {}
    ray_info["images_idx"] = torch.tensor(img_idx).type(torch.int64).cuda().reshape(-1, 1).expand(100, 1)
    ray_info["rays_x"] = torch.ones_like(ray_info["images_idx"]) * x
    ray_info["rays_y"] = torch.ones_like(ray_info["images_idx"]) * y
    ray_info["ray_depth"] = torch.ones_like(ray_info["rays_x"]) * depth
    nerf.kernelsnet.random_hwindow = 0
    rays, weights, _ = nerf.kernelsnet(H, W, K, None, ray_info)
    rays_d = rays[..., 1]
    poses = nerf.kernelsnet.poses
    r_inv = poses[img_idx:img_idx+1, :3, :3].inverse()
    rays_d = (r_inv[:, None] @ rays_d[..., None]).squeeze(-1)
    rays_d = rays_d / -rays_d[..., -1:]
    rays_x = rays_d[..., 0] * K[0, 0] + K[0, 2]
    rays_y = rays_d[..., 1] * K[1, 1] - K[1, 2]  # it's reversed, attention!
    rays_x = rays_x - ray_info["rays_x"]
    rays_y = rays_y + ray_info["rays_y"]

    import matplotlib.pyplot as plt
    colors = np.linspace(0, 255, rays_x.permute(1, 0).reshape(-1).shape[0]).astype(np.uint8) if color else None
    scale = (weights.permute(1, 0).reshape(-1).cpu().numpy() * 200).astype(np.uint8) if weight else None
    plt.scatter(rays_x.permute(1, 0).reshape(-1).cpu().numpy(),
                rays_y.permute(1, 0).reshape(-1).cpu().numpy(), scale, colors)
    plt.show()


@torch.no_grad()
def visualize_itsample(H, W, K, nerf: nn.Module, x=1, y=1, img_idx=1, ptnum=1000, color=False):
    if isinstance(nerf, nn.DataParallel):
        nerf = nerf.module

    nerf.cuda()
    ray_info = {}
    ray_info["images_idx"] = torch.tensor(img_idx).type(torch.int64).cuda().reshape(-1, 1)
    nerf.kernelsnet.num_pt = ptnum
    ray_info["rays_x"] = torch.ones_like(ray_info["images_idx"]) * x
    ray_info["rays_y"] = torch.ones_like(ray_info["images_idx"]) * y
    ray_info["ray_depth"] = torch.ones_like(ray_info["rays_x"]) * 0.5
    rays, weights, loss = nerf.kernelsnet(H, W, K, None, ray_info)
    rays_d = rays[..., 1]
    poses = nerf.kernelsnet.poses
    r_inv = poses[img_idx:img_idx+1, :3, :3].inverse()
    rays_d = (r_inv[:, None] @ rays_d[..., None]).squeeze(-1)
    rays_d = rays_d / -rays_d[..., -1:]
    rays_x = rays_d[..., 0] * K[0, 0] + K[0, 2]
    rays_y = rays_d[..., 1] * K[1, 1] - K[1, 2]  # it's reversed, attention!
    rays_x = rays_x - ray_info["rays_x"]
    rays_y = - rays_y - ray_info["rays_y"]

    import matplotlib.pyplot as plt
    colors = np.linspace(0, 255, rays_x.shape[1]).astype(np.uint8) if color else None
    plt.scatter(rays_x[0].cpu().numpy(), rays_y[0].cpu().numpy(), None, colors)
    plt.show()


@torch.no_grad()
def visualize_kmap(H, W, K, nerf: nn.Module, x=1, y=1, img_idx=1, softmax=False):
    if isinstance(nerf, nn.DataParallel):
        nerf = nerf.module

    nerf.cuda()
    self = nerf.kernelsnet
    img_embed = self.img_embed[img_idx][None, :]
    x = img_embed
    # forward
    x = x[..., None, None]
    x = self.cnns(x).squeeze(1)[0]
    w, h = x.shape
    if softmax:
        x = torch.softmax(x.reshape(-1), dim=0).reshape(w, h)
    x = x.cpu().numpy()

    import matplotlib.pyplot as plt
    plt.imshow(x)
    plt.show()


@torch.no_grad()
def visualize_motionposes(H, W, K, nerf: nn.Module, img_idx=1):
    if isinstance(nerf, nn.DataParallel):
        nerf = nerf.module

    nerf.cuda()
    assert hasattr(nerf.kernelsnet, "rotations")
    assert hasattr(nerf.kernelsnet, "trans")

    self = nerf.kernelsnet
    r_x = self.rotations[..., 0, :] / torch.norm(self.rotations[..., 0, :], dim=2, keepdim=True)
    r_y = self.rotations[..., 1, :] / torch.norm(self.rotations[..., 1, :], dim=2, keepdim=True)
    r_z = torch.cross(r_x, r_y, dim=2)
    rotations = torch.stack([r_x, r_y, r_z], dim=-1)
    delta_poses = torch.cat([rotations, self.trans[..., None]], dim=-1)
    visualize_pose(delta_poses[img_idx].cpu().numpy(), W / H, K[0, 0])


# Positional encoding (section 5.1)
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            self.freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                out_dim += d

        self.out_dim = out_dim

    def forward(self, inputs):
        # print(f"input device: {inputs.device}, freq_bands device: {self.freq_bands.device}")
        self.freq_bands = self.freq_bands.type_as(inputs)
        outputs = []
        if self.kwargs['include_input']:
            outputs.append(inputs)

        for freq in self.freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                outputs.append(p_fn(inputs * freq))
        return torch.cat(outputs, -1)


def get_embedder(multires, i=0, input_dim=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dim,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i + (HALF_PIX - K[0][2])) / K[0][0], -(j + (HALF_PIX - K[1][2])) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i + (HALF_PIX - K[0][2])) / K[0][0], -(j + (HALF_PIX - K[1][2])) / K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    See Paper supplementary for details
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def smart_load_state_dict(model: nn.Module, state_dict: dict):
    if "network_fn_state_dict" in state_dict.keys():
        state_dict_fn = {k.lstrip("module."): v for k, v in state_dict["network_fn_state_dict"].items()}
        state_dict_fn = {"mlp_coarse." + k: v for k, v in state_dict_fn.items()}

        state_dict_fine = {k.lstrip("module."): v for k, v in state_dict["network_fine_state_dict"].items()}
        state_dict_fine = {"mlp_fine." + k: v for k, v in state_dict_fine.items()}
        state_dict_fn.update(state_dict_fine)
        state_dict = state_dict_fn
    elif "network_state_dict" in state_dict.keys():
        state_dict = {k[7:]: v for k, v in state_dict["network_state_dict"].items()}
    else:
        state_dict = state_dict

    if isinstance(model, nn.DataParallel):
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
