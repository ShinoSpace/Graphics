import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']   # d=3 for 3d world coordinate (x, y, z)
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    """ build position encoding """
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


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
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
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
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    """ get rays parametric function in world space: a ref point `rays_o` and direction `rays_d`
        rays are start from camera optical center and emit to image pixel

    Args:
        c2w (Tensor): (3, 4) camera-to-world transform matrix.

    Returns:
        rays_o, rays_d (Tensor): rays origin point and direction vector in world space.
            both shape = (H, W, 3) where 3 means (Xw, Yw, Zw)
    """
    # torch meshgrid default indexing is "ij"
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H)) # (W, H) for both
    i = i.t()   # -> (H, W), W coordinate
    j = j.t()   # -> (H, W), H coordinate
    """ #^ #################### details of ray direction #################### ^#
        details of ray direction: when we face to image plane,
        the image pixel coordinate is u - right, v - down,
        then the raw camera coordinate will be X - right, Y - down, Z - points to image

        we need follow the camera coordinate convention in graphic:
        X - right, Y - up and camera gaze at -Z

        direction vector: points from camera origin to normalized image plane, i.e. Zc = 1
    """
    #^ direction vector in camera space, from origin to Zc = 1 plane ^#
    #^ dirs.shape = (H, W, 3), 3 means (Xc, Yc, Zc) ^#
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    #^ vector's origin is always at (0, 0, 0), so only rotate applied ^#
    #^ this is actually c2w.dot(each 3d direction vector), implemented by broadcast ^#
    #^ rays_d.shape = (H, W, 3) ^#
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    #^ camera coordinate's origin in world space ^#
    rays_o = c2w[:3,-1].expand(rays_d.shape)

    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    """ get rays parametric function in world space: a ref point `rays_o` and direction `rays_d`
        rays are start from camera optical center and emit to image pixel

    Args:
        c2w (Tensor): (3, 4) camera-to-world transform matrix.

    Returns:
        rays_o, rays_d (Tensor): rays origin point and direction vector in world space.
            both shape = (H, W, 3) where 3 means (Xw, Yw, Zw)
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    """ #^ #################### details of ray direction #################### ^#
        details of ray direction: when we face to image plane,
        the pixel coordinate of image will be u - right, v - down.
        then the raw camera coordinate will be X - right, Y - down, Z - points to image

        we need follow the camera coordinate convention in graphic:
        X - right, Y - up and camera gaze at -Z

        direction vector: points from camera origin to normalized image plane, i.e. Zc = 1
    """
    #^ direction vector in camera space, from origin to Zc = 1 plane ^#
    #^ dirs.shape = (H, W, 3), 3 means (Xc, Yc, Zc) ^#
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    #^ vector's origin is always at (0, 0, 0), so only rotate applied ^#
    #^ this is actually c2w.dot(each 3d direction vector), implemented by broadcast ^#
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    #^ camera coordinate's origin in world space ^#
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))

    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """ convert ray parametric function from world space to related NDC space.
        NDC space is only used in forward facing scenes and keep world space for
        360 degree scenes.

        see this for OpenGL 3d projection and NDC space:
        http://www.songho.ca/opengl/gl_projectionmatrix.html
        How the NDC works? NeRF paper's appendix C give the whole derivation.
        but we can see this for better explanation:
        https://yconquesty.github.io/blog/ml/nerf/nerf_ndc.html#background

    Args:
        rays_o, rays_d (Tensor): ray origin point and direction vector in world space.
            shape = (num_rays, 3)
    """

    """ #^ first shift ray origins to near plane before 3d projection. ^#
        the intention is: there's no object between optical center and near plane
        so we do not need sample location in this region

        NOTE: near, far > 0
    """
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    #^ NeRF appendix C, equation (25) ^#
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    #^ NeRF appendix C, equation (26) ^#
    d0 = -1./ (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1./ (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[...,2]

    #^ NeRF appendix C, equation (10) ^#
    #^ π(o + td) = o' + t'd', where o + td is in world space and o' + t'd' is in NDC space ^#
    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

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
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
