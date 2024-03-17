import torch
import torch.nn as nn
import functools
from .layers import ResnetBlockDDPM, Upsample, Downsample, ddpm_conv3x3, default_init, AttnBlock, get_timestep_embedding_linear, get_timestep_embedding_fourier

class Net(nn.Module):
  def __init__(self, config):
    super().__init__()
    # https://github.com/yang-song/score_sde_pytorch/blob/main/configs/vp/ddpm/cifar10.py
    # https://github.com/yang-song/score_sde_pytorch/blob/main/configs/default_cifar10_configs.py
    self.act = act = nn.SiLU()
    #self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    
    self.temb_func = get_timestep_embedding_linear if config["model"]["embedding_method"] == "linear" else get_timestep_embedding_fourier
    self.nf = nf = config["model"]["ch"]
    ch_mult = config["model"]["ch_mult"]
    self.num_res_blocks = num_res_blocks = config["model"]["num_res_blocks"]
    self.attn_resolutions = attn_resolutions = config["model"]["attention_resolutions"]
    dropout = config["model"]["dropout"]
    resamp_with_conv = config["model"]["resample_with_conv"]
    self.num_resolutions = num_resolutions = len(ch_mult)
    image_resolution = config["data_loader"]["image_size"]
    self.all_resolutions = all_resolutions = [image_resolution // (2 ** i) for i in range(num_resolutions)]

    AttnBlock_partial = functools.partial(AttnBlock)
    self.conditional = conditional = True
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    if conditional:
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)]
      modules[0].weight.data = default_init()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_init()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)

    # ours is in [0, 1]
    self.centered = False
    channels = 3

    # Downsampling block
    modules.append(ddpm_conv3x3(channels, nf))
    hs_c = [nf]
    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock_partial(channels=in_ch))
        hs_c.append(in_ch)
      if i_level != num_resolutions - 1:
        modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock_partial(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
        in_ch = out_ch
      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock_partial(channels=in_ch))
      if i_level != 0:
        modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

    assert not hs_c
    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    modules.append(ddpm_conv3x3(in_ch, channels, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

    self.scale_by_sigma = False

  def forward(self, x, labels):
    modules = self.all_modules
    m_idx = 0
    if self.conditional:
      # timestep/scale embedding
      timesteps = labels
      temb = self.temb_func(timesteps, self.nf)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    if self.centered:
      # Input is in [-1, 1]
      h = x
    else:
      # Input is in [0, 1]
      h = 2 * x - 1.

    # Downsampling block
    hs = [modules[m_idx](h)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(modules[m_idx](hs[-1]))
        m_idx += 1

    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1
      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1
      if i_level != 0:
        h = modules[m_idx](h)
        m_idx += 1

    assert not hs
    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    assert m_idx == len(modules)

    return h