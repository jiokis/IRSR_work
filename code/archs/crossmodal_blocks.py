"""Cross-modal (visible-light guided) blocks for IR SR: `veg` / `vsg` / `crm`.

Design (see docs/advisor/11):
- `CrossModalContext`: shared builder, computed ONCE per forward. Contains
    * VI light encoder (kept at LR resolution),
    * zero-parameter Sobel edge prior E (from VI),
    * VI semantic features guided by IR heat semantics (heat map + heat-distribution descriptor),
    * reliability r (from VI statistics) and spectral budget b (from IR low band).
- `BlockVEG`: edge-prior injection (Part 1: visible EDGE guidance).
- `BlockVSG`: semantic injection (Part 2: IR heat semantics -> VI semantic re-encoding).
- `BlockCRM`: marker block; its presence in `block_types` enables the reliability x budget gating
  (ablation switch: without it the injection scale is fixed to 1).

All injections are zero-initialised => at init the network output is EXACTLY the pure-IR baseline output.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _act(c):
    return nn.PReLU(c)


class CrossModalContext(nn.Module):
    """Shared cross-modal context builder (VI + IR -> edge / semantic / heat / reliability / budget)."""

    def __init__(self, c, vi_in=3):
        super().__init__()
        self.c = c
        # ---- VI light encoder (LR resolution) ----
        self.vi_stem = nn.Sequential(
            nn.Conv2d(vi_in, c // 2, 3, padding=1), _act(c // 2),
            nn.Conv2d(c // 2, c, 3, padding=1), _act(c))
        # ---- zero-parameter Sobel edge prior (per-channel) ----
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.0
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 4.0
        self.register_buffer('kx', kx.view(1, 1, 3, 3).repeat(c, 1, 1, 1))
        self.register_buffer('ky', ky.view(1, 1, 3, 3).repeat(c, 1, 1, 1))
        self.edge_proj = nn.Sequential(nn.Conv2d(1, c, 1), _act(c), nn.Conv2d(c, c, 1))
        # ---- VI semantic encoder + channel attention ----
        hid = max(c // 4, 8)
        self.sem = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), _act(c), nn.Conv2d(c, c, 1))
        self.sem_se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, hid, 1), _act(hid),
                                    nn.Conv2d(hid, c, 1), nn.Sigmoid())
        # ---- IR heat-semantic extractor ----
        self.heat = nn.Sequential(nn.Conv2d(vi_in, c // 2, 3, padding=1), _act(c // 2),
                                  nn.Conv2d(c // 2, 1, 1), nn.Sigmoid())
        self.heat_desc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(1, 32, 1), _act(32),
                                       nn.Conv2d(32, c, 1), nn.Tanh())
        # ---- reliability r (VI statistics) / spectral budget b (IR low band) ----
        self.rel = nn.Sequential(nn.Conv2d(c, 16, 3, padding=1), _act(16),
                                 nn.AdaptiveAvgPool2d(1), nn.Conv2d(16, 1, 1), nn.Sigmoid())
        self.budget = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(vi_in, 16, 1), _act(16),
                                    nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(self, ir_lq, vi):
        v = self.vi_stem(vi)
        c = v.size(1)
        gx = F.conv2d(v, self.kx, padding=1, groups=c)
        gy = F.conv2d(v, self.ky, padding=1, groups=c)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6).mean(dim=1, keepdim=True)   # (B,1,H,W)
        edge = self.edge_proj(edge)                                            # (B,C,H,W) edge prior
        sem = self.sem(v)
        sem = sem * self.sem_se(sem)                                           # VI semantic (channel attn)
        heat = self.heat(ir_lq)                                                # (B,1,H,W) heat source map
        desc = self.heat_desc(heat)                                            # (B,C,1,1) heat distribution
        sem_hat = sem * (1.0 + desc) * (1.0 + heat)      # IR heat semantics -> VI semantic re-encoding
        r = self.rel(v)                                  # (B,1,1,1) reliability
        b = self.budget(ir_lq)                           # (B,1,1,1) spectral budget
        return {'edge': edge, 'sem': sem_hat, 'vi_feat': v, 'heat': heat, 'desc': desc, 'r': r, 'b': b, 'gate': False}


class BlockVEG(nn.Module):
    """(veg) Visible-EDGE guidance (Part 1): zero-parameter edge prior injection, zero-init -> identity."""
    accepts_ctx = True

    def __init__(self, c):
        super().__init__()
        self.g = nn.Conv2d(2 * c, c, 1)
        self.r = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.g.weight); nn.init.zeros_(self.g.bias)
        nn.init.zeros_(self.r.weight); nn.init.zeros_(self.r.bias)
        self.act = nn.Tanh()

    def forward(self, x, ctx=None):
        if ctx is None:
            return x
        e = ctx['edge']
        s = ctx['r'] * ctx['b'] if ctx.get('gate', False) else 1.0
        y = x * self.act(self.g(torch.cat([e, x], dim=1))) + self.r(e)
        return x + s * y


class BlockVSG(nn.Module):
    """(vsg) Visible-SEMANTIC guidance (Part 2): IR heat-semantics-conditioned VI semantic injection."""
    accepts_ctx = True

    def __init__(self, c):
        super().__init__()
        self.g = nn.Conv2d(2 * c, c, 1)
        self.r = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.g.weight); nn.init.zeros_(self.g.bias)
        nn.init.zeros_(self.r.weight); nn.init.zeros_(self.r.bias)
        self.act = nn.Tanh()

    def forward(self, x, ctx=None):
        if ctx is None:
            return x
        e = ctx['sem']
        s = ctx['r'] * ctx['b'] if ctx.get('gate', False) else 1.0
        y = x * self.act(self.g(torch.cat([e, x], dim=1))) + self.r(e)
        return x + s * y


class BlockCRM(nn.Module):
    """(crm) Cross-modal Reliability modulator marker: presence enables reliability x budget gating."""
    accepts_ctx = False

    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x):
        return x


class CrossModalFusion(nn.Module):
    """TSCF (ours): Thermal-Semantic Conditioned Fusion = S1 rectify + S2 asymmetric condition + S3 budget-gated fuse.

    S1 (CMX-style, symmetric bidirectional channel reweighting) -- proven stable.
    S2 (ours, asymmetric): IR heat-semantics condition VI features; VI edges condition IR features.
    S3 (ours): CAFNet-style gate, but multiplied by reliability r and spectral budget b (hallucination control).
    Adapter-style: zero/hard-to-hurt? -> we train it with the IR backbone FROZEN, so the baseline is safe.
    """

    def __init__(self, c, hid=None):
        super().__init__()
        hid = hid or max(c // 4, 8)
        # S1: shared MLP for channel attention (avg+max pooling)
        self.mlp = nn.Sequential(nn.Conv2d(c, hid, 1), nn.PReLU(hid), nn.Conv2d(hid, c, 1))
        # S2: thermal-semantic conditioning (IR -> VI) and edge conditioning (VI -> IR)
        self.cond_heat = nn.Conv2d(c + 1, c, 1)      # heat map + descriptor channel
        self.cond_edge = nn.Conv2d(c, c, 1)
        # S3: gated fusion
        self.gate = nn.Sequential(nn.Conv2d(2 * c, c, 1), nn.PReLU(c), nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.out = nn.Conv2d(c, c, 1)
        # zero-init the fusion projection: the adapter starts EXACTLY as the frozen baseline (33.80)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, ctx):
        v = ctx['vi_feat']
        # ---- S1: symmetric bidirectional channel reweighting ----
        def stat(t):
            return t.mean(dim=(2, 3), keepdim=True) + t.amax(dim=(2, 3), keepdim=True)
        z_ir = self.mlp(stat(x)); z_vi = self.mlp(stat(v))
        f_ir = x * torch.sigmoid(z_vi)
        f_vi = v * torch.sigmoid(z_ir)
        # ---- S2: asymmetric physical-semantic conditioning ----
        heat = ctx['heat']                                    # (B,1,H,W) heat source map
        # IR heat semantics condition the VI features (IR -> VI)
        f_vi = f_vi * (1.0 + 0.5 * torch.tanh(self.cond_heat(torch.cat([f_vi, heat], dim=1))))
        # VI edges condition the IR features (VI -> IR)
        f_ir = f_ir * (1.0 + 0.5 * torch.tanh(self.cond_edge(ctx['edge'])))
        # ---- S3: budget x reliability gated fusion ----
        g = self.gate(torch.cat([f_ir, f_vi], dim=1))
        self.last_gate = float(g.mean().detach())
        scale = (ctx['r'] * ctx['b']) if ctx.get('gate', False) else 1.0
        fused = g * f_ir + (1.0 - g) * f_vi
        return x + scale * self.out(fused)


class CrossModalFusionSimple(nn.Module):
    """CAFNet-style gated cross-modal fusion (literature-exact control baseline, no budget/thermal conditioning)."""

    def __init__(self, c, hid=None):
        super().__init__()
        hid = hid or max(c // 4, 8)
        self.mlp = nn.Sequential(nn.Conv2d(c, hid, 1), nn.PReLU(hid), nn.Conv2d(hid, c, 1))
        self.gate = nn.Sequential(nn.Conv2d(2 * c, c, 1), nn.PReLU(c), nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.out = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # fair start = frozen baseline

    def forward(self, x, ctx):
        v = ctx['vi_feat']
        def stat(t):
            return t.mean(dim=(2, 3), keepdim=True) + t.amax(dim=(2, 3), keepdim=True)
        z_ir = self.mlp(stat(x)); z_vi = self.mlp(stat(v))
        f_ir = x * torch.sigmoid(z_vi)
        f_vi = v * torch.sigmoid(z_ir)
        g = self.gate(torch.cat([f_ir, f_vi], dim=1))
        self.last_gate = float(g.mean().detach())
        return x + self.out(g * f_ir + (1.0 - g) * f_vi)


class HFLS(nn.Module):
    """HFLS (ours): High-Frequency Latent-space Distillation Sampling.

    1) HF split (zero-param): x_hf = x - avgpool3x3(x)
    2) HF latent space: compress x_hf to a low-dim latent z
    3) Latent distillation (RFDN-style progressive channel-split 1x1 distillation inside the latent)
    4) Content-adaptive sub-pixel SAMPLING: predict per-pixel weights over the up*up sub-positions,
       applied to a learnable sub-pixel projection before PixelShuffle (replaces the fixed PixelShuffle mapping)
    Output projection is zero-initialised -> the model starts exactly at its baseline.
    """

    def __init__(self, c, upscale=4, latent=16, out_nc=3):
        super().__init__()
        self.up = upscale
        self.lat = nn.Conv2d(c, latent, 1)
        self.d1 = nn.Conv2d(latent, latent, 1); self.r1 = nn.Conv2d(latent, latent, 3, padding=1)
        self.d2 = nn.Conv2d(latent, latent, 1); self.r2 = nn.Conv2d(latent, latent, 3, padding=1)
        self.fuse = nn.Conv2d(latent * 3, latent, 1)
        self.proj = nn.Conv2d(latent, c * upscale * upscale, 1)      # learnable sub-pixel projection
        self.wgate = nn.Conv2d(latent, upscale * upscale, 1)          # per-pixel sub-pixel sampling weights
        self.out = nn.Conv2d(c, out_nc, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.act = nn.PReLU(latent)

    def forward(self, x):
        x_hf = x - F.avg_pool2d(x, 3, stride=1, padding=1)
        z = self.lat(x_hf)
        d1 = self.act(self.d1(z)); r1 = self.act(self.r1(z) + z)
        d2 = self.act(self.d2(r1)); r2 = self.act(self.r2(r1) + r1)
        z2 = self.act(self.fuse(torch.cat([d1, d2, r2], dim=1)))
        B, C, H, W = x.shape
        proj = self.proj(z2).view(B, C, self.up * self.up, H, W)       # (B,C,up^2,H,W)
        w = torch.softmax(self.wgate(z2), dim=1).unsqueeze(1)          # (B,1,up^2,H,W) content-adaptive sampling
        y = (proj * w).view(B, C * self.up * self.up, H, W)
        y = F.pixel_shuffle(y, self.up)
        return self.out(y)


class HFLS2(nn.Module):
    """HFLS-v2 (ours): HF latent + FREQUENCY-SPARSE distillation + LTE-style Fourier SAMPLING + latent reconstruction.

    Literature grounding:
      - LTE (Local Texture Estimator, CVPR'22): dominant-frequency (2D Fourier) estimation reconstructs HF details.
      - Frequency-Sparse Attention (2026): keep only the most informative frequency components (sparsity).
      - LUA / latent-diffusion implicit decoders (ECCV'26 / CVPR'24): upsampling in latent space via implicit sampling.
      - Ours: self-supervised LATENT INFORMATION RECONSTRUCTION (decode latent -> LR HF residual) to force HF encoding.

    Zero-init output projection => model starts exactly at its baseline.
    """

    def __init__(self, c, upscale=4, latent=16, K=6, out_nc=3, keep_ratio=0.5):
        super().__init__()
        self.up = upscale
        self.K = K
        self.lat = nn.Conv2d(c, latent, 1)
        # frequency-sparse distillation (RFDN FDC + top-k channel sparsity in the latent)
        self.d1 = nn.Conv2d(latent, latent, 1); self.r1 = nn.Conv2d(latent, latent, 3, padding=1)
        self.d2 = nn.Conv2d(latent, latent, 1); self.r2 = nn.Conv2d(latent, latent, 3, padding=1)
        self.fuse = nn.Conv2d(latent * 3, latent, 1)
        self.score = nn.Conv2d(latent, latent, 1)                     # sparsity scores
        self.keep = max(1, int(latent * keep_ratio))
        # LTE-style Fourier sampling: per-pixel K components (amplitude) + K dominant frequencies (fx,fy)
        self.amp = nn.Conv2d(latent, K, 1)
        self.freq = nn.Conv2d(latent, K * 2, 1)
        # sub-pixel synthesis: K*(up^2) -> c
        self.mix = nn.Conv2d(K * upscale * upscale, c * upscale * upscale, 1)
        # latent information reconstruction (aux): latent -> LR HF residual (c channels)
        self.rec = nn.Conv2d(latent, c, 1)
        # main output
        self.out = nn.Conv2d(c, out_nc, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.act = nn.PReLU(latent)

    def forward(self, x):
        B, C, H, W = x.shape
        up = self.up
        x_hf = x - F.avg_pool2d(x, 3, stride=1, padding=1)
        z = self.lat(x_hf)
        d1 = self.act(self.d1(z)); r1 = self.act(self.r1(z) + z)
        d2 = self.act(self.d2(r1)); r2 = self.act(self.r2(r1) + r1)
        z2 = self.act(self.fuse(torch.cat([d1, d2, r2], dim=1)))
        # --- frequency-sparse selection (top-k latent channels) ---
        sc = self.score(z2)
        if self.keep < sc.size(1):
            thr = sc.topk(self.keep, dim=1).values.min(dim=1, keepdim=True).values
            mask = (sc >= thr).float()
            z2 = z2 * mask
        # --- LTE-style Fourier sampling over the up^2 sub-pixel grid ---
        amp = self.amp(z2)                                            # (B,K,H,W)
        fxy = torch.tanh(self.freq(z2)) * 0.5                          # (B,2K,H,W) in (-0.5,0.5)
        fx, fy = fxy[:, :self.K], fxy[:, self.K:]
        offs = torch.arange(up, device=x.device, dtype=x.dtype)
        oy, ox = torch.meshgrid(offs, offs, indexing='ij')
        oy = (oy.reshape(-1) / up)[None, None, :, None, None]          # (1,1,up^2,1,1)
        ox = (ox.reshape(-1) / up)[None, None, :, None, None]
        ang = 2 * 3.141592653589793 * (fx.unsqueeze(2) * ox + fy.unsqueeze(2) * oy)   # (B,K,up^2,H,W)
        basis = torch.cos(ang)
        comp = (amp.unsqueeze(2) * basis)                              # (B,K,up^2,H,W)
        comp = comp.reshape(B, self.K * up * up, H, W)
        y = self.mix(comp)                                             # (B, c*up^2, H, W)
        y = F.pixel_shuffle(y, up)                                     # (B,c,upH,upW)
        self.aux_loss = F.l1_loss(self.rec(z2), x_hf)                  # latent information reconstruction
        return self.out(y)


class HFLS3(nn.Module):
    """HFLS-v3 (ours, RK3588-friendly): HF lifting split + HF latent distillation + LEARNABLE POLYPHASE upsampling.

    Theory:
      - Lifting scheme (predict-update): zero-param, perfect-reconstruction HF/LF split (adds/subs only).
      - Multirate / polyphase filter banks: any LINEAR upsampler is y = PixelShuffle(K * x); the fixed
        PixelShuffle is the impulse special case. Making K learnable == the optimal linear-reconstruction
        family (Wiener solution lies inside it) -> better HF reconstruction at ~c^2*up^2 params.
      - All ops are RKNN-supported: Conv2d / PReLU / Add / Concat / PixelShuffle (no softmax/top-k/cos-sin).
      - Output projection zero-initialised -> starts EXACTLY at the IR baseline.
    """

    def __init__(self, c, upscale=4, latent=16, out_nc=3):
        super().__init__()
        self.up = upscale
        # HF latent
        self.lat = nn.Conv2d(c, latent, 1)
        # latent distillation (RFDN-style FDC)
        self.d1 = nn.Conv2d(latent, latent, 1); self.r1 = nn.Conv2d(latent, latent, 3, padding=1)
        self.d2 = nn.Conv2d(latent, latent, 1); self.r2 = nn.Conv2d(latent, latent, 3, padding=1)
        self.fuse = nn.Conv2d(latent * 3, latent, 1)
        # learnable polyphase (sub-pixel) filter bank -> PixelShuffle
        self.poly = nn.Conv2d(latent, c * upscale * upscale, 1)
        self.out = nn.Conv2d(c, out_nc, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.act = nn.PReLU(latent)

    def forward(self, x):
        x_hf = x - F.avg_pool2d(x, 3, stride=1, padding=1)      # lifting-style predictor/update (zero-param)
        z = self.lat(x_hf)
        d1 = self.act(self.d1(z)); r1 = self.act(self.r1(z) + z)
        d2 = self.act(self.d2(r1)); r2 = self.act(self.r2(r1) + r1)
        z2 = self.act(self.fuse(torch.cat([d1, d2, r2], dim=1)))
        y = F.pixel_shuffle(self.poly(z2), self.up)             # learnable polyphase upsampling
        return self.out(y)


class HFLS4(nn.Module):
    """HFLS-v4b (ours, FIXED): PR-structured latent + LSQ rate-constrained quantisation + TRUE Lipschitz<=1 decoder.

    Fixes vs the failed v4a (val collapsed to 14 dB):
      1) LSQ-style learnable per-channel quantisation step -> the latent is no longer rounded to zero;
      2) output projection has NO bias and zero-init weight -> the branch can never add a constant offset
         (constant-offset corruption was the catastrophic failure mode);
      3) TRUE Lipschitz bound: the latent->HR decoder weight is explicitly rescaled by its spectral norm
         (1 power iteration, cached buffer) so ||W||_2 <= 1  => non-expansive HF decoder (provable bound);
      4) auxiliary loss = bounded quantisation DISTORTION (LSQ objective), not a rate proxy that drives z->0.

    All ops stay RKNN-friendly (Conv2d / DWConv / PReLU / Add / PixelShuffle / round-clamp).
    Output zero-init => starts exactly at the IR baseline.
    """

    def __init__(self, c, upscale=4, latent=16, out_nc=3, bits=4, lam_rate=0.001, use_quant=True, use_lip=True):
        super().__init__()
        self.up = upscale
        self.bits = bits
        self.lam_rate = lam_rate
        self.use_quant = use_quant
        self.use_lip = use_lip
        self.lat1 = nn.Conv2d(c, latent, 1)
        self.lat2 = nn.Conv2d(latent, latent, 1)
        self.d1 = nn.Conv2d(latent, latent, 1)
        self.dw1 = nn.Conv2d(latent, latent, 3, padding=1, groups=latent)
        self.fuse = nn.Conv2d(2 * latent, latent, 1)
        # decoder WITHOUT bias (cannot emit a constant) + zero-init
        self.poly = nn.Conv2d(latent, c * upscale * upscale, 1, bias=False)
        self.out = nn.Conv2d(c, out_nc, 1, bias=False)
        nn.init.zeros_(self.out.weight)
        self.act = nn.PReLU(latent)
        # LSQ learnable step (per latent channel) + Lipschitz power-iteration buffer
        self.qstep = nn.Parameter(torch.ones(1, latent, 1, 1))
        self.register_buffer('u', torch.randn(1, c * upscale * upscale).div_((c * upscale * upscale) ** 0.5))

    def _bounded_weight(self):
        """Rescale the decoder weight so its spectral norm is <= 1 (provable non-expansive map).

        The spectral-norm factor is estimated under no_grad (3 power iterations, cached buffer) and only
        used as a scalar rescaling; gradients flow through W itself.
        """
        W = self.poly.weight.view(self.poly.out_channels, -1)
        with torch.no_grad():
            Wd = W.detach()
            u = self.u.view(-1)
            v = torch.nn.functional.normalize(Wd.t() @ u, dim=0)
            for _ in range(2):
                u = torch.nn.functional.normalize(Wd @ v, dim=0)
                v = torch.nn.functional.normalize(Wd.t() @ u, dim=0)
            sigma = (u @ (Wd @ v)).clamp(min=1e-6)
            self.u.copy_(u.view_as(self.u))
        Wb = W / (sigma * 1.2).clamp(min=1.0)
        return Wb.view_as(self.poly.weight)

    def forward(self, x):
        d = x - torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1)   # lifting/PR detail band
        z = self.lat2(self.lat1(d))
        # LSQ quantisation: step s, quantise in the scaled domain, no collapse to zero
        s = self.qstep.abs() + 1e-4
        q = 2 ** (self.bits - 1) - 1
        zs = z / s
        zq = zs + (torch.round(zs.clamp(-q, q)) - zs).detach()
        zq = zq * s
        self.aux_loss = self.lam_rate * (zq - z).pow(2).mean()                # bounded distortion
        a1 = self.act(self.d1(zq)); a2 = self.act(self.dw1(a1))
        z2 = self.act(self.fuse(torch.cat([a1, a2], dim=1)))
        w = self._bounded_weight() if self.use_lip else self.poly.weight.view_as(self.poly.weight)
        y = torch.nn.functional.pixel_shuffle(torch.nn.functional.conv2d(z2, w), self.up)
        return self.out(y)




class StructureGatedHFFusion(nn.Module):
    """SG-HF: Structure-Gated High-Frequency cross-modal fusion (TSCF-v2, ours).

    Physical motivation: in thermal imaging the LOW frequency carries radiometric (temperature) content
    and must NOT be altered by the visible spectrum; only the HIGH-frequency band carries structure that
    the visible image can legitimately supply.  Therefore:
        1) LF/HF split with a zero-parameter 3x3 mean (lifting-style predictor/update);
        2) ASYMMETRIC transfer: only VI's HF band is injected into IR's HF band;
        3) the transfer is gated by a *directional gradient agreement* map
           a = (1 + <grad IR, grad VI> / (|grad IR||grad VI|)) / 2  in [0,1]
           which is also used as a SELF-SUPERVISED target for the gate (zero inference cost);
        4) zero-initialised projection => the model starts EXACTLY at the IR baseline.
    Ops: avg_pool2d / fixed Sobel conv / 1x1 conv / PReLU / Sigmoid / mul-add  (RKNN whitelist).
    """

    def __init__(self, c):
        super().__init__()
        h = max(c // 4, 8)
        self.proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self.gate = nn.Sequential(nn.Conv2d(3, h, 1), nn.PReLU(h), nn.Conv2d(h, 1, 1), nn.Sigmoid())
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.0
        sy = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 4.0
        self.register_buffer('kx', sx.view(1, 1, 3, 3))
        self.register_buffer('ky', sy.view(1, 1, 3, 3))

    def _grad(self, x):
        g = x.mean(1, keepdim=True)                      # channel-mean map (cheap, zero-param)
        gx = torch.nn.functional.conv2d(g, self.kx, padding=1)
        gy = torch.nn.functional.conv2d(g, self.ky, padding=1)
        return gx, gy

    def forward(self, f_ir, f_vi):
        if isinstance(f_vi, dict):                 # called as (feat, ctx) from FDRNet
            f_vi = f_vi['vi_feat']
        if f_vi.shape[-2:] != f_ir.shape[-2:]:
            f_vi = torch.nn.functional.interpolate(f_vi, size=f_ir.shape[-2:], mode='bilinear', align_corners=False)
        ir_lf = torch.nn.functional.avg_pool2d(f_ir, 3, 1, 1)
        vi_lf = torch.nn.functional.avg_pool2d(f_vi, 3, 1, 1)
        ir_hf, vi_hf = f_ir - ir_lf, f_vi - vi_lf
        gxi, gyi = self._grad(f_ir); gxv, gyv = self._grad(f_vi)
        ni = (gxi.pow(2) + gyi.pow(2) + 1e-6).sqrt()          # ★ eps 放 sqrt 内：避免 0 处导数无穷
        nv = (gxv.pow(2) + gyv.pow(2) + 1e-6).sqrt()          # ★ 同上
        denom = (ni * nv).clamp(min=1e-3)          # ★ 相对下限：平坦区乘积可达 1e-12 -> 必须夹住
        agree = ((gxi * gxv + gyi * gyv) / denom + 1.0) * 0.5
        agree = agree.clamp(0.0, 1.0)          # ★ 数值越界也夹紧，防止 aux 目标失真
        g = self.gate(torch.cat([agree, ni / (ni.mean() + 1e-6), nv / (nv.mean() + 1e-6)], dim=1))
        self.last_gate = float(g.mean().detach())
        # self-supervised gate target (training only; no inference cost)
        self.aux_loss = torch.nn.functional.l1_loss(g, agree.detach().clamp(0, 1))
        return ir_lf + ir_hf + g * self.proj(vi_hf)
