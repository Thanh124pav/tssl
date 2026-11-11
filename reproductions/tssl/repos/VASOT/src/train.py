import argparse
from copy import deepcopy
from statistics import mean
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os

from arc_reparam import arc_u, resample_uniform_u, inverse_warp_predictions
from bezier import smooth_X_bezier
from bspline import smooth_X as bspline_smooth_X
from gauss_smooth import gaussian_smooth

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from video_dataset import VideoDataset
import alignment_ot
import segmentation_ot
from action_bridge import action_alignment_ce, action_similarity
from visual_evidence import visual_affinity_loss, consistent_frame_pairs, refine_action_pseudolabels
from geometry_consensus import geometry_loss, soft_consensus, spectot_temporal_affinity_loss
from utils import *
from metrics import ClusteringMetrics, indep_eval_metrics

num_eps = 1e-11


MRT = {'decenter': 0.0, 'ot_noise': 0.0, 'hub': 'template', 'local': 0.0, 'local_warmup': 5, 'local_raw': False, 'target_power': 1.0}  # set from CLI before the model is built


class WindowedMLP(nn.Module):
    """Frame encoder that reads a temporal window instead of a single frame.

    A12 measured that within-action phase is decodable from one raw frame at
    R2 0.103 and from a +-4 frame window at 0.314, so the single-frame input is
    the binding constraint on phase information, not the projection or the
    latent space. The optimum window (9 frames) matches the 11-frame median
    FSeval action, i.e. the signal sits at action scale, which a per-frame
    encoder can never reach.

    This wraps the existing MLP so every call site keeps working unchanged,
    including the diagnostic scripts that call `model.mlp(raw)` directly. With
    window = 0 it is exactly the old module.
    """

    def __init__(self, mlp, window=0, feature_dim=None):
        super().__init__()
        self.mlp = mlp
        self.window = int(window)
        self.feature_dim = feature_dim

    def forward(self, x):
        if self.window <= 0:
            return self.mlp(x)
        flat = (x.dim() == 2)
        if flat:
            # call sites reshape (B, T, D) to (B*T, D); the project runs
            # batch_size 1, which main() enforces when a window is requested,
            # so the flattened axis is a single video's time axis
            x = x.unsqueeze(0)
        B, T, D_in = x.shape
        w = self.window
        offs = torch.arange(-w, w + 1, device=x.device)
        idx = (torch.arange(T, device=x.device).unsqueeze(1) + offs.unsqueeze(0)).clamp(0, T - 1)
        xw = x[:, idx.reshape(-1), :].reshape(B, T, (2 * w + 1) * D_in)
        out = self.mlp(xw)
        return out.squeeze(0) if flat else out


class VideoSSL(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-4, layer_sizes=[64, 128, 40], n_clusters=20, alpha_train=0.3, alpha_eval=0.3,
                 n_ot_train=[50, 1], n_ot_eval=[50, 1], step_size=None, train_eps=0.06, eval_eps=0.01, ub_frames=False, ub_actions=False,
                 lambda_frames_train=0.05, lambda_actions_train=0.05, lambda_frames_eval=0.05, lambda_actions_eval=0.01,
                 temp=0.1, radius_gw=0.04,learn_clusters=True, n_frames=256, rho=0.1, beta=1, exclude_cls=None, visualize=False, activity=None,
                 rho_seg=None, rho_align=None, fix_virtual_column=False,
                 input_window=0,
                 template_slots=0, template_rho=None, template_warmup_epochs=5,
                 template_eps=None, template_blend=1.0, template_soft=False,
                 align_target_blur=0, template_ema=0.0,
                 template_frozen=False,
                 ray_geometry=False, ray_mode='sphere', ray_videos=3,
                 ray_weight=0.1, ray_radius_weight=1.0, ray_group_weight=0.1,
                 ray_shell_cost_weight=0.0, ray_warmup_epochs=10, ray_window=15,
                 ray_stride=4, ray_cycle_tol=4, ray_centre_momentum=0.9,
                 ray_centre_margin=0.35, ray_update_centres=True,
                 action_bridge_cost_weight=0.0, action_bridge_loss_weight=0.0,
                 action_bridge_temp=0.1, action_bridge_warmup_epochs=5,
                 visual_affinity_weight=0.0, visual_affinity_stride=4,
                 visual_affinity_teacher_temp=0.07,
                 match_pseudo_weight=0.0, match_raw_topk=8,
                 match_cycle_tolerance=4, match_warmup_epochs=5,
                 geometry_weight=0.0, geometry_repulsion_weight=0.0,
                 spectot_tar_weight=0.0,
                 soft_consensus_weight=0.0, soft_consensus_blend=0.0,
                 proposal_videos=4,
                 arc_reparam=False, arc_warmup_epochs=5,
                 bspline=False, bspline_tol=0.05, bspline_kmax=0, bspline_warmup_epochs=5,
                 bspline_space='X', bspline_blend_max=1.0, bspline_blend_rampup=0,
                 latent_dim=None,
                 gauss_smooth=False, gauss_sigma=2.0, gauss_blend_max=1.0, gauss_warmup_epochs=5,
                 bezier=False, bezier_nseg=4, bezier_tol=0.05, bezier_maxseg=32,
                 bezier_warmup_epochs=5, bezier_space='X', bezier_blend_max=1.0, bezier_blend_rampup=0):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_clusters = n_clusters
        self.learn_clusters = learn_clusters
        self.layer_sizes = layer_sizes
        self.exclude_cls = exclude_cls
        self.visualize = visualize

        self.alpha_train = alpha_train
        self.alpha_eval = alpha_eval
        self.n_ot_train = n_ot_train
        self.n_ot_eval = n_ot_eval
        self.step_size = step_size
        self.train_eps = train_eps
        self.eval_eps = eval_eps
        self.radius_gw = radius_gw
        self.ub_frames = ub_frames
        self.ub_actions = ub_actions
        self.lambda_frames_train = lambda_frames_train
        self.lambda_actions_train = lambda_actions_train
        self.lambda_frames_eval = lambda_frames_eval
        self.lambda_actions_eval = lambda_actions_eval

        self.temp = temp
        # Margin-restoring transport (MRT): see _view/_teacher below.
        self.decenter = float(MRT['decenter'])
        self.ot_noise = float(MRT['ot_noise'])
        self.register_buffer('embed_mean', torch.zeros(layer_sizes[-1]))
        self.register_buffer('embed_mean_ready', torch.zeros((), dtype=torch.bool))
        self.register_buffer('mrt_cfg', torch.tensor([self.decenter, self.ot_noise]))
        self.n_frames = n_frames
        self.rho = rho
        self.rho_seg = rho if rho_seg is None else rho_seg
        self.rho_align = rho if rho_align is None else rho_align
        # see the virtual-frame block in training_step; OFF reproduces the
        # published behaviour bit-for-bit
        self.fix_virtual_column = fix_virtual_column
        # A09 ray/sphere geometry objectives. Centres and radii are BUFFERS, not
        # parameters: G00 measured that gradient descent does not move this kind of
        # parameter (prototypes stay at cosine 0.872 from their own init), so they
        # are re-estimated in closed form instead.
        self.ray_geometry = ray_geometry
        self.ray_mode = ray_mode
        self.ray_videos = ray_videos
        self.ray_weight = ray_weight
        self.ray_radius_weight = ray_radius_weight
        self.ray_group_weight = ray_group_weight
        self.ray_shell_cost_weight = ray_shell_cost_weight
        self.ray_warmup_epochs = ray_warmup_epochs
        self.ray_window = ray_window
        self.ray_stride = ray_stride
        self.ray_cycle_tol = ray_cycle_tol
        self.ray_centre_momentum = ray_centre_momentum
        self.ray_centre_margin = ray_centre_margin
        self.ray_update_centres = ray_update_centres
        self.action_bridge_cost_weight = action_bridge_cost_weight
        self.action_bridge_loss_weight = action_bridge_loss_weight
        self.action_bridge_temp = action_bridge_temp
        self.action_bridge_warmup_epochs = action_bridge_warmup_epochs
        self.visual_affinity_weight = float(visual_affinity_weight)
        self.visual_affinity_stride = int(visual_affinity_stride)
        self.visual_affinity_teacher_temp = float(visual_affinity_teacher_temp)
        self.match_pseudo_weight = float(match_pseudo_weight)
        self.match_raw_topk = int(match_raw_topk)
        self.match_cycle_tolerance = int(match_cycle_tolerance)
        self.match_warmup_epochs = int(match_warmup_epochs)
        self.geometry_weight = float(geometry_weight)
        self.geometry_repulsion_weight = float(geometry_repulsion_weight)
        self.spectot_tar_weight = float(spectot_tar_weight)
        self.soft_consensus_weight = float(soft_consensus_weight)
        self.soft_consensus_blend = float(soft_consensus_blend)
        self.proposal_videos = int(proposal_videos)
        self.beta = beta
        self.arc_reparam = arc_reparam
        self.arc_warmup_epochs = arc_warmup_epochs
        self.bspline = bspline
        self.bspline_tol = bspline_tol
        self.bspline_kmax = bspline_kmax if bspline_kmax > 0 else None
        self.bspline_warmup_epochs = bspline_warmup_epochs
        self.bspline_space = bspline_space   # 'X' (frame embedding), 'logits' (action-space via clusters), or 'latent' (dedicated learnable bottleneck)
        self.latent_dim = 16 if latent_dim is None else latent_dim
        self.bspline_blend_max = bspline_blend_max
        self.bspline_blend_rampup = bspline_blend_rampup
        self.gauss_smooth = gauss_smooth
        self.gauss_sigma = gauss_sigma
        self.gauss_blend_max = gauss_blend_max
        self.gauss_warmup_epochs = gauss_warmup_epochs
        self.bezier = bezier
        self.bezier_nseg = bezier_nseg
        self.bezier_tol = bezier_tol
        self.bezier_maxseg = bezier_maxseg
        self.bezier_warmup_epochs = bezier_warmup_epochs
        self.bezier_space = bezier_space
        self.bezier_blend_max = bezier_blend_max
        self.bezier_blend_rampup = bezier_blend_rampup
        # initialize MLP
        self.input_window = int(input_window)
        # A14: alignment factored through an ordered template of L slots, so
        # P(j|i) = sum_l P(l|i) P(j|l). Composing through a third video then
        # gives the SAME plan as the direct one, making cycle-consistency
        # structural rather than penalised. A02 measured the defect this fixes
        # (X->Y->Z disagrees with X->Z by 18.5 frames, 39% within 4).
        # L >> K on purpose: E13 did this with the K=12 action posterior and
        # barely helped, which A05's granularity result explains.
        self.template_slots = int(template_slots)
        self.template_rho = template_rho
        self.template_warmup_epochs = int(template_warmup_epochs)
        self.template_eps = template_eps
        self.template_blend = float(template_blend)
        self.template_soft = bool(template_soft)
        # A16: the parsimonious form of A14/A15. If a never-learned random
        # template works as well as a learned one (A15 frozen), the template is
        # not doing anything and the active ingredient is the block structure of
        # the target. Blurring the existing pairwise pseudo-label reproduces that
        # structure with no slots, no parameters and three lines.
        self.align_target_blur = int(align_target_blur)
        # A18: the template parameter never receives a gradient, because the plan
        # that uses it is built inside torch.no_grad (see section 8 of the A14
        # report). An EMA of each slot toward the frames assigned to it is how
        # the repo learns its other codebook-like objects, and is what A09 used
        # for the geometric centres. 0 keeps the random fixed template.
        self.template_ema = float(template_ema)
        self.template_frozen = bool(template_frozen)
        # a window of 2w+1 frames is concatenated, so only the input layer widens
        eff_sizes = list(layer_sizes)
        if self.input_window > 0:
            eff_sizes[0] = layer_sizes[0] * (2 * self.input_window + 1)
        layers = [nn.Sequential(nn.Linear(sz, sz1), nn.ReLU()) for sz, sz1 in zip(eff_sizes[:-2], eff_sizes[1:-1])]
        layers += [nn.Linear(eff_sizes[-2], eff_sizes[-1])]
        self.mlp = WindowedMLP(nn.Sequential(*layers), self.input_window,
                               feature_dim=layer_sizes[0])

        # initialize cluster centers/codebook
        d = self.layer_sizes[-1]
        self.clusters = nn.parameter.Parameter(data=F.normalize(torch.randn(self.n_clusters, d), dim=-1),
                                               requires_grad=learn_clusters)
        if self.template_slots > 0:
            # frozen is the strong control: if a never-learned random template
            # works as well, the factoring is the active ingredient and the
            # template's content is not -- the same logic as G01's random centre
            self.template = nn.parameter.Parameter(
                data=F.normalize(torch.randn(self.template_slots, d), dim=-1),
                requires_grad=not self.template_frozen)
        # geometric centres / radii for the A09 objectives (see __init__ comment)
        self.register_buffer('geo_centres', torch.zeros(self.n_clusters, d))
        self.register_buffer('geo_radii', torch.ones(self.n_clusters))
        self.register_buffer('geo_ready', torch.zeros(1))

        # Dedicated learnable latent basis for bspline_space='latent' (E10b).
        # Shape (latent_dim, d), unit-norm rows at init.  Trains through the
        # downstream CE loss just like `clusters` does.
        # Older checkpoints have no latent_basis. New E10a/E10b checkpoints
        # record latent_dim in hyperparameters and must retain the parameter
        # even when their smoothing space is not 'latent'.
        if latent_dim is not None or bspline_space == 'latent':
            self.latent_basis = nn.parameter.Parameter(
                data=F.normalize(torch.randn(self.latent_dim, d), dim=-1),
                requires_grad=True,
            )

        #TODO: Need to replace with evaluation metrics for video alignment
        self.activity = activity
        # initialize evaluation metrics
        self.mof = ClusteringMetrics(metric='mof')
        self.f1 = ClusteringMetrics(metric='f1')
        self.miou = ClusteringMetrics(metric='miou')
        self.save_hyperparameters()
        self.test_cache = []

    def _maybe_warp(self, features, mask):
        """Returns (features_warped, u) if arc-reparam is active, else (features, None).
        Warm-up: skip warp until self.current_epoch >= arc_warmup_epochs during training.
        At eval/test, always apply if arc_reparam=True.
        """
        if self.arc_reparam and not (self.training and self.current_epoch < self.arc_warmup_epochs):
            u = arc_u(features, mask)
            features = resample_uniform_u(features, u)
        else:
            u = None
        # Smoothing families (B-spline / Gaussian) apply ONLY during training
        # as a representation-shaping regulariser.  At eval/test the already
        # trained encoder is used as-is — this avoids the full-video (N≈11k)
        # adaptive fit which both slows evaluation massively and oversmoothes
        # to collapse when the encoder was trained to depend on per-frame
        # discriminability.
        if self.training and self.bspline and self.current_epoch >= self.bspline_warmup_epochs:
            if self.bspline_space == 'X':
                smoothed = bspline_smooth_X(features, mask, u_target=None,
                                            tol=self.bspline_tol, k_max=self.bspline_kmax)
            elif self.bspline_space == 'logits':
                L = features @ self.clusters.T.unsqueeze(0)
                L_smoothed = bspline_smooth_X(L, mask, u_target=None,
                                              tol=self.bspline_tol, k_max=self.bspline_kmax)
                smoothed = L_smoothed @ self.clusters.unsqueeze(0)
            elif self.bspline_space == 'latent':
                # E10b: dedicated learnable bottleneck. (latent_dim,) subspace.
                Z = features @ self.latent_basis.T.unsqueeze(0)
                Z_smoothed = bspline_smooth_X(Z, mask, u_target=None,
                                              tol=self.bspline_tol, k_max=self.bspline_kmax)
                smoothed = Z_smoothed @ self.latent_basis.unsqueeze(0)
            else:
                raise ValueError(f"unknown bspline_space={self.bspline_space!r}")
            blend = self._ramp(self.bspline_blend_max, self.bspline_warmup_epochs, self.bspline_blend_rampup)
            features = F.normalize((1 - blend) * features + blend * smoothed, dim=-1)
        if self.training and self.gauss_smooth and self.current_epoch >= self.gauss_warmup_epochs:
            smoothed = gaussian_smooth(features, sigma=self.gauss_sigma)
            blend = self._ramp(self.gauss_blend_max, self.gauss_warmup_epochs, 0)
            features = F.normalize((1 - blend) * features + blend * smoothed, dim=-1)
        if self.training and self.bezier and self.current_epoch >= self.bezier_warmup_epochs:
            if self.bezier_space == 'X':
                smoothed = smooth_X_bezier(features, mask, n_segments=self.bezier_nseg,
                                           tol=self.bezier_tol, max_segments=self.bezier_maxseg)
            else:
                L = features @ self.clusters.T.unsqueeze(0)
                L_smoothed = smooth_X_bezier(L, mask, n_segments=self.bezier_nseg,
                                             tol=self.bezier_tol, max_segments=self.bezier_maxseg)
                smoothed = L_smoothed @ self.clusters.unsqueeze(0)
            blend = self._ramp(self.bezier_blend_max, self.bezier_warmup_epochs, self.bezier_blend_rampup)
            features = F.normalize((1 - blend) * features + blend * smoothed, dim=-1)
        return features, u

    def _view(self, z):
        """Decentred view used for every similarity/cost: remove the shared
        (concentrated) direction of the embedding cloud, then renormalise."""
        if self.decenter <= 0 or not bool(self.embed_mean_ready):
            return z
        return F.normalize(z - self.decenter * self.embed_mean, dim=-1)

    def _teacher(self, z):
        """View used for OT pseudo-label costs: decentred, plus an isotropic
        Gaussian sample so one pseudo-label set is not locked in early."""
        if self.training and self.ot_noise > 0:
            z = F.normalize(z + self.ot_noise * torch.randn_like(z) / z.shape[-1] ** 0.5, dim=-1)
        return self._view(z)

    @torch.no_grad()
    def _update_embed_mean(self, feats, masks):
        m = torch.cat([f[k] for f, k in zip(feats, masks)]).mean(0)
        if bool(self.embed_mean_ready):
            self.embed_mean.lerp_(m, 0.05)
        else:
            self.embed_mean.copy_(m); self.embed_mean_ready.fill_(True)

    def _ramp(self, target: float, warmup_epochs: int, rampup_epochs: int) -> float:
        """Linear ramp of the blend coefficient after warmup. Eval → full target."""
        if not self.training:
            return target
        e = max(0, self.current_epoch - warmup_epochs)
        if rampup_epochs <= 0:
            return target
        return min(1.0, e / float(rampup_epochs)) * target

    def on_load_checkpoint(self, checkpoint):
        """Keep every checkpoint written before the A09 buffers existed loadable.

        Adding `geo_centres` / `geo_radii` / `geo_ready` as persistent buffers
        made `load_from_checkpoint` fail on all earlier checkpoints with
        "Missing key(s) in state_dict". Every frozen baseline and every
        diagnostic script loads through that path, so the buffers are filled in
        with their constructor defaults when a checkpoint predates them.
        """
        sd = checkpoint.get('state_dict')
        if sd is None:
            return
        if 'mrt_cfg' in sd:  # MRT settings travel with the weights, not the CLI
            self.decenter, self.ot_noise = (float(v) for v in sd['mrt_cfg'])
        # WindowedMLP wraps the old nn.Sequential, which renamed every encoder
        # parameter from `mlp.X` to `mlp.mlp.X`. Remap so earlier checkpoints
        # still load; the window itself carries no parameters.
        stale = [k for k in list(sd) if k.startswith('mlp.') and not k.startswith('mlp.mlp.')]
        for k in stale:
            sd['mlp.mlp.' + k[len('mlp.'):]] = sd.pop(k)
        for name in ('geo_centres', 'geo_radii', 'geo_ready', 'embed_mean', 'embed_mean_ready', 'mrt_cfg'):
            if name not in sd and hasattr(self, name):
                sd[name] = getattr(self, name).clone()

    def training_step(self, batch, batch_idx):
        # NOTE: The comments below assume batchsize=1 so we dont talk about the batch element dimension
        # features_raw [A 2D Tensor of size=self.n_frames(default 256) x frame embedding dims, representing the input frame embeddings of the sampled frames from a video]
        # mask [A 1D array of Trues/Falses of size=self.n_frames(default 256)]
        # gt [A 1D tensor of action-ids of size=self.n_frames(default 256)]
        # fname [Video's filename]
        # n_subactions [No. of unique action-classes amongst the randomly sampled frames/gt]

        features_raw_X, mask_X, gt_X, fname_X, n_subactions_X = batch[0]
        features_raw_Y, mask_Y, gt_Y, fname_Y, n_subactions_Y = batch[1]

        # MLP's last layer size (D=40 for penn_action)
        D = self.layer_sizes[-1]

        ## Process the features of video X
        # B(batch size), T(no. of frames or timesteps) and _(feature dim/frame embedding size, which is 1024 in penn_action)
        B_X, T_X, _ = features_raw_X.shape
        # Reshape features_raw from 3D(B x T x _) to 2D(B*T x _) tensor, while keeping the feature dim intact
        # Pass it through the MLP: input layer size=_ and output layer size=D
        # Reshape the MLP output from (B*T x D) to (B x T x D)
        # Normalize features along the last dimension(-1), so each feature vector along D and across B and T has unit norm, for stable training
        features_X = F.normalize(
            self.mlp(features_raw_X.reshape(-1, features_raw_X.shape[-1])).reshape(B_X, T_X, D),
            dim=-1
        )
        features_X, _ = self._maybe_warp(features_X, mask_X)

        ## Process the features of video Y
        B_Y, T_Y, _ = features_raw_Y.shape
        features_Y = F.normalize(
            self.mlp(features_raw_Y.reshape(-1, features_raw_Y.shape[-1])).reshape(B_Y, T_Y, D),
            dim=-1
        )
        features_Y, _ = self._maybe_warp(features_Y, mask_Y)

        self._update_embed_mean([features_X, features_Y], [mask_X, mask_Y])
        self.log('embed_mean_norm', self.embed_mean.norm())
        loss_visual = features_X.new_zeros(())
        if self.visual_affinity_weight > 0:
            loss_visual = visual_affinity_loss(
                features_raw_X, features_raw_Y, features_X, features_Y,
                mask_X, mask_Y, stride=self.visual_affinity_stride,
                teacher_temp=self.visual_affinity_teacher_temp,
            )
        self.log('train_loss_visual_affinity', loss_visual)

        with torch.no_grad():
            self.clusters.data = F.normalize(self.clusters.data, dim=-1)
            # Keep latent_basis rows unit-norm (parallels the clusters normalization).
            if hasattr(self, 'latent_basis'):
                self.latent_basis.data = F.normalize(self.latent_basis.data, dim=-1)

        ##FIND SEGMENTATION LOSS FOR X
        codes_segmentation_X = torch.exp(self._view(features_X) @ self._view(self.clusters).T[None, ...] / self.temp)
        codes_segmentation_X = codes_segmentation_X / codes_segmentation_X.sum(dim=-1, keepdim=True)

        with torch.no_grad():  # pseudo-labels from OT
            temp_prior_segmentation_X = segmentation_ot.temporal_prior(T_X, self.n_clusters, self.rho_seg,
                                                                         features_X.device)
            cost_matrix_segmentation_X = 1. - self._teacher(features_X) @ self._view(self.clusters).T.unsqueeze(0)
            cost_matrix_segmentation_X += temp_prior_segmentation_X
            # print(cost_matrix_segmentation_X.shape)
            #change ub_action to False (reminder)
            opt_codes_segmentation_X, _ = segmentation_ot.segment_ot(cost_matrix_segmentation_X, mask_X,
                                                                         eps=self.train_eps, alpha=self.alpha_train,
                                                                         radius=self.radius_gw,
                                                                         ub_frames=False, ub_actions=True,
                                                                         lambda_frames=self.lambda_frames_train,
                                                                         lambda_actions=self.lambda_actions_train,
                                                                         n_iters=self.n_ot_train,
                                                                         step_size=self.step_size)

        ##FIND SEGMENTATION LOSS FOR Y
        codes_segmentation_Y = torch.exp(self._view(features_Y) @ self._view(self.clusters).T[None, ...] / self.temp)
        codes_segmentation_Y = codes_segmentation_Y / codes_segmentation_Y.sum(dim=-1, keepdim=True)

        with torch.no_grad():  # pseudo-labels from OT
            temp_prior_segmentation_Y = segmentation_ot.temporal_prior(T_Y, self.n_clusters, self.rho_seg,
                                                                         features_Y.device)
            cost_matrix_segmentation_Y = 1. - self._teacher(features_Y) @ self._view(self.clusters).T.unsqueeze(0)
            cost_matrix_segmentation_Y += temp_prior_segmentation_Y
            opt_codes_segmentation_Y, _ = segmentation_ot.segment_ot(cost_matrix_segmentation_Y, mask_Y,
                                                                         eps=self.train_eps, alpha=self.alpha_train,
                                                                         radius=self.radius_gw,
                                                                         ub_frames=False, ub_actions=True,
                                                                         lambda_frames=self.lambda_frames_train,
                                                                         lambda_actions=self.lambda_actions_train,
                                                                         n_iters=self.n_ot_train,
                                                                         step_size=self.step_size)

        loss_geometry_new = features_X.new_zeros(())
        loss_soft_consensus = features_X.new_zeros(())
        if self.geometry_weight > 0 or self.geometry_repulsion_weight > 0 or self.spectot_tar_weight > 0 or self.soft_consensus_weight > 0 or self.soft_consensus_blend > 0:
            proposal_views = [features_X, features_Y]
            proposal_masks = [mask_X, mask_Y]
            proposal_targets = [opt_codes_segmentation_X, opt_codes_segmentation_Y]
            for item in batch[2:self.proposal_videos]:
                raw_extra, mask_extra = item[:2]
                B_extra, T_extra, _ = raw_extra.shape
                z_extra = F.normalize(
                    self.mlp(raw_extra.reshape(-1, raw_extra.shape[-1])).reshape(B_extra, T_extra, D), dim=-1)
                z_extra, _ = self._maybe_warp(z_extra, mask_extra)
                proposal_views.append(z_extra)
                proposal_masks.append(mask_extra)
                with torch.no_grad():
                    cost_extra = 1. - self._teacher(z_extra) @ self._view(self.clusters).T.unsqueeze(0)
                    cost_extra += segmentation_ot.temporal_prior(
                        T_extra, self.n_clusters, self.rho_seg, z_extra.device)
                    q_extra, _ = segmentation_ot.segment_ot(
                        cost_extra, mask_extra, eps=self.train_eps, alpha=self.alpha_train,
                        radius=self.radius_gw, ub_frames=False, ub_actions=True,
                        lambda_frames=self.lambda_frames_train,
                        lambda_actions=self.lambda_actions_train,
                        n_iters=self.n_ot_train, step_size=self.step_size)
                    proposal_targets.append(q_extra)
            if self.geometry_weight > 0 or self.geometry_repulsion_weight > 0:
                var, cov, repel = geometry_loss(proposal_views, proposal_masks, proposal_targets)
                loss_geometry_new = self.geometry_weight * (var + cov) + self.geometry_repulsion_weight * repel
                self.log('geometry_variance', var)
                self.log('geometry_covariance', cov)
                self.log('geometry_far_repulsion', repel)
            if self.spectot_tar_weight > 0:
                tar = spectot_temporal_affinity_loss(proposal_views, proposal_masks)
                loss_geometry_new = loss_geometry_new + self.spectot_tar_weight * tar
                self.log('spectot_tar_loss', tar)
            if self.soft_consensus_weight > 0 or self.soft_consensus_blend > 0:
                loss_soft_consensus, refined_targets = soft_consensus(
                    proposal_views, proposal_masks, proposal_targets,
                    blend=self.soft_consensus_blend)
                opt_codes_segmentation_X, opt_codes_segmentation_Y = refined_targets[:2]
                self.log('soft_consensus_loss', loss_soft_consensus)

        if self.match_pseudo_weight > 0 and self.current_epoch >= self.match_warmup_epochs:
            if len(batch) < 3:
                raise RuntimeError('matched pseudo-labels require a third training video')
            features_raw_Z, mask_Z = batch[2][:2]
            with torch.no_grad():
                B_Z, T_Z, _ = features_raw_Z.shape
                features_Z = F.normalize(
                    self.mlp(features_raw_Z.reshape(-1, features_raw_Z.shape[-1])).reshape(B_Z, T_Z, D),
                    dim=-1,
                )
                features_Z, _ = self._maybe_warp(features_Z, mask_Z)
                matched_x, matched_y, match_stats = consistent_frame_pairs(
                    features_raw_X, features_raw_Y, features_raw_Z,
                    features_X, features_Y, features_Z, mask_X, mask_Y, mask_Z,
                    raw_topk=self.match_raw_topk,
                    cycle_tolerance=self.match_cycle_tolerance,
                )
                opt_codes_segmentation_X, opt_codes_segmentation_Y = refine_action_pseudolabels(
                    opt_codes_segmentation_X, opt_codes_segmentation_Y,
                    matched_x, matched_y, strength=self.match_pseudo_weight,
                )
            self.log('match_pseudo_coverage', match_stats['coverage_x'])
            self.log('match_pseudo_pairs', float(match_stats['cycle_supported']))

        loss_ce_segmentation_X = -(
            (opt_codes_segmentation_X * torch.log(codes_segmentation_X + num_eps))
            * mask_X[..., None]
        ).sum(dim=2).mean()
        loss_ce_segmentation_Y = -(
            (opt_codes_segmentation_Y * torch.log(codes_segmentation_Y + num_eps))
            * mask_Y[..., None]
        ).sum(dim=2).mean()
        self.log('train_loss_segmentation_X', loss_ce_segmentation_X)
        self.log('train_loss_segmentation_Y', loss_ce_segmentation_Y)

        ##TOTAL SEGMENTATION LOSS
        loss_ce_segmentation = loss_ce_segmentation_X + loss_ce_segmentation_Y

        #######################END SEGMENTATION######################################

        # Eq (6)
        # codes represent a matrix P for each batch element
        # size of a matrix P is (no. of frames in X x no. of frames in Y)
        # P_ij represents the prob. of the frame_i in X being aligned with the frame_j in Y
        codes = torch.exp(self._view(features_X) @ self._view(features_Y).transpose(1, 2) / self.temp)
        codes = codes / codes.sum(dim=-1, keepdim=True)

        # Produce pseudo-labels using ASOT, note that we don't backpropagate through this part
        with torch.no_grad():
            # Calculate the KOT cost matrix from the paragraph above Eq (7)
            # ρR = rho * Temporal prior
            temp_prior = alignment_ot.temporal_prior(T_X, T_Y, self.rho_align, features_X.device)
            # Cost Matrix Ck from section 4.2, no need to divide by norms since both vectors were previously normalized with F.normalize()
            cost_matrix = 1. - self._teacher(features_X) @ self._teacher(features_Y).transpose(1, 2)
            bridge_active = self.current_epoch >= self.action_bridge_warmup_epochs
            if bridge_active and self.action_bridge_cost_weight > 0:
                # Segmentation -> alignment: frames with similar action
                # posteriors receive a lower alignment ground cost. This path
                # is deliberately detached together with the OT pseudo-label.
                cost_matrix += self.action_bridge_cost_weight * (
                    1.0 - action_similarity(codes_segmentation_X, codes_segmentation_Y)
                )
            # Ĉk = Ck + ρR
            cost_matrix += temp_prior

            ## Added for virtual frames
            B, N, K = cost_matrix.shape
            dev = cost_matrix.device
            # align_ot returns T[:, 1:, 1:], i.e. it drops index 0 on BOTH axes
            # as the virtual frame. The published code prepends the virtual row
            # but appends the virtual column, so the column that gets dropped is
            # a real Y frame and the virtual column survives in the pseudo-label.
            # That shifts opt_codes one frame relative to `codes`: two identical
            # videos align with a mean -0.95 frame offset instead of the
            # identity (reproduced in scripts/test_diagnostics.py).
            # The flag is OFF by default so every frozen baseline stays bit-identical.
            top_row = torch.ones(B, 1, K).to(dev) * 0.5
            cost_matrix = torch.cat((top_row, cost_matrix), dim=1)
            virtual_column = torch.ones(B, N + 1, 1).to(dev) * 0.5
            if self.fix_virtual_column:
                cost_matrix = torch.cat((virtual_column, cost_matrix), dim=2)
            else:
                cost_matrix = torch.cat((cost_matrix, virtual_column), dim=2)

            # opt_codes represent a matrix Tb for each batch element
            # size of a matrix Tb is (no. of frames in X x no. of frames in Y)
            # Tb are the (soft) pseudo-labels defined above Eq (7)
            # Tb_ij represents the prob. of the frame_i in X being aligned with the frame_j in Y

            opt_codes, _ = alignment_ot.align_ot(cost_matrix=cost_matrix, mask_X=mask_X, mask_Y=mask_Y,
                                                       eps=self.train_eps, alpha=self.alpha_train,
                                                       radius=self.radius_gw,
                                                       ub_frames=self.ub_frames, ub_actions=self.ub_actions,
                                                       lambda_frames=self.lambda_frames_train,
                                                       lambda_actions=self.lambda_actions_train,
                                                       n_iters=self.n_ot_train, step_size=self.step_size)

            # A14: replace the pairwise pseudo-label with one factored through the
            # ordered template, which makes the plan cycle-consistent by
            # construction. Blended so the arm is one factor at a time.
            opt_codes_pair = opt_codes  # frame-level plan, before any hub factoring
            template_active = (self.template_slots > 0
                               and self.current_epoch >= self.template_warmup_epochs)
            if template_active:
                if MRT['hub'] == 'clusters':
                    # Prototype hub: the two segmentation plans already map both
                    # videos onto the shared action prototypes, so alignment is
                    # their composition and needs no pairwise solve or template.
                    plan_x, plan_y = opt_codes_segmentation_X, opt_codes_segmentation_Y
                else:
                    plan_x = self._template_plan(features_X, mask_X)
                    plan_y = self._template_plan(features_Y, mask_Y)
                induced, induced_valid = self.induced_alignment(
                    plan_x, plan_y, mask_y=mask_Y, hard=not self.template_soft)
                w = max(0.0, min(1.0, self.template_blend))
                # where the template leaves the correspondence undefined, keep the
                # pairwise plan rather than supervising with an invented target
                gate = (w * induced_valid.to(induced.dtype)).unsqueeze(-1)
                opt_codes = (1.0 - gate) * opt_codes + gate * induced
                self.log('tmpl_rows_defined', float(induced_valid.float().mean()))
                if self.template_ema > 0:
                    for plan, feats in ((plan_x, features_X), (plan_y, features_Y)):
                        w = plan[0] / plan[0].sum(dim=0, keepdim=True).clamp_min(1e-8)
                        obs = w.T @ feats[0]
                        seen = plan[0].sum(dim=0) > 1e-5
                        upd = F.normalize(
                            self.template_ema * self.template.data[seen]
                            + (1 - self.template_ema) * obs[seen], dim=-1)
                        self.template.data[seen] = upd
                    self.log('tmpl_drift', float(
                        (self.template.data - F.normalize(self.template.data, dim=-1)).abs().max()))
                self.log('tmpl_slot_entropy', float(
                    -(plan_x / plan_x.sum(-1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12).log().mul(
                        plan_x / plan_x.sum(-1, keepdim=True).clamp_min(1e-12)).sum(-1).mean()))
                self.log('tmpl_slots_used', float(
                    (plan_x.sum(dim=1)[0] > 1e-4).float().sum()))
                self.log('tmpl_active', 1.0)
            else:
                self.log('tmpl_active', 0.0)

            if self.align_target_blur > 0:
                w = self.align_target_blur
                blurred = F.avg_pool2d(opt_codes.unsqueeze(1), kernel_size=2 * w + 1,
                                       stride=1, padding=w, count_include_pad=False).squeeze(1)
                blurred = blurred * mask_X.unsqueeze(2) * mask_Y.unsqueeze(1)
                opt_codes = blurred / blurred.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                self.log('blur_support', float(
                    (-(opt_codes.clamp_min(1e-12).log() * opt_codes).sum(-1)).exp().mean()))

            if MRT['target_power'] != 1.0:
                # T^gamma, each row rescaled to its previous mass: gamma < 1
                # softens the pseudo-label, gamma > 1 sharpens it.
                row_mass = opt_codes.sum(dim=-1, keepdim=True)
                powed = opt_codes.clamp_min(0).pow(MRT['target_power'])
                opt_codes = powed / powed.sum(dim=-1, keepdim=True).clamp_min(1e-12) * row_mass
                self.log('target_support', float(
                    (-(opt_codes.clamp_min(1e-12).log() * opt_codes).sum(-1)).exp().mean()))

        # Eq (7)
        loss_ce_alignment = -((opt_codes * torch.log(codes + num_eps)) * mask_X.unsqueeze(2) * mask_Y.unsqueeze(1)).sum(
            dim=2).mean()
        self.log('train_loss_alignment', loss_ce_alignment)

        loss_action_bridge = features_X.new_zeros(())
        if bridge_active and self.action_bridge_loss_weight > 0:
            # Alignment -> action space: the detached frame transport teaches
            # the differentiable action posteriors to agree across videos.
            loss_action_bridge = action_alignment_ce(
                opt_codes, codes_segmentation_X, codes_segmentation_Y,
                mask_X, mask_Y, temperature=self.action_bridge_temp,
            )
        self.log('train_loss_action_bridge', loss_action_bridge)

        # Weighted sum of the segmentation and alignment losses
        # total_loss_ce = (self.beta * loss_ce_segmentation) + loss_ce_alignment
        # print(abc)
        total_loss_ce = (
            loss_ce_segmentation
            + (self.beta * loss_ce_alignment)
            + (self.action_bridge_loss_weight * loss_action_bridge)
            + (self.visual_affinity_weight * loss_visual)
            + loss_geometry_new
            + (self.soft_consensus_weight * loss_soft_consensus)
        )

        loss_geometry = self._ray_geometry_step(batch, D, codes_segmentation_X)
        total_loss_ce = total_loss_ce + loss_geometry

        # Local (intra-action) contrast. The global losses above place a frame
        # relative to the action prototypes; here the assigned prototype is
        # subtracted and the residual -- where the frame sits inside its action
        # -- is matched across the two videos, among frames of the same
        # pseudo-action only, against the frame-level OT plan.
        if MRT['local'] > 0 and self.current_epoch >= MRT['local_warmup']:
            with torch.no_grad():
                ax = opt_codes_segmentation_X.argmax(-1)
                ay = opt_codes_segmentation_Y.argmax(-1)
                same = ((ax.unsqueeze(2) == ay.unsqueeze(1))
                        & mask_X.unsqueeze(2).bool() & mask_Y.unsqueeze(1).bool())
                t_loc = opt_codes_pair * same
                row = t_loc.sum(-1, keepdim=True)
                valid = (row.squeeze(-1) > 0.1 * opt_codes_pair.sum(-1)) & mask_X.bool()
                t_loc = t_loc / row.clamp_min(1e-12)
                protos = self.clusters.detach()
            res_x = features_X - protos[ax]
            res_y = features_Y - protos[ay]
            if not MRT['local_raw']:
                # cosine of residuals; the raw variant keeps their length, so
                # frames sitting on their prototype contribute ~0 instead of
                # having a tiny residual blown up to unit norm
                res_x, res_y = F.normalize(res_x, dim=-1), F.normalize(res_y, dim=-1)
            self.log('local_residual_norm', (features_X - protos[ax]).norm(dim=-1).mean())
            logp = (res_x @ res_y.transpose(1, 2) / self.temp).masked_fill(~same, -1e4).log_softmax(-1)
            loss_local = -((t_loc * logp).sum(-1) * valid).sum() / valid.sum().clamp_min(1)
            total_loss_ce = total_loss_ce + MRT['local'] * loss_local
            self.log('train_loss_local', loss_local)
            self.log('local_rows_used', valid.float().mean())

        self.log('train_loss', total_loss_ce)

        return total_loss_ce

    def _template_plan(self, features, mask):
        """P(slot | frame): the frames-to-template plan, from the public
        segmentation solver with L slots instead of K actions. The temporal
        prior at L slots is what keeps the slots ordered in time."""
        T_f = features.shape[1]
        L = self.template_slots
        rho = self.rho_seg if self.template_rho is None else self.template_rho
        eps = self.train_eps if self.template_eps is None else self.template_eps
        cost = 1. - features @ self.template.T.unsqueeze(0)
        cost = cost + segmentation_ot.temporal_prior(T_f, L, rho, features.device)
        plan, _ = segmentation_ot.segment_ot(
            cost, mask, eps=eps, alpha=self.alpha_train, radius=self.radius_gw,
            ub_frames=False, ub_actions=True,
            lambda_frames=self.lambda_frames_train,
            lambda_actions=self.lambda_actions_train,
            n_iters=self.n_ot_train, step_size=self.step_size)
        return plan

    def induced_alignment(self, plan_x, plan_y, mask_y=None, eps=1e-12, hard=True):
        """Factor the correspondence through the template:
            P(j | i) = sum_l P(l | i) P(j | l)

        `hard=True` assigns each frame to its argmax slot. That is not a
        convenience: composition is EXACT only for hard assignments. With soft
        plans, composing through a third video inserts an extra
        slot -> frame -> slot round trip, sum_j P(j|l) P(m|j), which is not the
        identity, and the unit test measured a 0.008 residual on random plans.
        With hard assignments on both sides P(j|l) is supported only on slot-l
        frames, that term collapses to delta_lm, and X->Y->Z reproduces X->Z
        exactly -- the property A02 found missing in the pairwise plan.

        The pseudo-label is detached in any case, so hardening costs no
        gradient. `hard=False` is kept so the soft variant can be measured
        rather than assumed. Both are checked in scripts/test_ray_geometry.py.
        """
        col = plan_y
        if mask_y is not None:
            col = col * mask_y.unsqueeze(-1)
        if hard:
            L = plan_x.shape[-1]
            sx = plan_x.argmax(dim=-1)
            p_l_given_i = F.one_hot(sx, L).to(plan_x.dtype)
            sy = col.argmax(dim=-1)
            occupancy = F.one_hot(sy, L).to(col.dtype)
            if mask_y is not None:
                occupancy = occupancy * mask_y.unsqueeze(-1)
            p_j_given_l = occupancy / occupancy.sum(dim=1, keepdim=True).clamp_min(eps)
        else:
            p_l_given_i = plan_x / plan_x.sum(dim=-1, keepdim=True).clamp_min(eps)
            p_j_given_l = col / col.sum(dim=1, keepdim=True).clamp_min(eps)
        induced = p_l_given_i @ p_j_given_l.transpose(1, 2)
        # A frame whose slot has no counterpart in Y has NO defined
        # correspondence. Inventing a uniform target there would supervise the
        # model with noise and would also destroy the exactness above, so those
        # rows are returned as invalid and the caller drops them from the loss.
        rows = induced.sum(dim=-1, keepdim=True)
        valid = (rows > eps).squeeze(-1)
        induced = induced / rows.clamp_min(eps)
        return induced, valid

    def _ray_geometry_step(self, batch, D, codes_segmentation_X):
        """A09 ray/sphere geometry objectives. Returns a zero tensor when off, so
        the baseline path is numerically untouched."""
        zero = self.clusters.new_zeros(())
        if (not self.ray_geometry) or len(batch) < 3:
            return zero
        if self.current_epoch < self.ray_warmup_epochs:
            return zero
        import ray_geometry as rg

        # embeddings for every video in the batch, WITH gradient
        feats, masks = [], []
        for rec in batch:
            raw, m = rec[0], rec[1]
            f = F.normalize(self.mlp(raw.reshape(-1, raw.shape[-1])).reshape(
                raw.shape[0], raw.shape[1], D), dim=-1)
            f, _ = self._maybe_warp(f, m)
            feats.append(f)
            masks.append(m)
        V = len(feats)
        drift = [rg.smooth_drift(f, self.ray_window) for f in feats]

        with torch.no_grad():
            # action posteriors from the public segmentation solver
            posts = []
            for f, m in zip(feats, masks):
                T_f = f.shape[1]
                cost = 1. - self._view(f) @ self._view(self.clusters).T.unsqueeze(0)
                cost = cost + segmentation_ot.temporal_prior(T_f, self.n_clusters,
                                                             self.rho_seg, f.device)
                plan, _ = segmentation_ot.segment_ot(
                    cost, m, eps=self.train_eps, alpha=self.alpha_train,
                    radius=self.radius_gw, ub_frames=False, ub_actions=True,
                    lambda_frames=self.lambda_frames_train,
                    lambda_actions=self.lambda_actions_train,
                    n_iters=self.n_ot_train, step_size=self.step_size)
                posts.append(plan / plan.sum(-1, keepdim=True).clamp_min(1e-12))

            # lazily initialise the centres at the posterior-weighted centroids
            if float(self.geo_ready) < 1.0:
                zc = torch.cat([d[0] for d in drift], dim=0)
                pc = torch.cat([p[0] for p in posts], dim=0)
                w = pc.sum(dim=0).clamp_min(1e-6)
                self.geo_centres.copy_((pc.T @ zc) / w.unsqueeze(-1))
                dist = (zc.unsqueeze(1) - self.geo_centres.unsqueeze(0)).norm(dim=-1)
                self.geo_radii.copy_(((dist * pc).sum(0) / w).clamp_min(1e-2))
                self.geo_ready.fill_(1.0)

            # reference video 0 against every other, both directions
            n_ref = feats[0].shape[1]
            fwd, bwd = {}, {}
            for v in range(1, V):
                p_fwd = self._align_argmax(feats[0], feats[v], masks[0], masks[v])
                p_bwd = self._align_argmax(feats[v], feats[0], masks[v], masks[0])
                fwd[v], bwd[v] = p_fwd, p_bwd
            idx, members, member_ok = rg.build_groups(
                fwd, bwd, n_ref, stride=self.ray_stride, cycle_tol=self.ray_cycle_tol)

        self.log('geo_group_yield', float(len(members)) / max(len(range(0, n_ref, self.ray_stride)), 1))
        if len(members) < 3:
            self.log('geo_active', 0.0)
            return zero

        ref_idx = [i for i, _ in zip(idx, range(len(idx)))]
        kept = [idx[j] for j in range(len(idx)) if j < len(members)]
        # group points: member 0 is the reference frame itself
        pts, msk = [], []
        for gi, i in enumerate(kept):
            row = [drift[0][0, i]] + [drift[v][0, int(members[gi][v - 1])] for v in range(1, V)]
            pts.append(torch.stack(row))
            msk.append(torch.tensor([True] + list(member_ok[gi]), device=feats[0].device))
        group_points = torch.stack(pts)
        group_mask = torch.stack(msk)

        # one action per group, from the reference frame's posterior
        k_of = posts[0][0][torch.as_tensor(kept, device=feats[0].device)].argmax(-1)

        total = zero
        diag = {}
        for k in k_of.unique():
            sel = (k_of == k)
            if int(sel.sum()) < 2:
                continue
            gp = group_points[sel]
            gm = group_mask[sel]
            centre = self.geo_centres[k]
            if self.ray_mode == 'sphere':
                loss_k, u, r = rg.sphere_alignment_loss(
                    gp, gm, centre, self.geo_radii[k],
                    radius_weight=self.ray_radius_weight)
            else:
                loss_k, u, r = rg.ray_alignment_loss(gp, gm, centre)
                loss_k = loss_k + self.ray_radius_weight * rg.radius_regularisation_loss(
                    r, float(self.geo_radii[k]), gm)
            nce, hinge, _ = rg.group_separation_loss(u, gm)
            total = total + loss_k + self.ray_group_weight * (nce + hinge)
            diag.setdefault('within', []).append(float(
                torch.einsum('gvd,gwd->gvw', u.detach(), u.detach())[
                    gm.unsqueeze(2) & gm.unsqueeze(1)].mean()))
            diag.setdefault('radius', []).append(float(r.detach().mean()))
            if self.ray_update_centres and int(sel.sum()) >= 3:
                anchor = (gp.detach() * gm.unsqueeze(-1)).sum(dim=(0, 1)) / \
                         gm.float().sum().clamp_min(1.0)
                info = rg.concurrency_centre_update(
                    self.geo_centres, int(k), gp.detach(), gm,
                    torch.ones(gp.shape[0], device=gp.device),
                    momentum=self.ray_centre_momentum, anchor=anchor)
                diag.setdefault('centre_updated', []).append(info['updated'])
                if 'conditioning' in info:
                    diag.setdefault('conditioning', []).append(info['conditioning'])

        with torch.no_grad():
            # keep the centres from merging; the shell cost cannot discriminate at
            # all when they sit closer together than the frames sit from them
            # (measured: nearest pair 0.268 vs radius 0.424)
            dmat = torch.cdist(self.geo_centres, self.geo_centres)
            k_n = self.geo_centres.shape[0]
            for a in range(k_n):
                for b in range(a + 1, k_n):
                    gap = float(dmat[a, b])
                    if gap < self.ray_centre_margin and gap > 1e-6:
                        push = (self.geo_centres[a] - self.geo_centres[b]) / gap
                        shift = 0.5 * (self.ray_centre_margin - gap)
                        self.geo_centres[a] += 0.25 * shift * push
                        self.geo_centres[b] -= 0.25 * shift * push
            rg.update_radii(self.geo_radii, torch.cat([d.detach() for d in drift], dim=0),
                            self.geo_centres, torch.cat([p for p in posts], dim=0),
                            momentum=0.95)

        self.log('geo_active', 1.0)
        self.log('geo_loss', total.detach())
        for key, vals in diag.items():
            if vals:
                self.log(f'geo_{key}', float(sum(vals) / len(vals)))
        self.log('geo_centre_nearest', float(
            torch.cdist(self.geo_centres, self.geo_centres).masked_fill(
                torch.eye(self.n_clusters, dtype=torch.bool, device=self.geo_centres.device),
                float('inf')).min()))
        self.log('geo_radius_mean', float(self.geo_radii.mean()))
        return self.ray_weight * total

    def _align_argmax(self, f_a, f_b, m_a, m_b):
        """Detached argmax correspondence a -> b from the public alignment solver."""
        n_a, n_b = f_a.shape[1], f_b.shape[1]
        cost = 1. - self._view(f_a) @ self._view(f_b).transpose(1, 2)
        cost = cost + alignment_ot.temporal_prior(n_a, n_b, self.rho_align, f_a.device)
        cost = torch.cat((torch.ones(1, 1, n_b, device=f_a.device) * 0.5, cost), dim=1)
        if self.fix_virtual_column:
            cost = torch.cat((torch.ones(1, n_a + 1, 1, device=f_a.device) * 0.5, cost), dim=2)
        else:
            cost = torch.cat((cost, torch.ones(1, n_a + 1, 1, device=f_a.device) * 0.5), dim=2)
        plan, _ = alignment_ot.align_ot(
            cost, m_a, m_b, eps=self.train_eps, alpha=self.alpha_train,
            radius=self.radius_gw, ub_frames=self.ub_frames, ub_actions=self.ub_actions,
            lambda_frames=self.lambda_frames_train, lambda_actions=self.lambda_actions_train,
            n_iters=self.n_ot_train, step_size=self.step_size)
        return plan[0].argmax(-1).cpu().tolist()

    def validation_step(self, batch, batch_idx):
        features_raw_X, mask_X, gt_X, fname_X, n_subactions_X = batch[0]
        features_raw_Y, mask_Y, gt_Y, fname_Y, n_subactions_Y = batch[1]

        D = self.layer_sizes[-1]
        
        B_X, T_X, _ = features_raw_X.shape
        features_X = F.normalize(
            self.mlp(features_raw_X.reshape(-1, features_raw_X.shape[-1])).reshape(B_X, T_X, D),
            dim=-1
        )
        features_X, u_X_eval = self._maybe_warp(features_X, mask_X)

        B_Y, T_Y, _ = features_raw_Y.shape
        features_Y = F.normalize(
            self.mlp(features_raw_Y.reshape(-1, features_raw_Y.shape[-1])).reshape(B_Y, T_Y, D),
            dim=-1
        )
        features_Y, u_Y_eval = self._maybe_warp(features_Y, mask_Y)

        temp_prior_segmentation_X = segmentation_ot.temporal_prior(T_X, self.n_clusters, self.rho_seg,
                                                                     features_X.device)
        cost_matrix_segmentation_X = 1. - self._view(features_X) @ self._view(self.clusters).T.unsqueeze(0)
        cost_matrix_segmentation_X += temp_prior_segmentation_X
        # print(cost_matrix_segmentation_X.shape)
        segmentation_X, _ = segmentation_ot.segment_ot(cost_matrix_segmentation_X, mask_X,
                                                                     eps=self.train_eps, alpha=self.alpha_train,
                                                                     radius=self.radius_gw,
                                                                     ub_frames=False, ub_actions=True,
                                                                     lambda_frames=self.lambda_frames_train,
                                                                     lambda_actions=self.lambda_actions_train,
                                                                     n_iters=self.n_ot_train,
                                                                     step_size=self.step_size)

        temp_prior_segmentation_Y = segmentation_ot.temporal_prior(T_Y, self.n_clusters, self.rho_seg,
                                                                     features_Y.device)
        cost_matrix_segmentation_Y = 1. - self._view(features_Y) @ self._view(self.clusters).T.unsqueeze(0)
        cost_matrix_segmentation_Y += temp_prior_segmentation_Y
        segmentation_Y, _ = segmentation_ot.segment_ot(cost_matrix_segmentation_Y, mask_Y,
                                                                     eps=self.train_eps, alpha=self.alpha_train,
                                                                     radius=self.radius_gw,
                                                                     ub_frames=False, ub_actions=True,
                                                                     lambda_frames=self.lambda_frames_train,
                                                                     lambda_actions=self.lambda_actions_train,
                                                                     n_iters=self.n_ot_train,
                                                                     step_size=self.step_size)
        # log clustering metrics over full epoch


        #TODO: Adjust the evaluation code below for video alignment
        segments_X = segmentation_X.argmax(dim=2)
        if u_X_eval is not None:
            segments_X = inverse_warp_predictions(segments_X, u_X_eval)
        self.mof.update(segments_X, gt_X, mask_X)
        self.f1.update(segments_X, gt_X, mask_X)
        self.miou.update(segments_X, gt_X, mask_X)

        segments_Y = segmentation_Y.argmax(dim=2)
        if u_Y_eval is not None:
            segments_Y = inverse_warp_predictions(segments_Y, u_Y_eval)
        self.mof.update(segments_Y, gt_Y, mask_Y)
        self.f1.update(segments_Y, gt_Y, mask_Y)
        self.miou.update(segments_Y, gt_Y, mask_Y)

        # log clustering metrics per video
        metrics_X = indep_eval_metrics(segments_X, gt_X, mask_X, ['mof', 'f1', 'miou'], exclude_cls=self.exclude_cls)
        self.log('val_mof_per_X', metrics_X['mof'])
        self.log('val_f1_per_X', metrics_X['f1'])
        self.log('val_miou_per_X', metrics_X['miou'])

        metrics_Y = indep_eval_metrics(segments_Y, gt_Y, mask_Y, ['mof', 'f1', 'miou'], exclude_cls=self.exclude_cls)
        self.log('val_mof_per_Y', metrics_Y['mof'])
        self.log('val_f1_per_Y', metrics_Y['f1'])
        self.log('val_miou_per_Y', metrics_Y['miou'])

        self.log('val_mof_per', mean([metrics_X['mof'], metrics_Y['mof']]))
        self.log('val_f1_per', mean([metrics_X['f1'], metrics_Y['f1']]))
        self.log('val_miou_per', mean([metrics_X['miou'],metrics_Y['miou']]))
        
        return None
    
    def test_step(self, batch, batch_idx):
        # NOTE: The comments below assume batchsize=1
        # features_raw [A 2D Tensor of size=no. of frames x frame embedding dims, representing the input frame embeddings of a video]
        # mask [A 1D array of Trues of size=no. of frames]
        # gt [A 1D tensor of action-ids of size=no. of frames]
        # fname [Video's filename]
        # n_subactions [No. of unique action-classes amongst the randomly sampled frames/gt]

        features_raw_X, mask_X, gt_X, fname_X, n_subactions_X = batch[0]
        features_raw_Y, mask_Y, gt_Y, fname_Y, n_subactions_Y = batch[1]

        D = self.layer_sizes[-1]

        B_X, T_X, _ = features_raw_X.shape
        features_X = F.normalize(
            self.mlp(features_raw_X.reshape(-1, features_raw_X.shape[-1])).reshape(B_X, T_X, D),
            dim=-1
        )
        features_X, u_X_eval = self._maybe_warp(features_X, mask_X)

        B_Y, T_Y, _ = features_raw_Y.shape
        features_Y = F.normalize(
            self.mlp(features_raw_Y.reshape(-1, features_raw_Y.shape[-1])).reshape(B_Y, T_Y, D),
            dim=-1
        )
        features_Y, u_Y_eval = self._maybe_warp(features_Y, mask_Y)

        temp_prior_segmentation_X = segmentation_ot.temporal_prior(T_X, self.n_clusters, self.rho_seg,
                                                                     features_X.device)
        cost_matrix_segmentation_X = 1. - self._view(features_X) @ self._view(self.clusters).T.unsqueeze(0)
        cost_matrix_segmentation_X += temp_prior_segmentation_X
        # print(cost_matrix_segmentation_X.shape)
        segmentation_X, _ = segmentation_ot.segment_ot(cost_matrix_segmentation_X, mask_X,
                                                           eps=self.train_eps, alpha=self.alpha_train,
                                                           radius=self.radius_gw,
                                                           ub_frames=False, ub_actions=True,
                                                           lambda_frames=self.lambda_frames_train,
                                                           lambda_actions=self.lambda_actions_train,
                                                           n_iters=self.n_ot_train,
                                                           step_size=self.step_size)

        temp_prior_segmentation_Y = segmentation_ot.temporal_prior(T_Y, self.n_clusters, self.rho_seg,
                                                                     features_Y.device)
        cost_matrix_segmentation_Y = 1. - self._view(features_Y) @ self._view(self.clusters).T.unsqueeze(0)
        cost_matrix_segmentation_Y += temp_prior_segmentation_Y
        segmentation_Y, _ = segmentation_ot.segment_ot(cost_matrix_segmentation_Y, mask_Y,
                                                           eps=self.train_eps, alpha=self.alpha_train,
                                                           radius=self.radius_gw,
                                                           ub_frames=False, ub_actions=True,
                                                           lambda_frames=self.lambda_frames_train,
                                                           lambda_actions=self.lambda_actions_train,
                                                           n_iters=self.n_ot_train,
                                                           step_size=self.step_size)

        #TODO: Adjust the evaluation code below for video alignment
        # segments = segmentation.argmax(dim=2)
        #
        # # NOTE: Uncomment this code to store the predictions as a file, this is used on the colab notebook to make visualizations
        # # segments_np = segments.cpu().numpy()
        # # output_dir = "preds"
        # # os.makedirs(output_dir, exist_ok=True)
        # # np.savetxt(os.path.join(output_dir, f'{fname[0]}.txt'), segments_np, fmt='%d', delimiter='\n')
        #
        # self.mof.update(segments, gt, mask)
        # self.f1.update(segments, gt, mask)
        # self.miou.update(segments, gt, mask)
        #
        # # log clustering metrics per video
        # metrics = indep_eval_metrics(segments, gt, mask, ['mof', 'f1', 'miou'], exclude_cls=self.exclude_cls)
        # self.log('test_mof_per', metrics['mof'])
        # self.log('test_f1_per', metrics['f1'])
        # self.log('test_miou_per', metrics['miou'])

        segments_X = segmentation_X.argmax(dim=2)
        if u_X_eval is not None:
            segments_X = inverse_warp_predictions(segments_X, u_X_eval)
        self.mof.update(segments_X, gt_X, mask_X)
        self.f1.update(segments_X, gt_X, mask_X)
        self.miou.update(segments_X, gt_X, mask_X)

        segments_Y = segmentation_Y.argmax(dim=2)
        if u_Y_eval is not None:
            segments_Y = inverse_warp_predictions(segments_Y, u_Y_eval)
        self.mof.update(segments_Y, gt_Y, mask_Y)
        self.f1.update(segments_Y, gt_Y, mask_Y)
        self.miou.update(segments_Y, gt_Y, mask_Y)

        # log clustering metrics per video
        metrics_X = indep_eval_metrics(segments_X, gt_X, mask_X, ['mof', 'f1', 'miou'], exclude_cls=self.exclude_cls)
        self.log('val_mof_per_X', metrics_X['mof'])
        self.log('val_f1_per_X', metrics_X['f1'])
        self.log('val_miou_per_X', metrics_X['miou'])

        metrics_Y = indep_eval_metrics(segments_Y, gt_Y, mask_Y, ['mof', 'f1', 'miou'], exclude_cls=self.exclude_cls)
        self.log('val_mof_per_Y', metrics_Y['mof'])
        self.log('val_f1_per_Y', metrics_Y['f1'])
        self.log('val_miou_per_Y', metrics_Y['miou'])

        self.log('val_mof_per', mean([metrics_X['mof'], metrics_Y['mof']]))
        self.log('val_f1_per', mean([metrics_X['f1'], metrics_Y['f1']]))
        self.log('val_miou_per', mean([metrics_X['miou'],metrics_Y['miou']]))
        # cache videos for plotting
        self.test_cache.append([metrics_X['mof'], segments_X, gt_X, mask_X, fname_X])
        self.test_cache.append([metrics_Y['mof'], segments_Y, gt_Y, mask_Y, fname_Y])

        return None
    
    def on_validation_epoch_end(self):
        #TODO: Adjust the evaluation code below for video alignment
        mof, pred_to_gt = self.mof.compute(exclude_cls=self.exclude_cls)
        f1, _ = self.f1.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        miou, _ = self.miou.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        self.log('val_mof_full', mof)
        self.log('val_f1_full', f1)
        self.log('val_miou_full', miou)
        self.mof.reset()
        self.f1.reset()
        self.miou.reset()

    def on_test_epoch_end(self):
        #TODO: Adjust the evaluation code below for video alignment
        mof, pred_to_gt = self.mof.compute(exclude_cls=self.exclude_cls)
        f1, _ = self.f1.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        miou, _  = self.miou.compute(exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)
        self.log('test_mof_full', mof)
        self.log('test_f1_full', f1)
        self.log('test_miou_full', miou)

        if self.visualize:
            
            # Make a dir to store visualizations
            output_dir = f'{self.activity}'
            os.makedirs(output_dir, exist_ok=True)

            for i, (mof, pred, gt, mask, fname) in enumerate(self.test_cache):
                self.test_cache[i][0] = indep_eval_metrics(pred, gt, mask, ['mof'], exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt)['mof']
            self.test_cache = sorted(self.test_cache, key=lambda x: x[0], reverse=True)

            for i, (mof, pred, gt, mask, fname) in enumerate(self.test_cache):
                fig = plot_segmentation_gt(gt, pred, mask, exclude_cls=self.exclude_cls, pred_to_gt=pred_to_gt,
                                           gt_uniq=np.unique(self.mof.gt_labels), name=f'{fname[0]}')
                fig_path = os.path.join(output_dir, f"test_segment_{i}.png")
                fig.savefig(fig_path)
                plt.close(fig)

        self.test_cache = []
        self.mof.reset()
        self.f1.reset()
        self.miou.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    


def init_action_prototypes(model, dataset, mode, seed=0, max_videos=20, verbose=True):
    """A01: replace the random prototype codebook with a different initialiser.

    All modes keep the unit-norm convention the training loop enforces. Data
    dependent modes read the TRAINING videos' features through the freshly
    initialised encoder; they use no labels and no ground truth, so they stay
    inside the self-supervised protocol. 'random' is the published behaviour and
    returns without touching anything.
    """
    if mode == 'random':
        return {'mode': 'random', 'changed': False}
    import numpy as _np
    k, d = model.clusters.shape
    g = torch.Generator().manual_seed(seed)
    info = {'mode': mode, 'changed': True, 'n_clusters': int(k), 'dim': int(d)}

    if mode == 'orthogonal':
        a = torch.randn(max(k, d), d, generator=g)
        q, _ = torch.linalg.qr(a)
        protos = q[:k] if k <= q.shape[0] else torch.cat([q] * (k // q.shape[0] + 1))[:k]
        info['note'] = ('exactly orthonormal only while k <= d; beyond that the rows '
                        'cannot all be orthogonal') if k > d else 'orthonormal rows'
    else:
        names = dataset.video_fnames[:max_videos]
        feats, positions = [], []
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for nm in names:
                raw, mask, _, _, _ = dataset._load_video_data(nm)
                z = F.normalize(model.mlp(raw.unsqueeze(0)), dim=-1)[0]
                keep = torch.as_tensor(_np.asarray(mask), dtype=torch.bool)
                z = z[keep]
                feats.append(z)
                positions.append(torch.linspace(0, 1, z.shape[0]))
        if was_training:
            model.train()
        Z = torch.cat(feats, dim=0)
        P = torch.cat(positions, dim=0)

        if mode == 'data':
            idx = torch.randperm(Z.shape[0], generator=g)[:k]
            protos = Z[idx]
            info['note'] = 'k randomly sampled frame embeddings'
        elif mode == 'temporal':
            edges = torch.linspace(0, 1, k + 1)
            rows = []
            for i in range(k):
                m = (P >= edges[i]) & (P <= edges[i + 1])
                rows.append(Z[m].mean(0) if int(m.sum()) > 0
                            else Z[torch.randint(Z.shape[0], (1,), generator=g)][0])
            protos = torch.stack(rows)
            info['note'] = 'mean embedding of each of k equal relative-time bins'
        elif mode == 'kmeans':
            from sklearn.cluster import KMeans
            sub = Z[torch.randperm(Z.shape[0], generator=g)[:min(20000, Z.shape[0])]].numpy()
            km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(sub)
            protos = torch.from_numpy(km.cluster_centers_).float()
            info['note'] = f'k-means on {sub.shape[0]} frame embeddings'
            info['kmeans_inertia'] = float(km.inertia_)
        else:
            raise ValueError(f'unknown cluster init {mode}')

    protos = F.normalize(protos, dim=-1).to(model.clusters.device)
    with torch.no_grad():
        model.clusters.data.copy_(protos)
    cos = protos.cpu() @ protos.cpu().T
    off = cos[~torch.eye(k, dtype=torch.bool)]
    info['offdiag_cosine_mean'] = float(off.mean())
    info['nearest_cosine_mean'] = float((cos - torch.eye(k) * 2).max(dim=-1).values.mean())
    if verbose:
        print(f"[cluster-init] {mode}: off-diagonal cosine mean "
              f"{info['offdiag_cosine_mean']:.4f}, nearest {info['nearest_cosine_mean']:.4f}")
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train representation learning pipeline")

    #TODO: Change args to only keep what we need and add any new args that we might need
    # FUGW OT segmentation parameters
    parser.add_argument('--alpha-train', '-at', type=float, default=0.3, help='weighting of KOT term on frame features in OT')
    parser.add_argument('--alpha-eval', '-ae', type=float, default=0.6, help='weighting of KOT term on frame features in OT')
    parser.add_argument('--ub-frames', '-uf', action='store_true',
                        help='relaxes balanced assignment assumption over frames, i.e., each frame is assigned')
    parser.add_argument('--ub-actions', '-ua', action='store_true',
                        help='relaxes balanced assignment assumption over actions, i.e., each action is uniformly represented in a video')
    parser.add_argument('--lambda-frames-train', '-lft', type=float, default=0.05, help='penalty on balanced frames assumption for training')
    parser.add_argument('--lambda-actions-train', '-lat', type=float, default=0.05, help='penalty on balanced actions assumption for training')
    parser.add_argument('--lambda-frames-eval', '-lfe', type=float, default=0.05, help='penalty on balanced frames assumption for test')
    parser.add_argument('--lambda-actions-eval', '-lae', type=float, default=0.01, help='penalty on balanced actions assumption for test')
    parser.add_argument('--eps-train', '-et', type=float, default=0.07, help='entropy regularization for OT during training')
    parser.add_argument('--eps-eval', '-ee', type=float, default=0.04, help='entropy regularization for OT during val/test')
    parser.add_argument('--radius-gw', '-r', type=float, default=0.04, help='Radius parameter for GW structure loss')
    parser.add_argument('--n-ot-train', '-nt', type=int, nargs='+', default=[25, 1], help='number of outer and inner iterations for ASOT solver (train)')
    parser.add_argument('--n-ot-eval', '-no', type=int, nargs='+', default=[25, 1], help='number of outer and inner iterations for ASOT solver (eval)')
    parser.add_argument('--step-size', '-ss', type=float, default=None,
                        help='Step size/learning rate for ASOT solver. Worth setting manually if ub-frames && ub-actions')

    # dataset params
    ## TODO: Not using this arg since we hardcoded it, add it back in later
    parser.add_argument('--base-path', '-p', type=str, default='/home/users/u6567085/data', help='base directory for dataset')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='dataset to use for training/eval (Breakfast, YTI, FSeval, FS, desktop_assembly)')
    parser.add_argument('--activity', '-ac', type=str, nargs='+', required=True, help='activity classes to select for dataset')
    parser.add_argument('--exclude', '-x', type=int, default=None, help='classes to exclude from evaluation. use -1 for YTI')
    parser.add_argument('--n-frames', '-f', type=int, default=256, help='number of frames sampled per video for train/val')
    parser.add_argument('--std-feats', '-s', action='store_true', help='standardize features per video during preprocessing')
    
    # representation learning params
    parser.add_argument('--n-epochs', '-ne', type=int, default=15, help='number of epochs for training')
    parser.add_argument('--batch-size', '-bs', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight-decay', '-wd', type=float, default=1e-4, help='weight decay for optimizer')
    parser.add_argument('--layers', '-ls', default=[64, 128, 40], nargs='+', type=int, help='layer sizes for MLP (in, hidden, ..., out)')
    parser.add_argument('--rho', type=float, default=0.1, help='Factor for global structure weighting term')
    parser.add_argument('--rho-seg', type=float, default=None,
                        help='Override temporal prior weight for segmentation (default: --rho)')
    parser.add_argument('--rho-align', type=float, default=None,
                        help='Override temporal prior weight for alignment (default: --rho)')
    parser.add_argument('--template-slots', type=int, default=0,
                        help='A14: factor alignment through an ordered template of L slots '
                             '(0 = off, the published pairwise plan). L >> n_clusters on purpose')
    parser.add_argument('--template-rho', type=float, default=None,
                        help='temporal prior weight for the frames-to-template plan (default: --rho-seg)')
    parser.add_argument('--template-eps', type=float, default=None)
    parser.add_argument('--template-warmup-epochs', type=int, default=5)
    parser.add_argument('--align-target-blur', type=int, default=0,
                        help='A16: box-blur the alignment pseudo-label over +-W frames on both '
                             'axes (0 = off). The template-free form of A14/A15')
    parser.add_argument('--template-ema', type=float, default=0.0,
                        help='A18: EMA momentum for adapting the template to the data '
                             '(0 = the random fixed template of A14/A15)')
    parser.add_argument('--template-soft', action='store_true',
                        help='A15: soft slot assignment; composition is then NOT exact (0.006 vs 3e-08)')
    parser.add_argument('--template-frozen', action='store_true',
                        help='A15 control: never learn the template. If this works too, the '
                             'factoring is the active ingredient, not the template content')
    parser.add_argument('--template-blend', type=float, default=1.0,
                        help='1.0 replaces the pairwise pseudo-label, 0.5 blends it')
    parser.add_argument('--input-window', type=int, default=0,
                        help='A13: encoder reads a +-w frame window of raw features '
                             '(0 = one frame, the published behaviour)')
    parser.add_argument('--ray-geometry', action='store_true',
                        help='A09: impose the ray/sphere correspondence geometry (default off)')
    parser.add_argument('--ray-mode', type=str, default='sphere', choices=['ray', 'sphere'])
    parser.add_argument('--ray-videos', type=int, default=3,
                        help='videos per training item; >=3 is required to define a group')
    parser.add_argument('--ray-weight', type=float, default=0.1)
    parser.add_argument('--ray-radius-weight', type=float, default=1.0)
    parser.add_argument('--ray-group-weight', type=float, default=0.1)
    parser.add_argument('--ray-shell-cost-weight', type=float, default=0.0)
    parser.add_argument('--ray-warmup-epochs', type=int, default=10)
    parser.add_argument('--ray-window', type=int, default=15)
    parser.add_argument('--ray-stride', type=int, default=4)
    parser.add_argument('--ray-cycle-tol', type=int, default=4)
    parser.add_argument('--ray-centre-momentum', type=float, default=0.9)
    parser.add_argument('--ray-centre-margin', type=float, default=0.35)
    parser.add_argument('--no-ray-centre-update', action='store_true',
                        help='ablate goal (2): keep the geometry loss, freeze the centres')
    parser.add_argument('--cluster-init', type=str, default='random',
                        choices=['random', 'orthogonal', 'data', 'temporal', 'kmeans'],
                        help='A01: action prototype initialisation (default random = published behaviour)')
    parser.add_argument('--fix-virtual-column', action='store_true',
                        help='prepend the virtual alignment column instead of appending it, so the '
                             'column align_ot drops is the virtual one and not a real Y frame '
                             '(default off: reproduces the published off-by-one)')
    parser.add_argument('--action-bridge-cost-weight', type=float, default=0.0,
                        help='Weight of action-posterior distance added to alignment OT cost')
    parser.add_argument('--action-bridge-loss-weight', type=float, default=0.0,
                        help='Weight of alignment-to-action auxiliary CE (0 = target-cost-only bridge)')
    parser.add_argument('--action-bridge-temp', type=float, default=0.1,
                        help='Temperature for action-induced frame alignment probabilities')
    parser.add_argument('--action-bridge-warmup-epochs', type=int, default=5,
                        help='Epochs before enabling action bridge cost/loss')
    parser.add_argument('--visual-affinity-weight', type=float, default=0.0,
                        help='Cross-video raw-visual affinity KL weight (0 = published model)')
    parser.add_argument('--visual-affinity-stride', type=int, default=4,
                        help='Subsample stride for visual affinity preservation')
    parser.add_argument('--visual-affinity-teacher-temp', type=float, default=0.07,
                        help='Temperature of the fixed raw-feature visual teacher')
    parser.add_argument('--match-pseudo-weight', type=float, default=0.0,
                        help='Blend strength for action pseudo-labels at 3-video-consistent frame matches')
    parser.add_argument('--match-raw-topk', type=int, default=8,
                        help='Raw-visual top-k support required for each frame match')
    parser.add_argument('--match-cycle-tolerance', type=int, default=4,
                        help='Allowed X→Z versus X→Y→Z disagreement in sampled frames')
    parser.add_argument('--match-warmup-epochs', type=int, default=5,
                        help='Epochs before enabling consistent-match pseudo-label transfer')
    parser.add_argument('--proposal-triplets', '--proposal-multiview',
                        dest='proposal_triplets', action='store_true',
                        help='Use the same n-video sampler for ablation controls')
    parser.add_argument('--geometry-weight', type=float, default=0.0,
                        help='VICReg-like variance and covariance regularization weight')
    parser.add_argument('--geometry-repulsion-weight', type=float, default=0.0,
                        help='Gated distant-frame cosine repulsion weight')
    parser.add_argument('--spectot-tar-weight', type=float, default=0.0,
                        help='SpecT-OT temporal affinity regularization ablation weight')
    parser.add_argument('--soft-consensus-weight', type=float, default=0.0,
                        help='Soft reciprocal and multi-path consistency weight')
    parser.add_argument('--soft-consensus-blend', type=float, default=0.0,
                        help='Soft confidence-weighted pseudo-label transfer strength')
    parser.add_argument('--align-hub', default='template', choices=['template', 'clusters'],
                        help='with --template-slots>0: factor alignment through the random template or through the action prototypes')
    parser.add_argument('--local-weight', type=float, default=0.0,
                        help='weight of the intra-action residual contrastive loss')
    parser.add_argument('--local-warmup-epochs', type=int, default=5)
    parser.add_argument('--local-raw', action='store_true',
                        help='local loss on un-normalised residuals (z - prototype)')
    parser.add_argument('--align-target-power', type=float, default=1.0,
                        help='raise the alignment pseudo-label to this power and renormalise '
                             'rows (<1 softens, >1 sharpens, 1 = off)')
    parser.add_argument('--decenter', type=float, default=0.0,
                        help='MRT: fraction of the EMA embedding mean removed before every cosine cost/logit')
    parser.add_argument('--ot-noise', type=float, default=0.0,
                        help='MRT: std of the Gaussian sample added to embeddings before OT pseudo-labelling')
    parser.add_argument('--proposal-videos', type=int, default=4,
                        help='Videos per training item for geometry and soft consensus')
    parser.add_argument('--n-clusters', '-c', type=int, default=8, help='number of actions/clusters')
    parser.add_argument('--beta', '-b', type=float, default=1,
                        help='the weight used when combining alignment and segmentation losses')

    # system/logging params
    parser.add_argument('--val-freq', '-vf', type=int, default=5, help='validation epoch frequency (epochs)')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='gpu id to use')
    parser.add_argument('--visualize', '-v', action='store_true', help='generate visualizations during logging')
    parser.add_argument('--seed', type=int, default=0, help='Random seed initialization')
    parser.add_argument('--ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--eval', action='store_true', help='run evaluation on test set only')
    parser.add_argument('--wandb', action='store_true', help='enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='vasot-repro', help='wandb project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='wandb run name (default: dataset_activity_ne<N>_seed<s>)')
    parser.add_argument('--wandb-tags', type=str, nargs='*', default=None, help='wandb tags')
    parser.add_argument('--arc-reparam', action='store_true', help='E1: arc-length re-parameterisation of X before OT')
    parser.add_argument('--arc-warmup-epochs', type=int, default=5, help='epochs of identity-warp warmup before arc-reparam kicks in')
    parser.add_argument('--bspline', action='store_true', help='E2: adaptive cubic B-spline fit of X before OT')
    parser.add_argument('--bspline-tol', type=float, default=0.05, help='reconstruction tolerance for adaptive knot insertion')
    parser.add_argument('--bspline-kmax', type=int, default=0, help='max number of control points (0 = N//8)')
    parser.add_argument('--bspline-warmup-epochs', type=int, default=5, help='epochs of identity before bspline kicks in')
    parser.add_argument('--bspline-space', type=str, default='X', choices=['X', 'logits', 'latent'],
                        help="fit B-spline on 'X' (frame embedding), 'logits' (action-space via clusters), or 'latent' (E10b: dedicated learnable bottleneck)")
    parser.add_argument('--latent-dim', type=int, default=None,
                        help="E10b: dim of dedicated learnable latent basis when --bspline-space=latent (default: 16)")
    parser.add_argument('--bspline-blend-max', type=float, default=1.0, help='residual blend coefficient at max (0=off, 1=full)')
    parser.add_argument('--bspline-blend-rampup', type=int, default=0, help='epochs over which blend ramps from 0 to blend-max')
    parser.add_argument('--gauss-smooth', action='store_true', help='E2d: Gaussian smoothing along time axis')
    parser.add_argument('--gauss-sigma', type=float, default=2.0, help='Gaussian kernel std (frames)')
    parser.add_argument('--gauss-blend-max', type=float, default=1.0, help='Gaussian residual blend coefficient')
    parser.add_argument('--gauss-warmup-epochs', type=int, default=5, help='epochs of identity before gauss kicks in')
    parser.add_argument('--bezier', action='store_true', help='E4: piecewise cubic Bézier fit of X before OT')
    parser.add_argument('--bezier-nseg', type=int, default=4, help='initial number of Bézier segments')
    parser.add_argument('--bezier-tol', type=float, default=0.05, help='reconstruction tolerance for adaptive segment splitting')
    parser.add_argument('--bezier-maxseg', type=int, default=32, help='max number of Bézier segments')
    parser.add_argument('--bezier-warmup-epochs', type=int, default=5, help='epochs of identity before bezier kicks in')
    parser.add_argument('--bezier-space', type=str, default='X', choices=['X', 'logits'],
                        help='fit Bézier on frame embedding X (default) or on action logits X·A^T')
    parser.add_argument('--bezier-blend-max', type=float, default=1.0, help='residual blend coefficient at max (0=off, 1=full)')
    parser.add_argument('--bezier-blend-rampup', type=int, default=0, help='epochs over which blend ramps from 0 to blend-max')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='explicit output directory for this run (logs, checkpoints, manifest)')
    parser.add_argument('--run-id', type=str, default=None,
                        help='unique run identifier (default: auto-generated from config)')
    args = parser.parse_args()
    MRT.update(target_power=args.align_target_power, decenter=args.decenter, ot_noise=args.ot_noise, hub=args.align_hub,
               local=args.local_weight, local_warmup=args.local_warmup_epochs, local_raw=args.local_raw)

    pl.seed_everything(args.seed)

    import json, socket, subprocess, hashlib
    from datetime import datetime, timezone

    act_str = '-'.join(args.activity) if isinstance(args.activity, list) else str(args.activity)
    run_id = args.run_id or f"{args.dataset}_{act_str}_s{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    run_dir = args.run_dir or os.path.join(
        os.environ.get('VASOT_ARTIFACTS_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'artifacts')),
        'runs', run_id)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_subdir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_subdir, exist_ok=True)

    data_val = VideoDataset(args.base_path, args.dataset, args.n_frames, standardise=args.std_feats, random=False, action_class=args.activity)
    new_proposal = (args.geometry_weight > 0 or args.geometry_repulsion_weight > 0
                    or args.spectot_tar_weight > 0
                    or args.soft_consensus_weight > 0 or args.soft_consensus_blend > 0)
    if args.ray_geometry or args.match_pseudo_weight > 0 or args.proposal_triplets or new_proposal:
        from video_dataset import MultiVideoDataset
        data_train = MultiVideoDataset(args.base_path, args.dataset, args.n_frames,
                                       standardise=args.std_feats, random=True,
                                       action_class=args.activity,
                                       n_videos_per_item=max(args.proposal_videos if (new_proposal or args.proposal_triplets) else 2,
                                                             args.ray_videos if args.ray_geometry else 2,
                                                             3 if args.match_pseudo_weight > 0
                                                             or args.proposal_triplets else 2))
    else:
        data_train = VideoDataset(args.base_path, args.dataset, args.n_frames, standardise=args.std_feats, random=True, action_class=args.activity)
    data_test = VideoDataset(args.base_path, args.dataset, None, standardise=args.std_feats, random=False, action_class=args.activity)
    if (args.input_window > 0 or args.visual_affinity_weight > 0
            or args.match_pseudo_weight > 0 or new_proposal) and args.batch_size != 1:
        raise SystemExit('--input-window requires --batch-size 1: the call sites flatten '
                         '(B, T, D) to (B*T, D), so a larger batch would window across videos')
    if args.visual_affinity_weight < 0 or not 0 <= args.match_pseudo_weight <= 1:
        raise SystemExit('visual-affinity-weight must be nonnegative and match-pseudo-weight in [0, 1]')
    if (args.proposal_videos < 3 or args.geometry_weight < 0 or args.geometry_repulsion_weight < 0
            or args.spectot_tar_weight < 0
            or args.soft_consensus_weight < 0 or not 0 <= args.soft_consensus_blend <= 1):
        raise SystemExit('invalid geometry/soft-consensus proposal hyperparameters')
    val_loader = DataLoader(data_val, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False)

    if args.ckpt is not None:
        ssl = VideoSSL.load_from_checkpoint(args.ckpt)
    else:
        ssl = VideoSSL(layer_sizes=args.layers, n_clusters=args.n_clusters, alpha_train=args.alpha_train, alpha_eval=args.alpha_eval,
                       ub_frames=args.ub_frames, ub_actions=args.ub_actions, lambda_frames_train=args.lambda_frames_train, lambda_frames_eval=args.lambda_frames_eval,
                       lambda_actions_train=args.lambda_actions_train, lambda_actions_eval=args.lambda_actions_eval, step_size=args.step_size,
                       train_eps=args.eps_train, eval_eps=args.eps_eval, radius_gw=args.radius_gw, n_ot_train=args.n_ot_train, n_ot_eval=args.n_ot_eval,
                       n_frames=args.n_frames, lr=args.learning_rate, weight_decay=args.weight_decay, rho=args.rho, rho_seg=args.rho_seg, rho_align=args.rho_align, fix_virtual_column=args.fix_virtual_column, input_window=args.input_window,
                       template_slots=args.template_slots, template_rho=args.template_rho,
                       template_eps=args.template_eps,
                       template_warmup_epochs=args.template_warmup_epochs,
                       template_blend=args.template_blend, template_soft=args.template_soft,
                       align_target_blur=args.align_target_blur, template_ema=args.template_ema,
                       template_frozen=args.template_frozen,
                       ray_geometry=args.ray_geometry, ray_mode=args.ray_mode,
                       ray_videos=args.ray_videos, ray_weight=args.ray_weight,
                       ray_radius_weight=args.ray_radius_weight,
                       ray_group_weight=args.ray_group_weight,
                       ray_shell_cost_weight=args.ray_shell_cost_weight,
                       ray_warmup_epochs=args.ray_warmup_epochs, ray_window=args.ray_window,
                       ray_stride=args.ray_stride, ray_cycle_tol=args.ray_cycle_tol,
                       ray_centre_momentum=args.ray_centre_momentum,
                       ray_centre_margin=args.ray_centre_margin,
                       ray_update_centres=not args.no_ray_centre_update, beta=args.beta, exclude_cls=args.exclude, visualize=args.visualize, activity=args.activity,
                       action_bridge_cost_weight=args.action_bridge_cost_weight,
                       action_bridge_loss_weight=args.action_bridge_loss_weight,
                       action_bridge_temp=args.action_bridge_temp,
                       action_bridge_warmup_epochs=args.action_bridge_warmup_epochs,
                       visual_affinity_weight=args.visual_affinity_weight,
                       visual_affinity_stride=args.visual_affinity_stride,
                       visual_affinity_teacher_temp=args.visual_affinity_teacher_temp,
                       match_pseudo_weight=args.match_pseudo_weight,
                       match_raw_topk=args.match_raw_topk,
                       match_cycle_tolerance=args.match_cycle_tolerance,
                       match_warmup_epochs=args.match_warmup_epochs,
                       geometry_weight=args.geometry_weight,
                       geometry_repulsion_weight=args.geometry_repulsion_weight,
                       spectot_tar_weight=args.spectot_tar_weight,
                       soft_consensus_weight=args.soft_consensus_weight,
                       soft_consensus_blend=args.soft_consensus_blend,
                       proposal_videos=args.proposal_videos,
                       arc_reparam=args.arc_reparam, arc_warmup_epochs=args.arc_warmup_epochs,
                       bspline=args.bspline, bspline_tol=args.bspline_tol, bspline_kmax=args.bspline_kmax, bspline_warmup_epochs=args.bspline_warmup_epochs,
                       bspline_space=args.bspline_space, bspline_blend_max=args.bspline_blend_max, bspline_blend_rampup=args.bspline_blend_rampup,
                       latent_dim=args.latent_dim,
                       gauss_smooth=args.gauss_smooth, gauss_sigma=args.gauss_sigma,
                       gauss_blend_max=args.gauss_blend_max, gauss_warmup_epochs=args.gauss_warmup_epochs,
                       bezier=args.bezier, bezier_nseg=args.bezier_nseg, bezier_tol=args.bezier_tol,
                       bezier_maxseg=args.bezier_maxseg, bezier_warmup_epochs=args.bezier_warmup_epochs,
                       bezier_space=args.bezier_space, bezier_blend_max=args.bezier_blend_max,
                       bezier_blend_rampup=args.bezier_blend_rampup)

    cluster_init_info = init_action_prototypes(
        ssl, data_train, args.cluster_init, seed=args.seed)

    logger = None
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger
        run_name = args.wandb_run_name or f"{args.dataset}_{act_str}_ne{args.n_epochs}_seed{args.seed}"
        logger = WandbLogger(project=args.wandb_project, name=run_name, tags=args.wandb_tags,
                             save_dir=os.environ.get('WANDB_DIR', './wandb'), config=vars(args))

    accel = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = 1 if accel == 'gpu' else 'auto'

    from pytorch_lightning.callbacks import ModelCheckpoint
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_subdir,
        filename=f"{run_id}" + '-{epoch}',
        save_last=True,
        every_n_epochs=max(1, args.n_epochs),
    )

    trainer = pl.Trainer(
        devices=devices, accelerator=accel,
        check_val_every_n_epoch=args.val_freq, max_epochs=args.n_epochs,
        log_every_n_steps=50, logger=logger, num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
        default_root_dir=run_dir,
    )

    git_hash = 'unknown'
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        pass

    manifest = {
        'run_id': run_id,
        'dataset': args.dataset,
        'activity': args.activity,
        'seed': args.seed,
        'config': vars(args),
        'run_dir': os.path.abspath(run_dir),
        'git_commit': git_hash,
        'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID', None),
        'accelerator': accel,
        'start_time': datetime.now(timezone.utc).isoformat(),
        'status': 'started',
    }
    manifest_path = os.path.join(run_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    try:
        if not args.eval:
            trainer.validate(ssl, val_loader)
            trainer.fit(ssl, train_loader, val_loader)
        test_results = trainer.test(ssl, dataloaders=test_loader)

        manifest['end_time'] = datetime.now(timezone.utc).isoformat()
        manifest['status'] = 'completed'
        if test_results:
            manifest['test_metrics'] = test_results[0]
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        metrics_path = os.path.join(run_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(test_results[0] if test_results else {}, f, indent=2)

    except Exception as exc:
        manifest['end_time'] = datetime.now(timezone.utc).isoformat()
        manifest['status'] = 'failed'
        manifest['error'] = str(exc)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        raise
