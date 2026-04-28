"""
STG-NF model, based on awesome previous work by https://github.com/y0ast/Glow-PyTorch
"""

import math
import torch
import torch.nn as nn

from predictors.STG_NF.modules_pose import (
    Conv2d,
    Conv2dZeros,
    ActNorm2d,
    InvertibleConv1x1,
    Permute2d,
    SqueezeLayer,
    Split2d,
    gaussian_likelihood,
    gaussian_sample,
)
from predictors.STG_NF.utils import split_feature
from predictors.STG_NF.graph import Graph
from predictors.STG_NF.stgcn import st_gcn

def nan_throw(tensor, name="tensor"):
    stop = False
    if ((tensor != tensor).any()):
        print(name + " has nans")
        stop = True
    if (torch.isinf(tensor).any()):
        print(name + " has infs")
        stop = True
    if stop:
        print(name + ": " + str(tensor))


def get_stgcn(in_channels, hidden_channels, out_channels,
              temporal_kernel_size=9, spatial_kernel_size=2, first=False):
    kernel_size = (temporal_kernel_size, spatial_kernel_size)
    if hidden_channels == 0:
        block = nn.ModuleList((
            st_gcn(in_channels, out_channels, kernel_size, 1, residual=(not first)),
        ))
    else:
        block = nn.ModuleList((
            st_gcn(in_channels, hidden_channels, kernel_size, 1, residual=(not first)),
            st_gcn(hidden_channels, out_channels, kernel_size, 1, residual=(not first)),
        ))

    return block


def get_block(in_channels, out_channels, hidden_channels):
    block = nn.Sequential(
        Conv2d(in_channels, hidden_channels),
        nn.ReLU(inplace=False),
        Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(inplace=False),
        Conv2dZeros(hidden_channels, out_channels),
    )
    return block


class FlowStep(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            actnorm_scale,
            flow_permutation,
            flow_coupling,
            LU_decomposed,
            A=None,
            temporal_kernel_size=4,
            edge_importance_weighting=False,
            last=False,
            first=False,
            strategy='uniform',
            max_hops=8,
            device='cuda:0'
    ):
        super().__init__()
        self.device = device
        self.flow_coupling = flow_coupling
        if A is None:
            g = Graph(strategy=strategy, max_hop=max_hops)
            self.A = torch.from_numpy(g.A).float().to(device)

        self.actnorm = ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (
                self.shuffle(z, rev),
                logdet,
            )
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (
                self.reverse(z, rev),
                logdet,
            )

        # 3. coupling
        if flow_coupling == "additive":
            self.block = get_stgcn(in_channels // 2, in_channels // 2, hidden_channels,
                                   temporal_kernel_size=temporal_kernel_size, spatial_kernel_size=self.A.size(0),
                                   first=first
                                   )
        elif flow_coupling == "affine":
            self.block = get_stgcn(in_channels // 2, hidden_channels, in_channels,
                                   temporal_kernel_size=temporal_kernel_size, spatial_kernel_size=self.A.size(0),
                                   first=first)

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.block
            ])
        else:
            self.edge_importance = [1] * len(self.block)

    def forward(self, input, logdet=None, reverse=False, label=None):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):

        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)

        # 3. coupling
        z1, z2 = split_feature(z, "split")
        if self.flow_coupling == "additive":
            z2 = z2 + self.block(z1)
        elif self.flow_coupling == "affine":
            if len(z1.shape) == 3:
                z1 = z1.unsqueeze(dim=1)
            if len(z2.shape) == 3:
                z2 = z2.unsqueeze(dim=1)
            h = z1.clone()
            for gcn, importance in zip(self.block, self.edge_importance):
                # h = gcn(h)
                h, _ = gcn(h, self.A * importance)
            shift, scale = split_feature(h, "cross")
            if len(scale.shape) == 3:
                scale = scale.unsqueeze(dim=1)
            if len(shift.shape) == 3:
                shift = shift.unsqueeze(dim=1)
            scale = torch.sigmoid(scale + 2.) + 1e-6
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1.coupling
        z1, z2 = split_feature(input, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            if len(z1.shape) == 3:
                z1 = z1.unsqueeze(dim=1)
            if len(z2.shape) == 3:
                z2 = z2.unsqueeze(dim=1)
            h = z1.clone()
            for gcn, importance in zip(self.block, self.edge_importance):
                # h = gcn(h)
                h, _ = gcn(h, self.A * importance)
            # h = self.block(z1)
            shift, scale = split_feature(h, "cross")
            if len(scale.shape) == 3:
                scale = scale.unsqueeze(dim=1)
            if len(shift.shape) == 3:
                shift = shift.unsqueeze(dim=1)
            scale = torch.sigmoid(scale + 2.0) + 1e-6
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


class FlowNet(nn.Module):
    def __init__(
            self,
            pose_shape,
            hidden_channels,
            K, # default 8
            L, # default 1
            actnorm_scale,
            flow_permutation,
            flow_coupling,
            LU_decomposed,
            edge_importance=False,
            temporal_kernel_size=None,
            strategy='uniform',
            max_hops=8,
            device='cuda:0',
    ):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList()
        self.output_shapes = []

        self.K = K

        C, T, V = pose_shape
        for i in range(L):
            if i > 1:
                # 1. Squeeze
                C, T, V = C * 2, T // 2, V
                self.layers.append(SqueezeLayer(factor=2))
                self.output_shapes.append([-1, C, T, V])
            if temporal_kernel_size is None:
                temporal_kernel_size = T // 2 + 1
            # 2. K FlowStep
            for k in range(K):
                last = (k == K - 1)
                first = (k == 0)
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                        temporal_kernel_size=temporal_kernel_size,
                        edge_importance_weighting=edge_importance,
                        last=last,
                        first=first,
                        strategy=strategy,
                        max_hops=max_hops,
                        device=device,
                    )
                )
                self.output_shapes.append([-1, C, T, V])

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, z, logdet=0.0):
        logdet = torch.zeros(z.shape[0]).to(self.device)
        for i, (layer, shape) in enumerate(zip(self.layers, self.output_shapes)):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, logdet = layer(z, logdet=0, reverse=True, temperature=temperature)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z


class STG_NF(nn.Module):
    def __init__(
            self,
            pose_shape,
            hidden_channels,
            K,
            L,
            actnorm_scale,
            flow_permutation,
            flow_coupling,
            LU_decomposed,
            learn_top,
            R=0,
            edge_importance=False,
            temporal_kernel_size=None,
            strategy='uniform',
            max_hops=8,
            device='cuda:0'
    ):
        super().__init__()
        self.flow = FlowNet(
            pose_shape=pose_shape,
            hidden_channels=hidden_channels,
            K=K,
            L=L,
            actnorm_scale=actnorm_scale,
            flow_permutation=flow_permutation,
            flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed,
            edge_importance=edge_importance,
            temporal_kernel_size=temporal_kernel_size,
            strategy=strategy,
            max_hops=max_hops,
            device=device,
        )
        self.R = R
        self.learn_top = learn_top

        # learned prior
        if learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        print("self.flow.output_shapes", self.flow.output_shapes) # [[-1, 2, 16, 18], [-1, 2, 16, 18], [-1, 2, 16, 18], [-1, 2, 16, 18], [-1, 2, 16, 18], [-1, 2, 16, 18], [-1, 2, 16, 18], [-1, 2, 16, 18]]

        self.register_buffer(
            "prior_h",
            torch.zeros(
                [
                    1,
                    self.flow.output_shapes[-1][1] * 2,
                    self.flow.output_shapes[-1][2],
                    self.flow.output_shapes[-1][3],
                ]
            ),
        )
        self.register_buffer(
            "prior_h_normal",
            torch.concat(
                (
                    torch.ones([self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2],
                                self.flow.output_shapes[-1][3]]) * self.R,

                    torch.zeros([self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2],
                                 self.flow.output_shapes[-1][3]]),
                ), dim=0
            ))
        self.register_buffer(
            "prior_h_abnormal",
            torch.concat(
                (
                    torch.ones([self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2],
                                self.flow.output_shapes[-1][3]]) * self.R * -1,

                    torch.zeros([self.flow.output_shapes[-1][1], self.flow.output_shapes[-1][2],
                                 self.flow.output_shapes[-1][3]]),
                ), dim=0
            ))

    def prior(self, data, label=None):
        # self.prior_h torch.Size([1, 4, 16, 18]) the 2 first are the means and the 2 last are the log_stds
        # self.prior_h_normal torch.Size([4, 16, 18]) the 2 first are the means (self.R) and the 2 last are the log_stds (0s)
        # self.prior_h_abnormal torch.Size([4, 16, 18]) the 2 first are the means (self.R * -1) and the 2 last are the log_stds (0s)
        if data is not None:
            if label is not None:
                h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
                h[label == 1] = self.prior_h_normal
                h[label == -1] = self.prior_h_abnormal
            else:
                h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            # Hardcoded a batch size of 32 here
            h = self.prior_h_normal.repeat(32, 1, 1, 1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        return split_feature(h, "split")

    def forward(self, x=None, z=None, temperature=None, reverse=False, label=None, score=1, return_z=False):
        if reverse:
            return self.reverse_flow(z, temperature)
        else:
            # print("x", x.shape) # torch.Size([256, 2, 16, 18])
            # print("label", label.shape) # torch.Size([256])
            # print("score", score.shape) # torch.Size([256])
            return self.normal_flow(x, label, score, return_z)

    def normal_flow(self, x, label, score, return_z = False):
        b, c, t, v = x.shape

        z, objective = self.flow(x, reverse=False)
        # print("z", z.shape) # torch.Size([256, 2, 16, 18])
        # print("objective", objective.shape) # the objective is of shape (B,) and each element is actually the sum of the log|det(df_i/df_i-1)| for each flow step i

        mean, logs = self.prior(x, label)
        objective += gaussian_likelihood(mean, logs, z) # then we sum the objective with the gaussian_likelihood log(p_z(f(x))) with z = f(x)
        
        # all of this gives us the objective = log(p_X(x)) = log(p_Z(f(x))) + log|det(df_i/df_i-1)| the log likelihood of the data x passed throught the model wrt to the prior (that is diffrent depending on the label)

        # NB log(p_X(x)) is the likelihood of x being in distribution X (normal)
        
        # Full objective - converted to bits per dimension
        # converted to nll
        nll = (-objective) / (math.log(2.0) * c * t * v)
        
        if return_z:
            return z, nll
        else:
            return nll

    def reverse_flow(self, z, temperature):
        with torch.no_grad():
            if z is None:
                mean, logs = self.prior(z)
                z = gaussian_sample(mean, logs, temperature)
            x = self.flow(z, temperature=temperature, reverse=True)
        return x

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True
