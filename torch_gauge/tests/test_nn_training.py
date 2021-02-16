"""
Test training a minimal SO(3) model on synthetic data
"""

import pytest
import torch

from torch_gauge.nn import IELin, RepNorm1d, Swish
from torch_gauge.o3 import O3Tensor, SphericalTensor
from torch_gauge.o3.clebsch_gordan import CGPCoupler


class mini2d_so3(torch.nn.Module):
    def __init__(self, metadata, n_channels):
        super().__init__()
        self.onebody_ielin = IELin(metadata, metadata, group="so3")
        self.onebody_mlp = torch.nn.Sequential(
            torch.nn.Linear(n_channels, n_channels),
            Swish(),
            torch.nn.Linear(n_channels, n_channels),
        )
        self.onebody_norm = RepNorm1d(num_channels=n_channels)
        self.twobody_linear = torch.nn.Linear(4, 4, bias=False)
        self.twobody_gate = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            Swish(),
        )

    def forward(self, x_1d: SphericalTensor, x_2d: SphericalTensor):
        n, g = self.onebody_norm(x_1d)
        n = self.onebody_mlp(n)
        g = self.onebody_ielin(g)
        x_1d_new = g.scalar_mul(n, inplace=True)
        fold_1d = x_1d_new.fold(stride=4)
        a = (
            x_2d.ten.mul(fold_1d.ten.unsqueeze(1))
            .sum(dim=2)
            .mul(fold_1d.ten)
            .sum(dim=1)
        )
        a = self.twobody_gate(a)
        new_2d_ten = (self.twobody_linear(x_2d.ten)).mul(a.unsqueeze(1).unsqueeze(2))
        x_2d_new = x_2d.self_like(new_2d_ten)
        return x_1d_new, x_2d_new


def test_train_cpu_mini2d_so3():
    torch.manual_seed(42)
    metadata1d = torch.LongTensor([[12, 4, 8, 4, 4]])
    metadata2d = torch.LongTensor([[3, 1, 2, 1, 1], [3, 1, 2, 1, 1]])
    x_1d = SphericalTensor(torch.rand(128, 128), (1,), metadata1d)
    x_2d = SphericalTensor(torch.rand(128, 32, 32, 4), (1, 2), metadata2d)
    # Create synthetic labels
    rs_1d = x_1d.ten.view(128, 32, 4)
    labels = torch.sin(
        (x_2d.ten * rs_1d.unsqueeze(1) * rs_1d.unsqueeze(2))
        .pow(2)
        .mean(dim=(1, 2))
        .sqrt()
    ).sum(1)

    mods = torch.nn.ModuleList(
        [
            mini2d_so3(torch.LongTensor([12, 4, 8, 4, 4]), 32),
            mini2d_so3(torch.LongTensor([12, 4, 8, 4, 4]), 32),
            mini2d_so3(torch.LongTensor([12, 4, 8, 4, 4]), 32),
        ]
    )

    loss_fn = torch.nn.MSELoss()
    mae_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(mods.parameters(), lr=1e-3)

    loss, epoch_mae = 0.0, 0.0
    for epoch in range(50):
        epoch_mae = 0.0
        for iter in range(4):
            batch_idx = torch.randint(128, (32,))
            batch_label = labels[batch_idx]
            x_1d_new = x_1d.self_like(x_1d.ten[batch_idx])
            x_2d_new = x_2d.self_like(x_2d.ten[batch_idx])
            for mod in mods:
                x_1d_new, x_2d_new = mod(x_1d_new, x_2d_new)
            out = x_2d_new.invariant().sum(dim=(1, 2, 3))
            optimizer.zero_grad()
            loss = loss_fn(out, batch_label)
            mae = mae_fn(out, batch_label)
            loss.backward()
            optimizer.step()
            epoch_mae += mae.item()
        print(f"Epoch: {epoch+1}, Loss: {loss}, MAE: {epoch_mae/4}")

    assert loss < 0.01
    assert epoch_mae / 4 < 0.1


class mini2d_o3(torch.nn.Module):
    def __init__(self, metadata, n_channels):
        super().__init__()
        self.onebody_ielin1 = IELin(metadata, metadata, group="o3")
        self.onebody_mlp = torch.nn.Sequential(
            torch.nn.Linear(n_channels, n_channels),
            Swish(),
            torch.nn.Linear(n_channels, n_channels),
        )
        self.onebody_norm = RepNorm1d(num_channels=n_channels, n_invariant_channels=metadata[0])
        self.onebody_coupling = CGPCoupler(metadata, metadata, dtype=torch.float)
        self.onebody_ielin2 = IELin(
            self.onebody_coupling.metadata_out, metadata, group="o3"
        )
        self.twobody_linear = torch.nn.Linear(4, 4, bias=False)
        self.twobody_gate = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            Swish(),
        )

    def forward(self, x_1d: O3Tensor, x_2d: O3Tensor):
        n, g = self.onebody_norm(x_1d)
        n = self.onebody_mlp(n)
        g = self.onebody_ielin1(g)
        g_cgp = self.onebody_coupling(g, g)
        g = self.onebody_ielin2(g_cgp) + g
        x_1d_new = g.scalar_mul(n, inplace=True)
        fold_1d = x_1d_new.fold(stride=4)
        a = (
            x_2d.ten.mul(fold_1d.ten.unsqueeze(1))
            .sum(dim=2)
            .mul(fold_1d.ten)
            .sum(dim=1)
        )
        a = self.twobody_gate(a)
        new_2d_ten = (self.twobody_linear(x_2d.ten)).mul(a.unsqueeze(1).unsqueeze(2))
        x_2d_new = x_2d.self_like(new_2d_ten)
        return x_1d_new, x_2d_new


def test_train_cpu_mini2d_o3():
    torch.manual_seed(42)
    metadata1d = torch.LongTensor([[12, 4, 8, 4, 4, 4]])
    metadata2d = torch.LongTensor([[3, 1, 2, 1, 1, 1], [3, 1, 2, 1, 1, 1]])
    x_1d = O3Tensor(torch.rand(128, 92), (1,), metadata1d)
    x_2d = O3Tensor(torch.rand(128, 23, 23, 4), (1, 2), metadata2d)
    # Create synthetic labels
    rs_1d = x_1d.ten.view(128, 23, 4)
    labels = torch.sin(
        (x_2d.ten * rs_1d.unsqueeze(1) * rs_1d.unsqueeze(2))
        .pow(2)
        .mean(dim=(1, 2))
        .sqrt()
    ).sum(1)

    mods = torch.nn.ModuleList(
        [
            mini2d_o3(torch.LongTensor([12, 4, 8, 4, 4, 4]), 36),
            mini2d_o3(torch.LongTensor([12, 4, 8, 4, 4, 4]), 36),
            mini2d_o3(torch.LongTensor([12, 4, 8, 4, 4, 4]), 36),
        ]
    )

    loss_fn = torch.nn.MSELoss()
    mae_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(mods.parameters(), lr=1e-3)

    loss, epoch_mae = 0.0, 0.0
    for epoch in range(50):
        epoch_mae = 0.0
        for iter in range(4):
            batch_idx = torch.randint(128, (32,))
            batch_label = labels[batch_idx]
            x_1d_new = x_1d.self_like(x_1d.ten[batch_idx])
            x_2d_new = x_2d.self_like(x_2d.ten[batch_idx])
            for mod in mods:
                x_1d_new, x_2d_new = mod(x_1d_new, x_2d_new)
            out = x_2d_new.invariant().sum(dim=(1, 2, 3))
            optimizer.zero_grad()
            loss = loss_fn(out, batch_label)
            mae = mae_fn(out, batch_label)
            loss.backward()
            optimizer.step()
            epoch_mae += mae.item()
        print(f"Epoch: {epoch+1}, Loss: {loss}, MAE: {epoch_mae/4}")

    assert loss < 0.02
    assert epoch_mae / 4 < 0.1


def test_train_cuda_mini2d_o3():
    if not torch.cuda.is_available():
        pytest.skip()
    device = torch.device("cuda")
    torch.manual_seed(42)
    metadata1d = torch.LongTensor([[12, 4, 8, 4, 4, 4]])
    metadata2d = torch.LongTensor([[3, 1, 2, 1, 1, 1], [3, 1, 2, 1, 1, 1]])
    x_1d = O3Tensor(torch.rand(128, 92), (1,), metadata1d).to(device)
    x_2d = O3Tensor(torch.rand(128, 23, 23, 4), (1, 2), metadata2d).to(device)
    # Create synthetic labels
    rs_1d = x_1d.ten.view(128, 23, 4)
    labels = torch.sin(
        (x_2d.ten * rs_1d.unsqueeze(1) * rs_1d.unsqueeze(2))
        .pow(2)
        .mean(dim=(1, 2))
        .sqrt()
    ).sum(1)

    mods = torch.nn.ModuleList(
        [
            mini2d_o3(torch.LongTensor([12, 4, 8, 4, 4, 4]), 36),
            mini2d_o3(torch.LongTensor([12, 4, 8, 4, 4, 4]), 36),
            mini2d_o3(torch.LongTensor([12, 4, 8, 4, 4, 4]), 36),
        ]
    ).to(device)

    loss_fn = torch.nn.MSELoss()
    mae_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(mods.parameters(), lr=1e-3)

    loss, epoch_mae = 0.0, 0.0
    for epoch in range(50):
        epoch_mae = 0.0
        for iter in range(4):
            batch_idx = torch.randint(128, (32,))
            batch_label = labels[batch_idx]
            x_1d_new = x_1d.self_like(x_1d.ten[batch_idx])
            x_2d_new = x_2d.self_like(x_2d.ten[batch_idx])
            for mod in mods:
                x_1d_new, x_2d_new = mod(x_1d_new, x_2d_new)
            out = x_2d_new.invariant().sum(dim=(1, 2, 3))
            optimizer.zero_grad()
            loss = loss_fn(out, batch_label)
            mae = mae_fn(out, batch_label)
            loss.backward()
            optimizer.step()
            epoch_mae += mae.cpu().item()
        print(f"Epoch: {epoch+1}, Loss: {loss.cpu().item()}, MAE: {epoch_mae/4}")

    assert loss < 0.01
    assert epoch_mae / 4 < 0.1
