"""
Test training a minimal SO(3) model on synthetic data
"""

import torch

from torch_gauge.nn import IELin, RepNorm1d, Swish
from torch_gauge.o3.spherical import SphericalTensor


class mini2d(torch.nn.Module):
    def __init__(self, metadata, n_channels):
        super().__init__()
        self.onebody_ielin = IELin(metadata, metadata)
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


def test_train_mini2d():
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
            mini2d(torch.LongTensor([12, 4, 8, 4, 4]), 32),
            mini2d(torch.LongTensor([12, 4, 8, 4, 4]), 32),
            mini2d(torch.LongTensor([12, 4, 8, 4, 4]), 32),
        ]
    )

    loss_fn = torch.nn.MSELoss()
    mae_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(mods.parameters(), lr=1e-3)

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
        print(f"Epoch: {epoch+1}, Loss: {loss}, MAE: {epoch_mae/32}")

    assert loss < 0.01
    assert mae < 0.1


if __name__ == "__main__":
    test_train_mini2d()
