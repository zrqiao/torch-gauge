import torch

from torch_gauge.verlet_list import VerletList


# The i-j inversion should return a transposed matrix when the graph is dense
def test_verletlist_inversion_dense():
    data = torch.rand(17, 17, 8, 6)
    mask = torch.ones(17, 17, dtype=torch.bool)
    vl_test = VerletList().from_mask(mask, 17, 17, {}, {"test_edata": data})
    inversed = vl_test.to_src_first_view(vl_test.edata["test_edata"])
    assert torch.all(data.permute(1, 0, 2, 3).eq(inversed))


def test_verletlist_inversion_sparse():
    data = torch.rand(17, 17, 8, 6)
    mask_pre = torch.rand(17, 17)
    mask = (mask_pre + mask_pre.t()) > 0.5
    vl_test = VerletList().from_mask(mask, 32, 17, {}, {"test_edata": data})
    assert torch.all(
        data[mask, ...].eq(vl_test.edata["test_edata"][vl_test.edge_mask, ...])
    )

    inversed = vl_test.to_src_first_view(vl_test.edata["test_edata"])
    sparse1 = torch.zeros_like(data)
    sparse1[mask, ...] = data[mask, ...]
    sparse2 = torch.zeros_like(data)
    sparse2[mask, ...] = inversed[vl_test.edge_mask, ...]
    assert torch.all(sparse1.permute(1, 0, 2, 3).eq(sparse2))


# When the graph is dense, query_src inserts a dimension
def test_verletlist_querysrc_dense():
    data = torch.rand(17, 8, 6)
    mask = torch.ones(17, 17, dtype=torch.bool)
    vl_test = VerletList().from_mask(mask, 17, 17, {"test_ndata": data}, {})
    src_data = vl_test.query_src(vl_test.ndata["test_ndata"])
    assert torch.all(data.unsqueeze(0).expand(17, 17, 8, 6).eq(src_data))


if __name__ == "__main__":
    test_verletlist_inversion_dense()
    test_verletlist_querysrc_dense()
