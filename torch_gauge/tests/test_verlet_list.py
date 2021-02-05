import torch

from torch_gauge import VerletList
from torch_gauge.o3 import O3Tensor


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


def test_verletlist_o3():
    vls = []
    for _ in range(4):
        dten1 = torch.rand(17, 101, 7)
        metadata = torch.LongTensor([[3, 4, 5, 4, 1, 0, 2, 3, 2, 1]])
        test1 = O3Tensor(dten1, (1,), metadata)
        dten2 = torch.rand(17, 17, 101, 101, 7)
        test2 = O3Tensor(dten2, (2, 3), metadata.repeat(2, 1))
        mask_pre = torch.rand(17, 17)
        mask = (mask_pre + mask_pre.t()) > 0.5
        vl_test = VerletList().from_mask(
            mask, 17, 17, {"test_ndata": test1}, {"test_edata": test2}
        )
        vls.append(vl_test)
    batched_vls = VerletList.batch(vls)
    assert batched_vls.ndata["test_ndata"].rep_dims == (1,)
    assert batched_vls.ndata["test_ndata"].ten.shape == (17 * 4, 101, 7)
    assert batched_vls.edata["test_edata"].rep_dims == (
        2,
        3,
    )
    assert batched_vls.edata["test_edata"].ten.shape == (17 * 4, 17, 101, 101, 7)
    expanded_n = batched_vls.query_src(batched_vls.ndata["test_ndata"])
    expanded_e = batched_vls.query_src(batched_vls.edata["test_edata"])
    assert expanded_n.rep_dims == (2,)
    assert expanded_n.ten.shape == (17 * 4, 17, 101, 7)
    assert expanded_e.rep_dims == (3, 4)
    assert expanded_e.ten.shape == (17 * 4, 17, 17, 101, 101, 7)
    inverted_e = batched_vls.to_src_first_view(batched_vls.edata["test_edata"])
    assert inverted_e.rep_dims == (2, 3)
    assert inverted_e.ten.shape == (17 * 4, 17, 101, 101, 7)


if __name__ == "__main__":
    test_verletlist_inversion_dense()
    test_verletlist_querysrc_dense()
