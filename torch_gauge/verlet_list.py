from dataclasses import dataclass
from typing import List

import torch

from torch_gauge.o3.spherical import SphericalTensor


@dataclass
class VerletList:
    def __init__(self):
        """
        Assuming in-edges and out-edges are identical (undirected graph)
        WARNING: the current Verlet-list scheme will break vanilla edge BatchNorm
        """
        self.neighbor_idx = None
        self.ndata = {}
        self.edata = {}
        self.edge_mask = None
        self.n_nodes = None
        self.PADSIZE = None
        self.batch_num_nodes = None
        self._dst_edim_locators = None

    def from_mask(
        self,
        verlet_mask: torch.BoolTensor,
        padding_size,
        num_nodes,
        one_body_data,
        two_body_data,
    ):
        self.PADSIZE = padding_size
        self.n_nodes = num_nodes
        self.batch_num_nodes = torch.LongTensor([self.n_nodes])

        in_degrees = verlet_mask.long().sum(1)
        src_raw = (
            torch.arange(num_nodes, dtype=torch.long)
            .unsqueeze(0)
            .expand(num_nodes, num_nodes)[verlet_mask]
        )

        # Scatter the src node-ids to the (N*PADDING_SIZE) padded table
        src_locators_1d = torch.LongTensor(
            [
                dstid * self.PADSIZE + src_location
                for dstid in range(self.n_nodes)
                for src_location in range(in_degrees[dstid])
            ]
        )
        src_edim_locators_flattened = torch.LongTensor(
            [
                src_location
                for dstid in range(self.n_nodes)
                for src_location in range(in_degrees[dstid])
            ]
        )
        src_edim_locators = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
        src_edim_locators[verlet_mask] = src_edim_locators_flattened

        self.neighbor_idx = (
            torch.zeros(self.n_nodes * self.PADSIZE, dtype=torch.long)
            .scatter_(
                dim=0,
                index=src_locators_1d,
                src=src_raw,
            )
            .view(self.n_nodes, self.PADSIZE)
        )

        self.edge_mask = (
            torch.zeros(self.n_nodes * self.PADSIZE, dtype=torch.bool)
            .scatter_(
                dim=0,
                index=src_locators_1d,
                src=torch.ones(src_raw.size(0), dtype=torch.bool),
            )
            .view(self.n_nodes, self.PADSIZE)
        )

        self._dst_edim_locators = (
            torch.zeros(self.n_nodes, self.PADSIZE, dtype=torch.long)
            .masked_scatter_(
                mask=self.edge_mask,
                source=src_edim_locators.t()[verlet_mask],
            )
            .view(self.n_nodes, self.PADSIZE)
        )

        self.ndata = one_body_data
        # edata can be understood as node data with additional "neighborhood" dimensions
        # To make padding easier, always unsqueeze at least one trailing dimension
        self.edata = {
            k: self._scatter_efeat(edata, verlet_mask, src_raw)
            for k, edata in two_body_data.items()
        }

        return self

    def from_dgl(self, g: "dgl.DGLGraph", padding_size, nkeys, ekeys):
        """
        The interface for generating Verlet-list from dgl.DGLGraph
        Only scalar type tensors are supported
        """
        self.PADSIZE = padding_size
        self.n_nodes = g.num_nodes()
        self.batch_num_nodes = torch.LongTensor([self.n_nodes])

        src_raw, dst_raw, eid_raw = g.in_edges(torch.arange(self.n_nodes), form="all")
        in_degrees = g.in_degrees(torch.arange(self.n_nodes))

        # Scatter the src node-ids to the (N*PADDING_SIZE) padded table
        src_locators_1d = torch.LongTensor(
            [
                dstid * self.PADSIZE + src_location
                for dstid in range(self.n_nodes)
                for src_location in range(in_degrees[dstid])
            ]
        )
        src_edim_locators = torch.LongTensor(
            [
                src_location
                for dstid in range(self.n_nodes)
                for src_location in range(in_degrees[dstid])
            ]
        )
        g.edata["dst_edim_locators"] = torch.zeros_like(src_edim_locators)
        g.edata["dst_edim_locators"][g.edge_ids(src_raw, dst_raw)] = src_edim_locators
        self._dst_edim_locators = (
            torch.zeros(self.n_nodes * self.PADSIZE, dtype=src_raw.dtype)
            .scatter_(
                dim=0,
                index=src_locators_1d,
                src=g.edata["dst_edim_locators"][eid_raw],
            )
            .view(self.n_nodes, self.PADSIZE)
        )

        self.neighbor_idx = (
            torch.zeros(self.n_nodes * self.PADSIZE, dtype=src_raw.dtype)
            .scatter_(
                dim=0,
                index=src_locators_1d,
                src=src_raw,
            )
            .view(self.n_nodes, self.PADSIZE)
        )

        self.edge_mask = (
            torch.zeros(self.n_nodes * self.PADSIZE, dtype=torch.bool)
            .scatter_(
                dim=0,
                index=src_locators_1d,
                src=torch.ones(eid_raw.size(0), dtype=torch.bool),
            )
            .view(self.n_nodes, self.PADSIZE)
        )

        self.ndata = {k: g.ndata[k] for k in nkeys}
        self.edata = {
            k: torch.zeros(
                self.n_nodes * self.PADSIZE,
                g.edata[k][0, ...].numel(),
                dtype=g.edata[k].dtype,
            )
            .scatter_(
                dim=0,
                index=src_locators_1d.unsqueeze(1).expand(
                    -1, g.edata[k][0, ...].numel()
                ),
                src=g.edata[k][eid_raw, ...].view(g.edata[k].shape[0], -1),
            )
            .view(self.n_nodes, self.PADSIZE, *g.edata[k].shape[1:])
            for k in ekeys
        }
        return self

    def _scatter_efeat(self, edata, verlet_mask, src_raw):
        if isinstance(edata, torch.Tensor):
            return (
                torch.zeros(
                    self.n_nodes,
                    self.PADSIZE,
                    *edata.shape[2:],
                    dtype=edata.dtype,
                )
                .view(self.n_nodes, self.PADSIZE, -1)
                .masked_scatter_(
                    mask=self.edge_mask.unsqueeze(-1),
                    source=edata[verlet_mask, ...].view(src_raw.shape[0], -1),
                )
                .view(self.n_nodes, self.PADSIZE, *edata.shape[2:])
            )
        elif isinstance(edata, SphericalTensor):
            out_ten = (
                torch.zeros(
                    self.n_nodes,
                    self.PADSIZE,
                    *edata.shape[2:],
                    dtype=edata.ten.dtype,
                )
                .view(self.n_nodes, self.PADSIZE, -1)
                .masked_scatter_(
                    mask=self.edge_mask.unsqueeze(-1),
                    source=edata.ten[verlet_mask, ...].view(src_raw.shape[0], -1),
                )
                .view(self.n_nodes, self.PADSIZE, *edata.shape[2:])
            )
            return SphericalTensor(
                out_ten,
                rep_dims=edata.rep_dims,
                metadata=edata.metadata,
                rep_layout=edata.rep_layout,
                num_channels=edata.num_channels,
            )

    def query_src(self, src_feat):
        """
        Returns the src-node data scattered into the neighbor-list frame.
        When applied to an edge-data tensor, this function generates a higher-order
         view (k+1 hop) of the underlying k-hop graph structure.
        src_feat must be contiguous.
        """
        flattened_neighboridx = self.neighbor_idx.view(-1)
        if isinstance(src_feat, torch.Tensor):
            flattened2d_src = src_feat.view(src_feat.shape[0], -1)
            flattened_out = (
                flattened2d_src[flattened_neighboridx, ...]
                .view(*self.neighbor_idx.shape, flattened2d_src.shape[1])
                .mul_(self.edge_mask.unsqueeze(2))
            )
            return flattened_out.view(*self.neighbor_idx.shape, *src_feat.shape[1:])
        elif isinstance(src_feat, SphericalTensor):
            flattened2d_src = src_feat.ten.view(src_feat.ten.shape[0], -1)
            flattened_out = (
                flattened2d_src[flattened_neighboridx, ...]
                .view(*self.neighbor_idx.shape, flattened2d_src.shape[1])
                .mul_(self.edge_mask.unsqueeze(2))
            )
            return SphericalTensor(
                flattened_out.view(*self.neighbor_idx.shape, *src_feat.shape[1:]),
                rep_dims=tuple(dim + 1 for dim in src_feat.rep_dims),
                metadata=src_feat.metadata,
                rep_layout=src_feat.rep_layout,
                num_channels=src_feat.num_channels,
            )

    def to_src_first_view(self, data):
        """
        Flipping src / dst node indexing order without inverting the data
        """
        scatter_ten = (self.neighbor_idx * self.PADSIZE + self._dst_edim_locators)[
            self.edge_mask
        ]
        if isinstance(data, torch.Tensor):
            out_ten = torch.zeros_like(data).view(self.n_nodes * self.PADSIZE, -1)
            out_ten[scatter_ten, :] = data.view(self.n_nodes, self.PADSIZE, -1)[
                self.edge_mask, :
            ]
            return out_ten.view_as(data)
        elif isinstance(data, SphericalTensor):
            out_ten = torch.zeros_like(data.ten).view(self.n_nodes * self.PADSIZE, -1)
            out_ten[scatter_ten, :] = data.ten.view(self.n_nodes, self.PADSIZE, -1)[
                self.edge_mask, :
            ]
            return SphericalTensor(
                out_ten.view_as(data.ten),
                rep_dims=data.rep_dims,
                metadata=data.metadata,
                rep_layout=data.rep_layout,
                num_channels=data.num_channels,
            )
        else:
            raise NotImplementedError

    def to(self, device):
        self.neighbor_idx = self.neighbor_idx.to(device)
        self.ndata = {k: v.to(device) for k, v in self.ndata.items()}
        self.edata = {k: v.to(device) for k, v in self.edata.items()}
        self.edge_mask = self.edge_mask.to(device)
        self._dst_edim_locators = self._dst_edim_locators.to(device)
        self.batch_num_nodes = self.batch_num_nodes.to(device)
        return self

    @staticmethod
    def batch(vls: List["VerletList"]):
        """
        WARNING: In the current version, taking batch of batches will
         break the offset indices
        """
        batched_vl = VerletList()
        batched_vl.PADSIZE = vls[0].PADSIZE
        batched_vl.batch_num_nodes = torch.cat([vl.batch_num_nodes for vl in vls])
        batched_vl.n_nodes = torch.sum(batched_vl.batch_num_nodes)
        bnn_offsets = torch.repeat_interleave(
            torch.cat(
                [torch.LongTensor([0]), torch.cumsum(batched_vl.batch_num_nodes, dim=0)]
            )[:-1],
            batched_vl.batch_num_nodes,
        )
        batched_vl.neighbor_idx = torch.cat(
            [vl.neighbor_idx for vl in vls], dim=0
        ) + bnn_offsets.unsqueeze(1)
        batched_vl.edge_mask = torch.cat([vl.edge_mask for vl in vls], dim=0)
        batched_vl.ndata = {
            nk: torch.cat([vl.ndata[nk] for vl in vls], dim=0)
            for nk in vls[0].ndata.keys()
        }
        batched_vl.edata = {
            ek: torch.cat([vl.edata[ek] for vl in vls], dim=0)
            for ek in vls[0].edata.keys()
        }
        batched_vl._dst_edim_locators = torch.cat(
            [vl._dst_edim_locators for vl in vls], dim=0
        )

        return batched_vl
