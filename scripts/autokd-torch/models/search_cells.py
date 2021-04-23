""" CNN cell for architecture search """
import torch
import torch.nn as nn
from models import ops


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, n_nodes, C_pp, C_p, C):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
        """
        super().__init__()
        self.n_nodes = n_nodes

        # # If previous cell is reduction cell, current input size does not match with
        # # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        # self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        # self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)

        # generate dag in the stacked cell
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2+i):  # include 2 input nodes
                op = ops.MixedOp(C, C, stride=1)
                self.dag[i].append(op)

    def forward(self, s0, s1, w_dag, att_inner_nodes):
        # s0 = self.preproc0(s0)
        # s1 = self.preproc1(s1)

        inner_node_states = [s0, s1]
        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(inner_node_states, w_list)))
            inner_node_states.append(s_cur)

        # s_out = torch.cat(states[2:], dim=1)

        # vanilla version to blend inner states: averaging
        s1 = inner_node_states[-1]
        s_out = torch.zeros(s1.shape, dtype=s1.dtype, device=s1.device)
        for i, s in enumerate(inner_node_states[2:]):
            s_out += att_inner_nodes[i] * s
        # return s_out / self.n_nodes
        return s_out

