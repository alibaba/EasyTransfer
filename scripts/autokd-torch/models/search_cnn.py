""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.search_cells import SearchCell
import genotypes as gt
from torch.nn.parallel._functions import Broadcast
import logging
from . import multi_head_attention
from models import ops

# Tags for ablation test
USING_2D_INPUT = False
CELL_WEIGHTS_SHARED = False
OUT_AVE_LAYER = True
DEBUG_EMB = False
FIXED_STRUCTURE = False


def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class SearchCNN(nn.Module):
    """ Search CNN model """
    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=3, stem_multiplier=4,
                 emb_size=768, hid_size=128, num_heads=None, droprate=0.2):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell stem_multiplier
            emb_size: the input embedding size
            hid_size: the projected embedding size
            stem_multiplier: used for 2D input, adapt to the input/output shape
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.max_layers = n_layers

        # stem_multiplier = 4 if using_2d_input else 1
        C_cur = stem_multiplier * C
        C_cur_out = 1

        hid_size = 768 if DEBUG_EMB else hid_size

        self.dropout = nn.Dropout(droprate)
        self.projected_emb_linear = nn.Linear(emb_size, hid_size)
        self.layer_norm = nn.LayerNorm(hid_size)
        self.cells = nn.ModuleList()

        self.cell_wise_blend_weights = nn.Parameter(1e-2 * torch.randn(1, n_nodes)) if CELL_WEIGHTS_SHARED \
            else nn.Parameter(1e-2 * torch.randn(self.max_layers, n_nodes))

        if OUT_AVE_LAYER:
            self.layer_wise_blend_weights = nn.ParameterList()
            for i in range(self.max_layers + 1):  # including the embedding layer
                layer_wise_blend_weights_i_layer = nn.Parameter(1e-2*torch.randn(i+1))
                self.layer_wise_blend_weights.append(layer_wise_blend_weights_i_layer)

        if USING_2D_INPUT:
            for _ in range(self.max_layers + 1):
                cell = SearchCell(n_nodes, C_pp, C_p, C_cur)
                self.cells.append(cell)
                C_cur_out = C_cur * n_nodes
                C_pp, C_p = C_p, C_cur_out
        elif CELL_WEIGHTS_SHARED:
            # using only one cell, whose parameter weights (e.g., convolutional parameters) are shared for all layers
            cell = SearchCell(n_nodes, hid_size, hid_size, hid_size)
            self.cells.append(cell)
        elif FIXED_STRUCTURE:
            # test for fixed structure, simple KIM-CNN
            windows = [1, 3, 5]
            for win in windows:
                # to make the input size as the same as the output size
                dilation = 2 if win % 2 == 0 else 1
                cell = ops.TextConv(hid_size, hid_size, win, dilation=dilation)
                self.cells.append(cell)
            C_cur_out = len(windows)
        else:  # every layers use different parameters
            for _ in range(self.max_layers):
                cell = SearchCell(n_nodes, hid_size, hid_size, hid_size)
                self.cells.append(cell)

        num_heads = hid_size / 16 if num_heads is None else hid_size
        embed_attention_size = hid_size if not USING_2D_INPUT else C_cur_out * hid_size
        self.self_attn = multi_head_attention.MultiheadAttention(embed_attention_size, int(num_heads), dropout=0.2)

        # probes for teacher and the student
        bert_hidden = 768
        bert_layer = 12
        self.teacher_probes = nn.ModuleList()
        teacher_probe_emb_input = nn.Linear(bert_hidden, n_classes)
        self.teacher_probes.append(teacher_probe_emb_input)
        self.probes = nn.ModuleList()
        probe_emb_input = nn.Linear(hid_size, n_classes)
        self.probes.append(probe_emb_input)
        min_probe_num = min(n_layers-1, bert_layer-1)
        for _ in range(min_probe_num):
            teacher_probe = nn.Linear(bert_hidden, n_classes)
            self.teacher_probes.append(teacher_probe)
            probe = nn.Linear(hid_size, n_classes)
            self.probes.append(probe)

        self.classifiers = nn.ModuleList()
        # train N classifers for the varied, searched layer n
        for i in range(self.max_layers):
            classifier = nn.Linear(C_cur_out * hid_size, n_classes) if OUT_AVE_LAYER \
                else nn.Linear((i+1) * C_cur_out * hid_size, n_classes)  # concatenation
            self.classifiers.append(classifier)
        # self.classifier_simple = nn.Linear(hid_size, n_classes)

    # un-used in the submission methodology, not work in preliminary experiments, leaving as future work
    def self_att_transform1d(self, x_in):
        # input x_in shape: [batch_size, hid_size, seq_length]
        # MultiheadAttention input shape: [seq_length, batch_size, embedding_size]
        x_att = x_in.permute(2, 0, 1)
        x_att, _ = self.self_attn(x_att, x_att, x_att)
        x_att = x_att.permute(1, 2, 0)
        return x_att

    # un-used in the submission methodology part, not work in preliminary experiments, leaving as future work
    def self_att_transform2d(self, x_in):
        # input x_in shape: [batch_size, input_channel, seq_length, hid_size]
        # to use nn.MultiheadAtt, transform into x_att, whose shape: [seq_length, batch_size, input_channel, hid_size]
        x_att = x_in.permute(2, 0, 1, 3)
        channel = x_att.size(2)
        # MultiheadAttention input shape: [seq_length, batch_size, embedding_size]
        x_att = x_att.reshape(x_att.size(0), x_att.size(1), -1)
        x_att, _ = self.self_attn(x_att, x_att, x_att)
        # restore the x_att indices as the same as x_in
        x_att = x_att.reshape(x_att.size(0), x_att.size(1), channel, -1)
        x_att = x_att.permute(1, 2, 0, 3)
        return x_att

    # def forward(self, x_pair, sampled_weights_normal, sampled_weights_layer):
    def forward(self, x_pair, sampled_weights_normal):
        x1, x2 = x_pair
        hid_states = []
        # return self.classifier_simple(x.mean(2).squeeze()), hid_states
        # input x_in_2d shape: [batch_size, input_channel, seq_length, hid_size]
        if USING_2D_INPUT:
            x1 = x1 if DEBUG_EMB else self.projected_emb_linear(x1)
            x1 = self.layer_norm(x1)
            x1 = self.dropout(x1)
            hid_states.append(x1)
            x1 = self.stem(x1)
            # s0 = s1 = self.self_att_transform2d(x1)
        else:
            # input x_in_1d shape: [batch_size, hid_size, seq_length]
            x1 = x1.permute(0, 2, 1) if DEBUG_EMB else self.projected_emb_linear(x1.permute(0, 2, 1))
            x1 = self.layer_norm(x1)
            s0 = self.dropout(x1).permute(0, 2, 1)
            x2 = x2.permute(0, 2, 1) if DEBUG_EMB else self.projected_emb_linear(x2.permute(0, 2, 1))
            x2 = self.layer_norm(x2)
            s1 = self.dropout(x2).permute(0, 2, 1)
            hid_states.append((s0+s1)/2)  # embedding layer
            # s1 = self.self_att_transform1d(s0)

        w_cell = F.softmax(self.cell_wise_blend_weights, dim=1)
        if USING_2D_INPUT:
            for cell in self.cells:
                weights = sampled_weights_normal
                s0, s1 = s1, cell(s0, s1, weights)
                hid_states.append(s1)
        elif CELL_WEIGHTS_SHARED:
            #for _ in range(self.sampled_layers):
            for _ in range(self.max_layers):
                s0, s1 = s1, self.cells[0](s0, s1, sampled_weights_normal, w_cell)
                hid_states.append(s1)
        else:
            #for i in range(self.sampled_layers):
            for i in range(self.max_layers):
                s0, s1 = s1, self.cells[i](s0, s1, sampled_weights_normal, w_cell[i])
                hid_states.append(s1)

        # the shape of logits is K, w.r.t K classifiers of the AdaBERT.
        logits = []
        for i in range(self.max_layers):
            # the shape of hid_states is K+1, hid_states[0] indicates the input embedding layers.
            if OUT_AVE_LAYER:
                # hid_states shape: [batch, hid, sequence]
                # out shape: [layer, batch, hid, sequence]
                out = torch.stack(hid_states[0:i+2])
                # out shape: [layer, batch, hid, sequence]  ->  [batch, hid, layer]
                out = out.mean(3).permute(1, 2, 0)
                # attentively summed output over layers
                out = torch.mul(F.softmax(self.layer_wise_blend_weights[i], dim=0), out)
                out = torch.sum(out, 2)
            else:
                # concatenation
                # hid_states shape: [batch, hid, sequence]
                out_layers = torch.cat(hid_states[0:i+2], dim=1)
                out = out_layers.mean(2)

            out = torch.tanh(out.view(out.size(0), -1))  # flatten
            out = self.dropout(out)
            logits_i = self.classifiers[i](out)
            logits.append(logits_i)
        return logits, hid_states


class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self, C_in, C, n_classes, N_layers, criterion, droprate, n_nodes=3, stem_multiplier=4,
                 device_ids=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_layer = nn.ParameterList()
        self.sampled_weights_normal = None
        self.sampled_weights_layer = None
        self.max_layers = N_layers
        self.sampled_layer = N_layers

        for i in range(n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3*torch.randn(i+2, n_ops)))
            # self.alpha_normal.append(nn.Parameter(1e-2 * torch.randn(i + 2, n_ops)))
        self.alpha_layer.append(nn.Parameter(1e-3 * torch.randn(N_layers)))
        # self.alpha_layer.append(nn.Parameter(1e-2 * torch.randn(N_layers)))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        self.net = SearchCNN(C_in, C, n_classes, N_layers, n_nodes, stem_multiplier, droprate=droprate)

    def forward(self, s1, s2, temp_alpha):
        if self.training:
            # self.sampled_weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
            self.sampled_weights_normal = [F.gumbel_softmax(alpha, tau=temp_alpha, hard=True, dim=-1)
                                           for alpha in self.alpha_normal]
            self.sampled_weights_layer = F.gumbel_softmax(self.alpha_layer[0], tau=temp_alpha, hard=True, dim=-1)
            # item() is used to transform tensor to int
            self.sampled_layer = int(torch.dot(self.sampled_weights_layer,
                                torch.tensor([i + 1 for i in range(self.max_layers)],
                                dtype=self.sampled_weights_layer.dtype, device=self.sampled_weights_layer.device)).item())
        else:
            self.sampled_weights_normal = [torch.argmax(alpha, 0, keepdim=True) for alpha in self.alpha_normal]
            # self.sampled_weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
            self.sampled_weights_layer = torch.argmax(self.alpha_layer[0], 0, keepdim=True)
            #self.sampled_layer = int(torch.argmax(self.sampled_weights_layer).item())
            self.sampled_layer = self.sampled_weights_layer.item() + 1


        #if len(self.device_ids) == 1:
        #    return self.net((s1, s2), self.sampled_weights_normal, self.sampled_weights_layer)
        if len(self.device_ids) == 1:
            return self.net((s1, s2), self.sampled_weights_normal)

        # scatter x
        xs = nn.parallel.scatter((s1, s2), self.device_ids)
        # broadcast weights
        wnormal_copies = broadcast_list(self.sampled_weights_normal, self.device_ids)
        #wlayer_copies = broadcast_list(self.sampled_weights_layer, self.device_ids)

        # replicate modules
        replicas = nn.parallel.replicate(self.net, self.device_ids)
        # outputs = nn.parallel.parallel_apply(replicas,
        #                                     list(zip(xs, wnormal_copies, wlayer_copies)), devices=self.device_ids)
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wnormal_copies)), devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])

    def loss(self, X, y):
        logits = self.forward(X)[0]
        return self.criterion(logits, y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))
        logger.info("# Alpha - layer")
        logger.info(F.softmax(self.alpha_layer[0], dim=-1))

        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        concat = range(2, 2+self.n_nodes)  # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
