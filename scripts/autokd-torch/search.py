""" Search cell """
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import (BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)
from scipy.special import logsumexp
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset

import utils
from PyTransformer.utils_glue import (compute_metrics, convert_examples_to_features,
                                      output_modes, processors)
from architect import Architect
from config import SearchConfig
from models.search_cnn import SearchCNNController
from visualize import plot



# FLags used for ablation study
USING_2D_INPUT = False   # character level model
MASK_CLS_TOKEN = True
LEARN_FROM_HID = True
LEARN_HID_MODE_FOCAL = True
LEARN_OP_EFFICIENCY_LOSS = False
# BI_LEVEL_OPTIMIZATION = True

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

config = SearchConfig()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.task_name)))
config.print_params(logger.info)





def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    torch.cuda.set_device(config.gpus[0] if torch.cuda.is_available() else -1)

    torch.backends.cudnn.benchmark = True

    # Prepare GLUE task
    config.task_name = config.task_name.lower()
    if config.task_name not in processors:
        raise ValueError("Task not found: %s" % config.task_name)
    processor = processors[config.task_name]()
    config.output_mode = output_modes[config.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    eval_task_names = ("mnli", "mnli-mm") if config.task_name == "mnli" else (config.task_name,)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[config.model_type]
    model_config = config_class.from_pretrained(config.config_name if config.config_name else config.model_name_or_path,
                                                num_labels=num_labels, finetuning_task=config.task_name)
    tokenizer = tokenizer_class.from_pretrained(config.output_dir, do_lower_case=config.do_lower_case)
    bert_model = model_class.from_pretrained(config.model_name_or_path,
                                             from_tf=bool('.ckpt' in config.model_name_or_path), config=model_config)
    train_data = load_and_cache_examples(config, eval_task_names[0], tokenizer, evaluate=False, data_aug=config.data_aug)
    valid_data = load_and_cache_examples(config, eval_task_names[0], tokenizer, evaluate=True)

    bert_model.to(device)
    net_crit = nn.CrossEntropyLoss().to(device)
    input_channels = 1
    model = SearchCNNController(input_channels, config.init_channels, num_labels, config.layers,
                                net_crit, droprate=config.droprate, device_ids=config.gpus)
    model = model.to(device)

    OP_COST_LOOKUP = count_OP_cost(model, 128, 128)

    # weights optimizer  (parameters of candidate operations)
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    teacher_probe_parameters = nn.ParameterList()
    for probe in model.net.teacher_probes:
        teacher_probe_parameters.append(probe.weight)
    # teacher_probe_optimizer, the parameters is reffered to the BERT fine-tuning
    teacher_probe_optim = torch.optim.Adam(teacher_probe_parameters, config.w_lr, eps=config.adam_epsilon)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)
    if config.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install nvidia apex to use mixed precision.")
        [model, bert_model], [w_optim, alpha_optim, teacher_probe_optim] = \
            amp.initialize([model, bert_model], [w_optim, alpha_optim, teacher_probe_optim], opt_level=config.fp16_opt_level)

    # # split training data to train/alpha_validation, used in original DARTS setting, here we adopt all training data
    # n_train = len(train_data)
    # indices = list(range(n_train))
    # split = n_train // 2
    # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    # alpha_valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    # small_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:3000])
    # train_sampler = small_sampler
    # alpha_valid_sampler = small_sampler
    train_sampler = torch.utils.data.sampler.RandomSampler(train_data)
    alpha_valid_sampler = torch.utils.data.sampler.RandomSampler(train_data)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    alpha_valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=alpha_valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                                     batch_size=config.batch_size,
                                                     sampler=torch.utils.data.sampler.RandomSampler(valid_data),
                                                     num_workers=config.workers,
                                                     pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    best_result = 0.
    # for m in model.modules():
    #     if isinstance(m, nn.Conv1d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    if LEARN_FROM_HID and config.layers > 1:
        # training teacher_probes
        for epoch in range(config.teacher_probe_train_epochs):
            logger.info("Train teacher probes: [{:2d}/{}]".format(epoch, config.teacher_probe_train_epochs))
            train_teacher_probe(epoch, train_loader, model, teacher_probe_optim, teacher_probe_parameters,  bert_model=bert_model)
        for parameters in teacher_probe_parameters:
            parameters.detach()

    # training and searching
    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)

        cur_step = (epoch+1) * len(train_loader)
        gumbel_temp = np.maximum(config.initial_gumbel_temp * np.exp(-config.gumbel_anneal_rate * epoch),
                                 config.min_gumbel_temp)

        # training
        train(train_loader, alpha_valid_loader, model, gumbel_temp,
                    w_optim, alpha_optim, lr, epoch, bert_model=bert_model, OP_COST_LOOKUP=OP_COST_LOOKUP)

        # validation
        # result = validate(valid_loader, model, epoch, cur_step, bert_embeddings=bert_model.bert.embeddings)
        result = validate(valid_loader, model, epoch, cur_step, bert_model=bert_model, gumbel_temp=gumbel_temp)

        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        plot(genotype.normal, plot_path + "-normal", caption)
        # plot(genotype.reduce, plot_path + "-reduce", caption)

        # save
        if best_result < result:
            best_result = result
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)
        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_result))
    logger.info("Best Genotype = {}".format(best_genotype))


def count_OP_cost(model, hid_size, length):
    def np_softmax(x, axis=None):
        # compute in log space for numerical stability
        return np.exp(x - logsumexp(x, axis=axis, keepdims=True))
    def pool_input(input_res):
        return {'x': torch.ones(()).new_empty((1, *input_res), device=device)}

    OP_SIZE = []
    OP_FLOPs = []
    for op in model.net.cells[0].dag[0][0]._ops:
        if op.__class__.__name__ in ["Zero", "Identity"]:
            # smoothing
            OP_SIZE.append(1)
            OP_FLOPs.append(1)
        elif "Pool" in op.__class__.__name__:
            flops, size = utils.get_model_complexity_info(op, (hid_size, length), print_per_layer_stat=False,
                                                          as_strings=False, input_constructor=pool_input)
            OP_SIZE.append(size)
            OP_FLOPs.append(flops)
        else:
            flops, size = utils.get_model_complexity_info(op, (hid_size, length), as_strings=False)
            OP_SIZE.append(size)
            OP_FLOPs.append(flops)
    OP_SIZE_COST = np_softmax(np.array(OP_SIZE).reshape(1, -1)/100000, axis=1)
    OP_FLOPs_COST = np_softmax(np.array(OP_FLOPs).reshape(1, -1) / 10000000, axis=1)
    # OP_SIZE_COST = np_softmax(np.array(OP_SIZE).reshape(1, -1)/1000000, axis=1)
    # OP_FLOPs_COST = np_softmax(np.array(OP_FLOPs).reshape(1, -1) / 100000000, axis=1)
    op_dtype = torch.half if config.fp16 else torch.float32
    OP_COST_LOOKUP = torch.tensor((OP_SIZE_COST + OP_FLOPs_COST)/2, device=device, dtype=op_dtype)
    return OP_COST_LOOKUP


def train_teacher_probe(epoch, train_loader, model, teacher_probe_optim, teacher_probe_parametes, bert_model=None):

    model.train()
    losses = utils.AverageMeter()

    bert_embeddings = bert_model.bert.embeddings
    assert bert_embeddings is not None

    hid_out_dtype = torch.half if config.fp16 else torch.float

    for step, trn_batch in enumerate(train_loader):
        trn_batch = tuple(t.to(device, non_blocking=True) for t in trn_batch)

        bert_embeddings.dropout.training = False
        bert_model.eval()
        bert_model.bert.encoder.output_hidden_states = True
        bert_outputs = bert_model(input_ids=trn_batch[0], token_type_ids=trn_batch[2], attention_mask=trn_batch[1])
        hid_states_bert = bert_outputs[1]

        teacher_probe_optim.zero_grad()

        bert_layer = 12
        min_teacher_probe_num = min(config.layers - 1, bert_layer - 1)
        teacher_probe_losses = []
        for i in range(min_teacher_probe_num):
            proportional_layer_idx = math.ceil(i * 12 / config.layers)
            # the first layer (embedding layer) use the average pooling over all words, since the bert_pooler uses
            # first token, which is the same for all instance, in the embedding layer
            proportional_hid_state = hid_states_bert[proportional_layer_idx].type(hid_out_dtype)
            hid_out_bert = bert_model.bert.pooler(proportional_hid_state) if \
                proportional_layer_idx != 0 else torch.tanh(proportional_hid_state.mean(1))
            teacher_probe_i = model.net.teacher_probes[i]
            hid_logits_bert = teacher_probe_i(hid_out_bert)
            teach_probe_loss = model.criterion(hid_logits_bert, trn_batch[3]).type(hid_out_dtype)
            teacher_probe_losses.append(teach_probe_loss)
        teacher_probe_losses = torch.stack(teacher_probe_losses)
        teacher_probe_losses = torch.sum(teacher_probe_losses)
        if config.fp16:
            from apex import amp
            with amp.scale_loss(teacher_probe_losses, teacher_probe_optim) as scaled_loss:
                scaled_loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(amp.master_params(teacher_probe_optim), config.w_grad_clip)
        else:
            teacher_probe_losses.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(teacher_probe_parametes, config.w_grad_clip)
        losses.update(teacher_probe_losses)
        teacher_probe_optim.step()
        if step % 5 == 0:
            logger.info("Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} ".format(
                    epoch, config.teacher_probe_train_epochs, step, len(train_loader) - 1, losses=losses))


def train(train_loader, alpha_valid_loader, model, gumbel_temp, w_optim, alpha_optim, lr, epoch, bert_model=None,
          OP_COST_LOOKUP=None):
    result = utils.AverageMeter()
    result_bert = utils.AverageMeter()
    glue_metric = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()

    bert_embeddings = bert_model.bert.embeddings
    assert bert_embeddings is not None

    for step, (trn_batch, alpha_val_batch) in enumerate(zip(train_loader, alpha_valid_loader)):
        trn_batch = tuple(t.to(device, non_blocking=True) for t in trn_batch)
        alpha_val_batch = tuple(t.to(device, non_blocking=True) for t in alpha_val_batch)
        trn_y = trn_batch[3]
        val_y = alpha_val_batch[3]
        N = trn_y.shape[0]

        with torch.no_grad():
            s1_input_embedding, s2_input_embedding = bert_embedding_to_adabert(bert_embeddings, trn_batch)
            # val_s1_input_embedding, val_s2_input_embedding = bert_embedding_to_adabert(bert_embeddings, alpha_val_batch)

            # teacher model
            bert_model.eval()
            bert_model.bert.encoder.output_hidden_states = True
            bert_outputs = bert_model(input_ids=trn_batch[0], token_type_ids=trn_batch[2], attention_mask=trn_batch[1])
            logits_bert = bert_outputs[0]
            hid_states_bert = bert_outputs[1]

        if MASK_CLS_TOKEN:
            # mask = trn_batch[1]
            # input x_in_1d shape: [batch_size, hid_size, seq_length]
            pad = (torch.zeros(N, s1_input_embedding.shape[1], 1)).to(device)
            s1_input_embedding = torch.cat((pad.permute(2, 0, 1), s1_input_embedding[:, :, 1:].permute(2, 0, 1)))
            s1_input_embedding = s1_input_embedding.permute(1, 2, 0)
            pad = (torch.zeros(N, s2_input_embedding.shape[1], 1)).to(device)
            s2_input_embedding = torch.cat((pad.permute(2, 0, 1), s2_input_embedding[:, :, 1:].permute(2, 0, 1)))
            s2_input_embedding = s2_input_embedding.permute(1, 2, 0)


        # # bi-level optimization of darts, not used in the latter version.
        # if BI_LEVEL_OPTIMIZATION:
        #     # phase 2. architect step (alpha)
        #     alpha_optim.zero_grad()
        #     if config.unrolled:
        #         architect.unrolled_backward(trn_input_embedding, trn_y, val_input_embedding, val_y, lr, w_optim)
        #     else:
        #         # first-order
        #         loss_alpha = architect.v_net.loss(val_input_embedding, val_y)
        #         loss_alpha.backward()
        #     alpha_optim.step()

        # phase 1. child network step (w)
        alpha_optim.zero_grad()
        w_optim.zero_grad()
        logits, hid_states = model(s1_input_embedding, s2_input_embedding, gumbel_temp)
        loss_ce_all = []
        loss_last_layer_all = []
        for i in range(model.net.max_layers):
            # k loss_ce and k loss_last_layer to update the alpha k
            loss_ce_i = model.criterion(logits[i], trn_batch[3])
            loss_ce_all.append(loss_ce_i)

            ce_fct = nn.KLDivLoss(reduction='batchmean')
            loss_last_layer = ce_fct(F.log_softmax(logits[i]/config.temperature_kd, dim=-1),
                                       F.softmax(logits_bert/config.temperature_kd, dim=-1))
            loss_last_layer_all.append(loss_last_layer)
        # sampled_weights_layer is a one-hot vector
        loss_dtype = torch.half if config.fp16 else torch.float
        loss_ce_all = torch.stack(loss_ce_all)
        loss_ce_all = loss_ce_all.type(loss_dtype)
        loss_ce = torch.dot(model.sampled_weights_layer, loss_ce_all)
        loss_last_layer_all = torch.stack(loss_last_layer_all)
        loss_last_layer_all = loss_last_layer_all.type(loss_dtype)
        loss_last_layer = torch.dot(model.sampled_weights_layer, loss_last_layer_all)

        sampled_layer = model.sampled_layer
        if LEARN_FROM_HID and sampled_layer > 1:
            hid_knowledge_loss = []
            knowledge_weights = []

            for i in range(sampled_layer - 1):
                proportional_layer_idx = int(i * 12 / config.layers)

                # the first layer (embedding layer) use the average pooling over all words, since the bert_pooler uses
                # first token, which is the same for all instance, in the embedding layer
                hid_state_bert_j = hid_states_bert[proportional_layer_idx] if not config.fp16 else\
                    hid_states_bert[proportional_layer_idx].type(torch.half)
                hid_out_bert = bert_model.bert.pooler(hid_state_bert_j) if \
                    proportional_layer_idx != 0 else torch.tanh(hid_state_bert_j.mean(1))
                teacher_probe_i = model.net.teacher_probes[i]
                if config.fp16:
                    hid_out_bert = hid_out_bert.type(torch.half)
                hid_logits_bert = teacher_probe_i(hid_out_bert)

                hid_out = hid_states[i] if not config.fp16 else hid_states[i].type(torch.half)
                # hid_out = F.max_pool1d(hid_out, hid_out.size(2)).squeeze()
                hid_out = F.avg_pool1d(hid_out, hid_out.size(2)).squeeze() if not USING_2D_INPUT \
                    else hid_out.mean(2).view(config.batch_size, -1)
                probe_i = model.net.probes[i]
                # hid_logits = probe_i(model.net.dropout(hid_out))
                hid_logits = probe_i(torch.tanh(hid_out))

                loss_kd_hid = ce_fct(F.log_softmax(hid_logits / config.temperature_kd, dim=-1),
                                     F.softmax(hid_logits_bert / config.temperature_kd, dim=-1))
                hid_knowledge_loss.append(loss_kd_hid)
                if LEARN_HID_MODE_FOCAL:  # negative CE loss of BERT_t
                    knowledge_weights.append(-model.criterion(hid_logits_bert, trn_batch[3]))
            hid_knowledge_loss = torch.stack(hid_knowledge_loss)
            hid_knowledge_loss = hid_knowledge_loss.type(loss_dtype)
            knowledge_weights = torch.stack(knowledge_weights)
            knowledge_weights = knowledge_weights.type(loss_dtype)

            if LEARN_HID_MODE_FOCAL:   # Attentive hierarchical transfer
                weights = F.softmax(knowledge_weights)
                hid_loss = torch.dot(weights, hid_knowledge_loss)
            else:
                hid_loss = hid_knowledge_loss.mean()

        hard_loss_transition_point = config.epochs * 1
        # first learn most knowledge from teacher, then slightly fit the target label
        weight_loss_ce = 0.2 if epoch < hard_loss_transition_point else 0.4
        weight_loss_final_layer_kd = 0.4 if epoch < hard_loss_transition_point else 0.5
        beta = 4
        loss = weight_loss_ce * loss_ce + weight_loss_final_layer_kd * loss_last_layer
        if LEARN_FROM_HID and sampled_layer > 1:
            loss += (1 - weight_loss_ce - weight_loss_final_layer_kd) * hid_loss
        efficiency_cost = 0
        for op_vector in model.sampled_weights_normal:
            op_vector = torch.mean(op_vector, 0)
            efficiency_cost += torch.dot(op_vector, OP_COST_LOOKUP.squeeze())
        loss_e = sampled_layer / config.layers * efficiency_cost / len(model.sampled_weights_normal)
        loss = loss + beta * loss_e if LEARN_OP_EFFICIENCY_LOSS else loss
        if config.fp16:
            from apex import amp
            with amp.scale_loss(loss, [w_optim, alpha_optim]) as scaled_loss:
                scaled_loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(amp.master_params(w_optim), config.w_grad_clip)
            nn.utils.clip_grad_norm_(amp.master_params(alpha_optim), config.w_grad_clip)
        else:
            loss.backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
            nn.utils.clip_grad_norm_(model.alphas(), config.w_grad_clip)
        w_optim.step()
        alpha_optim.step()


        preds = None
        if preds is None:
            preds = logits[sampled_layer-1].detach().cpu().numpy()
            out_label_ids = trn_y.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits[sampled_layer-1].detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, trn_y.detach().cpu().numpy(), axis=0)
        if config.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif config.output_mode == "regression":
            preds = np.squeeze(preds)
        glue_result = compute_metrics(config.task_name, preds, out_label_ids)
        prec1 = utils.accuracy(logits[sampled_layer-1], trn_y)[0]
        prec1_bert = utils.accuracy(logits_bert, trn_y)[0]
        losses.update(loss.item(), N)
        result.update(prec1.item(), N)
        result_bert.update(prec1_bert.item(), N)
        glue_metric.update(glue_result, N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "GLUE Metric: {glue_metric.avg:.3f} "
                "Prec@(1,1_bert) ({result.avg:.1%}, {result_bert.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses, glue_metric=glue_metric,
                    result=result, result_bert=result_bert))

        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/result', prec1.item(), cur_step)
        writer.add_scalar('train/result_bert', prec1_bert.item(), cur_step)
        glue_val = list(glue_result.values())[0] if isinstance(glue_result, dict) else glue_result
        writer.add_scalar('train/glue_metric', glue_val, cur_step)
        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final Prec@1 {:.4%} GLUE_Metric {:.4%}".format(
        epoch+1, config.epochs, result.avg, glue_metric.avg))


# def validate(valid_loader, model, epoch, cur_step, bert_embeddings):
def validate(valid_loader, model, epoch, cur_step, bert_model, gumbel_temp):
    results = utils.AverageMeter()
    result_bert = utils.AverageMeter()
    losses = utils.AverageMeter()
    glue_metric = utils.AverageMeter()

    model.eval()

    bert_embeddings = bert_model.bert.embeddings

    with torch.no_grad():
        for step, val_batch in enumerate(valid_loader):
            val_batch = tuple(t.to(device, non_blocking=True) for t in val_batch)
            val_y = val_batch[3]

            s1_input_embedding, s2_input_embedding = bert_embedding_to_adabert(bert_embeddings, val_batch)
            # teacher model
            bert_model.eval()
            bert_outputs = bert_model(input_ids=val_batch[0], token_type_ids=val_batch[2], attention_mask=val_batch[1])
            logits_bert = bert_outputs[0]

            if MASK_CLS_TOKEN:
                # input x_in_1d shape: [batch_size, hid_size, seq_length]
                pad = (torch.zeros(N, s1_input_embedding.shape[1], 1)).to(device)
                s1_input_embedding = torch.cat((pad.permute(2, 0, 1), s1_input_embedding[:, :, 1:].permute(2, 0, 1)))
                s1_input_embedding = s1_input_embedding.permute(1, 2, 0)
                pad = (torch.zeros(N, s2_input_embedding.shape[1], 1)).to(device)
                s2_input_embedding = torch.cat((pad.permute(2, 0, 1), s2_input_embedding[:, :, 1:].permute(2, 0, 1)))
                s2_input_embedding = s2_input_embedding.permute(1, 2, 0)

            logits, _ = model(s1_input_embedding, s2_input_embedding, gumbel_temp)
            logits = logits[model.sampled_layer - 1]
            loss = model.criterion(logits, val_y)

            preds = None
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = val_y.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, val_y.detach().cpu().numpy(), axis=0)
            if config.output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            elif config.output_mode == "regression":
                preds = np.squeeze(preds)
            result = compute_metrics(config.task_name, preds, out_label_ids)

            prec1 = utils.accuracy(logits, val_y)[0]
            prec1_bert = utils.accuracy(logits_bert, val_y)[0]
            N = val_y.shape[0]
            losses.update(loss.item(), N)
            results.update(prec1.item(), N)
            result_bert.update(prec1_bert.item(), N)
            glue_val = list(result.values())[0] if isinstance(result, dict) else result
            glue_metric.update(glue_val, N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "GLUE Metric: {glue_metric.avg:.3f} "
                    "Prec@(1,1_bert) ({results.avg:.1%}, {result_bert.avg:.1%})".format(
                    epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses, glue_metric=glue_metric,
                    results=results, result_bert=result_bert))

        writer.add_scalar('val/loss', losses.avg, cur_step)
        writer.add_scalar('val/results', results.avg, cur_step)
        writer.add_scalar('val/result_bert', result_bert.avg, cur_step)
        writer.add_scalar('val/glue_metric', glue_metric.avg, cur_step)
        cur_step += 1

    logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%} GLUE_Metric {:.4%}".format(
        epoch+1, config.epochs, results.avg, glue_metric.avg))

    return results.avg


# an element is the first nonzero element if it is nonzero and the cumulative sum of a nonzero indicator is 1
def first_nonzero(x, axis=0):
    nonz = (x > 0)
    return ((nonz.cumsum(axis) == 1) & nonz).max(axis)


def bert_embedding_to_adabert(bert_embeddings, batch_input):
    # batch_input format: (all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
    #                                              all_input_ids_s1, all_input_ids_s2)
    bert_embeddings.dropout.training = False
    s1_input_embedding = bert_embeddings.forward(input_ids=batch_input[4])
    text_pair_task = sum(batch_input[2][0]) != 0
    if text_pair_task:
        s2_input_embedding = bert_embeddings.forward(input_ids=batch_input[5])
    if USING_2D_INPUT:
        input_shape = s1_input_embedding.shape
        # input shape: [batch_size, input_channel, seq_length, embedding_size]
        reshape_size = [-1, 1, input_shape[1], input_shape[2]]
        s1_input_embedding = s1_input_embedding.reshape(reshape_size)
        s2_input_embedding = s2_input_embedding.reshape(reshape_size) if text_pair_task else s1_input_embedding
    else:
        # input x_in_1d shape: [batch_size, hid_size, seq_length]
        s1_input_embedding = s1_input_embedding.permute(0, 2, 1)
        s2_input_embedding = s2_input_embedding.permute(0, 2, 1) if text_pair_task else s1_input_embedding

    return s1_input_embedding, s2_input_embedding


def load_and_cache_examples(config, task, tokenizer, evaluate=False, data_aug=False):
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if config.local_rank not in [-1, 0] and not evaluate:
         torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    train_type = 'train_aug' if data_aug else 'train'
    cached_features_file = os.path.join(config.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else train_type,
        list(filter(None, config.model_name_or_path.split('/'))).pop(),
        str(config.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and os.path.exists(cached_features_file + "text_s1") and \
            os.path.exists(cached_features_file + "text_s2"):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if os.path.exists(cached_features_file+"text_s1"):
            logger.info("Loading features from cached file %s", cached_features_file+ "text_s1")
            features_s1 = torch.load(cached_features_file+"text_s1")
        if os.path.exists(cached_features_file + "text_s2"):
            logger.info("Loading features from cached file %s", cached_features_file + "text_s2")
            features_s2 = torch.load(cached_features_file + "text_s2")
    else:
        logger.info("Creating features from dataset file at %s", config.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and config.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1] 
        examples = processor.get_dev_examples(config.data_dir) if evaluate else processor.get_train_examples(config.data_dir, data_aug)
        features = convert_examples_to_features(examples, label_list, config.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(config.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if config.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(config.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(config.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if config.model_type in ['xlnet'] else 0,
        )
        features_s1 = convert_examples_to_features(examples, label_list, config.max_seq_length, tokenizer, output_mode,
                                                   cls_token_at_end=bool(config.model_type in ['xlnet']),
                                                   cls_token=tokenizer.cls_token,
                                                   cls_token_segment_id=2 if config.model_type in ['xlnet'] else 0,
                                                   sep_token=tokenizer.sep_token,
                                                   sep_token_extra=bool(config.model_type in ['roberta']),
                                                   pad_on_left=bool(config.model_type in ['xlnet']),
                                                   pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                   pad_token_segment_id=4 if config.model_type in ['xlnet'] else 0,
                                                   text_pair_choose_half=True, choose_s2=False
                                                   )
        features_s2 = convert_examples_to_features(examples, label_list, config.max_seq_length, tokenizer, output_mode,
                                                   cls_token_at_end=bool(config.model_type in ['xlnet']),
                                                   cls_token=tokenizer.cls_token,
                                                   cls_token_segment_id=2 if config.model_type in ['xlnet'] else 0,
                                                   sep_token=tokenizer.sep_token,
                                                   sep_token_extra=bool(config.model_type in ['roberta']),
                                                   pad_on_left=bool(config.model_type in ['xlnet']),
                                                   pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                   pad_token_segment_id=4 if config.model_type in ['xlnet'] else 0,
                                                   text_pair_choose_half=True, choose_s2=True
                                                   )

        if config.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            logger.info("Saving features_s1 into cached file %s", cached_features_file+"text_s1")
            torch.save(features_s1, cached_features_file+"text_s1")
            logger.info("Saving features_s2 into cached file %s", cached_features_file+"text_s2")
            torch.save(features_s2, cached_features_file+"text_s2")

    if config.local_rank == 0 and not evaluate:
        torch.distributed.barrier()
        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache

    # Convert to Tensors and build dataset
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_input_ids_s1 = torch.tensor([f.input_ids for f in features_s1], dtype=torch.long)
    all_input_ids_s2 = torch.tensor([f.input_ids for f in features_s2], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                             all_input_ids_s1, all_input_ids_s2)

    return dataset



if __name__ == "__main__":
    main()
