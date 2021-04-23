""" Config class for search/augment """
import argparse
import os
import genotypes as gt
from functools import partial
import torch


from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)
from PyTransformer.utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)


ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class SearchConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        # parser.add_argument('--w_lr', type=float, default=0.025, help='lr for weights')
        parser.add_argument('--w_lr', type=float, default=2e-2, help='lr for weights')
        # parser.add_argument('--w_lr_min', type=float, default=0.001, help='minimum lr for weights')
        parser.add_argument('--w_lr_min', type=float, default=5e-4, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=3e-4,
                            help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=70, help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=1)
        parser.add_argument('--layers', type=int, default=8, help='# of layers')
        parser.add_argument('--workers', type=int, default=1, help='# of workers')
        parser.add_argument('--alpha_lr', type=float, default=3e-4, help='lr for alpha')
        parser.add_argument('--unrolled', type=bool, default=True,
                            help='second-order optimization for architecture parameters in darts')
        parser.add_argument('--alpha_weight_decay', type=float, default=1e-3,
                            help='weight decay for alpha')

        # arguments inherited from the PyTransformer
        parser.add_argument("--data_dir", default=None, type=str, required=True,
                            help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
        parser.add_argument("--model_type", default=None, type=str, required=True,
                            help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
        parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                            help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                                ALL_MODELS))
        parser.add_argument("--task_name", default=None, type=str, required=True,
                            help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
        parser.add_argument("--output_dir", default=None, type=str, required=True,
                            help="The output directory where the model predictions and checkpoints will be written.")

        parser.add_argument("--temperature_kd", default=1., type=float,
                            help="Temperature for the softmax temperature in knowledge distillation.")
        parser.add_argument("--initial_gumbel_temp", default=1., type=float,
                            help="Initial temperature for the gumbel softmax temperature.")
        parser.add_argument("--min_gumbel_temp", default=0.2, type=float,
                            help="Min temperature for the gumbel softmax temperature.")
        parser.add_argument("--gumbel_anneal_rate", default=0.02, type=float,
                            help="Anneal rate for temperature in the gumbel softmax temperature.")
        parser.add_argument("--droprate", default=0.3, type=float,
                            help="Dropout rate used in the searched nn.")

        parser.add_argument('--genotype', required=False, help='Cell genotype')

        ## Other parameters
        parser.add_argument("--config_name", default="", type=str,
                            help="Pretrained config name or path if not the same as model_name")
        parser.add_argument("--tokenizer_name", default="", type=str,
                            help="Pretrained tokenizer name or path if not the same as model_name")
        parser.add_argument("--cache_dir", default="", type=str,
                            help="Where do you want to store the pre-trained models downloaded from s3")
        parser.add_argument("--max_seq_length", default=128, type=int,
                            help="The maximum total input sequence length after tokenization. Sequences longer "
                                 "than this will be truncated, sequences shorter will be padded.")
        parser.add_argument("--do_train", action='store_true',
                            help="Whether to run training.")
        parser.add_argument("--do_eval", action='store_true',
                            help="Whether to run eval on the dev set.")
        parser.add_argument("--evaluate_during_training", action='store_true',
                            help="Rul evaluation during training at each logging step.")
        parser.add_argument("--do_lower_case", action='store_true',
                            help="Set this flag if you are using an uncased model.")
        parser.add_argument("--data_aug", type=str2bool, nargs='?', const=True, required=True,
                            help="The output directory where the model predictions and checkpoints will be written.")

        parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                            help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--weight_decay", default=0.0, type=float,
                            help="Weight deay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float,
                            help="Max gradient norm.")
        parser.add_argument("--teacher_probe_train_epochs", default=5, type=int,
                            help="Total number of training epochs for probe classifiers.")
        parser.add_argument("--num_train_epochs", default=3.0, type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--max_steps", default=-1, type=int,
                            help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
        parser.add_argument("--warmup_steps", default=0, type=int,
                            help="Linear warmup over warmup_steps.")

        parser.add_argument('--logging_steps', type=int, default=50,
                            help="Log every X updates steps.")
        parser.add_argument('--save_steps', type=int, default=50,
                            help="Save checkpoint every X updates steps.")
        parser.add_argument("--eval_all_checkpoints", action='store_true',
                            help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
        parser.add_argument("--no_cuda", action='store_true',
                            help="Avoid using CUDA when available")
        parser.add_argument('--overwrite_output_dir', action='store_true',
                            help="Overwrite the content of the output directory")
        parser.add_argument('--overwrite_cache', action='store_true',
                            help="Overwrite the cached training and evaluation sets")

        parser.add_argument('--fp16', action='store_true',
                            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
        parser.add_argument('--fp16_opt_level', type=str, default='O2',
                            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                 "See details at https://nvidia.github.io/apex/amp.html")
        parser.add_argument("--local_rank", type=int, default=-1,
                            help="For distributed training: local_rank")
        parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
        parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        # self.data_path = './data/'
        self.path = os.path.join('searchs', self.task_name)
        self.plot_path = os.path.join(self.path, 'plots')
        self.gpus = parse_gpus(self.gpus)
        # self.genotype = gt.from_str(self.genotype)

