import os
import argparse
import logging

logger = logging.getLogger(__name__)


class Params:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # quantization setting
        self.parser.add_argument("--n_gpu", type=int, required=True, help='set to 0 if want to only use cpu')
        self.parser.add_argument("--benchmarks", type=str, default='baseline', 
                                    help='benchmark tasks split by comma. \
                                    For example: "baseline,quantization,fp16,pruning,distillation"')

        # Distillation training
        self.parser.add_argument('--student_model_name', type=str, default=None, help='set this \
                                        for distillation training, for example "resnet10"')
        self.parser.add_argument("--student_test_model_path", type=str, default=None)
        self.parser.add_argument('--distill_method', type=str, default='logit', choices={'logit', 'soft'})
        self.parser.add_argument('--distill_weight', type=float, default=0.1)
        self.parser.add_argument('--temperature', type=int, default=4)

        # General
        self.parser.add_argument('--model_name', type=str, default='resnet18')
        self.parser.add_argument('--num_epochs', type=int, default=60)
        self.parser.add_argument('--batch_size', type=int, default=256)
        self.parser.add_argument('--step_size', type=int, default=1, help='at which epoch does lr starts to decay by *gamma')
        self.parser.add_argument('--optim', type=str, default='sgd')
        self.parser.add_argument('--warmup_steps', type=int, default=1)
        self.parser.add_argument('--lr', type=float, default=0.1)
        self.parser.add_argument('--weight_decay', type=float, default=5e-4)
        self.parser.add_argument('--nesterov', action='store_const', default=False, const=True)
        self.parser.add_argument('--momentum', type=float, default=0.9)
        self.parser.add_argument('--scheduler', type=str, default='steplr')
        self.parser.add_argument('--gamma', type=float, default=0.2)

        self.parser.add_argument("--do_train", action='store_true')
        self.parser.add_argument("--do_eval", action='store_true')
        self.parser.add_argument('--output_dir', type=str, default=None)

        self.parser.add_argument("--test_model_path", type=str, default=None)
        self.parser.add_argument("--best_pt_name", type=str, default='best.pt')

        self.args = self.parser.parse_args()

    
params = Params()
args = params.args

# sanity check
if args.test_model_path and '.pt' not in args.test_model_path:
    raise ValueError('test_model_path should be path with file name')

for arg in vars(args):
    print(' {}: {}'.format(arg, getattr(args, arg)))
