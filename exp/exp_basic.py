import os
import torch
from models import NIS, DLinear, iTransformer, Transformer, NN, NISp, RNIS


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'DLinear': DLinear,
            'iTransformer': iTransformer,
            'Transformer': Transformer,
            'NN': NN,
            'NIS': NIS,
            'NISp':NISp,
            "RNIS":RNIS
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            print(f'CUDA_VISIBLE_DEVICES:{os.environ["CUDA_VISIBLE_DEVICES"]}')
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                print('Use Apple Silicon GPU')
            else:
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
