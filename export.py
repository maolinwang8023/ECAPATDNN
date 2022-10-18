"""export model"""
import argparse
import numpy as np
import mindspore
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context
from src.model import ECAPA_TDNN
from src.config import Config_gpu, Config_ascend

parser = argparse.ArgumentParser(description='ECAPATDNN export')
parser.add_argument("--device_id", type=int, default=0, help="device id")
parser.add_argument("--checkpoint_path", type=str, required=True, help="checkpoint file path.")
parser.add_argument("--file_name", type=str, default="ECAPATDNN", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["MINDIR"], default='MINDIR', help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU"], default="Ascend", help="device target")


if __name__ == "__main__":
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    if args.device_target == "GPU":
        cfg = Config_gpu()
    else:
        cfg = Config_ascend()
        context.set_context(device_id=args.device_id)

    net = ECAPA_TDNN(input_size=cfg.in_channels,
                     channels=(cfg.channels, cfg.channels, cfg.channels, cfg.channels, cfg.channels * 3),
                     lin_neurons=cfg.emb_size)
    param_dict = load_checkpoint(args.checkpoint_path)
    load_param_into_net(net, param_dict)
    input_arr = Tensor(np.ones((1, cfg.in_channels, cfg.length)), mindspore.float32)
    export(net, input_arr, file_name=args.file_name, file_format=args.file_format)
