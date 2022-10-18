"""parameter configuration"""
from mindspore import context


class Config_ascend:
    """Ascend"""
    def __init__(self):
        # Random seed
        self.seed = 0

        # Hardware equipment selection
        self.mode = context.GRAPH_MODE
        self.device_target = "Ascend"
        self.device_id = 0
        self.rank = 0
        self.group_size = 1
        # 8p True 1p False
        self.run_distribute = False

        # data
        self.train_list = "/disk3/dataset/tdnnDataset/Voxceleb2/train.txt"
        self.train_path = "/disk3/dataset/tdnnDataset/Voxceleb2/train/wav"
        self.musan_path = "/disk3/dataset/tdnnDataset/Others/musan_split"
        self.rir_path = "/disk3/dataset/tdnnDataset/Others/RIRS_NOISES/simulated_rirs"
        self.num_frames = 200
        self.shuffle = True

        # train
        self.in_channels = 80
        self.channels = 1024
        self.emb_size = 192

        self.base_lrate = 0.000001
        # 8p 0.00015 1p 0.0001
        self.max_lrate = 0.0001
        self.weight_decay = 0.000002
        self.num_epochs = 80
        self.minibatch_size = 100
        self.class_num = 5994
        self.margin = 0.2
        self.scale = 30.0
        self.ckpt_save_dir = "/disk3/hit_wml/Finally/model_1p"
        self.keep_checkpoint_max = 100

        # eval
        self.eval_list = "/disk3/dataset/tdnnDataset/Voxceleb2/eval.txt"
        self.eval_path = "/disk3/dataset/tdnnDataset/Voxceleb2/eval/wav"
        self.model_path = "/disk3/hit_wml/Finally/model/pretrain.ckpt"

        # export
        self.length = 1199


class Config_gpu:
    """GPU"""
    def __init__(self):
        # Random seed
        self.seed = 0

        # Hardware equipment selection
        self.mode = context.GRAPH_MODE
        self.device_target = "GPU"
        self.device_id = 0
        self.rank = 0
        self.group_size = 1
        # 8p True 1p False
        self.run_distribute = False

        # data
        self.train_list = "/disk3/dataset/tdnnDataset/Voxceleb2/train.txt"
        self.train_path = "/disk3/dataset/tdnnDataset/Voxceleb2/train/wav"
        self.musan_path = "/disk3/dataset/tdnnDataset/Others/musan_split"
        self.rir_path = "/disk3/dataset/tdnnDataset/Others/RIRS_NOISES/simulated_rirs"
        self.num_frames = 200
        self.shuffle = True

        # train
        self.in_channels = 80
        self.channels = 1024
        self.emb_size = 192

        self.base_lrate = 0.00001
        self.max_lrate = 0.001
        self.weight_decay = 0.00002
        self.num_epochs = 80
        self.minibatch_size = 200
        self.class_num = 5994
        self.margin = 0.2
        self.scale = 30.0
        self.ckpt_save_dir = "/disk3/hit_wml/Finally/model"
        self.keep_checkpoint_max = 80

        # eval
        self.eval_list = "/disk3/dataset/tdnnDataset/Voxceleb2/eval.txt"
        self.eval_path = "/disk3/dataset/tdnnDataset/Voxceleb2/eval/wav"
        self.model_path = "/disk3/hit_wml/Finally/model/pretrain.ckpt"

        # export
        self.length = 1199
