"""train"""
import argparse
import os
import time
from datetime import datetime
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import Tensor, context
from mindspore.nn import FixedLossScaleUpdateCell
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import RunContext, _InternalCallbackParam
from mindspore.context import ParallelMode
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank, get_group_size

from src.dataset import DatasetGenerator
from src.model import ECAPA_TDNN, Classifier
from src.loss import AdditiveAngularMargin
from src.util import learning_rate_clr_triangle_function
from src.util import TrainOneStepWithLossScaleCellv2 as TrainOneStepWithLossScaleCell
from src.sampler import DistributedSampler
from src.config import Config_gpu, Config_ascend
set_seed(0)


class BuildTrainNetwork(nn.Cell):
    """train construct"""
    def __init__(self, network, classifier, lossfunction, criterion, train_batch_size, class_num_):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.classifier = classifier
        self.criterion = criterion
        self.loss = lossfunction
        self.output = ms.Parameter(Tensor(np.ones((train_batch_size, class_num_)), ms.float32), requires_grad=False)
        self.onehot = ms.nn.OneHot(depth=class_num_, axis=-1, dtype=ms.float32)

    def construct(self, input_data, label):
        output = self.network(input_data)
        onehot = self.onehot(label)
        logits = self.classifier(output)
        output = self.loss(logits, onehot)
        self.output = output
        loss0 = self.criterion(output, onehot)
        return loss0


def update_average(loss_, avg_loss, step):
    avg_loss -= avg_loss / step
    avg_loss += loss_ / step
    return avg_loss


def train_net(rank, model, epoch_max, data_train, ckpt_cb, steps_per_epoch, train_batch_size):
    """train network"""
    cb_params = _InternalCallbackParam()
    cb_params.train_network = model
    cb_params.epoch_num = epoch_max
    cb_params.batch_num = steps_per_epoch
    cb_params.cur_epoch_num = 0
    cb_params.cur_step_num = 0
    run_context = RunContext(cb_params)
    if rank == 0:
        print("============== Starting Training ==============")
        ckpt_cb.begin(run_context)

    for epoch in range(epoch_max):
        t_start = time.time()
        train_loss = 0
        avg_loss = 0
        train_loss_cur = 0
        print_dur = 1
        for idx, (data, gt_classes) in enumerate(data_train):
           # start = time.time()
            model.set_train()
            batch_loss, _, _, _ = model(data, gt_classes)
            train_loss += batch_loss
            train_loss_cur += batch_loss
            avg_loss = update_average(batch_loss, avg_loss, idx+1)
            if rank == 0 and idx % print_dur == 0:
                cur_loss = train_loss_cur.asnumpy()
                total_avg = train_loss.asnumpy() / float(idx+1)
                if idx > 0:
                    cur_loss = train_loss_cur.asnumpy()/float(print_dur)
                print(f"{datetime.now()}, epoch:{epoch + 1}/{epoch_max}, iter-{idx}/{steps_per_epoch},"
                      f'cur loss:{cur_loss:.4f}, aver loss:{avg_loss.asnumpy():.4f},'
                      f'total_avg loss:{total_avg:.4f}')
                train_loss_cur = 0
            cb_params.cur_step_num += 1
            if rank == 0:
               # end = time.time()
               # print(f"train time per step: {(end - start)*1000} ms/step")
                ckpt_cb.step_end(run_context)

        cb_params.cur_epoch_num += 1
        my_train_loss = train_loss/steps_per_epoch
        time_used = time.time() - t_start
        fps = train_batch_size * steps_per_epoch / time_used
        if rank == 0:
            print('epoch[{}], {:.2f} step/sec'.format(epoch, fps))
            print('Train Loss:', my_train_loss)


def train(cfg):
    """train function"""
    if cfg.run_distribute:
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(mode=cfg.mode, device_target=cfg.device_target, device_id=device_id, save_graphs=False)
        init()
        cfg.rank = get_rank()
        cfg.group_size = get_group_size()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True,
                                          device_num=8,
                                          parameter_broadcast=True)
        num_parallel_workers = 1
    else:
        context.set_context(mode=cfg.mode, device_target=cfg.device_target, device_id=cfg.device_id, save_graphs=False)
        cfg.rank = 0
        cfg.group_size = 1
        num_parallel_workers = 8
    # dataset
    dataset_generator = DatasetGenerator(cfg.train_list, cfg.train_path, cfg.musan_path, cfg.rir_path, cfg.num_frames)
    distributed_sampler = None
    if cfg.run_distribute:
        distributed_sampler = DistributedSampler(len(dataset_generator), cfg.group_size, cfg.rank, shuffle=True)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=True, sampler=distributed_sampler)
    dataset = dataset.batch(batch_size=cfg.minibatch_size,
                            drop_remainder=True,
                            num_parallel_workers=num_parallel_workers)
    steps_per_epoch_train = int(dataset.get_dataset_size())
    print(f'group_size:{cfg.group_size}, data total len:{steps_per_epoch_train}')

    # model
    model = ECAPA_TDNN(input_size=cfg.in_channels,
                       channels=(cfg.channels, cfg.channels, cfg.channels, cfg.channels, cfg.channels * 3),
                       lin_neurons=cfg.emb_size)

    # loss function
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')

    # optimizer
    my_classifier = Classifier(1, 0, cfg.emb_size, cfg.class_num)
    aam = AdditiveAngularMargin(cfg.margin, cfg.scale)
    lr_list = []
    lr_list_total = steps_per_epoch_train * cfg.num_epochs
    for i in range(lr_list_total):
        lr_list.append(learning_rate_clr_triangle_function(steps_per_epoch_train, cfg.max_lrate, cfg.base_lrate, i))

    loss_scale_manager = FixedLossScaleUpdateCell(loss_scale_value=2**14)
    # loss_scale_manager = DynamicLossScaleUpdateCell(loss_scale_value=2**10, scale_factor=2, scale_window=1000)
    model_constructed = BuildTrainNetwork(model, my_classifier, aam, loss, cfg.minibatch_size, cfg.class_num)
    opt = nn.Adam(model_constructed.trainable_params(), learning_rate=lr_list, weight_decay=cfg.weight_decay)

    model_constructed = TrainOneStepWithLossScaleCell(model_constructed, opt, scale_sense=loss_scale_manager)
    config_ck = CheckpointConfig(save_checkpoint_steps=steps_per_epoch_train,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="ECAPATDNN", directory=cfg.ckpt_save_dir, config=config_ck)
    train_net(cfg.rank,
              model_constructed,
              cfg.num_epochs,
              dataset,
              ckpoint_cb,
              steps_per_epoch_train,
              cfg.minibatch_size)


def get_args():
    """ get args"""
    parser = argparse.ArgumentParser(description='Train ECAPATDNN')
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend.')
    parser.add_argument('--device_target', type=str, default="Ascend", help='device target.')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.device_target == "GPU":
        config = Config_gpu()
    else:
        config = Config_ascend()
    config.device_id = args.device_id
    s = time.time()
    train(config)
    e = time.time()
    print(f"total time:{e - s}")
