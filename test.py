from argparse import ArgumentParser
import torch
from data.data import DataLoaderX, Dense_SIRST
from model.model import SSCFNet
from model.model_utils.metric import SigmoidMetric, SamplewiseSigmoidMetric, ROCMetric, SeRankDet_PD_FA, PD_FA
from utils.tools import denormalize, set_seed, init_env
from torch.utils.tensorboard import SummaryWriter
import os
import time
import logging
from tqdm import tqdm
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def parse_args():
    parser = ArgumentParser(description='parse args')

    # log
    parser.add_argument('--log_root', type=str, default="logs", help='log dir')
    parser.add_argument('--exp_name', type=str, default="SSCFNet", help='experiment name')
    parser.add_argument('--phase_name', type=str, default="test", help='phase name')
    parser.add_argument('--log_name', type=str, default="log.log", help='log name')
    parser.add_argument('--weight_path', type=str, default="logs/SSCFNet/train/20250616092902/best_miou.pth",
                        help='weight for testing')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size for training')  # must be 1 in inference stage & considering saveing image

    # environment
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument("--seed", type=int, default=3407, help="Torch seed 3407 is all you need")

    args = parser.parse_args()

    return args


class Tester(object):

    def __init__(self, args):
        self.args = args

        self.trainset = Dense_SIRST(mode='train')
        self.testset = Dense_SIRST(mode='test')
        self.train_data_loader = DataLoaderX(self.trainset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                             num_workers=8, prefetch_factor=8)
        self.test_data_loader = DataLoaderX(self.testset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                            num_workers=8, prefetch_factor=8)

        # dir
        self.log_dir = os.path.join(args.log_root, args.exp_name, args.phase_name,
                                    time.strftime('%Y%m%d%H%M%S', time.localtime()))
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # log
        self.log_file = os.path.join(self.log_dir, args.log_name)
        logging.basicConfig(filename=self.log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()
        # tensorboard
        self.writer = SummaryWriter(self.log_dir)
        self.writer.add_text(self.log_dir, 'Args:%s' % args)
        for i, (data, label,name) in enumerate(
                tqdm(self.test_data_loader, desc='Export testset to TensorBoard      ', position=0, leave=True)):
            self.writer.add_images('img/original', denormalize(data, self.testset.mean, self.testset.std), i,
                                   dataformats='NCHW')
            self.writer.add_images('img/processed', data, i, dataformats='NCHW')
            self.writer.add_images('label/original', label, i, dataformats='NCHW')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model init
        self.net = SSCFNet(mode='test')
        self.net.model.load_state_dict(torch.load(args.weight_path).state_dict())

        self.net.eval()
        self.net = self.net.to(self.device)  # all parameter including extra parameter
        self.net.model = self.net.model.to(self.device)  # only model parameter

        # metric
        self.miou_metric = SigmoidMetric(score_thresh=0.5)
        self.nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0.5,
                                                   do_sigmoid=False)  # do_sigmoid=False, as the output has do sigmoid
        self.roc_metric = ROCMetric(nclass=1, bins=100,
                                    do_sigmoid=False)  # do_sigmoid=False, as the output has do sigmoid
        self.SeRankDet_pd_fa_metric = SeRankDet_PD_FA(1, 100)
        # Modified based on BasicIRSTD, as a reference
        self.pd_fa_metric = PD_FA(1, 100)

    def testing(self):
        self.miou_metric.reset()
        self.nIoU_metric.reset()
        with torch.no_grad():
            tbar = tqdm(self.test_data_loader, desc='Testing...', position=0, leave=True)
            for i, (data, label,name) in enumerate(tbar):
                data = data[:, 1:2, :, :]

                data = data.to(self.device)
                label = label.to(self.device)

                output = self.net(data)  # （1,1,512,512）

                self.miou_metric.update(output, label)
                self.nIoU_metric.update(output, label)
                self.roc_metric.update(output, label)
                self.SeRankDet_pd_fa_metric.update(output, label, output[0, 0].shape)
                self.pd_fa_metric.update(output, label, output[0, 0].shape)
                # save output
                self.writer.add_images('result/output', output, i, dataformats='NCHW')
                os.makedirs(os.path.join(self.log_dir, 'output'), exist_ok=True)
                img_name=name[0]

                save_image(output, os.path.join(self.log_dir, 'output', f'{img_name}.png'))

                # save output_0_5_threshold
                output_0_5_threshold = output.clone()
                output_0_5_threshold[output_0_5_threshold > 0.5] = 1
                output_0_5_threshold[output_0_5_threshold <= 0.5] = 0
                self.writer.add_images('result/output_0_5_threshold', output_0_5_threshold, i, dataformats='NCHW')
                os.makedirs(os.path.join(self.log_dir, 'output_0_5_threshold'), exist_ok=True)

                save_image(output_0_5_threshold, os.path.join(self.log_dir, 'output_0_5_threshold', f'{img_name}.png'))

                # save output_0_5_threshold_visual
                overlay_color = torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1).to(self.device)
                alpha = 0.6
                mask_3c = output_0_5_threshold.repeat(1, 3, 1, 1)
                output_0_5_threshold_visual = denormalize(data, self.testset.mean, self.testset.std) * (
                            1 - alpha * mask_3c) + overlay_color * (alpha * mask_3c)
                self.writer.add_images('result/output_0_5_threshold_visual', output_0_5_threshold_visual, i,
                                       dataformats='NCHW')
                os.makedirs(os.path.join(self.log_dir, 'output_0_5_threshold_visual'), exist_ok=True)

                save_image(output_0_5_threshold_visual, os.path.join(self.log_dir, 'output_0_5_threshold_visual', f'{img_name}.png'))

                _, miou = self.miou_metric.get()
                _, nIoU = self.nIoU_metric.get()
                tp_rates, fp_rates, recall, precision, f1_score, f1_score_all = self.roc_metric.get()
                auc_value = auc(fp_rates, tp_rates)
                Final_PD, Final_PD_All, Final_FA, Final_FA_All = self.pd_fa_metric.get()

                tbar.set_description(
                    'Testing...  mIoU:%f, nIoU:%f, auc_value:%f, f1_score:%f, pd:%f, fa:%f' % (
                    miou, nIoU, auc_value, f1_score, Final_PD, Final_FA))

        _, miou = self.miou_metric.get()
        _, nIoU = self.nIoU_metric.get()
        tp_rates, fp_rates, recall, precision, f1_score, f1_score_all = self.roc_metric.get()
        auc_value = auc(fp_rates, tp_rates)
        SeRankDet_Final_PD, SeRankDet_Final_PD_All, SeRankDet_Final_FA, SeRankDet_Final_FA_All = self.SeRankDet_pd_fa_metric.get()
        Final_PD, Final_PD_All, Final_FA, Final_FA_All = self.pd_fa_metric.get()

        miou = miou * 1e+2
        nIoU = nIoU * 1e+2

        f1_score = f1_score * 1e+2
        f1_score_all = f1_score_all * 1e+2

        SeRankDet_Final_PD = SeRankDet_Final_PD * 1e+2
        SeRankDet_Final_PD_All = SeRankDet_Final_PD_All * 1e+2
        Final_PD = Final_PD * 1e+2
        Final_PD_All = Final_PD_All * 1e+2

        SeRankDet_Final_FA = SeRankDet_Final_FA * 1e+8
        SeRankDet_Final_FA_All = SeRankDet_Final_FA_All * 1e+8
        Final_FA = Final_FA * 1e+8
        Final_FA_All = Final_FA_All * 1e+8

        self.logger.info(
            '\r\n mIoU:\r\n%f\r\n nIoU:\r\n%f\r\n auc_value:\r\n%f\r\n f1_score:\r\n%f\r\n f1_score_all:\r\n%s\r\n tp_rates:\r\n%s\r\n fp_rates:\r\n%s\r\n recall:\r\n%s\r\n precision:\r\n%s\r\n SeRankDet_Final_PD:\r\n%s\r\n SeRankDet_Final_PD_All:\r\n%s\r\n SeRankDet_Final_FA:\r\n%s\r\n SeRankDet_Final_FA_All:\r\n%s\r\n Final_PD:\r\n%s\r\n Final_PD_All:\r\n%s\r\n Final_FA:\r\n%s\r\n Final_FA_All:\r\n%s\r\n' % (
            miou, nIoU, auc_value, f1_score, f1_score_all, tp_rates, fp_rates, recall, precision, SeRankDet_Final_PD,
            SeRankDet_Final_PD_All, SeRankDet_Final_FA, SeRankDet_Final_FA_All, Final_PD, Final_PD_All, Final_FA,
            Final_FA_All))
        self.writer.add_text('mIoU', 'mIoU:%f' % (miou))
        self.writer.add_text('nIoU', 'nIoU:%f' % (nIoU))
        self.writer.add_text('auc_value', 'auc_value: %f' % (auc_value))
        self.writer.add_text('f1_score', 'f1_score:%f' % (f1_score))
        self.writer.add_text('f1_score_all', 'f1_score_all:%s' % (f1_score_all))
        self.writer.add_text('tp_rates', 'tp_rates:%s' % (tp_rates))
        self.writer.add_text('fp_rates', 'fp_rates:%s' % (fp_rates))
        self.writer.add_text('recall', 'recall:%s' % (recall))
        self.writer.add_text('precision', 'precision:%s' % (precision))
        self.writer.add_text('SeRankDet_Final_PD', 'SeRankDet_Final_PD:%s' % (SeRankDet_Final_PD))
        self.writer.add_text('SeRankDet_Final_PD_All', 'SeRankDet_Final_PD_All:%s' % (SeRankDet_Final_PD_All))
        self.writer.add_text('SeRankDet_Final_FA', 'SeRankDet_Final_FA:%s' % (SeRankDet_Final_FA))
        self.writer.add_text('SeRankDet_Final_FA_All', 'SeRankDet_Final_FA_All:%s' % (SeRankDet_Final_FA_All))
        self.writer.add_text('Final_PD', 'Final_PD:%s' % (Final_PD))
        self.writer.add_text('Final_PD_All', 'Final_PD_All:%s' % (Final_PD_All))
        self.writer.add_text('Final_FA', 'Final_FA:%s' % (Final_FA))
        self.writer.add_text('Final_FA_All', 'Final_FA_All:%s' % (Final_FA_All))

        self.writer.close()


if __name__ == '__main__':
    args = parse_args()

    init_env(args.gpu_ids)

    # must set seed as use dataloader twice
    set_seed(args.seed)

    tester = Tester(args)
    tester.testing()
