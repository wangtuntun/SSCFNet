import torch
import torch.nn.functional as F
import numpy as np
from skimage import measure
from tqdm import tqdm


class SigmoidMetric():
    def __init__(self, score_thresh=0.5):
        self.score_thresh = score_thresh
        self.reset()

    # 进来的是 (1,1,height,width)
    def update(self, pred, labels):
        correct, labeled = self.batch_pix_accuracy(pred, labels)
        inter, union = self.batch_intersection_union(pred, labels)

        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        """Gets the current evaluation result."""
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

    def batch_pix_accuracy(self, output, target):
        assert output.shape == target.shape
        output = output.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        # P map
        predict = (output > self.score_thresh).astype('int64')
        # TP sum: sum(P map == T map) 预测正确且为正类的个数
        pixel_correct = np.sum((predict == target) * (target > 0))

        pixel_labeled = np.sum(target > 0)
        
        assert pixel_correct <= pixel_labeled
        return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):
        mini = 1
        maxi = 1  # nclass
        nbins = 1  # nclass

        predict = (output.cpu().detach().numpy() > self.score_thresh).astype('int64')
        # T map
        target = target.cpu().numpy().astype('int64')

        intersection = predict * (predict == target)

        area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
        # count of P map
        area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
        # count of T map
        area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))

        # count of union map
        area_union = area_pred + area_lab - area_inter
        assert ( area_inter <= area_union ).all()

        return area_inter, area_union

class SamplewiseSigmoidMetric():
    def __init__(self, nclass, score_thresh=0.5, do_sigmoid=True):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.reset()
        self.do_sigmoid = do_sigmoid

    def update(self, preds, labels):
        inter_arr, union_arr = self.batch_intersection_union(preds, labels, self.nclass, self.score_thresh)
        # np.append([ [1], [2], [3] ],[ [4], [5], [6] ]) = [1 2 3 4 5 6], numpy.ndarray数据类型一样
        self.total_inter = np.append(self.total_inter, inter_arr)
        self.total_union = np.append(self.total_union, union_arr)

    def get(self):
        """Gets the current evaluation result."""
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        # 这是为了对多张图片取mean
        mIoU = IoU.mean()
        return IoU, mIoU

    def reset(self):
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])

    def batch_intersection_union(self, output, target, nclass, score_thresh):
        """mIoU"""
        # inputs are tensor
        # the category 0 is ignored class, typically for background / boundary
        mini = 1
        maxi = 1  # nclass
        nbins = 1  # nclass

        # P map
        if(self.do_sigmoid):
            predict = (F.sigmoid(output).cpu().detach().numpy() > score_thresh).astype('int64')
        else:
            predict = (output.cpu().detach().numpy() > score_thresh).astype('int64')
        # T map
        target = target.cpu().detach().numpy().astype('int64')  # T
        # P=T map
        intersection = predict * (predict == target)  # TP

        num_sample = intersection.shape[0]
        area_inter_arr = np.zeros(num_sample)
        area_pred_arr = np.zeros(num_sample)
        area_lab_arr = np.zeros(num_sample)
        area_union_arr = np.zeros(num_sample)

        for b in range(num_sample):
            area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
            area_inter_arr[b] = area_inter[0]

            area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
            area_pred_arr[b] = area_pred[0]

            area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
            area_lab_arr[b] = area_lab[0]

            area_union = area_pred + area_lab - area_inter
            area_union_arr[b] = area_union[0]

            assert (area_inter <= area_union).all()

        return area_inter_arr, area_union_arr


class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass, bins, do_sigmoid=True):
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)
        self.reset()
        self.do_sigmoid = do_sigmoid

    def update(self, preds, labels):
        for iBin in range(self.bins + 1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg, i_class_pos = self.cal_tp_pos_fp_neg(preds, labels, self.nclass, score_thresh)
            self.tp_arr[iBin] += i_tp
            self.pos_arr[iBin] += i_pos
            self.fp_arr[iBin] += i_fp
            self.neg_arr[iBin] += i_neg
            self.class_pos[iBin] += i_class_pos

    def get(self):
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)

        recall = self.tp_arr / (self.pos_arr + 0.001)
        precision = self.tp_arr / (self.class_pos + 0.001)
        f1_score = (2.0 * recall[int(self.bins/2)] * precision[int(self.bins/2)]) / (recall[int(self.bins/2)] + precision[int(self.bins/2)] + 0.00001)
        f1_score_all = (2.0 * recall * precision) / (recall + precision + 0.00001)

        return tp_rates, fp_rates, recall, precision, f1_score, f1_score_all

    def reset(self):
        self.tp_arr = np.zeros(self.bins + 1)
        self.pos_arr = np.zeros(self.bins + 1)
        self.fp_arr = np.zeros(self.bins + 1)
        self.neg_arr = np.zeros(self.bins + 1)
        self.class_pos = np.zeros(self.bins + 1)

    
    def cal_tp_pos_fp_neg(self, output, target, nclass, score_thresh):

        if(self.do_sigmoid):
            predict = (torch.sigmoid(output) > score_thresh).float()
        else:
            predict = ( output > score_thresh).float()
        if len(target.shape) == 3:
            # 加一个维度 使得target与 output的size一致
            target = np.expand_dims(target.float(), axis=1)
        elif len(target.shape) == 4:
            target = target.float()
        else:
            raise ValueError("Unknown target dimension")

        intersection = predict * ((predict == target).float())
        tp = intersection.sum()
        fp = (predict * ((predict != target).float())).sum()
        tn = ((1 - predict) * ((predict == target).float())).sum()
        fn = (((predict != target).float()) * (1 - predict)).sum()
        pos = tp + fn
        neg = fp + tn
        class_pos = tp + fp
        return tp, pos, fp, neg, class_pos


# used in most projects
class SeRankDet_PD_FA():
    def __init__(self, nclass, bins):
        super(SeRankDet_PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)
        self.reset()

    def update(self, preds, labels, size):

        for iBin in range(self.bins + 1):
            score_thresh = (iBin + 0.0) / self.bins
            batch = preds.size()[0]
            for b in range(batch):
                predits = np.array((preds[b, :, :, :] > score_thresh).cpu()).astype('int64')
                labelss = np.array((labels[b, :, :, :]).cpu()).astype('int64')
                image = measure.label(predits, connectivity=2)
                coord_image = measure.regionprops(image)
                label = measure.label(labelss, connectivity=2)
                coord_label = measure.regionprops(label)

                self.target[iBin] += len(coord_label)
                self.image_area_total = []
                self.image_area_match = []
                self.distance_match = []
                self.dismatch = []

                for K in range(len(coord_image)):
                    area_image = np.array(coord_image[K].area)
                    self.image_area_total.append(area_image)

                for i in range(len(coord_label)):
                    centroid_label = np.array(list(coord_label[i].centroid))
                    for m in range(len(coord_image)):
                        centroid_image = np.array(list(coord_image[m].centroid))
                        distance = np.linalg.norm(centroid_image - centroid_label)
                        area_image = np.array(coord_image[m].area)
                        if distance < 3:
                            self.distance_match.append(distance)
                            self.image_area_match.append(area_image)

                            del coord_image[m]
                            break

                self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
                self.FA[iBin] += np.sum(self.dismatch)
                self.all_pixel += size[0] * size[1]
                self.PD[iBin] += len(self.distance_match)

    def get(self):
        Final_FA = self.FA / self.all_pixel
        Final_PD = self.PD / self.target

        return Final_PD[int(self.bins/2)], Final_PD, Final_FA[int(self.bins/2)], Final_FA

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])
        self.all_pixel = 0
        self.target = np.zeros(self.bins + 1)


# Not used
class BasicIRSTD_PD_FA():
    def __init__(self, nclass, bins):
        super(BasicIRSTD_PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)
        self.reset()

    def update(self, preds, labels, size):
        for iBin in range(self.bins + 1):
            score_thresh = (iBin + 0.0) / self.bins
            batch = preds.size()[0]
            for b in range(batch):
                predits = np.array((preds[b, :, :, :] > score_thresh).cpu()).astype('int64')
                labelss = np.array((labels[b, :, :, :]).cpu()).astype('int64')  # P
                image = measure.label(predits, connectivity=2)
                coord_image = measure.regionprops(image)

                label = measure.label(labelss, connectivity=2)
                coord_label = measure.regionprops(label)

                self.target[iBin] += len(coord_label)
                self.image_area_total = []
                self.image_area_match = []
                self.distance_match = []

                self.dismatch = []

                for K in range(len(coord_image)):
                    area_image = np.array(coord_image[K].area)
                    self.image_area_total.append(area_image)

                true_img = np.zeros(predits.shape)
                for i in range(len(coord_label)):
                    centroid_label = np.array(list(coord_label[i].centroid))
                    for m in range(len(coord_image)):
                        centroid_image = np.array(list(coord_image[m].centroid))
                        distance = np.linalg.norm(centroid_image - centroid_label)
                        if distance < 3:
                            self.distance_match.append(distance)
                            # self.image_area_match.append(area_image)
                            true_img[coord_image[m].coords[:,0], coord_image[m].coords[:,1]] = 1

                            del coord_image[m]
                            break

                self.FA[iBin] += (predits - true_img).sum()
                self.all_pixel += size[0] * size[1]
                self.PD[iBin] += len(self.distance_match)

    def get(self):
        Final_FA = self.FA / self.all_pixel

        Final_PD = self.PD / self.target

        return Final_PD[int(self.bins/2)], Final_PD, Final_FA[int(self.bins/2)], Final_FA

    def reset(self):

        self.FA = np.zeros([self.bins + 1])

        self.PD = np.zeros([self.bins + 1])

        self.all_pixel = 0
        self.target = np.zeros(self.bins + 1)

class PD_FA():
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins + 1)
        self.PD = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)

        self.reset()

    def update(self, preds, labels, size):

        for iBin in range(self.bins + 1):
            score_thresh = (iBin + 0.0) / self.bins
            batch = preds.size()[0]
            for b in range(batch):
                predits = np.array((preds[b, :, :, :] > score_thresh).cpu()).astype('int64')
                labelss = np.array((labels[b, :, :, :]).cpu()).astype('int64')  # P

                image = measure.label(predits, connectivity=2)
                coord_image = measure.regionprops(image)

                label = measure.label(labelss, connectivity=2)
                coord_label = measure.regionprops(label)

                self.target[iBin] += len(coord_label)

                self.image_area_total = []
                self.image_area_match = []
                self.distance_match = []


                for K in range(len(coord_image)):
                    area_image = np.array(coord_image[K].area)
                    self.image_area_total.append(area_image)

                true_img = np.zeros(labelss.shape)
                for i in range(len(coord_label)):
                    centroid_label = np.array(list(coord_label[i].centroid))
                    for m in range(len(coord_image)):
                        centroid_image = np.array(list(coord_image[m].centroid))
                        distance = np.linalg.norm(centroid_image - centroid_label)
                        if distance < 3: # 这个3作为一个参数传入会更好点
                            self.distance_match.append(distance)
                            true_img[coord_label[i].coords[:,0], coord_label[i].coords[:,1]] = 1

                            del coord_image[m]
                            break

                self.FA[iBin] += (predits * ((predits != true_img).astype('int64'))).sum()

                self.all_pixel += size[0] * size[1]
                self.PD[iBin] += len(self.distance_match)

    def get(self):

        Final_FA = self.FA / self.all_pixel

        Final_PD = self.PD / self.target

        return Final_PD[int(self.bins/2)], Final_PD, Final_FA[int(self.bins/2)], Final_FA

    def reset(self):
        self.FA = np.zeros([self.bins + 1])
        self.PD = np.zeros([self.bins + 1])

        self.all_pixel = 0
        self.target = np.zeros(self.bins + 1)


if __name__ == '__main__':
    m1 = SigmoidMetric(score_thresh=0.5)
    m2 = SamplewiseSigmoidMetric(nclass=1, score_thresh=0.5)

    m3 = ROCMetric(1,10)
    m4 = SeRankDet_PD_FA(1,10)
    
    for i in tqdm( range(100) ):
        
        pred = torch.rand(16, 1, 256, 256)
        target = torch.ones(16, 1, 256, 256)

        m1.update(pred, target)
        m2.update(pred, target)
        Pd_or_Recall, mIoU = m1.get()
        Single_IoU_List, nIoU = m2.get()
        print("Pd_or_Recall:{}\r\n mIoU:{}\r\n Single_IoU_List:{}\r\n nIoU:{}\r\n".format(Pd_or_Recall, mIoU, Single_IoU_List, nIoU))

        m3.update(pred, target)
        m4.update(pred, target, pred[0, 0].shape)

        tp_rates, fp_rates, recall, precision, f1_score, f1_score_all = m3.get()
        print("tp_rates:{}\r\n fp_rates:{}\r\n recall:{}\r\n precision:{}\r\n f1_score:{}\r\n f1_score_all:{}\r\n".format(tp_rates, fp_rates, recall, precision, f1_score, f1_score_all))
        Final_FA, Final_FA_All, Final_PD ,Final_PD_All= m4.get()
        print("Final_FA:{}\r\n Final_PD:{}\r\n".format(Final_FA, Final_PD))