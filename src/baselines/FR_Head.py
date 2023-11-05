import torch
import torch.nn as nn
import torch.nn.functional as F


class CL_Module(nn.Module):
    """
    Contrastive Learning Module, main functions includes:
        calculate Global Prototype, Local FN & FP Prototypes
        calculate distances between these prototypes and original features
        InfoNCE Loss calculation
    """
    def __init__(self, channels, classes, in_channels=256, alp=0.125, tmp=0.125, mom=0.9, pred_threshold=0.0, use_p_map=True):
        super(CL_Module, self).__init__()
        self.channels = in_channels
        self.classes = classes

        self.alp = alp
        self.tmp = tmp
        self.mom = mom

        # global prototype of all the classes (each column is a prototype)
        self.global_prototype = torch.randn(in_channels, classes, requires_grad=False)  # [C, K]

        self.embed = nn.Linear(channels, in_channels)

        self.use_p_map = use_p_map

        self.infoNCE = nn.CrossEntropyLoss(reduction='none')

        self.pred_threshold = pred_threshold

    def onehot(self, labels):
        """
        :param labels:      true labels of this batch of data
                                                             [B]
        :return:
                            one hot format labels
        """
        label = labels.clone()
        ones = torch.sparse.torch.eye(self.classes).to(labels.device)
        ones = ones.index_select(0, label.long())
        return ones.float()

    def get_mask_fn_fp(self, lbl_one, pred_one, logit):
        """
        :param lbl_one:                 true labels in onehot format
        :param pred_one:                predict labels in onehot format
        :param logit:                   predict probabilities
                            All in shape [N, K], N stands for batch size while K means categories
        :return:

        """
        tp = lbl_one * pred_one  # only true positive samples will remain 1 in this matrix
        fn = lbl_one - tp  # remain the true labels of those misclassified samples
        fp = pred_one - tp  # remain the predicted labels of those misclassified samples

        tp = tp * (logit > self.pred_threshold).float()

        num_fn = fn.sum(0).unsqueeze(1)  # [K, 1] the num of FN samples of each category
        has_fn = (num_fn > 1e-8).float()  #
        num_fp = fp.sum(0).unsqueeze(1)  # [K, 1]
        has_fp = (num_fp > 1e-8).float()
        return tp, fn, fp, has_fn, has_fp

    def local_avg_tp_fn_fp(self, f, mask, fn, fp):
        """
        :param f:               backbone network features [N, C]
        :param mask:            TP matrix                 [N, K]
        :param fn:              FN matrix                 [N, K]
        :param fp:
        :return:
                                local prototypes of FN and FP samples
                                global prototypes of all categories
        """
        global_prototype = self.global_prototype.detach().to(f.device)

        features = torch.permute(f, [1, 0])  # [C, N]
        # column map matrix multiplication helps quickly understand this    [C, K]

        """
        divide 0 error
        """
        # local_avg_fn = torch.matmul(features, fn)
        # local_avg_fn = local_avg_fn / torch.permute(num_fn, [1, 0])
        #
        # local_avg_fp = torch.matmul(features, fp)
        # local_avg_fp = local_avg_fp / torch.permute(num_fp, [1, 0])

        fn = F.normalize(fn, p=1, dim=0)
        fp = F.normalize(fp, p=1, dim=0)

        local_avg_fn = torch.matmul(features, fn)
        local_avg_fp = torch.matmul(features, fp)

        local_avg_tp = torch.matmul(features, mask)
        num_tp = torch.sum(mask, dim=0, keepdim=True)
        local_avg_tp = local_avg_tp / (num_tp + 1e-12)

        # if there's no prototype of some categories in this batch, then do not update rather than use '0' to update
        tp_mask = (num_tp > 1e-8).float()  # [1, K]
        tp_mask[tp_mask > 0.1] = self.mom
        tp_mask[tp_mask <= 0.1] = 1.0

        # global will gradually converge during training process,
        # local_avg_tp are prototypes of all the classes in this batch
        # tp_mask is EMA moving average coefficient
        global_prototype = global_prototype * tp_mask + local_avg_tp * (1 - tp_mask)
        with torch.no_grad():
            self.global_prototype = global_prototype.cpu()

        return global_prototype, local_avg_fn, local_avg_fp

    def get_score(self, feature, lbl_one, logit, global_prototype, local_avg_fn, local_avg_fp, s_fn, s_fp, tp):
        """
        :param feature:                 [N, C]
        :param lbl_one:                 [N, K]
        :param logit:                   [N, k]
        :param global_prototype:        [C, K]
        :param local_avg_fn:            [C, K]
        :param local_avg_fp:            [C, K]
        :param s_fn:                    FN mask, [K, 1] (0/1 matrix) whether each category has FN samples
        :param s_fp:                    FP mask
        :param tp:                      TP Matrix [N, K]
        :return:
        """
        # L2 normalization
        feature = feature / torch.norm(feature, p=2, dim=1, keepdim=True) + 1e-1

        global_prototype = torch.permute(global_prototype, [1, 0])
        global_prototype = global_prototype / (torch.norm(global_prototype, p=2, dim=1, keepdim=True) + 1e-12)

        local_avg_fp = torch.permute(local_avg_fp, [1, 0])
        local_avg_fp = local_avg_fp / (torch.norm(local_avg_fp, p=2, dim=1, keepdim=True) + 1e-12)

        local_avg_fn = torch.permute(local_avg_fn, [1, 0])
        local_avg_fn = local_avg_fn / (torch.norm(local_avg_fn, p=2, dim=1, keepdim=True) + 1e-12)

        """
        with above code, each row of the matrix is the L2 normalized prototype of corresponding category
        """

        if self.use_p_map:
            p_map = (1 - logit) * self.alp  # N, K
        else:
            p_map = self.alp  # N, K

        # distance of global prototypes to all the samples
        # each row represents the distance between one sample and all the global prototypes
        score_global_prototype = torch.matmul(feature, global_prototype.permute(1, 0))

        # directly use 'V0' as the baseline
        # each row represents the distance between one sample and all the mean FP & FN
        score_fn = torch.matmul(feature, local_avg_fn.permute(1, 0)) - 1
        score_fp = -torch.matmul(feature, local_avg_fp.permute(1, 0)) - 1

        fn_map = score_fn * p_map * tp  # only confident samples effect
        fp_map = score_fp * p_map * tp

        score_fn_cl = (score_global_prototype + fn_map) / self.tmp
        score_fp_cl = (score_global_prototype + fp_map) / self.tmp

        return score_fn_cl, score_fp_cl

    def forward(self, feature, lbl, logit):
        feature = self.embed(feature)

        pred = logit.max(1)[1]
        pred_one = self.onehot(pred)
        lbl_one = self.onehot(lbl)

        logit = torch.softmax(logit, 1)
        mask, fn, fp, has_fn, has_fp= self.get_mask_fn_fp(lbl_one, pred_one, logit)
        f_mem, f_fn, f_fp = self.local_avg_tp_fn_fp(feature, mask, fn, fp)
        score_cl_fn, score_cl_fp = self.get_score(feature, lbl_one, logit, f_mem, f_fn, f_fp, has_fn, has_fp, mask)

        return (self.infoNCE(score_cl_fn, lbl) + self.infoNCE(score_cl_fp, lbl)).mean()


class FR_Head(nn.Module):
    def __init__(self, in_channels, frames, joints, classes, h_channel=256):
        super(FR_Head, self).__init__()

        self.spatio_squeeze = nn.Sequential(nn.Conv2d(in_channels, h_channel // joints, kernel_size=1),
                                            nn.BatchNorm2d(h_channel // joints),
                                            nn.ReLU(True))

        self.tempor_squeeze = nn.Sequential(nn.Conv2d(in_channels, h_channel // frames, kernel_size=1),
                                            nn.BatchNorm2d(h_channel // frames),
                                            nn.ReLU(True))

        self.spatial_cl_net = CL_Module(h_channel // joints * joints, classes)

        self.temporal_cl_net = CL_Module(h_channel // frames * frames, classes)

    def forward(self, features, labels, logits):
        """
        :param features:      Featured generated by backbone network which can be ST-GCN etc.
                                                            [batch, channels, frames, joints]
        :param labels:        True Labels of this batch
                                                            [batch]
        :param logits:        Output of backbone network
                                                            [batch, classes]
        :return:
                Contrastive Learning Loss
        """
        spatial_features = torch.mean(features, dim=-2, keepdim=True)  # [b, c, 1, j]
        spatial_features = self.spatio_squeeze(spatial_features)  # [b, h // j, 1, j]
        spatial_features = torch.flatten(spatial_features, start_dim=1)  # [b, h]
        temporal_features = torch.mean(features, dim=-1, keepdim=True)
        temporal_features = self.tempor_squeeze(temporal_features)
        temporal_features = torch.flatten(temporal_features, start_dim=1)  # [b, h]

        spatial_cl_loss = self.spatial_cl_net(spatial_features, labels, logits)
        temporal_cl_loss = self.temporal_cl_net(temporal_features, labels, logits)

        return spatial_cl_loss + temporal_cl_loss
