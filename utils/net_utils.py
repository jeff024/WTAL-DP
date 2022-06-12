#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
import torch
import random
import numpy as np
import torch.nn as nn


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def weights_init(model):
    if isinstance(model, nn.Conv1d):
        model.weights.data.normal_(0.0, 0.001)
    elif isinstance(model, nn.Linear):
        model.weights.data.normal_(0.0, 0.001)



class clfLoss(nn.Module):

    def __init__(self):
        super(clfLoss, self).__init__()

    def forward(self, inputs, label):
        return - torch.mean(torch.sum(torch.log(inputs) * label, dim=1))


class WtadLoss(nn.Module):
    def __init__(self, args):
        super(WtadLoss, self).__init__()
        self.batch_size = args.batch_size
        self.device = args.device
        self.bce = clfLoss()
        self.dataset = args.dataset
        self.feat_margin = 50

    def forward(self, preds, v_label, attentions, sub_cas, cas, features):

        self.batch_size = preds[0].shape[0]

        back_label = torch.hstack(
            (torch.zeros_like(v_label),
             torch.ones((self.batch_size, 1), device=self.device)
             )
        )
        cont_label = torch.hstack(
            (v_label,
             torch.ones((self.batch_size, 1), device=self.device),
             )
        )
        inst_label = torch.hstack(
            (v_label,
             torch.zeros((self.batch_size, 1), device=self.device),
             )
        )

        back_label = back_label / torch.sum(back_label, dim=1, keepdim=True)
        cont_label = cont_label / torch.sum(cont_label, dim=1, keepdim=True)
        inst_label = inst_label / torch.sum(inst_label, dim=1, keepdim=True)

        back_loss = self.bce(preds[0], back_label)
        cont_loss = self.bce(preds[1], cont_label)
        inst_loss = self.bce(preds[2], inst_label)

        if self.dataset == "ActivityNet":
            clf_loss = sum([back_loss, cont_loss, 5*inst_loss])
        else:
            clf_loss = sum([back_loss, cont_loss, inst_loss])

        #####################################################################
        norm = torch.norm(attentions, p=1, dim=2)
        att_loss = torch.mean(torch.sum(torch.abs(1-norm), dim=1))

        ########################################################################

        cas_sep_loss = (1 - sub_cas[2][:, :, -1]) - \
            attentions[:, :, 0].detach()
        cas_sep_loss = torch.mean(torch.sum(torch.abs(cas_sep_loss), dim=1))

        back_sep_loss = sub_cas[0][:, :, -1]-attentions[:, :, 2].detach()
        back_sep_loss = torch.mean(torch.sum(torch.abs(back_sep_loss), dim=1))

        cont_sep_loss = sub_cas[1][:, :, -1] - attentions[:, :, 1].detach()
        cont_sep_loss = torch.mean(torch.sum(torch.abs(cont_sep_loss), dim=1))

        guid_loss = cas_sep_loss + back_sep_loss + cont_sep_loss
        ###############################################################################
        act_inst_feat = torch.mean(features[2], dim=1)
        act_cont_feat = torch.mean(features[1], dim=1)
        act_back_feat = torch.mean(features[0], dim=1)

        act_inst_feat_norm = torch.norm(act_inst_feat, p=2, dim=1)
        act_cont_feat_norm = torch.norm(act_cont_feat, p=2, dim=1)
        act_back_feat_norm = torch.norm(act_back_feat, p=2, dim=1)

        feat_loss_1 = self.feat_margin - act_inst_feat_norm + act_cont_feat_norm
        feat_loss_1[feat_loss_1 < 0] = 0
        feat_loss_2 = self.feat_margin - act_cont_feat_norm + act_back_feat_norm
        feat_loss_2[feat_loss_2 < 0] = 0
        feat_loss_3 = act_back_feat_norm
        feat_loss = torch.mean((feat_loss_1 + feat_loss_2 + feat_loss_3)**2)
        ##################################################################################
        att_loss = 0.01*att_loss
        guid_loss = 0.002*guid_loss
        feat_loss = 5e-5 * feat_loss

        loss = clf_loss + att_loss + guid_loss + feat_loss
        ##############################################################################
        loss_dict = {}
        loss_dict['clf_loss'] = clf_loss
        loss_dict['att_loss'] = att_loss
        loss_dict['guid_loss'] = guid_loss
        loss_dict['feat_loss'] = feat_loss

        return loss, loss_dict


