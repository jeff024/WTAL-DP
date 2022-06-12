import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
# from pyrsistent import T
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from config.model_config import build_args
from dataset.dataset_class import build_dataset
from model.WtadNet import WtadNet
from utils.net_evaluation import (ANETDetection, get_proposal_oic, grouping,
                                  nms, result2json, upgrade_resolution)
from utils.net_utils import WtadLoss, set_random_seed

"""   
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
#                                              TRAIN FUNCTION                                                   #
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
"""


def train(args, model, dataloader, criterion, optimizer):
    model.train()
    print("-------------------------------------------------------------------------------")
    device = args.device

    # train_process

    total_loss = {
        'loss': [],
        'clf_loss': [],
        'att_loss': [],
        'guid_loss': [],
        'feat_loss': []
    }
    total_acc = []
    for input_feature, vid_label_t, sample_ratio in tqdm(dataloader):

        vid_label_t = vid_label_t.to(device)
        input_feature = input_feature.to(device)

        preds, attentions, sub_cas, cas, features = model(input_feature)
        loss, loss_dict = criterion(
            preds, vid_label_t, attentions,  sub_cas, cas, features)

        optimizer.zero_grad()
        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()

        total_loss['loss'].append(loss.item())
        for key in loss_dict:
            total_loss[key].append(loss_dict[key].item())
        pred = (preds[-1] > args.cls_threshold).float()
        acc = torch.sum(torch.sum((pred[:, :args.action_cls_num] == vid_label_t),
                        dim=1) == vid_label_t.shape[1]).item()/input_feature.shape[0]
        total_acc.append(acc)

    for key in total_loss:
        total_loss[key] = np.mean(total_loss[key])
    acc = np.mean(total_acc)
    print("\ntrain loss: ", total_loss['loss'])
    print("train acc: ", acc)
    return total_loss, acc


def oic(args, proposal_by_class, inst_score, lamb=0.1):
    proposals = []
    nums_class = len(proposal_by_class)
    for c_id in range(nums_class):
        c_temp = []
        temp_list = proposal_by_class[c_id][0].cpu().numpy()
        grouped_temp_list = grouping(temp_list)
        for p_id in range(len(grouped_temp_list)):

            if len(grouped_temp_list) < 2:
                continue
            inner_score = torch.mean(inst_score[grouped_temp_list[p_id], c_id])

            len_proposal = len(grouped_temp_list[p_id])
            inner_s = grouped_temp_list[p_id][0]
            inner_e = grouped_temp_list[p_id][-1]
            outer_s = max(0, int(inner_s - lamb*len_proposal))
            outer_e = min(
                int(inst_score.shape[0] - 1), int(inner_e + lamb*len_proposal))

            outer_temp_list = list(range(outer_s, inner_s)) + \
                list(range(int(inner_e + 1), outer_e + 1))

            if len(outer_temp_list) == 0:
                outer_score = 0
            else:
                outer_score = torch.mean(inst_score[outer_temp_list, c_id])

            clf_score = inner_score - outer_score
            # clf_score = inner_score
            t_start = inner_s * 16/25
            t_end = (inner_e+1)*16/25
            if t_start != t_end:
                c_temp.append([c_id, clf_score.item(
                ), t_start/args.test_upgrade_scale, t_end/args.test_upgrade_scale])
        proposals.append(c_temp)
    return proposals


def inference(args, preds, inst_feature, attentions, t_factor, clf_thresholds=np.arange(0.1, 0.6, 0.05), att_thresholds=np.arange(0.1, 1.00, 0.05), nms_threshold=0.55):

    temp_cas = inst_feature
    temp_att = attentions
    act_inst_cls = preds[2]
    #--------------------------------------------------------------------------#
    #--------------------------------------------------------------------------#
    fg_score = act_inst_cls[:, :args.action_cls_num]
    score_np = fg_score.cpu().numpy()
    # #--------------------------------------------------------------------------#
    #--------------------------------------------------------------------------#
    # GENERATE PROPORALS.
    temp_cls_score_np = temp_cas[:, :, :args.action_cls_num].cpu().numpy()
    temp_cls_score_np = np.reshape(
        temp_cls_score_np, (temp_cas.shape[1], args.action_cls_num, 1))
    temp_att_ins_score_np = temp_att[:, :, 0].unsqueeze(
        2).expand([-1, -1, args.action_cls_num]).cpu().numpy()
    temp_att_con_score_np = temp_att[:, :, 1].unsqueeze(
        2).expand([-1, -1, args.action_cls_num]).cpu().numpy()
    temp_att_ins_score_np = np.reshape(
        temp_att_ins_score_np, (temp_cas.shape[1], args.action_cls_num, 1))
    temp_att_con_score_np = np.reshape(
        temp_att_con_score_np, (temp_cas.shape[1], args.action_cls_num, 1))

    score_np = np.reshape(score_np, (-1))
    if score_np.max() > args.cls_threshold:
        cls_prediction = np.array(np.where(score_np > args.cls_threshold)[0])
    else:
        cls_prediction = np.array([np.argmax(score_np)], dtype=np.int)

    temp_cls_score_np = temp_cls_score_np[:, cls_prediction]
    temp_att_ins_score_np = temp_att_ins_score_np[:, cls_prediction]
    temp_att_con_score_np = temp_att_con_score_np[:, cls_prediction]

    int_temp_cls_scores = upgrade_resolution(
        temp_cls_score_np, args.test_upgrade_scale)
    int_temp_att_ins_score_np = upgrade_resolution(
        temp_att_ins_score_np, args.test_upgrade_scale)

    proposal_dict = {}
    # CAS based proposal generation
    for act_thresh in clf_thresholds:

        tmp_int_cas = int_temp_cls_scores.copy()
        zero_location = np.where(tmp_int_cas < act_thresh)
        tmp_int_cas[zero_location] = 0

        tmp_seg_list = []
        for c_idx in range(len(cls_prediction)):
            pos = np.where(tmp_int_cas[:, c_idx] >= act_thresh)
            tmp_seg_list.append(pos)

        props_list = get_proposal_oic(tmp_seg_list, (0.70*tmp_int_cas + 0.30*int_temp_att_ins_score_np),
                                      cls_prediction, score_np, t_factor, lamb=0.150, gamma=0.0)

        for i in range(len(props_list)):
            if len(props_list[i]) == 0:
                continue
            class_id = props_list[i][0][0]

            if class_id not in proposal_dict.keys():
                proposal_dict[class_id] = []

            proposal_dict[class_id] += props_list[i]

    # att_act_thresh = []
    for att_thresh in att_thresholds:

        tmp_int_att = int_temp_att_ins_score_np.copy()
        zero_location = np.where(tmp_int_att < att_thresh)
        tmp_int_att[zero_location] = 0

        tmp_seg_list = []
        for c_idx in range(len(cls_prediction)):
            pos = np.where(tmp_int_att[:, c_idx] >= att_thresh)
            tmp_seg_list.append(pos)

        props_list = get_proposal_oic(tmp_seg_list, (0.70*int_temp_cls_scores + 0.30*tmp_int_att),
                                      cls_prediction, score_np, t_factor, lamb=0.150, gamma=0.250)

        for i in range(len(props_list)):
            if len(props_list[i]) == 0:
                continue
            class_id = props_list[i][0][0]

            if class_id not in proposal_dict.keys():
                proposal_dict[class_id] = []

            proposal_dict[class_id] += props_list[i]

    # NMS
    final_proposals = []

    for class_id in proposal_dict.keys():
        final_proposals.append(nms(proposal_dict[class_id], nms_threshold))

    return final_proposals


"""   
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
#                                              TEST FUNCTION                                                    #
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
"""


def test(args, model, dataloader, criterion, phase="test"):

    model.eval()
    device = args.device
    test_mAP = 0
    print("-------------------------------------------------------------------------------")
    total_loss = {
        'loss': [],
        'clf_loss': [],
        'att_loss': [],
        'guid_loss': [],
        'feat_loss': []
    }
    total_acc = []

    ###########################################################################################
    test_final_result = dict()
    test_final_result['version'] = 'VERSION 1.3'
    test_final_result['results'] = {}
    test_final_result['external_data'] = {
        'used': True, 'details': 'Features from I3D Net'}
    ###############################################################################################

    with torch.no_grad():
        for vid_name, input_feature, vid_label_t, vid_len, vid_duration, sample_ratio in tqdm(dataloader):
            vid_label_t = vid_label_t.to(device)
            input_feature = input_feature.to(device)
            t_factor = (args.segment_frames_num * vid_len) / \
                (args.frames_per_sec *
                 args.test_upgrade_scale * input_feature.shape[1])

            preds, attentions, sub_cas, cas, features = model(input_feature)
            loss, loss_dict = criterion(
                preds, vid_label_t, attentions,  sub_cas, cas, features)
            total_loss['loss'].append(loss.item())
            for key in loss_dict:
                total_loss[key].append(loss_dict[key].item())
            pred = (preds[-1] > args.cls_threshold).float()
            pred = pred[:, :args.action_cls_num]
            acc = torch.sum(torch.sum((pred == vid_label_t), dim=1)
                            == vid_label_t.shape[1]).item()/input_feature.shape[0]
            total_acc.append(acc)

        # ####################################
            if phase == "test":
                ######################################
                # inference
                inst_feature = sub_cas[2]
                final_proposals = inference(args, preds, inst_feature, attentions, t_factor,
                                                clf_thresholds=args.clf_thresh, att_thresholds=args.att_thresh, nms_threshold=args.nms_thresh)
                test_final_result['results'][vid_name[0]] = result2json(
                    final_proposals, args.class_name_lst)

    ########################################
    # evaluations
    if phase == "test":
        test_final_json_path = os.path.join(
            args.save_dir, "final_test_{}_result.json".format(args.dataset))
        with open(test_final_json_path, 'w') as f:
            json.dump(test_final_result, f)

        anet_detection = ANETDetection(ground_truth_file=args.test_gt_file_path,
                                       prediction_file=test_final_json_path,
                                       tiou_thresholds=args.tiou_thresholds,
                                       subset="val")

        test_mAP = anet_detection.evaluate()
    #######################################

    for key in total_loss:
        total_loss[key] = np.mean(total_loss[key])
    acc = np.mean(total_acc)
    print("\ntest loss: ", total_loss['loss'])
    print("test acc: ", acc)

    return total_loss, acc, test_mAP


"""   
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
#                                              MAIN FUNCTION                                                    #
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
"""


def main(args):
    os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_time = time.localtime()[0:5]
    this_dir = os.path.join(os.path.dirname(__file__), ".")
    if not args.test:
        save_dir = os.path.join(this_dir, "checkpoints_act", "checkpoints_act_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}"
                                .format(local_time[0], local_time[1], local_time[2],
                                        local_time[3], local_time[4]))
    else:
        save_dir = os.path.dirname(args.checkpoint)

    args.save_dir = save_dir
    args.device = device

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model = WtadNet(args)

    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(
            args.checkpoint, map_location=torch.device(args.device))
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    if not args.test:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(
            0.9, 0.999), weight_decay=args.weight_decay)

        train_dataset = build_dataset(args, phase="train", sample="random")
        test_dataset = build_dataset(args, phase="test", sample="uniform")

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                      num_workers=1, drop_last=False)

        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True,
                                     num_workers=1, drop_last=False)

        criterion = WtadLoss(args)

        best_test_mAP = 0

        train_loss = {
            'loss': [],
            'clf_loss': [],
            'att_loss': [],
            'guid_loss': [],
            'feat_loss': []
        }
        train_acc = []
        test_loss = {
            'loss': [],
            'clf_loss': [],
            'att_loss': [],
            'guid_loss': [],
            'feat_loss': []
        }
        test_acc = []
        test_maps = []

        for epoch_idx in tqdm(range(args.start_epoch, args.epochs)):
            t_loss, t_acc = train(
                args, model, train_dataloader, criterion, optimizer)
            te_loss, te_acc, test_mAP = test(
                args, model, test_dataloader, criterion, phase="test")

            for key in t_loss:
                train_loss[key].append(t_loss[key])
            train_acc.append(t_acc)
            for key in te_loss:
                test_loss[key].append(te_loss[key])
            test_acc.append(te_acc)
            test_maps.append(test_mAP)

            if test_mAP > best_test_mAP:
                best_test_mAP = test_mAP
                checkpoint_file = "{}_best_checkpoint.pth".format(args.dataset)
                torch.save({
                    'epoch': epoch_idx,
                    'model_state_dict': model.state_dict()
                }, os.path.join(save_dir, checkpoint_file))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
        fig.suptitle('Train Results Acc')
        ax1.plot(train_acc)
        ax1.set_title("train_acc")
        ax2.plot(test_acc)
        ax2.set_title("test_acc")
        plt.savefig("figs_act/train_results_acc.jpg")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
        fig.suptitle('Train Results Loss')
        for key in train_loss:
            ax1.plot(train_loss[key], label=key)
        ax1.legend()
        ax1.set_title("train_loss")
        for key in test_loss:
            ax2.plot(test_loss[key], label=key)
        ax2.legend()
        ax2.set_title("test_loss")
        plt.savefig("figs_act/train_results_loss.jpg")

        fig = plt.figure()
        plt.title('test_mAP')
        plt.plot(test_maps)
        plt.savefig("figs_act/test_mAPs.jpg")

        checkpoint_file = "{}_latest_checkpoint.pth".format(args.dataset)
        torch.save({
            'epoch': epoch_idx,
            'model_state_dict': model.state_dict()
        }, os.path.join(save_dir, checkpoint_file))
        path = os.path.join(save_dir, checkpoint_file)
        path.replace('\\', '/')
        print(
            f"python main_act.py --test --checkpoint {path[2:]}")


        # final eval
        with torch.no_grad():
            _, _, _ = test(args, model, test_dataloader, criterion)

    else:
        test_dataset = build_dataset(args, phase="plot_test", sample="uniform")
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                     pin_memory=True, drop_last=False)
        criterion = WtadLoss(args)

        with torch.no_grad():
            _, _, _ = test(args, model, test_dataloader, criterion)


if __name__ == "__main__":

    set_random_seed()
    args = build_args(dataset="ActivityNet")
    print(args)
    main(args)
