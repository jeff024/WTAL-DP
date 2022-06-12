import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        # vid_label_t = torch.hstack((vid_label_t, torch.ones((input_feature.shape[0], 1), device=device)))
        acc = torch.sum(torch.sum((pred[:, :args.action_cls_num] == vid_label_t),
                        dim=1) == vid_label_t.shape[1]).item()/input_feature.shape[0]
        total_acc.append(acc)

    for key in total_loss:
        total_loss[key] = np.mean(total_loss[key])
    acc = np.mean(total_acc)
    print("\ntrain loss: ", total_loss['loss'])
    print("train acc: ", acc)
    return total_loss, acc


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
        cls_prediction = np.array([np.argmax(score_np)], dtype=int)

    temp_cls_score_np = temp_cls_score_np[:, cls_prediction]
    temp_att_ins_score_np = temp_att_ins_score_np[:, cls_prediction]
    temp_att_con_score_np = temp_att_con_score_np[:, cls_prediction]

    int_temp_cls_scores = upgrade_resolution(
        temp_cls_score_np, args.test_upgrade_scale)
    int_temp_att_ins_score_np = upgrade_resolution(
        temp_att_ins_score_np, args.test_upgrade_scale)
    # int_temp_att_con_score_np = upgrade_resolution(temp_att_con_score_np, args.test_upgrade_scale)

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

        props_list = get_proposal_oic(tmp_seg_list, (1.0*tmp_int_cas + 0.0*int_temp_att_ins_score_np),
                                      cls_prediction, score_np, t_factor, lamb=0.2, gamma=0.0)

        for i in range(len(props_list)):
            if len(props_list[i]) == 0:
                continue
            class_id = props_list[i][0][0]

            if class_id not in proposal_dict.keys():
                proposal_dict[class_id] = []

            proposal_dict[class_id] += props_list[i]

    for att_thresh in att_thresholds:

        tmp_int_att = int_temp_att_ins_score_np.copy()
        zero_location = np.where(tmp_int_att < att_thresh)
        tmp_int_att[zero_location] = 0

        tmp_seg_list = []
        for c_idx in range(len(cls_prediction)):
            pos = np.where(tmp_int_att[:, c_idx] >= att_thresh)
            tmp_seg_list.append(pos)

        props_list = get_proposal_oic(tmp_seg_list, (1.0*int_temp_cls_scores +
                                      0.0*tmp_int_att), cls_prediction, score_np, t_factor, lamb=0.2, gamma=0.0)

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

    acc_vid_list = []
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
            if phase == "test" or phase == 'plot_test':
                if phase == "plot_test":
                    acc_vid_list.append(
                        [vid_name[0], (acc == 1), args.class_name_lst[np.argmax(vid_label_t.cpu().numpy())]])
            ######################################
                # inference
                inst_cas = sub_cas[2]
                ################################################################
                # feature fusion
                back_cas = sub_cas[0]
                inst_cas_fuse = inst_cas[:,:,:-1] * (1 - back_cas[:,:,-1].unsqueeze(2))
                inst_cas_fuse = torch.cat((inst_cas_fuse, inst_cas[:,:,-1].unsqueeze(2)), dim=2)
                inst_cas = inst_cas_fuse
                ##################################################################
                final_proposals = inference(args, preds, inst_cas, attentions, t_factor,
                                                clf_thresholds=args.clf_thresh, att_thresholds=args.att_thresh, nms_threshold=args.nms_thresh)
                test_final_result['results'][vid_name[0]] = result2json(
                    final_proposals, args.class_name_lst)

    ########################################
    # evaluations
    if phase == "test" or phase == 'plot_test':
        test_final_json_path = os.path.join(
            args.save_dir, "final_test_{}_result.json".format(args.dataset))
        with open(test_final_json_path, 'w') as f:
            json.dump(test_final_result, f)

        anet_detection = ANETDetection(ground_truth_file=args.test_gt_file_path,
                                       prediction_file=test_final_json_path,
                                       tiou_thresholds=args.tiou_thresholds,
                                       subset="test")

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
    torch.multiprocessing.set_sharing_strategy('file_system')

    os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    # device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # exit(0)

    local_time = time.localtime()[0:5]
    this_dir = os.path.join(os.path.dirname(__file__), ".")
    if not args.test:
        save_dir = os.path.join(this_dir, "checkpoints_thumos", "checkpoints_thumos_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}"
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
                                      num_workers=args.num_workers, drop_last=False)

        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True,
                                     num_workers=args.num_workers, drop_last=False)

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

            print("best test mAP: ",best_test_mAP)
            checkpoint_file = "{}_best_checkpoint.pth".format(args.dataset)
            path = os.path.join(save_dir, checkpoint_file)
            path.replace('\\', '/')
            print(
                f"python main_thu.py --test --checkpoint {path[2:]}")

            latest_checkpoint_file = "{}_latest_checkpoint.pth".format(args.dataset)
            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': model.state_dict()
            }, os.path.join(save_dir, latest_checkpoint_file))
            latest_checkpoint_file = os.path.join(save_dir, latest_checkpoint_file)
            latest_checkpoint_file.replace('\\', '/')
            print(f"python main_thu.py --test --checkpoint {latest_checkpoint_file[2:]}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
        fig.suptitle('Train Results Acc')
        ax1.plot(train_acc)
        ax1.set_title("train_acc")
        ax2.plot(test_acc)
        ax2.set_title("test_acc")
        plt.savefig("figs_thu/train_results_acc.jpg")

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
        plt.savefig("figs_thu/train_results_loss.jpg")

        fig = plt.figure()
        plt.title('test_mAP')
        plt.plot(test_maps)
        plt.savefig("figs_thu/test_mAPs.jpg")

        latest_checkpoint_file = "{}_latest_checkpoint.pth".format(args.dataset)
        torch.save({
            'epoch': epoch_idx,
            'model_state_dict': model.state_dict()
        }, os.path.join(save_dir, latest_checkpoint_file))

        checkpoint_file = "{}_best_checkpoint.pth".format(args.dataset)
        path = os.path.join(save_dir, checkpoint_file)
        path.replace('\\', '/')
        latest_path = os.path.join(save_dir, latest_checkpoint_file)
        latest_path.replace('\\', '/')

        print(
            f"python main_thu.py --test --checkpoint {path[2:]} > figs_thu/map@tiou.txt")
        print(
            f"python main_thu.py --test --checkpoint {latest_path[2:]}")

        print("best_mAP =", best_test_mAP)


        checkpoint_file = "{}_best_checkpoint.pth".format(args.dataset)
        path = os.path.join(save_dir, checkpoint_file)
        checkpoint = torch.load(
            path, map_location=torch.device(args.device))
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        with torch.no_grad():
            te_loss, te_acc, test_mAP = test(
                    args, model, test_dataloader, criterion, phase="plot_test")

    else:
        test_dataset = build_dataset(args, phase="test", sample="uniform")
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                     pin_memory=True, drop_last=False)
        criterion = WtadLoss(args)

        with torch.no_grad():
            _, _, _ = test(args, model, test_dataloader, criterion, phase='plot_test')


if __name__ == "__main__":
    set_random_seed()
    args = build_args(dataset="THUMOS")
    print(args)
    main(args)


