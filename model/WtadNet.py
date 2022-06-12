import torch
import torch.nn as nn


class Myembedder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.feature_dim
        if args.dataset == "THUMOS":
            self.feature_embedding = nn.Sequential(
                # nn.Dropout(args.dropout),
                nn.Conv1d(in_channels=args.feature_dim,
                          out_channels=args.feature_dim, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm1d(args.feature_dim),
                nn.ReLU()
            )
        else:
            # We add more strong regularization operation to the ActivityNet dataset since this dataset contains much more diverse videos.
            self.feature_embedding = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Conv1d(in_channels=args.feature_dim,
                          out_channels=args.feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(args.feature_dim),
                nn.ReLU()
            )
               

    def forward(self, x):
        x = self.feature_embedding(x)
        return x



class WtadNet(nn.Module):
    def __init__(self, args):
        super(WtadNet, self).__init__()
        self.dataset = args.dataset
        self.feature_dim = args.feature_dim
        self.action_cls_num = args.action_cls_num + 1
        self.nums_att = 3
        self.n_encoder = 1

        self.dropout = nn.Dropout(args.dropout)
        self.feature_embedding = Myembedder(args)

        self.clf = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(in_features=self.feature_dim, out_features=(self.action_cls_num)),
        )
        self.conv_clf = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim, out_channels=self.action_cls_num, kernel_size=1),
        )
        self.attentions = nn.Sequential(
            nn.Conv1d(in_channels=self.feature_dim,
                      out_channels=self.nums_att, kernel_size=1, padding=0),
        )


    def forward(self, input_features):
        device = input_features.device
        batch_size, temp_len = input_features.shape[0], input_features.shape[1]
        mask = torch.ones((batch_size, 1, temp_len), device=device)

        input_features = input_features.permute(0, 2, 1)  # (batch, 2048, 750)
        input_features = self.feature_embedding(input_features)  # (batch, 2048, 750)

        ################################################################
        attentions = self.attentions(input_features)
        attentions = attentions.permute(0, 2, 1)  # (batch, 750, 3)
        attentions = torch.sigmoid(attentions)

        inst_att = attentions[:, :, 0].unsqueeze(2)  # (batch,750,1)
        cont_att = attentions[:, :, 1].unsqueeze(2)
        back_att = attentions[:, :, 2].unsqueeze(2)

        input_features = input_features.permute(0, 2, 1)  # (batch, 750, 2048)

        ###############################################################
        cas = self.clf((input_features))        
        ###############################################################
        back_cas = cas * back_att
        cont_cas = cas * cont_att
        inst_cas = cas * inst_att
        ###########################################################
        softmax_back_att = torch.softmax(back_att, dim=1)
        pred_back = torch.sum(softmax_back_att * back_cas, dim=1)
        softmax_cont_att = torch.softmax(cont_att, dim=1)
        pred_cont = torch.sum(softmax_cont_att * cont_cas, dim=1)
        softmax_inst_att = torch.softmax(inst_att, dim=1)
        pred_inst = torch.sum(softmax_inst_att * inst_cas, dim=1)
        ################################################################
        pred_back = torch.softmax(pred_back, dim=1)
        pred_cont = torch.softmax(pred_cont, dim=1)
        pred_inst = torch.softmax(pred_inst, dim=1)

        ################################################################
        back_cas = torch.softmax(back_cas, dim=2)
        cont_cas = torch.softmax(cont_cas, dim=2)
        inst_cas = torch.softmax(inst_cas, dim=2)

        ###############################################################
        back_feature = input_features * back_att
        cont_feature = input_features * cont_att
        inst_feature = input_features * inst_att
        ###############################################################
        cas = torch.softmax(cas, dim=2)



        return [pred_back, pred_cont, pred_inst], attentions, [back_cas, cont_cas, inst_cas], None, [back_feature, cont_feature, inst_feature]