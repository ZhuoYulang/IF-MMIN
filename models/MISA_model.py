from models.utils import ReverseLayerF
from models.utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD

import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.shared import SharedEncoder
from models.networks.project import ProjectEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN

gpu_id = None


def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)

# let's define a simple model that can deal with multimodal variable length sequence
class MISAModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='visual input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
                            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
                            help='visual embedding method,last,mean or atten')
        parser.add_argument('--cls_layers', type=str, default='128,128',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--modality', type=str, help='which modality to use for model')
        return parser

    def __init__(self, config):
        super().__init__(config)

        self.config = config  # 结尾标记#！的，都是需要从config中直接读取数据写进来的
        self.text_size = config.input_dim_l
        self.visual_size = config.input_dim_v
        self.acoustic_size = config.input_dim_a

        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = nn.ReLU()  # 激活函数
        self.tanh = nn.Tanh()

        config.hidden_size = config.embd_size_v

        self.loss_names = []
        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.loss_diff_func = DiffLoss()
        self.loss_recon_func = MSE()
        self.loss_cmd_func = CMD()

        # 以下是MMIN配置文件的内容
        self.model_names = ['private_t', 'private_v', 'private_a', 'shared', 'recon_t', 'recon_v', 'recon_a',
                            'sp_discriminator', 'fusion', 'tlayer_norm', 'vlayer_norm', 'alayer_norm',
                            'transformer_encoder']
        self.modality = config.modality

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(config.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # shared save_dir
        self.save_shared_dir = os.path.join(self.save_shared_dir, str(config.cvNo))
        if not os.path.exists(self.save_shared_dir):
            os.makedirs(self.save_shared_dir)

        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU

        # acoustic model
        self.netA = LSTMEncoder(config.input_dim_a, config.embd_size_a, embd_method=config.embd_method_a)
        self.model_names.append('A')
        # lexical model 文本
        self.netL = TextCNN(config.input_dim_l, config.embd_size_l)
        self.model_names.append('L')
        # visual model
        self.netV = LSTMEncoder(config.input_dim_v, config.embd_size_v, config.embd_method_v)
        self.model_names.append('V')

        # self.netL1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
        # self.netL2 = rnn(2 * hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        #
        # self.netV1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        # self.netV2 = rnn(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        #
        # self.netA1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        # self.netA2 = rnn(2 * hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        ##########################################
        # mapping modalities to same sized space        多模态信息映射到同一尺寸的空间
        ##########################################
        # self.netproject_t = nn.Sequential()
        # self.netproject_t.add_module('project_t',
        #                              nn.Linear(in_features=hidden_sizes[0] * 4, out_features=config.embd_size_l))
        # self.netproject_t.add_module('project_t_activation', self.activation)
        # self.netproject_t.add_module('project_t_layer_norm', nn.LayerNorm(config.embd_size_l))
        #
        # self.netproject_v = nn.Sequential()
        # self.netproject_v.add_module('project_v',
        #                              nn.Linear(in_features=hidden_sizes[1] * 4, out_features=config.embd_size_v))
        # self.netproject_v.add_module('project_v_activation', self.activation)
        # self.netproject_v.add_module('project_v_layer_norm', nn.LayerNorm(config.embd_size_v))
        #
        # self.netproject_a = nn.Sequential()
        # self.netproject_a.add_module('project_a',
        #                              nn.Linear(in_features=hidden_sizes[2] * 4, out_features=config.embd_size_a))
        # self.netproject_a.add_module('project_a_activation', self.activation)
        # self.netproject_a.add_module('project_a_layer_norm', nn.LayerNorm(config.embd_size_a))

        ##########################################
        # private encoders          私有编码器
        ##########################################
        self.netprivate_t = nn.Sequential()
        self.netprivate_t.add_module('private_t_1',
                                     nn.Linear(in_features=config.embd_size_l, out_features=config.embd_size_l))
        self.netprivate_t.add_module('private_t_activation_1', nn.Sigmoid())

        self.netprivate_v = nn.Sequential()
        self.netprivate_v.add_module('private_v_1',
                                     nn.Linear(in_features=config.embd_size_v, out_features=config.embd_size_v))
        self.netprivate_v.add_module('private_v_activation_1', nn.Sigmoid())

        self.netprivate_a = nn.Sequential()
        self.netprivate_a.add_module('private_a_3',
                                     nn.Linear(in_features=config.embd_size_a, out_features=config.embd_size_a))
        self.netprivate_a.add_module('private_a_activation_3', nn.Sigmoid())

        ##########################################
        # shared encoder            共享编码器
        ##########################################
        self.netshared = SharedEncoder(config)
        ##########################################
        # reconstruct               重建
        ##########################################
        self.netrecon_t = nn.Sequential()
        self.netrecon_t.add_module('recon_t_1',
                                   nn.Linear(in_features=config.embd_size_l, out_features=config.embd_size_l))
        self.netrecon_v = nn.Sequential()
        self.netrecon_v.add_module('recon_v_1',
                                   nn.Linear(in_features=config.embd_size_v, out_features=config.embd_size_v))
        self.netrecon_a = nn.Sequential()
        self.netrecon_a.add_module('recon_a_1',
                                   nn.Linear(in_features=config.embd_size_a, out_features=config.embd_size_a))

        ##########################################
        # shared space adversarial discriminator            共享空间对抗鉴别器
        ##########################################
        if not self.config.use_cmd_sim:
            self.netdiscriminator = nn.Sequential()
            self.netdiscriminator.add_module('discriminator_layer_1',
                                             nn.Linear(in_features=config.embd_size_a, out_features=config.embd_size_a))
            self.model_names.append('discriminator')

        ##########################################
        # shared-private collaborative discriminator        公私协作鉴别器
        ##########################################

        self.netsp_discriminator = nn.Sequential()  # 一个全连接层
        self.netsp_discriminator.add_module(
            'sp_discriminator_layer_1',
            nn.Linear(in_features=config.embd_size_a, out_features=256))
        self.netsp_discriminator.add_module(
            'sp_discriminator_layer_2',
            nn.Linear(in_features=256, out_features=4))
        self.netsp_discriminator.add_module(
            'sp_discriminator_layer_1_dropout',
            nn.Dropout(dropout_rate))
        self.netsp_discriminator.add_module('sp_discriminator_layer_1_activation', self.activation)


        self.netfusion = nn.Sequential()
        self.netfusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size * 6,
                                                              out_features=self.config.hidden_size * 3))
        self.netfusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.netfusion.add_module('fusion_layer_1_activation', self.activation)
        self.netfusion.add_module('fusion_layer_3',
                                  nn.Linear(in_features=self.config.hidden_size * 3, out_features=output_size))

        self.nettlayer_norm = nn.LayerNorm((hidden_sizes[0] * 2))
        self.netvlayer_norm = nn.LayerNorm((hidden_sizes[1] * 2))
        self.netalayer_norm = nn.LayerNorm((hidden_sizes[2] * 2))

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2, dropout=0.5)
        self.nettransformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=config.lr, betas=(config.beta1, 0.999),
                                              weight_decay=config.weight_decay)
            self.optimizers.append(self.optimizer)
            self.output_dim = config.output_dim


    # 提取特征
    def extract_features(self, sequence, type, rnn1, rnn2, layer_norm):
        sequence = sequence.permute(1, 0, 2)

        # lstm和GRU的输出不一样
        if type == 'textcnn':
            print('sequence.size is:', sequence.size())
            packed_h1 = final_h1 = rnn1(sequence)
            print('packed_h1.size is:', packed_h1.size())
            normed_h1 = layer_norm(packed_h1)
            print('normed_h1.size is:', normed_h1.size())
            _, final_h2 = rnn2(normed_h1)
            pass
        else:
            if self.config.rnncell == "lstm":
                packed_h1, (final_h1, _) = rnn1(sequence)
            else:
                packed_h1, final_h1 = rnn1(sequence)

            # print('packed_h1.size is:', packed_h1.size())
            normed_h1 = layer_norm(packed_h1)  # 层归一化
            # print('normed_h1.size is:', normed_h1.size())

            if self.config.rnncell == "lstm":
                _, (final_h2, _) = rnn2(normed_h1)
            else:
                _, final_h2 = rnn2(normed_h1)

        return final_h1, final_h2

    # 重建
    def reconstruct(self, ):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.netrecon_t(self.utt_t)
        self.utt_v_recon = self.netrecon_v(self.utt_v)
        self.utt_a_recon = self.netrecon_a(self.utt_a)

    # 共享_特有
    def shared_private(self, utterance_t, utterance_v, utterance_a):

        # # Projecting to same sized space
        # self.utt_t_orig = utterance_t = self.netproject_t(utterance_t)
        # self.utt_v_orig = utterance_v = self.netproject_v(utterance_v)
        # self.utt_a_orig = utterance_a = self.netproject_a(utterance_a)
        self.utt_t_orig = utterance_t
        self.utt_v_orig = utterance_v
        self.utt_a_orig = utterance_a

        # Private-shared components         每个话语向量um投射到两个不同的表示中：独特子空间和公共子空间
        self.utt_private_t = self.netprivate_t(utterance_t)
        self.utt_private_v = self.netprivate_v(utterance_v)
        self.utt_private_a = self.netprivate_a(utterance_a)

        # print('data.size is  ',utterance_t.size())
        self.utt_shared_t = self.netshared(utterance_t)
        self.utt_shared_v = self.netshared(utterance_v)
        self.utt_shared_a = self.netshared(utterance_a)


    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        # print('=======================================================')
        # print(self.config.dataset_mode)
        # print('*******************************************************')
        self.lengths = input['lengths'].to(self.device)
        if self.modality == 'AVL':
            if 'A' in self.modality:
                self.acoustic = input['A_feat'].float().to(self.device)
            if 'L' in self.modality:
                self.lexical = input['L_feat'].float().to(self.device)
            if 'V' in self.modality:
                self.visual = input['V_feat'].float().to(self.device)

            self.label = input['label'].to(self.device)
        else:
            acoustic = input['A_feat'].float().to(self.device)
            lexical = input['L_feat'].float().to(self.device)
            visual = input['V_feat'].float().to(self.device)
            self.missing_index = input['missing_index'].long().to(self.device)
            if self.isTrain:
                self.label = input['label'].to(self.device)
                # A modality
                self.A_miss_index = self.missing_index[:, 0].unsqueeze(1).unsqueeze(2)
                self.acoustic = acoustic * self.A_miss_index
                # L modality
                self.L_miss_index = self.missing_index[:, 2].unsqueeze(1).unsqueeze(2)
                self.lexical = lexical * self.L_miss_index
                # V modality
                self.V_miss_index = self.missing_index[:, 1].unsqueeze(1).unsqueeze(2)
                self.visual = visual * self.V_miss_index
            else:
                self.acoustic = acoustic
                self.visual = visual
                self.lexical = lexical

    def forward(self):
        """
        :param sentences:   文本序列
        :param visual:      视频序列
        :param acoustic:    音频序列
        :param lengths:     长度
        :param bert_sent:
        :param bert_sent_type:
        :param bert_sent_mask:
        :return:
        """
        batch_size = self.lengths.size(0)

        # final_h1t, final_h2t = self.extract_features(self.lexical, 'lstm', self.netL1, self.netL2,
        #                                              self.nettlayer_norm)
        # # # permute()对张量的维度进行换位；contiguous()开辟一块新内存，将tensor变成在内存中连续分布的状态
        # utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        utterance_text = self.netL(self.lexical)
        # extract features from visual modality
        # final_h1v, final_h2v = self.extract_features(self.visual, 'lstm', self.netV1, self.netV2,
        #                                              self.netvlayer_norm)
        # utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        utterance_video = self.netV(self.visual)

        # extract features from acoustic modality
        # final_h1a, final_h2a = self.extract_features(self.acoustic, 'lstm', self.netA1, self.netA2,
        #                                              self.netalayer_norm)
        # utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        utterance_audio = self.netA(self.acoustic)

        # Shared-private encoders       对特征进行分流
        self.shared_private(utterance_text, utterance_video, utterance_audio)

        if not self.config.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)

            self.domain_label_t = self.netdiscriminator(reversed_shared_code_t)
            self.domain_label_v = self.netdiscriminator(reversed_shared_code_v)
            self.domain_label_a = self.netdiscriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None

        # 特性特征
        self.shared_or_private_p_t = self.netsp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.netsp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.netsp_discriminator(self.utt_private_a)
        # 共性特征  就一个全连接层？
        self.shared_or_private_s = self.netsp_discriminator(
            (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a) / 3.0)

        # For reconstruction
        self.reconstruct()

        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t,
                         self.utt_shared_v, self.utt_shared_a), dim=0)

        # 进行自我关注
        h = self.nettransformer_encoder(h)
        # 进行简单拼接
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        # 融合
        self.logits = self.netfusion(h)
        self.pred = F.softmax(self.logits, dim=-1)


    def backward(self):
        criterion = nn.CrossEntropyLoss(reduction="mean")
        # criterion = nn.MSELoss(reduction="mean")
        self.loss_cls = criterion(self.logits, self.label)
        if 'cls' not in self.loss_names:
            self.loss_names.append('cls')

        self.loss_diff = self.get_diff_loss()
        if 'diff' not in self.loss_names:
            self.loss_names.append('diff')


        self.loss_domain = self.get_domain_loss() * self.config.sim_weight
        self.loss_cmd = self.get_cmd_loss()

        self.loss_recon = self.get_recon_loss()
        if 'recon' not in self.loss_names:
            self.loss_names.append('recon')

        if self.config.use_cmd_sim:
            similarity_loss = self.loss_cmd
            if 'cmd' not in self.loss_names:
                self.loss_names.append('cmd')
        else:
            similarity_loss = self.loss_domain
            if 'domain' not in self.loss_names:
                self.loss_names.append('domain')

        loss = self.config.cls_weight * self.loss_cls + \
               self.config.diff_weight * self.loss_diff + \
               self.config.sim_weight * similarity_loss + \
               self.config.recon_weight * self.loss_recon

        loss.backward()

        # 梯度裁剪,防止梯度爆炸
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)
        self.optimizer.step()

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        # backward
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_domain_loss(self, ):

        if self.config.use_cmd_sim:
            return 0.0

        # Predicted domain labels
        domain_pred_t = self.domain_label_t
        domain_pred_v = self.domain_label_v
        domain_pred_a = self.domain_label_a

        # True domain labels
        domain_true_t = to_gpu(torch.LongTensor([0] * domain_pred_t.size(0)), gpu_id)
        domain_true_v = to_gpu(torch.LongTensor([1] * domain_pred_v.size(0)), gpu_id)
        domain_true_a = to_gpu(torch.LongTensor([2] * domain_pred_a.size(0)), gpu_id)

        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_t, domain_pred_v, domain_pred_a), dim=0)
        domain_true = torch.cat((domain_true_t, domain_true_v, domain_true_a), dim=0)

        return self.domain_loss_criterion(domain_pred, domain_true)

    def get_cmd_loss(self, ):

        if not self.config.use_cmd_sim:
            return 0.0

        # losses between shared states
        loss = self.loss_cmd_func(self.utt_shared_t, self.utt_shared_v, 1)
        loss += self.loss_cmd_func(self.utt_shared_t, self.utt_shared_a, 1)
        loss += self.loss_cmd_func(self.utt_shared_a, self.utt_shared_v, 1)
        loss = loss / 3.0

        return loss

    def get_diff_loss(self):
        shared_t = self.utt_shared_t
        shared_v = self.utt_shared_v
        shared_a = self.utt_shared_a
        private_t = self.utt_private_t
        private_v = self.utt_private_v
        private_a = self.utt_private_a

        # between private and shared
        loss = self.loss_diff_func(private_a, shared_a)
        loss += self.loss_diff_func(private_v, shared_v)
        loss += self.loss_diff_func(private_t, shared_t)

        # across privates
        loss += self.loss_diff_func(private_t, private_v)
        loss += self.loss_diff_func(private_t, private_a)
        loss += self.loss_diff_func(private_v, private_a)

        return loss

    def get_recon_loss(self, ):

        loss = self.loss_recon_func(self.utt_t_recon, self.utt_t_orig)
        loss += self.loss_recon_func(self.utt_v_recon, self.utt_v_orig)
        loss += self.loss_recon_func(self.utt_a_recon, self.utt_a_orig)
        loss = loss / 3.0
        return loss
