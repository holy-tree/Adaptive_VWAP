import torch
import torch.nn as nn
import math

class Generator_gru(nn.Module):
    def __init__(self, input_size, out_size, hidden_dim = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size, hidden_dim, batch_first=True)  # 仅保留一层GRU，隐藏单元数为256
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.linear_2 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.linear_3 = nn.Linear(hidden_dim//4, 4)
        self.dropout = nn.Dropout(0.2)

        # 添加分类头，输入维度为256，输出3类别
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim//2, 3)
        )

    def forward(self, x):
        device = x.device
        # 初始化GRU隐藏状态
        h0 = torch.zeros(1, x.size(0), self.hidden_dim, device=device)
        # 通过GRU层
        out, _ = self.gru(x, h0)
        # 取序列最后一个时间步的输出，并经过dropout处理
        last_feature = self.dropout(out[:, -1, :])

        # 原始输出（例如生成或回归任务）
        gen = self.linear_1(last_feature)
        gen = self.linear_2(gen)
        gen = self.linear_3(gen)

        # 分类输出
        cls = self.classifier(last_feature)

        return gen, cls


class Generator_lstm(nn.Module):
    def __init__(self, input_size, out_size, hidden_size=128, num_layers=1, dropout=0.1):
        """
        Args:
            input_size (int): 输入特征数
            out_size (int): 输出目标维度（例如用于生成回归结果）
            hidden_size (int): LSTM 的隐藏单元数
            num_layers (int): LSTM 层数，默认 1 层（减少计算量）
            dropout (float): LSTM 内部 dropout 系数，默认为 0.1
        """
        super().__init__()
        # 使用深度可分离卷积：先进行 depthwise, 再 pointwise 转换
        self.depth_conv = nn.Conv1d(in_channels=input_size, out_channels=input_size,
                                    kernel_size=3, padding=1, groups=input_size)
        self.point_conv = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=1)
        self.act = nn.ReLU()

        # LSTM 部分：输入通道数为 (input_size * 4)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        # 直接使用最后一个时间步的输出进行线性映射
        self.linear = nn.Linear(hidden_size, 4)
        # # 添加分类头，输入维度为256，输出3类别
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size//2),
        #     nn.ReLU(),
        #     nn.Dropout(0.05),
        #     nn.Linear(hidden_size//2, 3)
        # )

        self.classifier = nn.Linear(hidden_size, 3)


    def forward(self, x, hidden=None):
        """
        Args:
            x (torch.Tensor): 输入，形状 (batch_size, seq_len, input_size)
            hidden: 可选的 LSTM 初始状态
        Returns:
            torch.Tensor: 输出，形状 (batch_size, out_size)
        """
        # 调整维度：将输入从 (B, T, F) 转为 (B, F, T) 以适应Conv1d
        x = x.permute(0, 2, 1)  # (B, input_size, T)
        # 深度卷积
        x = self.depth_conv(x)
        # 点卷积
        x = self.point_conv(x)
        x = self.act(x)
        # 转回 (B, T, F')
        x = x.permute(0, 2, 1)
        # LSTM 前向传播：这里使用默认最后一时刻状态作为输出
        lstm_out, hidden = self.lstm(x, hidden)
        # 直接取最后一个时间步输出作为特征（避免额外池化操作）
        last_out = lstm_out[:, -1, :]
        out = self.linear(last_out)
        cls = self.classifier(last_out)

        return out, cls

# 位置编码器
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        """
        model_dim: 模型的特征向量维度
        max_len: 支持的最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, model_dim)

        # 位置索引
        positions = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]

        # 维度索引，使用指数函数缩放
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))

        # 偶数位置使用 sin，奇数位置使用 cos
        self.encoding[:, 0::2] = torch.sin(positions * div_term)
        self.encoding[:, 1::2] = torch.cos(positions * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # 增加 batch 维度：[1, max_len, model_dim]

    def forward(self, x):
        """
        x: 输入特征 [batch_size, seq_len, model_dim]
        """
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)  # 只取对应长度的位置信息


class Generator_transformer(nn.Module):
    def __init__(self, input_dim, feature_size=128, num_layers=2, num_heads=8, dropout=0.1, output_len=1):
        """
        input_dim: 数据特征维度
        feature_size: 模型特征维度
        num_layers: 编码器层数
        num_heads: 注意力头数目
        dropout: dropout概率
        output_len: 预测时间步长度（原始任务输出维度）
        """
        super().__init__()
        self.feature_size = feature_size
        self.output_len = output_len
        self.input_projection = nn.Linear(input_dim, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)
        # 添加 batch_first=True
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout,
                                                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 4)  # 原始任务输出
        # 添加分类头：输入feature_size，输出3类别
        # # 添加分类头，输入维度为256，输出3类别
        # self.classifier = nn.Sequential(
        #     nn.Linear(feature_size, feature_size//4),
        #     nn.ReLU(),
        #     nn.Dropout(0.05),
        #     nn.Linear(feature_size//4, 3)
        # )
        self.classifier = nn.Linear(feature_size, 3)

        self._init_weights()
        self.src_mask = None

    def _init_weights(self):
        init_range = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, src_mask=None):
        batch_size, seq_len, _ = src.size()
        src = self.input_projection(src)
        src = self.pos_encoder(src)

        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)

        output = self.transformer_encoder(src, src_mask)
        # 取最后一个时间步作为特征表示 [batch_size, feature_size]
        last_feature = output[:, -1, :]

        # 原始任务输出
        gen = self.decoder(last_feature)
        # 分类输出
        cls = self.classifier(last_feature)

        return gen, cls

    def _generate_square_subsequent_mask(self, seq_len):
        # 生成上三角掩码
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

# RNN生成器模型
class Generator_rnn(nn.Module):
    def __init__(self, input_size):
        super(Generator_rnn, self).__init__()
        self.rnn_1 = nn.RNN(input_size, 1024, batch_first=True)
        self.rnn_2 = nn.RNN(1024, 512, batch_first=True)
        self.rnn_3 = nn.RNN(512, 256, batch_first=True)
        self.linear_1 = nn.Linear(256, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        use_cuda = 1
        device = x.device
        h0_1 = torch.zeros(1, x.size(0), 1024).to(device)
        out_1, _ = self.rnn_1(x, h0_1)
        out_1 = self.dropout(out_1)
        h0_2 = torch.zeros(1, x.size(0), 512).to(device)
        out_2, _ = self.rnn_2(out_1, h0_2)
        out_2 = self.dropout(out_2)
        h0_3 = torch.zeros(1, x.size(0), 256).to(device)
        out_3, _ = self.rnn_3(out_2, h0_3)
        out_3 = self.dropout(out_3)
        out_4 = self.linear_1(out_3[:, -1, :])
        out_5 = self.linear_2(out_4)
        out = self.linear_3(out_5)
        return out

class Discriminator3(nn.Module):
    def __init__(self, input_dim, out_size, num_cls):
        """
        input_dim: 每个时间步的特征数，比如你是21
        out_size: 你想输出几个预测值，比如5
        """
        super().__init__()
        # 回归值处理分支
        self.label_embedding = nn.Embedding(num_cls, 32)
        self.conv_x = nn.Conv1d(4, 32, kernel_size=3, padding='same')
        # Label嵌入处理分支
        self.conv_label = nn.Conv1d(32, 32, kernel_size=3, padding='same')

        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same')

        self.linear1 = nn.Linear(128, 220)
        self.batch1 = nn.BatchNorm1d(220)
        self.linear2 = nn.Linear(220, 220)
        self.batch2 = nn.BatchNorm1d(220)
        self.linear3 = nn.Linear(220, out_size)

        self.leaky = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, label_indices):
        """
                x: [B, W, 1] 回归值
                labels: [B, W] hard label (整数类型)
                """
        # 处理回归值
        x = x.permute(0, 2, 1)  # [B, 1, W]
        x_feat = self.leaky(self.conv_x(x))  # [B, 32, W]

        # 处理label嵌入
        embedded = self.label_embedding(label_indices)  # [B, W, embedding_dim]
        embedded = embedded.squeeze().permute(0, 2, 1)  # [B, embedding_dim, W]
        label_feat = self.leaky(self.conv_label(embedded))  # [B, 32, W]

        # 合并特征
        combined = torch.cat([x_feat, label_feat], dim=1)  # [B, 64, W]
        conv2 = self.leaky(self.conv2(combined))  # [B, 64, W]
        conv3 = self.leaky(self.conv3(conv2))  # [B, 128, W]

        # 聚合时间信息，取平均
        pooled = torch.mean(conv3, dim=2)  # [B, 128]

        out = self.leaky(self.linear1(pooled))  # [B, 220]
        out = self.relu(self.linear2(out))     # [B, 220]
        out = self.sigmoid(self.linear3(out))  # [B, out_size]

        return out


import torch
import torch.nn as nn
from models.src.itransformer.layers.Transformer_EncDec import Encoder, EncoderLayer
from models.src.itransformer.layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.src.itransformer.layers.Embed import DataEmbedding_inverted


class Generator_itransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Generator_itransformer, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # 添加新的分类头
        self.classifier = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model // 2, configs.num_classes)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out,enc_out, attns

    # def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
    #     dec_out,enc_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
    #
    #     # 从 enc_out 中计算分类结果
    #     # 这里使用对所有 tokens 求均值，得到一个全局表示
    #     cls_input = enc_out.mean(1)
    #     cls_out = self.classifier(cls_input)
    #
    #     # 假设你的目标是预测 pred_len 个时间步，并且每个时间步有 N 个变量
    #     # dec_out 的形状是 [B, pred_len, N]
    #     # 但你的 val_y 形状是 [B, N]，这暗示你的目标只有 1 个时间步
    #     # 因此，我们只取 dec_out 的最后一个时间步作为预测结果
    #     final_predictions = dec_out[:, -1, :]  # 形状变为 [B, N]
    #
    #     if self.output_attention:
    #         return final_predictions, cls_out, attns
    #     else:
    #         return final_predictions, cls_out  # [B, N], [B, num_classes]
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # 打印输入 x_enc 的形状
        # print(f"输入 x_enc 形状: {x_enc.shape}")

        dec_out, enc_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # 打印 forecast 的输出 dec_out 的形状
        # print(f"forecast 输出 dec_out 形状: {dec_out.shape}")

        # 从 enc_out 中计算分类结果
        # 这里使用对所有 tokens 求均值，得到一个全局表示
        cls_input = enc_out.mean(1)
        cls_out = self.classifier(cls_input)

        # 打印分类头的输出 cls_out 的形状
        # print(f"分类头输出 cls_out 形状: {cls_out.shape}")

        # --- 这是关键的修改部分 ---

        # 1. 先从 dec_out (形状 [B, pred_len, 12]) 中选择最后一个时间步的预测
        final_predictions_all_features = dec_out[:, -1, :]  # 形状变为 [B, 12]

        # 2. 【新增】从这12个特征中，只选择前4个作为最终输出
        #    因为你的目标变量 (target_columns) 有4个
        num_target_features = 4  # 根据你的需求，这里是4
        final_predictions = final_predictions_all_features[:, :num_target_features]  # 形状变为 [B, 4]

        # --- 修改结束 ---

        # 打印最终预测的形状
        # print(f"最终预测 final_predictions 形状: {final_predictions.shape}")

        if self.output_attention:
            return final_predictions, cls_out, attns
        else:
            return final_predictions, cls_out  # [B, N], [B, num_classes]



class Generator_FITS(nn.Module):
    def __init__(self, configs):
        super(Generator_FITS, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in  # 原始代码中通常为1，现在应为4（OHLC）
        self.num_classes = configs.num_classes  # 新增：分类类别的数量

        self.dominance_freq = configs.cut_freq
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        # 频率上采样器，现在处理多个通道
        if self.individual:
            # 独立处理每个通道
            self.freq_upsampler_real = nn.ModuleList()
            self.freq_upsampler_imag = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler_real.append(
                    nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)))
                self.freq_upsampler_imag.append(
                    nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio)))
        else:
            # 为实部和虚部创建独立的线性层
            self.freq_upsampler_real = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio))
            self.freq_upsampler_imag = nn.Linear(self.dominance_freq, int(self.dominance_freq * self.length_ratio))

        self.output_dim = 4
        pred_len_upsampled = int(self.dominance_freq * self.length_ratio)

        # ✅ 线性头输入 = 未来片段长度 * 通道数（与实际喂入严格一致）
        in_feats = self.pred_len * configs.enc_in
        self.regression_head = nn.Linear(in_features=in_feats, out_features=self.output_dim)
        self.classification_head = nn.Linear(in_features=in_feats, out_features=configs.num_classes)

    def forward(self, x):
        # RIN（Relative Input Normalization）
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:, self.dominance_freq:] = 0
        low_specx = low_specx[:, 0:self.dominance_freq, :]

        # 分别提取实部和虚部
        low_specx_real = low_specx.real.permute(0, 2, 1)
        low_specx_imag = low_specx.imag.permute(0, 2, 1)

        # 频率上采样
        if self.individual:
            low_specxy_real = torch.zeros(
                [low_specx_real.size(0), int(self.dominance_freq * self.length_ratio), low_specx_real.size(2)],
                dtype=torch.float32).to(low_specx.device)
            low_specxy_imag = torch.zeros(
                [low_specx_imag.size(0), int(self.dominance_freq * self.length_ratio), low_specx_imag.size(2)],
                dtype=torch.float32).to(low_specx.device)

            for i in range(self.channels):
                low_specxy_real[:, :, i] = self.freq_upsampler_real[i](low_specx_real[:, :, i])
                low_specxy_imag[:, :, i] = self.freq_upsampler_imag[i](low_specx_imag[:, :, i])

            low_specxy_ = torch.complex(low_specxy_real, low_specxy_imag).permute(0, 2, 1)
        else:
            # 分别对实部和虚部进行上采样
            low_specxy_real = self.freq_upsampler_real(low_specx_real)
            low_specxy_imag = self.freq_upsampler_imag(low_specx_imag)
            # 重新组合成复数
            low_specxy_ = torch.complex(low_specxy_real, low_specxy_imag).permute(0, 2, 1)

        # 零填充并进行逆FFT
        low_specxy = torch.zeros(
            [low_specxy_.size(0), int((self.seq_len + self.pred_len) / 2 + 1), low_specxy_.size(2)],
            dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.length_ratio  # 能量补偿

        # 反向 RIN
        pred_ohlc_full = low_xy * torch.sqrt(x_var) + x_mean  # [B, seq_len+pred_len, C]

        # ✅ 只取未来片段，再展平
        pred_future = pred_ohlc_full[:, -self.pred_len:, :]  # [B, pred_len, C]
        flattened = pred_future.reshape(pred_future.size(0), -1)  # [B, pred_len*C]

        # 线性头
        new_pred_ohlc = self.regression_head(flattened)  # [B, 4]
        classification_output = self.classification_head(flattened)  # [B, num_classes]
        return new_pred_ohlc, classification_output



import torch
import torch.nn as nn
from models.src.PatchTST.layers.Transformer_EncDec import Encoder, EncoderLayer
from models.src.PatchTST.layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.src.PatchTST.layers.Embed import DataEmbedding_inverted,PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Generator_ptransformer(nn.Module):
    """
    修改后的PatchTST模型，显式参数传入，仅保留预测和分类任务
    参考论文: https://arxiv.org/pdf/2211.14730.pdf
    """
    def __init__(self,
                 input_dim,  # 输入特征维度（对应原enc_in）
                 seq_len,  # 输入序列长度
                 output_len=1,  # 预测序列长度（输出长度）
                 feature_size=512,  # 模型内部特征维度（对应原d_model）
                 num_layers=2,  # 编码器层数（对应原e_layers）
                 num_heads=8,  # 注意力头数（对应原n_heads）
                 d_ff=2048,  # 前馈网络维度（通常为feature_size的4倍）
                 dropout=0.1,  # dropout概率
                 activation='gelu',  # 激活函数
                 factor=5,  # 注意力因子（原FullAttention的factor）
                 patch_len=96,  # patch长度
                 stride=96,  # patch滑动步长
                 num_cls=3  # 分类任务类别数（固定为3类）
                 ):
        super().__init__()
        self.input_dim = input_dim  # 输入特征维度
        self.seq_len = seq_len  # 输入序列长度
        self.output_len = output_len  # 预测长度
        self.feature_size = feature_size  # 模型内部特征维度
        self.num_cls = num_cls  # 分类类别数

        # 1. Patch嵌入层（核心组件，用于序列分块）
        padding = stride  # 保持原逻辑的padding设置
        self.patch_embedding = PatchEmbedding(
            d_model=feature_size,
            patch_len=patch_len,
            stride=stride,
            padding=padding,
            dropout=dropout
        )

        # 2. 编码器（堆叠多层EncoderLayer）
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False,  # 不使用掩码
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=True
                        ),
                        d_model=feature_size,
                        n_heads=num_heads
                    ),
                    d_model=feature_size,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(num_layers)  # 堆叠num_layers层
            ],
            # 归一化层（保留原转置+BN逻辑）
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(feature_size), Transpose(1, 2))
        )

        # 3. 计算预测头的输入维度（head_nf）
        self.patch_num = int((self.seq_len - patch_len) / stride + 2)  # 分块数量
        self.head_nf = feature_size * self.patch_num  # 预测头的输入特征数

        # 4. 预测任务头（时间序列预测）
        self.pred_head = FlattenHead(
            n_vars=input_dim,
            nf=self.head_nf,
            target_window=output_len,
            head_dropout=dropout
        )

        # 5. 分类任务头（3类分类）
        self.cls_flatten = nn.Flatten(start_dim=-2)  # 展平特征
        self.cls_dropout = nn.Dropout(dropout)
        self.cls_projection = nn.Linear(
            self.head_nf * input_dim,  # 输入维度=head_nf * 特征数
            num_cls  # 输出3类
        )
        self.output_projection = nn.Linear(input_dim, 4)

    def _normalize(self, x):
        """序列归一化（保留原Non-stationary Transformer逻辑）"""
        means = x.mean(1, keepdim=True).detach()  # 按时间步求均值
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)  # 标准差
        x /= stdev
        return x, means, stdev

    def _denormalize(self, x, means, stdev):
        """反归一化"""
        stdev = stdev[:, 0, :].unsqueeze(1).repeat(1, self.output_len, 1)
        means = means[:, 0, :].unsqueeze(1).repeat(1, self.output_len, 1)
        return x * stdev + means

    def forecast(self, x_enc):
        """时间序列预测任务"""
        # 归一化
        x_enc, means, stdev = self._normalize(x_enc)  # x_enc: [B, seq_len, input_dim]

        # Patch嵌入：调整维度并分块
        x_enc = x_enc.permute(0, 2, 1)  # 转为 [B, input_dim, seq_len]（适配分块逻辑）
        enc_out, n_vars = self.patch_embedding(x_enc)  # enc_out: [B*input_dim, patch_num, feature_size]

        # 编码器特征提取
        enc_out, _ = self.encoder(enc_out)  # [B*input_dim, patch_num, feature_size]

        # 维度调整：恢复批次和特征维度
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))  # [B, input_dim, patch_num, feature_size]
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, input_dim, feature_size, patch_num]

        # 预测头输出
        gen = self.pred_head(enc_out)  # [B, input_dim, pred_len]
        gen = gen.permute(0, 2, 1)  # 转为 [B, pred_len, input_dim]

        # # 反归一化
        gen = self._denormalize(gen, means, stdev)
        gen = gen.squeeze(1)  # [B, 12]
        final_gen = self.output_projection(gen)  # [B, 4]

        return final_gen

    def classification(self, x_enc):
        """分类任务（预测3类）"""
        # # 归一化
        # x_enc, _, _ = self._normalize(x_enc)  # 仅归一化，不保留均值标准差
        # print(x_enc.shape)
        # Patch嵌入
        x_enc = x_enc.permute(0, 2, 1)  # [B, input_dim, seq_len]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [B*input_dim, patch_num, feature_size]

        # 编码器特征提取
        enc_out, _ = self.encoder(enc_out)  # [B*input_dim, patch_num, feature_size]

        # 维度调整
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))  # [B, input_dim, patch_num, feature_size]
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, input_dim, feature_size, patch_num]

        # 分类头输出
        cls_feat = self.cls_flatten(enc_out)  # 展平为 [B, input_dim, feature_size*patch_num]
        cls_feat = self.cls_dropout(cls_feat)
        cls_feat = cls_feat.reshape(cls_feat.shape[0], -1)  # [B, input_dim*feature_size*patch_num]
        cls = self.cls_projection(cls_feat)  # [B, num_cls]
        return cls

    def forward(self, x_enc):
        """前向传播：同时输出预测和分类结果"""
        # 预测结果（gen）
        gen = self.forecast(x_enc)  # [B, pred_len, input_dim]
        # 分类结果（cls）
        cls = self.classification(x_enc)  # [B, num_cls]
        return gen, cls