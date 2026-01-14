import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer

'''
三个工具函数，用于快速构建 Transformer 组件、线性注意力组件和初始化好的 1D 卷积层，
是后续网络结构的 “积木”。
'''
#构建标准的 PyTorch Transformer 编码器（基于自注意力的序列建模组件）
def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )#创建单个 Transformer 编码器层
    '''
    d_model=channels：指定输入特征的维度，必须和输入张量的维度匹配；
    nhead=heads：指定多头注意力的头数；
    dim_feedforward=64：前馈网络（Feed Forward）的隐藏层维度，这里固定为 64；
    activation="gelu"：激活函数使用 GELU（高斯误差线性单元），比 ReLU 更适合 Transformer 架构。
    '''
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)
#构建线性注意力 Transformer(高效的线性注意力 Transformer 层)
def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):
    return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )
    '''
    dim=channels：模型维度，和输入特征维度匹配；
    depth=layers：Transformer 的层数，默认 1；
    heads=heads：注意力头数；
    max_seq_len=256：限制模型能处理的最大序列长度；
    n_local_attn_heads=0/local_attn_window_size=0：关闭局部注意力，仅使用全局线性注意力。
    '''
#创建并初始化好权重的 1D 卷积层
def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    #三个参数：输入通道数、输出通道数（等价于卷积核的数量）、卷积核大小(滑动窗口大小)；可以继续指定padding、stride
    # 输入张量的三维 (B, in_channels, seq_len),输出张量的三维 (B, out_channels, seq_len')
    nn.init.kaiming_normal_(layer.weight)#使用Kaiming 正态初始化, 针对 ReLU、GELU 等激活函数能缓解梯度消失问题，让模型更容易收敛
    return layer

'''
编码「扩散步骤 t」
（注意：main_model.py中的time_embedding函数：编码「序列的时间步 pos」）
扩散步骤的嵌入层（也叫时间步嵌入层），作用是将离散的扩散步骤（比如第 t 步扩散）转化为高维的连续特征嵌入，
让模型能够学习到不同扩散步骤的特征模式，从而适应扩散过程中不同阶段的分布变化。
'''
class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(#继承的工具方法,注册缓冲区张量，保证设备同步
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,#指定缓冲区不被保存到模型的state_dict中
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)#对投影后的特征再次映射，增强嵌入的表达能力

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]#对正弦余弦嵌入表索引
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    # 构建基础的正弦余弦嵌入表( 自定义的内部方法（下划线开头）)
    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        #对于全局特征来说，sin 和 cos 的排列顺序并不重要，只要包含这些频率特征，就能捕捉到索引的信息
        return table

#整个扩散模型的核心网络
class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):#inputdim：无条件为1（仅加噪数据），有条件为2（观测值+加噪目标）
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        #三个卷积层的核大小都是 1x1 ，在不改变序列长度的前提下，实现通道数的映射 / 变换（相当于对每个位置的特征做线性变换）
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)#将高维特征映射到 1 维，得到最终的预测结果
        nn.init.zeros_(self.output_projection2.weight)#避免初始值过大导致模型训练不稳定，零初始化能让模型更平稳地开始学习

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        #cond_info即side_info(B, *, K, L)
        B, inputdim, K, L = x.shape #(B, inputdim, K, L),inputdim取值1/2

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)#得到(B,self.channels,K*L)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)#恢复原始形状(B,self.channels,K,L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)#扩散步嵌入，维度(B, projection_dim)

        skip = []
        #关键细节：每次循环的x会被更新为当前 Block 的输出x，实现残差的逐层传递，保证梯度能回传到底层；
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)#x (B, self.channels, K, L)  
            skip.append(skip_connection)
        #x维度不变

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))#融合多层 Skip 特征
        #torch.stack会在新的维度（dim=0）上堆叠这些张量（config["layers"]个），形成一个包含 “层维度” 的新张量；
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        #side_dim是侧边信息维度，即side_info(B, *, K, L)的*
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)#扩散步骤投影层(不改变序列长度，仅调整通道数，比全连接层更适合序列数据)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)#侧边信息投影层
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)#中间特征投影层
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)#输出投影层
        self.is_linear = is_linear
        # 时间维度注意力层：根据is_linear选择注意力类型
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)#处理时间自注意力
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)#处理特征自注意力
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    #对每个变量的时间序列做自注意力建模，捕捉时序依赖
    def forward_time(self, y, base_shape):
        # base_shape：输入x的形状，即(B, channel, K, L)
        #y是forward方法中融合扩散步数嵌入的结果(B, channel, K*L)
        B, channel, K, L = base_shape
        if L == 1:# 若时间步L=1，无时序依赖，直接返回
            return y
        #每个变量的时间序列作为独立 batch，序列长度为L
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)#得到(B * K, channel, L)
        '''
        注意力层只能对 “单个序列” 计算自注意力：
        把2 个班级（B=2）、3 个学生（K=3）、4 次考试成绩（L=4），拆成6 个独立的 “学生样本”，
        每个样本只有自己的 4 次成绩，方便老师单独分析每个学生的成绩变化
        '''

        if self.is_linear:#线性注意力机制：输入(B', L, channel) → 输出(与输入维度完全一致)后还原维度
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:#标准自注意力机制:输入(L, B', channel) → 输出(与输入维度完全一致)后还原维度
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        #注意力处理后(B * K, channel, L)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)#还原为(B, channel, K * L)
        return y


    def forward_feature(self, y, base_shape):
        # y (B, channel, K * L)
        # base_shape：输入x的形状，即(B, channel, K, L)
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)#返回(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        ##cond_info即side_info(B, *, K, L)
        #x (B, self.channels, K, L) 
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        #融合扩散步数嵌入
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        #双维度注意力建模
        y = self.forward_time(y, base_shape) #返回(B, channel, K * L)
        y = self.forward_feature(y, base_shape)  # 返回(B,channel,K*L)

        #中间特征投影（扩展维度为2*channel，为门控做准备）
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        #融合侧边信息
        _, cond_dim, _, _ = cond_info.shape #(B, cond_dim, K, L)
        cond_info = cond_info.reshape(B, cond_dim, K * L)#返回(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info  # (B,2*channel,K*L)

        '''
        对特征做动态筛选（门控激活），保留有用信息，丢弃噪声；
        设计残差连接缓解深度网络的梯度消失；
        设计Skip 连接融合多层特征，提升模型表达能力；
        '''
        #门控激活（Gated Activation）—— 增强非线性表达(软阈值)
        gate, filter = torch.chunk(y, 2, dim=1) #拆分(B,2*channel,K*L)为两部分：门控信号和过滤信号
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y) # (B,2*channel,K*L)

        residual, skip = torch.chunk(y, 2, dim=1) #拆分(B,2*channel,K*L)为两部分：残差和跳跃连接
        x = x.reshape(base_shape) #恢复原始形状(B, channel, K, L)
        residual = residual.reshape(base_shape) #恢复原始形状(B, channel, K, L)
        skip = skip.reshape(base_shape) #恢复原始形状(B, channel, K, L)
        return (x + residual) / math.sqrt(2.0), skip
        #设计残差连接缓解深度网络的梯度消失；残差连接的对象是当前 Block 的输入`x`，即`x + residual`
        #设计Skip 连接融合多层特征，提升模型表达能力；Skip 连接的对象是当前 Block 的输出skip（共config["layers"]个Block）

