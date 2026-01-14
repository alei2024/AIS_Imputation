import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI#扩散模型的核心网络,负责在扩散过程中预测噪声


class CSDI_base(nn.Module):# CSDI 模型的基类，实现了通用的扩散流程和插补逻辑
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim # 时间序列的特征数（K）

        # 嵌入维度配置
        self.emb_time_dim = config["model"]["timeemb"]  # 时间嵌入维度
        self.emb_feature_dim = config["model"]["featureemb"]  # 指定 “嵌入后每个离散标识的维度”
        self.is_unconditional = config["model"]["is_unconditional"]  # 是否无条件（无观测值约束）
        self.target_strategy = config["model"]["target_strategy"]  # 掩码生成策略（random/hist/mix）

        #总嵌入维度：时间嵌入 + 特征嵌入（有条件时加1维掩码）
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        # 实例化特征嵌入层：为每个特征维度生成嵌入向量
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        # 扩散模型配置：设置侧边信息维度
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        # 扩散模型输入维度：无条件为1（仅加噪数据），有条件为2（观测值+加噪目标）
        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)  # 初始化扩散模型(diff_CSDI需要接收侧边信息side info作为输入)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":#二次调度（quad）
            #让\(\beta\)的增长速度先慢后快（因为平方后，前期的小数值增长更慢，后期的大数值增长更快），这样前向加噪的过程更平滑，模型更容易学习
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        # self.alpha_torch的最终形状是(num_steps, 1, 1)，后续计算中输入数据形状是(B, K, L)（B：批次大小，K：特征数，L：时间步），
        #采样批次的时间步t（形状(B,)），取出alpha_troch（形状(B, 1, 1)）,
        #保证了形状满足广播规则，能和observed_data直接相乘

    def time_embedding(self, pos, d_model=128):#输出维度(B, L, d_model)
        #输入pos：形状为(B, L)的张量（B：批次大小，L：时间步长）
        #d_model：对应self.emb_time_dim
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)#在第二维添加新维度，形状从(B, L)扩展为(B, L, 1)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe #(B, L, d_model)

    #三种掩码生成策略：随机掩码、历史掩码、混合掩码
    def get_randmask(self, observed_mask):
        #observed_mask(B, K, L),原始的观测掩码，1 表示该位置有数据，0 表示该位置原本就缺失；训练数据常全1
        #cond_mask 条件掩码
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1) #展平张量为二维(B, K*L)
        for i in range(len(observed_mask)): #遍历批量内每个样本
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()#样本i观测位置的总数，item讲张量转换为Python原生数值
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1#indices为索引
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()#还原(B, K, L)
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":#支持混合策略（mix）
            rand_mask = self.get_randmask(observed_mask)
        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] #借鉴前一个样本的掩码模式
        return cond_mask

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask

    def get_side_info(self, observed_tp, cond_mask):#结合时间、特征、条件掩码作为侧边信息
        #observed_tp(B, L)
        #cond_mask(B, K, L)
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)#-1表示保持原维度大小不变，仅将第 2 维从 1 扩展为 K
        #time_embed(B, L, K, emb_time_dim)
        #调用实例（魔术函数）
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        #feature_embed(B, L, K, emb_feature_dim)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)#维度重排（Permute）
        #将张量的维度顺序从(B, L, K, *)改为(B, *, K, L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)#结合时间、特征、条件掩码作为侧边信息

        return side_info #(B, *, K, L)/(B, *+1, K, L)

    '''
    计算训练集 loss的整个流程（含反向传播、参数更新）是完整的训练过程；
    计算验证集 loss只是复用了训练过程的 loss 计算逻辑，属于评估环节，而非训练过程。
    '''
    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
    #observed_data	(B, K, L)
    #cond_mask(B, K, L)
    #side_info(B, *, K, L)
    #is_train 整数（1/0:训练/推理）
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()#断开损失的计算图（验证阶段不需要计算梯度，节省内存）
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation固定t评估的全面性和准确性（calc_loss_valid外层循环遍历了t）
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:#随机采样 t 能高效训练模型，覆盖不同噪声水平
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise #xt

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)#构建扩散模型的总输入

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L) 模型预测噪声predicted

        target_mask = observed_mask - cond_mask#只让模型学习预测 “被掩盖的真实数据区域”(cond_mask=1是observed_mask=1的子集)
        residual = (noise - predicted) * target_mask #需要计算损失的有效位置
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)#MSE
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        #noisy_data(B, K, L)
        #observed_data(B, K, L)
        #cond_mask(B, K, L)
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1) # (B,1,K,L)条件观测值
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1) # (B,1,K,L)噪声观测值（模型需要预测的所有位置）
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    '''
    推理过程：
    基于训练好的扩散模型，从纯高斯噪声开始，通过反向去噪过程，
    生成多个补全的样本（n_samples），实现对缺失数据的插补
    '''
    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        #要生成的插补样本数量n_samples
        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data#从原始观测数据开始，逐步添加噪声，模拟正向扩散过程
                noisy_cond_history = []#保存每个 t 步的 “含噪声观测数据 × 条件掩码”，即只保留模型可利用的位置的含噪声数据；

                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise#模拟扩散过程，逐步增加噪声
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data) #xt_hat
            #反向去噪过程，从最后一步开始，逐步恢复原始数据
            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                                #观测位置:使用预先计算的带噪声的观测数据
                                #缺失位置:使用当前生成的样本
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)#有条件：观测数据使用原始无噪声
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)#反向过程均值μ

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise#采样得到x_{t-1}的样本

            imputed_samples[:, i] = current_sample.detach() #imputed samples[:,i]选择了所有批次(B)的第i个样本位置,形状(B，K，L)

        return imputed_samples

    def forward(self, batch, is_train=1):
        #解析出模型所需的核心数据
        (
            observed_data,
            observed_mask,
            observed_tp,#用于生成时间嵌入,(B, L)
            gt_mask,#真实掩码,标记需要预测的区域,(B, K, L)
            for_pattern_mask,#模式掩码
            _,
        ) = self.process_data(batch)
        #生成条件掩码cond_mask（核心设计，分三个分支）
        if is_train == 0:#验证
            cond_mask = gt_mask#验证阶段用固定的gt_mask，保证评估的一致性（模拟真实场景中 “用历史数据预测未来缺失数据”）
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,#标记每个样本需要裁剪的起始区域(B,),(i,)表示第 i 个样本需要裁剪的时间步数量
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask#验证阶段用固定的gt_mask
            target_mask = observed_mask - cond_mask#只让模型学习预测 “被掩盖的真实数据区域”

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
                #i表示K维 ，...维度通配符代表K维，0 : cut_length[i].item()表示L维时间上从0到cut_length[i]的索引
        return samples, observed_data, target_mask, observed_mask, observed_tp
        #(B, n_samples, K, L)生成的多个插补样本


class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_AIS(CSDI_base):
    """
    AIS trajectory interpolation (Latitude, Longitude, SOG, COG).
    Dataset returns observed_data in shape (L, K); base model expects (B, K, L) after permute.
    """

    def __init__(self, config, device, target_dim=4):
        super(CSDI_AIS, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )



class CSDI_Forecasting(CSDI_base):
    def __init__(self, config, device, target_dim):
        super(CSDI_Forecasting, self).__init__(target_dim, config, device)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        feature_id=torch.arange(self.target_dim_base).unsqueeze(0).expand(observed_data.shape[0],-1).to(self.device)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            feature_id, 
        )        

    def sample_features(self,observed_data, observed_mask,feature_id,gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []
        
        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k,ind[:size]])
            extracted_mask.append(observed_mask[k,ind[:size]])
            extracted_feature_id.append(feature_id[k,ind[:size]])
            extracted_gt_mask.append(gt_mask[k,ind[:size]])
        extracted_data = torch.stack(extracted_data,0)
        extracted_mask = torch.stack(extracted_mask,0)
        extracted_feature_id = torch.stack(extracted_feature_id,0)
        extracted_gt_mask = torch.stack(extracted_gt_mask,0)
        return extracted_data, extracted_mask,extracted_feature_id, extracted_gt_mask


    def get_side_info(self, observed_tp, cond_mask,feature_id=None):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            )  # (K,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = self.embed_layer(feature_id).unsqueeze(1).expand(-1,L,-1,-1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask,feature_id,gt_mask = \
                    self.sample_features(observed_data, observed_mask,feature_id,gt_mask)
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if is_train == 0:
            cond_mask = gt_mask
        else: #test pattern
            cond_mask = self.get_test_pattern_mask(
                observed_mask, gt_mask
            )

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)



    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id, 
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1-gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp
