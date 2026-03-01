from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn

# FIXME
from transformers.models.mamba.modeling_mamba import logger, is_fast_path_available, mamba_inner_fn, causal_conv1d_fn, causal_conv1d_update, selective_state_update, selective_scan_fn
from transformers.models.mamba.modeling_mamba import MambaRMSNorm, MambaPreTrainedModel, MambaCache, MambaOutput, MambaMixer
from transformers import MambaConfig
from transformers.activations import ACT2FN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define the gating model

class MemoryTransformer(nn.Module):
    def __init__(self, dropout, d_model, n_head):
        super(MemoryTransformer, self).__init__()

        self.head = n_head
        self.d_model = d_model
        self.scale = torch.sqrt(torch.FloatTensor([d_model // n_head])).cuda()

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)


        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, input1, input2, input3, mask=None):
        query = self.w_q(input2)
        key = self.w_k(torch.cat((input1, input3), 1))
        value = self.w_v(torch.cat((input1, input3), 1))
        b = query.shape[0]

        query_view = query.view(b, -1 , self.head, self.d_model // self.head).permute(0, 2, 1, 3)
        key_view = key.view(b, -1 , self.head, self.d_model // self.head).permute(0, 2, 1, 3)
        value_view = value.view(b, -1 , self.head, self.d_model // self.head).permute(0, 2, 1, 3)

        attention = torch.matmul(query_view, key_view.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(attention, dim=-1)

        attn_output = torch.matmul(attention, value_view).permute(0, 2, 1, 3).contiguous()

        attn_output = attn_output.view(b, -1, self.head * (self.d_model // self.head))

        output = query + self.dropout1(attn_output)

        output = self.norm1(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)
        return output

class TextSemanticFusion(nn.Module):
    def __init__(self, in_dim=512, out_dim=768):

        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )
    def forward(self, image_features, prob_distribution, class_text_features):
        weighted_text = torch.matmul(prob_distribution, class_text_features)
        combined = torch.cat([image_features, weighted_text], dim=1)
        gate = self.gate(combined)
        fused_features = gate * image_features + (1 - gate) * weighted_text
        fused_features = self.transform(fused_features)

        return fused_features

class MambaMixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config, layer_idx, head_num=1):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = config.time_step_rank
        self.layer_idx = layer_idx
        self.head_num = head_num
        self.use_conv_bias = config.use_conv_bias
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )
        self.conv1d_back = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        self.in_proj_dif = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        self.x_proj_back = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        self.x_proj_dif = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        self.x_proj_dif_back = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)
        self.dt_proj_back = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)
        self.dt_proj_dif = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)
        self.dt_proj_dif_back = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        self.linear_hid2 = nn.Linear(self.intermediate_size,2*self.intermediate_size, bias=True)
        self.linear_hid2_back = nn.Linear(self.intermediate_size, 2*self.intermediate_size, bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.A_log_back = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.D_back = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(1*self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.out_LN = nn.LayerNorm(self.intermediate_size)
        self.use_bias = config.use_bias

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                " is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

    def cuda_kernels_forward(self, hidden_states: torch.Tensor, hidden_states_dif: torch.Tensor, cache_params: Optional[MambaCache] = None, cache_params_2: Optional[MambaCache] = None):
        # 1. Gated MLP's linear projection
        batch_size, seq_len, _ = hidden_states.shape
        flag_one = False
        if hidden_states_dif is None:
            flag_one = True
            hidden_states_dif = hidden_states
        hidden_states = torch.cat([hidden_states, hidden_states_dif], dim=0)
        # input_hidden_states_dif = hidden_states_dif
        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        projected_states_dif = self.in_proj_dif(hidden_states).transpose(1, 2)

        # process
        hidden_states, gate = projected_states.chunk(2, dim=1)
        hidden_states_dif, gate_dif = projected_states_dif.chunk(2, dim=1)
        # gate = gate[:batch_size]
        gate = gate_dif[batch_size:] if not flag_one else gate[:batch_size]

        gate_back = gate.flip(-1)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        # conv_weights_back = self.conv1d_back.weight.view(self.conv1d_back.weight.size(0), self.conv1d_back.weight.size(2))
        hidden_states_cat = causal_conv1d_fn(
            hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
        )
        # hidden_states_back = hidden_states.flip(-1)
        # hidden_states_back_cat = causal_conv1d_fn(
        #     hidden_states_back, conv_weights_back, self.conv1d_back.bias, activation=self.activation
        # )
        hidden_states_back_cat = hidden_states_cat.flip(-1)

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        hidden_states = hidden_states_cat[:batch_size]  # [batch, seq_len, intermediate_size]

        ## 反向排序：
        hidden_states_back = hidden_states_back_cat[:batch_size]  # [batch, seq_len, intermediate_size]

        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)
        # 反向：
        ssm_parameters_back = self.x_proj_back(hidden_states_back.transpose(1, 2))
        time_step_back, B_back, C_back = torch.split(
            ssm_parameters_back, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step_back = self.dt_proj_back.weight @ time_step_back.transpose(1, 2)


        A = -torch.exp(self.A_log.float())
        A_back = -torch.exp(self.A_log_back.float())
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
        time_proj_bias_back = self.dt_proj_back.bias.float() if hasattr(self.dt_proj_back, "bias") else None

        scan_outputs, ssm_state = selective_scan_fn(
            hidden_states,
            discrete_time_step,
            A,
            B.transpose(1, 2),
            C.transpose(1, 2),
            self.D.float(),
            gate,#None,
            time_proj_bias,
            delta_softplus=True,
            return_last_state=True,
        )
        scan_outputs_back, ssm_state_back = selective_scan_fn(
            hidden_states.flip(-1),
            discrete_time_step_back,
            A_back,
            B_back.transpose(1, 2),
            C_back.transpose(1, 2),
            self.D_back.float(),
            gate_back, #None, #
            time_proj_bias_back,
            delta_softplus=True,
            return_last_state=True,
        )

        # 4. Final linear projection
        contextualized_states = self.out_proj((scan_outputs+scan_outputs_back.flip(-1)).transpose(1, 2))
        return contextualized_states


    def forward(self, hidden_states, hidden_states_2, cache_params: Optional[MambaCache] = None, cache_params_2: Optional[MambaCache] = None):
        if is_fast_path_available and "cuda" in self.x_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, hidden_states_2, cache_params, cache_params_2)
        else:
            raise NotImplementedError("The fast path is not available")

class CaMambaBlock(nn.Module):
    def __init__(self, config, layer_idx, head_num=1, length=49, craft=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.head_num = head_num
        self.length = length
        self.craft = craft
        self.config.intermediate_size = config.intermediate_size
        self.mixer = MambaMixer(config, layer_idx=layer_idx)

    def forward(self, hidden_states, hidden_states_2, cache_params: Optional[MambaCache] = None, cache_params_2: Optional[MambaCache] = None,):
        residual = hidden_states
        # if hidden_states_2==None:
        #     residual = hidden_states[:,:,:768]
        #     hidden_states = self.linear(hidden_states)
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        # hidden_states_2 = self.norm(hidden_states_2.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, hidden_states_2, cache_params=cache_params, cache_params_2=cache_params_2)
        hidden_states = residual + hidden_states

        return hidden_states

class CaMambaModel(MambaPreTrainedModel):
    def __init__(self, config, head_num=1, length=49, craft=False):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([CaMambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds: Optional[torch.LongTensor] = None,
        inputs_embeds_2: Optional[torch.LongTensor] = None,
        cache_params: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # `attention_mask` is passed by the tokenizer and we don't want it
    ) -> Union[Tuple, MambaOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FIXME
        cache_params = None
        use_cache = False
        cache_params_2 = cache_params

        hidden_states = inputs_embeds
        hidden_states_2 = inputs_embeds_2
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            assert len(self.layers)==1
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(mixer_block.__call__, hidden_states, cache_params)
            else:
                hidden_states = mixer_block(hidden_states, hidden_states_2, cache_params=cache_params, cache_params_2=cache_params_2)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]
            cache_params_2.seqlen_offset += inputs_embeds_2.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )




class MambaSelf(nn.Module):
    def __init__(self, config, layer_idx, head_num=1):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = config.time_step_rank
        self.layer_idx = layer_idx
        self.head_num = head_num
        self.use_conv_bias = config.use_conv_bias
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )
        self.conv1d_back = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        self.in_proj_dif = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        self.x_proj_back = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        self.x_proj_dif = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        self.x_proj_dif_back = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)
        self.dt_proj_back = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)
        self.dt_proj_dif = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)
        self.dt_proj_dif_back = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        self.linear_hid2 = nn.Linear(self.intermediate_size,2*self.intermediate_size, bias=True)
        self.linear_hid2_back = nn.Linear(self.intermediate_size, 2*self.intermediate_size, bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.A_log_back = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.D_back = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(1*self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.out_LN = nn.LayerNorm(self.intermediate_size)
        self.use_bias = config.use_bias

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                " is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

    def cuda_kernels_forward(self, hidden_states: torch.Tensor, cache_params: Optional[MambaCache] = None, cache_params_2: Optional[MambaCache] = None):
        # 1. Gated MLP's linear projection
        batch_size, seq_len, _ = hidden_states.shape

        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        # process
        hidden_states, gate = projected_states.chunk(2, dim=1)

        gate_back = gate.flip(-1)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        hidden_states_cat = causal_conv1d_fn(
            hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
        )

        hidden_states_back_cat = hidden_states_cat.flip(-1)

        # 3. State Space Model sequence transformation
        # 3.a. input varying initialization of time_step, B and C
        hidden_states = hidden_states_cat[:batch_size]  # [batch, seq_len, intermediate_size]
        ## 反向排序：
        hidden_states_back = hidden_states_back_cat[:batch_size]  # [batch, seq_len, intermediate_size]

        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)
        # 反向：
        ssm_parameters_back = self.x_proj_back(hidden_states_back.transpose(1, 2))
        time_step_back, B_back, C_back = torch.split(
            ssm_parameters_back, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step_back = self.dt_proj_back.weight @ time_step_back.transpose(1, 2)


        A = -torch.exp(self.A_log.float())
        A_back = -torch.exp(self.A_log_back.float())
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
        time_proj_bias_back = self.dt_proj_back.bias.float() if hasattr(self.dt_proj_back, "bias") else None

        scan_outputs, ssm_state = selective_scan_fn(
            hidden_states,
            discrete_time_step,
            A,
            B.transpose(1, 2),
            C.transpose(1, 2),
            self.D.float(),
            gate,#None,
            time_proj_bias,
            delta_softplus=True,
            return_last_state=True,
        )
        scan_outputs_back, ssm_state_back = selective_scan_fn(
            hidden_states.flip(-1),
            discrete_time_step_back,
            A_back,
            B_back.transpose(1, 2),
            C_back.transpose(1, 2),
            self.D_back.float(),
            gate_back, #None, #
            time_proj_bias_back,
            delta_softplus=True,
            return_last_state=True,
        )

        # 4. Final linear projection
        contextualized_states = self.out_proj((scan_outputs+scan_outputs_back.flip(-1)).transpose(1, 2))
        return contextualized_states

    def forward(self, hidden_states, cache_params: Optional[MambaCache] = None, cache_params_2: Optional[MambaCache] = None):
        if is_fast_path_available and "cuda" in self.x_proj.weight.device.type:
            return self.cuda_kernels_forward(hidden_states, cache_params, cache_params_2)
        else:
            raise NotImplementedError("The fast path is not available")

class MambaBlock(nn.Module):
    def __init__(self, config, layer_idx, head_num=1, length=49, craft=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.head_num = head_num
        self.length = length
        self.craft = craft
        self.config.intermediate_size = config.intermediate_size
        self.mixer = MambaSelf(config, layer_idx=layer_idx)

    def forward(self, hidden_states, cache_params: Optional[MambaCache] = None, cache_params_2: Optional[MambaCache] = None,):
        residual = hidden_states

        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, cache_params=cache_params, cache_params_2=cache_params_2)
        hidden_states = residual + hidden_states

        return hidden_states

class Mamba(MambaPreTrainedModel):
    def __init__(self, config, head_num=1, length=49, craft=False):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # `attention_mask` is passed by the tokenizer and we don't want it
    ) -> Union[Tuple, MambaOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FIXME
        cache_params = None
        use_cache = False
        cache_params_2 = cache_params

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for mixer_block in self.layers:
            assert len(self.layers) == 1
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(mixer_block.__call__, hidden_states, cache_params)
            else:
                hidden_states = mixer_block(hidden_states, cache_params=cache_params, cache_params_2=cache_params_2)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


if __name__ == "__main__":
    config = MambaConfig(num_hidden_layers=1)
    model = CaMambaModel(config, head_num=1, length=49, craft=True)
    model = model.to(device)
    model.eval()
    input_embeds = torch.randn(4, 49, 768).to(device)
    input_embeds_2 = torch.randn(4, 49, 768).to(device)
    out1 = model(inputs_embeds=input_embeds, inputs_embeds_2=input_embeds_2).last_hidden_state
    print('last_hidden_state:')

