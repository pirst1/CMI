import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional
from torch.nn import functional as F
from torch.autograd import Variable
import math


class PatchEmbedding(nn.Module):
    """
    Args:
        d_model (int): The dimension of model
        patch_size (int): The size of small patch
        n_features (int): The number of features
    Inputs: inputs, mask
        - **inputs** (batch, time, n_features): Tensor containing input vector
    Returns:
        - **outputs** (batch, patched_time, dim): Tensor produces by linear projection.
    """
    def __init__(self, patch_size: int = 12, n_features: int = 7, d_model: int = 128):
        super().__init__()
        self.reduction = nn.Sequential(
            nn.Conv1d(n_features, d_model // 2, kernel_size=patch_size, stride=1, padding=patch_size//2),
            Swish(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm1d(d_model)
        )
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            Swish(),
            nn.Linear(d_model * 2, d_model)
        )
                
    def forward(self, x: Tensor) -> Tensor:
        x = self.reduction(x)
        x = x.transpose(1, 2)
        x = self.projection(x)
        
        return x


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class Swish(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()


class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module.
    Args:
        d_model: Embedding dimension.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : Input tensor B X T X C
        Returns:
            torch.Tensor: Encoded tensor B X T X C
        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
        dropout_p: float = 0.1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embedding: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 3, 1)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 3, 1)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key)
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding)
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score


class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size = inputs.size(0)
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)


class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    ref : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: False
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    """
    When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution.
    This operation often used to match dimensions.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ConvModule(nn.Module):
    """
    Convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout
    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by squeezeformer convolution module.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.pw_conv_1 = PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True)
        self.act1 = GLU(dim=1)
        self.dw_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.ln = nn.BatchNorm1d(in_channels)
        self.act2 = Swish()
        self.pw_conv_2 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)
        self.do = nn.Dropout(p=dropout_p)

    def forward(self, x):
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        Reference for masking : https://github.com/Ascend/ModelZoo-PyTorch/blob/master/PyTorch/built-in/audio/Wenet_Conformer_for_Pytorch/wenet/transformer/convolution.py#L26
        """
        x = x.transpose(1, 2)
        x = self.pw_conv_1(x)
        x = self.act1(x)
        x = self.dw_conv(x)
        x = self.ln(x)
        x = self.act2(x)
        x = self.pw_conv_2(x)
        x = self.do(x)
        x = x.transpose(1, 2)
        return x


class FeedForwardModule(nn.Module):
    """
    Feed Forward Module follow pre-norm residual units and apply layer normalization within the residual unit
    and on the input before the first linear layer. This module also apply Swish activation and dropout, which helps
    regularizing the network.

    Args:
        encoder_dim (int): Dimension of squeezeformer encoder
        expansion_factor (int): Expansion factor of feed forward module.
        dropout_p (float): Ratio of dropout
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        - **outputs** (batch, time, dim): Tensor produces by feed forward module.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        expansion_factor: int = 4,
        dropout_p: float = 0.1,
    ) -> None:
        super(FeedForwardModule, self).__init__()
        
        self.ffn1 = nn.Linear(encoder_dim, encoder_dim * expansion_factor, bias=True)
        self.act = Swish()
        self.do1 = nn.Dropout(p=dropout_p)
        self.ffn2 = nn.Linear(encoder_dim * expansion_factor, encoder_dim, bias=True)
        self.do2 = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.act(x)
        x = self.do1(x)
        x = self.ffn2(x)
        x = self.do2(x)
        
        return x


def make_scale(encoder_dim):
    scale = torch.nn.Parameter(torch.tensor([1.] * encoder_dim)[None,None,:])
    bias = torch.nn.Parameter(torch.tensor([0.] * encoder_dim)[None,None,:])
    return scale, bias


class SqueezeformerBlock(nn.Module):
    """
    SqueezeformerBlock is a simpler block structure similar to the standard Transformer block,
    where the MHA and convolution modules are each directly followed by a single feed forward module.

    Args:
        encoder_dim (int, optional): Dimension of squeezeformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of squeezeformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of squeezeformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by squeezeformer block.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 2,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
    ):
        super().__init__()
        
        self.scale_mhsa, self.bias_mhsa = make_scale(encoder_dim)
        self.scale_ff_mhsa, self.bias_ff_mhsa = make_scale(encoder_dim)
        self.scale_conv, self.bias_conv = make_scale(encoder_dim)
        self.scale_ff_conv, self.bias_ff_conv = make_scale(encoder_dim)
        
        self.mhsa = MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,)
        self.ln_mhsa = nn.LayerNorm(encoder_dim)
        self.ff_mhsa = FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                )
        self.ln_ff_mhsa = nn.LayerNorm(encoder_dim)
        self.conv = ConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                )
        self.ln_conv = nn.LayerNorm(encoder_dim)
        self.ff_conv = FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                )
        self.ln_ff_conv = nn.LayerNorm(encoder_dim)


    def forward(self, x):
        
        skip = x
        x = x * self.scale_mhsa + self.bias_mhsa
        x = skip + self.mhsa(x)
        x = self.ln_mhsa(x)

        skip = x
        x = x * self.scale_ff_mhsa + self.bias_ff_mhsa
        x = skip + self.ff_mhsa(x)
        x = self.ln_ff_mhsa(x)
        
        skip = x
        x = x * self.scale_conv + self.bias_conv
        x = skip + self.conv(x)
        x = self.ln_conv(x)
        
        skip = x
        x = x * self.scale_ff_conv + self.bias_ff_conv
        x = skip + self.ff_conv(x)
        x = self.ln_ff_conv(x)
        
        return x


class SqueezeformerEncoder(nn.Module):
    """
    Squeezeformer encoder first processes the input with a convolution subsampling layer and then
    with a number of squeezeformer blocks.

    Args:
        encoder_dim (int, optional): Dimension of squeezeformer encoder
        num_layers (int, optional): Number of squeezeformer blocks
        reduce_layer_index (int, optional): The layer index to reduce sequence length
        recover_layer_index (int, optional): The layer index to recover sequence length
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of squeezeformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of squeezeformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by squeezeformer encoder.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        num_layers: int = 16,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 2,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                SqueezeformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                )
            )

    def forward(self, x: Tensor):
        """
        Forward propagate a `inputs` for  encoder training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
        Returns:
            (Tensor, Tensor)
            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        """

        for block in self.blocks:
            x = block(x)

        return x

class LayerNormGRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, elementwise_affine=False):
        super().__init__()

        self.i_rz = torch.nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h_rz = torch.nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.ln_i_rz = torch.nn.LayerNorm(2 * hidden_size, elementwise_affine=elementwise_affine)
        self.ln_h_rz = torch.nn.LayerNorm(2 * hidden_size, elementwise_affine=elementwise_affine)
        self.sigmoid = torch.nn.Sigmoid()
        
        self.i_n = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.h_n = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.ln_i_n = torch.nn.LayerNorm(hidden_size, elementwise_affine=elementwise_affine)
        self.ln_h_n = torch.nn.LayerNorm(hidden_size, elementwise_affine=elementwise_affine)
        self.tanh = torch.nn.Tanh()
        
        self.reset_parameters(hidden_size)

    def reset_parameters(self, hidden_size):
        std = 1.0 / math.sqrt(hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h):

        ### Gate ###
        i_rz = self.i_rz(x)
        h_rz = self.h_rz(h)
        i_rz = self.ln_i_rz(i_rz)
        h_rz = self.ln_h_rz(h_rz)
        rz = self.sigmoid(i_rz + h_rz)
        r, z = rz.chunk(2, dim=1)

        ### candidate ###
        i_n = self.i_n(x)
        h_n = self.h_n(h)
        i_n = self.ln_i_n(i_n)
        h_n = self.ln_h_n(h_n)
        n = self.tanh(i_n + r * h_n)

        ### hidden state ###
        h = (1 - z) * n + z * h

        return h

class LayerNormGRUModel(nn.Module) :
    def __init__(self, input_dim, hidden_dim, bias=True, elementwise_affine=False, bidirectional=False, cuda=True) :
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.cuda = cuda
        self.gru_cell = LayerNormGRUCell(input_dim, hidden_dim, bias, elementwise_affine)
        if bidirectional:
            self.rgru_cell = LayerNormGRUCell(input_dim, hidden_dim, bias, elementwise_affine)
        
        
    def forward(self, x) :
        batch_size, sequence_len, _ = x.size()
        if self.bidirectional:
            if self.cuda:
                h = Variable(torch.zeros((batch_size, self.hidden_dim), device='cuda'))
                rh = Variable(torch.zeros((batch_size, self.hidden_dim), device='cuda'))
                hs = torch.zeros((batch_size, sequence_len, self.hidden_dim * 2), device='cuda')
            else:
                h = Variable(torch.zeros((batch_size, self.hidden_dim)))
                rh = Variable(torch.zeros((batch_size, self.hidden_dim)))
                hs = torch.zeros((batch_size, sequence_len, self.hidden_dim * 2))
            for seq in range(sequence_len): 
                h = self.gru_cell(x[:, seq, :], h)
                hs[:, seq, :self.hidden_dim] = h
                rh = self.rgru_cell(x[:, sequence_len - 1 - seq, :], rh)
                hs[:, sequence_len - 1 - seq, self.hidden_dim:] = rh
        else:
            if self.cuda:
                h = Variable(torch.zeros((batch_size, self.hidden_dim), device='cuda'))
                hs = torch.zeros((batch_size, sequence_len, self.hidden_dim), device='cuda')
            else:
                h = Variable(torch.zeros((batch_size, self.hidden_dim)))
                hs = torch.zeros((batch_size, sequence_len, self.hidden_dim))
            for seq in range(sequence_len): 
                h = self.gru_cell(x[:, seq, :], h)
                hs[:, seq, :self.hidden_dim] = h
        
        return hs

class BiGRUBlock(nn.Module):
    def __init__(
        self, 
        input_d_model: int = 128, 
        output_d_model: int = 128,
    ):
        super().__init__()
        self.bigru = nn.GRU(input_d_model, input_d_model, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(input_d_model * 2, output_d_model)

    def forward(self, x):
        x = self.bigru(x)[0]
        x = self.linear(x)

        return x

class Net(nn.Module):
    def __init__(
        self, 
        patch_size: int = 12,
        n_features: int = 7,
        d_model: int = 128,
        n_outputs: int = 4, 
    ):
        super().__init__()

        ### init ###
        self.initial = PatchEmbedding(
            patch_size = patch_size,
            n_features = n_features, 
            d_model = d_model
        )
        self.ln_init = nn.LayerNorm(d_model)

        ### down1 ###
        self.down1 = nn.Sequential(
            BiGRUBlock(input_d_model = d_model, output_d_model = d_model),
            BiGRUBlock(input_d_model = d_model, output_d_model = 2 * d_model),
        )
        self.pool_down1 = nn.MaxPool1d(15)
        self.ln_down1 = nn.LayerNorm(104)

        ### down2 ###
        self.down2 = nn.Sequential(
            BiGRUBlock(input_d_model = 2 * d_model, output_d_model = 2 * d_model),
            BiGRUBlock(input_d_model = 2 * d_model, output_d_model = 4 * d_model),
        )
        self.pool_down2 = nn.MaxPool1d(2)
        self.ln_down2 = nn.LayerNorm(52)

        ### bottleneck ###
        self.bottleneck = nn.Sequential(
            BiGRUBlock(input_d_model = 4 * d_model, output_d_model = 4 * d_model),
            BiGRUBlock(input_d_model = 4 * d_model, output_d_model = 4 * d_model),
        )

        ### up2 ###
        self.up2 = nn.Sequential(
            BiGRUBlock(input_d_model = 4 * d_model, output_d_model = 2 * d_model),
            BiGRUBlock(input_d_model = 2 * d_model, output_d_model = 2 * d_model),
        )

        ### up1 ###
        self.up1 = nn.Sequential(
            BiGRUBlock(input_d_model = 2 * d_model, output_d_model = d_model),
            BiGRUBlock(input_d_model = d_model, output_d_model = d_model),
        )

        ### final ###
        self.ln_finl = nn.LayerNorm(d_model)
        self.final = nn.Sequential(
            nn.Linear(d_model, d_model//4),
            nn.ReLU(),
            nn.Linear(d_model//4, n_outputs),
        )

    def forward(self, x):
        
        ### init ###
        # x: (batch_size, n_features, n_timestamp)
        x = self.initial(x)
        x = self.ln_init(x)
        # x: (batch_size, n_timestamp, d_model)

        ### down1 ###
        x = self.down1(x)
        skip1 = x
        
        x = x.transpose(1, 2)
        x = self.pool_down1(x)
        x = self.ln_down1(x)
        x = x.transpose(1, 2)

        ### down2 ###
        x = self.down2(x)
        skip2 = x
        
        x = x.transpose(1, 2)
        x = self.pool_down2(x)
        x = self.ln_down2(x)
        x = x.transpose(1, 2)

        ### bottleneck ###
        x = self.bottleneck(x)
        

        ### up2 ###
        x = x.transpose(1, 2)
        x = F.interpolate(x, scale_factor=2, mode="linear")
        x = x.transpose(1, 2)
        
        x = x + skip2
        x = self.up2(x)

        ### up1 ###
        x = x.transpose(1, 2)
        x = F.interpolate(x, scale_factor=15, mode='linear')
        x = x.transpose(1, 2)
        
        x = x + skip1
        x = self.up1(x)

        ### final ###
        x = self.ln_finl(x)
        x = self.final(x)
        # x: (batch_size, n_timestamp, 1)
        x = x.squeeze(2)
        # x: (batch_size, n_timestamp)
        
        return x[:, 60:-60]