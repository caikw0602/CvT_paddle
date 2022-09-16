import  paddle
import paddle.nn as nn
from paddle.nn.initializer import Constant, TruncatedNormal, XavierNormal
import paddle.nn.functional as F

from numpy import repeat
import os
from droppath import DropPath

_trunc_norm = TruncatedNormal(std=.02)
_xavier_init = XavierNormal()
zero = Constant(0.)
one = Constant(1.)

class QuickGELU(nn.Layer):
    '''
    Rewrite GELU function to increase processing speed
    '''

    def forward(self, x: paddle.Tensor):
        return x * F.sigmoid(1.702 * x)

class ConvEmbed(nn.Layer):
    """ Image to Conv Embedding
    using nn.Conv2D and norm_layer to embedd the input.
    Ops: conv -> norm.
    Attributes:
        conv: nn.Conv2D
        norm: nn.LayerNorm
    nn.LayerNorm handle thr input with one dim, so we should
    stretch 2D input into 1D

    ConvEmbed: Before Stages, to replace pos_embed of ViT
    """
    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super(ConvEmbed, self).__init__()
        patch_size = tuple(repeat(patch_size, 2))
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.stride = stride
        self.padding = padding
        self.norm_layer = norm_layer

        self.conv = nn.Conv2D(in_channels=self.in_chans,
                              out_channels=self.embed_dim,
                              kernel_size=self.patch_size,
                              stride=self.stride,
                              padding=self.padding)

        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        # 饼 -> 条
        x = paddle.reshape(x, [B, C, -1])
        x = paddle.transpose(x, [0, 2, 1])
        if self.norm:
            x = self.norm(x)
        # 条 -> 饼
        x = paddle.reshape(x, [B, H, W, C])
        x = paddle.transpose(x, [0, 3, 1, 2])
        return x

class Mlp(nn.Layer):
    """ MLP module
     Impl using nn.Linear and activation is GELU, dropout is applied.
     Ops: fc -> act -> dropout -> fc -> dropout
     Attributes:
         fc1: nn.Linear
         fc2: nn.Linear
         act: GELU
         dropout1: dropout after fc1
         dropout2: dropout after fc2
     """
    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 act_layer=nn.GELU,
                 drop_out=0.):
        super(Mlp, self).__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.hidden_dim = hidden_dim
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(in_features=embed_dim,
                             out_features=self.hidden_dim,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)
        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(in_features=self.hidden_dim,
                             out_features=embed_dim,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)

        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_out)
        self.drop2 = nn.Dropout(drop_out)

    def _init_weights(self):
        w_attr = paddle.ParamAttr(initializer=_trunc_norm)
        b_attr = paddle.ParamAttr(initializer=zero)
        return w_attr, b_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x    # x: 条



class Attention(nn.Layer):
    """ Attention module
    Attention module for CvT.
    using conv to calculate q,k,v
    Attributes:
        num_heads: number of heads
        qkv: a nn.Linear for q, k, v mapping
            dw_bn: nn.Conv2D -> nn.BatchNorm
            avg: nn.AvgPool2D
            linear: None
        scales: 1 / sqrt(single_head_feature_dim)
        attn_drop: dropout for attention
        proj_drop: final dropout before output
        out: projection of multi-head attention
    """

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 kernel_size=3,
                 stride_kv=2,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 **kwargs
                 ):
        super(Attention, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        # self.attn_drop = attn_drop
        # self.proj_drop = proj_drop
        self.kernel_size = kernel_size
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.padding_kv = padding_kv
        self.padding_q = padding_q
        self.with_cls_token = with_cls_token
        self.scale = dim_out ** (-0.5)

        # calculate conv proj of qkv using depth conv (without wise conv)
        self.conv_proj_q = self._build_proj_depth_conv_bn(self.dim_in, self.dim_out, self.kernel_size,
                                                          self.stride_q, self.padding_q)
        self.conv_proj_k = self._build_proj_depth_conv_bn(self.dim_in, self.dim_out, self.kernel_size,
                                                          self.stride_kv, self.padding_kv)
        self.conv_proj_v = self._build_proj_depth_conv_bn(self.dim_in, self.dim_out, self.kernel_size,
                                                          self.stride_kv, self.padding_kv)

        # calculate conv proj of qkv by using wise conv
        w_attr_q, b_attr_q = self._init_weights()
        w_attr_k, b_attr_k = self._init_weights()
        w_attr_v, b_attr_v = self._init_weights()
        # nn.Linear is equal to 1x1 conv
        self.proj_q = nn.Linear(dim_in, dim_out, weight_attr=w_attr_q, bias_attr=b_attr_q if qkv_bias else None)
        self.proj_k = nn.Linear(dim_in, dim_out, weight_attr=w_attr_k, bias_attr=b_attr_k if qkv_bias else None)
        self.proj_v = nn.Linear(dim_in, dim_out, weight_attr=w_attr_v, bias_attr=b_attr_v if qkv_bias else None)

        # init other parameters
        self.attn_drop = nn.Dropout(attn_drop)
        w_attr, b_attr = self._init_weights()
        self.proj = nn.Linear(dim_out, dim_out, weight_attr=w_attr, bias_attr=b_attr)
        self.proj_dropout = nn.Dropout(proj_drop)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=_trunc_norm)
        bias_attr = paddle.ParamAttr(initializer=zero)
        return weight_attr, bias_attr

    def _build_proj_depth_conv_bn(self,
                               dim_in,
                               dim_out,
                               kernel_size,
                               stride,
                               padding):
        proj = nn.Sequential(
            nn.Conv2D(in_channels=dim_in,
                      out_channels=dim_in,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias_attr=False,
                      groups=dim_in),
            nn.BatchNorm2D(dim_in)
        )

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = paddle.split(x, [1, h*w], 1)
        # 条 -> 饼
        B, L, C = x.shape  # L is length of tensor
        x = paddle.reshape(x, [B, h, w, C])
        x = paddle.transpose(x, [0, 3, 1, 2])

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
            # 饼 -> 条
            B, C, H, W = q.shape
            q = paddle.reshape(q, [B, C, H*W])
            q = paddle.transpose(q, [0, 2, 1])
        else:
            # 饼 -> 条
            B, C, H, W = x.shape
            q = paddle.reshape(x, [B, C, H * W])
            q = paddle.transpose(q, [0, 2, 1])

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
            # 饼 -> 条
            B, C, H, W = k.shape
            k = paddle.reshape(k, [B, C, H*W])
            k = paddle.transpose(k, [0, 2, 1])
        else:
            # 饼 -> 条
            B, C, H, W = x.shape
            k = paddle.reshape(x, [B, C, H * W])
            k = paddle.transpose(k, [0, 2, 1])

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
            # 饼 -> 条
            B, C, H, W = v.shape
            v = paddle.reshape(v, [B, C, H*W])
            v = paddle.transpose(v, [0, 2, 1])
        else:
            # 饼 -> 条
            B, C, H, W = x.shape
            v = paddle.reshape(x, [B, C, H * W])
            v = paddle.transpose(v, [0, 2, 1])

        if self.with_cls_token:
            q = paddle.concat([cls_token, q], axis=1)
            k = paddle.concat([cls_token, k], axis=1)
            v = paddle.concat([cls_token, v], axis=1)

        return q, k, v

    def forward(self, x, h, w):
        if (self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None):
            q, k, v = self.forward_conv(x, h, w)

        # 条: now q, k, v is b (h w) c
        heads = self.num_heads
        q = self.proj_q(q)
        B, T, L = q.shape      # B T L -> b (h w) c
        q = paddle.reshape(q, [B, T, heads, -1])
        q = paddle.transpose(q, [0, 2, 1, 3])

        k = self.proj_k(k)
        B, T, L = k.shape
        k = paddle.reshape(k, [B, T, heads, -1])
        k = paddle.transpose(k, [0, 2, 1, 3])

        v = self.proj_k(v)
        B, T, L = v.shape
        v = paddle.reshape(v, [B, T, heads, -1])
        v = paddle.transpose(v, [0, 2, 1, 3])

        # calculate attn_score
        attn = paddle.matmul(q, k, transpose_y=True) * self.scale     # attn: [B heads T T], T: h * w
        attn_score = F.softmax(attn, axis=-1)
        attn_score = self.attn_drop(attn_score)

        x = paddle.matmul(attn_score, v)
        x = paddle.transpose(x, [0, 2, 1, 3])
        x = paddle.reshape(x, [0, 0, -1])

        x = self.proj(x)
        x = self.proj_dropout(x)

        return x

class Block(nn.Layer):
    ''' Block moudule
    Ops: token -> multihead attention (reshape token to a grap) ->Mlp->token
    '''
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs
                 ):
        super(Block, self).__init__()

        self.with_cls_token = kwargs['with_cls_token']

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop, **kwargs)
        if drop_path > 0.:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(dim_out,
                       mlp_ratio,
                       act_layer=act_layer,
                       drop_out=drop)

    def forward(self, x, h, w):
        res = x
        x = self.norm1(x)
        attn = self.attn(x, h, w)
        x = res + self.drop_path(attn)
        x = self.norm2(x)
        x = x + self.drop_path(self.mlp(x))

        return x

class VisionTransformer(nn.Layer):
    """ VisionTransformer moudule
    Vision Transformer with support for patch or hybrid CNN input stage
    Ops: intput -> conv_embed -> depth*block -> out
    Attribute:
        input: raw picture
        out: features, cls_token

    """
    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=QuickGELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs
                 ):
        super(VisionTransformer, self).__init__()
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim

        self.conv_embed = ConvEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            stride=patch_stride,
            padding=patch_padding,
            norm_layer=norm_layer
        )

        with_cls_token = kwargs['with_cls_token']

        if with_cls_token:
            self.cls_token = paddle.create_parameter(
                shape=[1, 1, embed_dim],
                dtype='float32',
                default_initializer=_trunc_norm
            )
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(dim_in=embed_dim,
                      dim_out=embed_dim,
                      num_heads=num_heads,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias,
                      drop=drop_rate,
                      attn_drop=attn_drop_rate,
                      drop_path=dpr[j],
                      act_layer=act_layer,
                      norm_layer=norm_layer,
                      **kwargs
                      )
            )
        self.blocks = nn.LayerList(blocks)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            _xavier_init(m.weight)
            if m.bias is not None:
                zero(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D)):
            one(m.weight)
            zero(m.bias)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            _trunc_norm(m.weight)
            if m.bias is not None:
                zero(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D)):
            one(m.weight)
            zero(m.bias)

    def forward(self, x):
        x = self.conv_embed(x)     # x: [B, C, H, W]  饼
        B, C, H, W = x.shape
        x = paddle.reshape(x, [B, C, H*W])
        x = paddle.transpose(x, [0, 2, 1])     # x: [B, H*W, C]  条
        cls_tokens = None
        if self.cls_token is not None:
            cls_tokens = paddle.expand(self.cls_token, [B, -1, -1])
            x = paddle.concat([cls_tokens, x], axis=1)
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)
        if self.cls_token is not None:
            cls_tokens, x = paddle.split(x, [1, H*W], 1)
        B, L, C = x.shape     # L is length of tensor
        x = paddle.reshape(x, [B, H, W, C])
        x = paddle.transpose(x, [0, 3, 1, 2])

        return x, cls_tokens

class ConvolutionalVisionTransformer(nn.Layer):
    '''CvT model
    Introducing Convolutions to Vision Transformers
    Args:
        in_chans: int, input image channels, default: 3
        num_classes: int, number of classes for classification, default: 1000
        num_stage: int, numebr of stage, length of array of parameters should be given, default:3
        patch_size: int[], patch size, default: [7, 3, 3]
        patch_stride: int[], patch_stride ,default: [4, 2, 2]
        patch_padding: int[], patch padding,default: [2, 1, 1]
        embed_dim: int[], embedding dimension (patch embed out dim), default: [64, 192, 384]
        depth: int[], number ot transformer blocks, default: [1, 2, 10]
        num_heads: int[], number of attention heads, default: [1, 3, 6]
        drop_rate: float[], Mlp layer's droppath rate for droppath layers, default: [0.0, 0.0, 0.0]
        attn_drop_rate: float[], attention layer's droppath rate for droppath layers, default: [0.0, 0.0, 0.0]
        drop_path_rate: float[], each block's droppath rate for droppath layers, default: [0.0, 0.0, 0.1]
        with_cls_token: bool[], if image have cls_token, default: [False, False, True]
    '''
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 num_stage=3,
                 patch_size=[7, 3, 3],
                 patch_stride=[4, 2, 2],
                 patch_padding=[2, 1, 1],
                 embed_dim=[64, 192, 384],
                 depth=[1, 2, 10],
                 num_heads=[1, 3, 6],
                 drop_rate=[0.0, 0.0, 0.0],
                 attn_drop_rate=[0.0, 0.0, 0.0],
                 drop_path_rate=[0.0, 0.0, 0.1],
                 with_cls_token=[False, False, True],
                 ):
        super(ConvolutionalVisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_stage = num_stage
        self.stages = nn.LayerList()
        for i in range(self.num_stage):
            stage = VisionTransformer(
                patch_size=patch_size[i],
                patch_stride=patch_stride[i],
                patch_padding=patch_padding[i],
                in_chans=in_chans,
                embed_dim=embed_dim[i],
                depth=depth[i],
                num_heads=num_heads[i],
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=drop_rate[i],
                attn_drop_rate=attn_drop_rate[i],
                drop_path_rate=drop_path_rate[i],
                with_cls_token=with_cls_token[i]
            )
            self.stages.append(stage)
            in_chans = embed_dim[i]

        dim_embed = embed_dim[-1]
        self.norm = nn.LayerNorm(dim_embed)
        self.cls_token = with_cls_token[-1]

        # classifier head
        self.head = nn.Linear(in_features=dim_embed,
                              out_features=num_classes,
                              ) if num_classes > 0 else nn.Identity()

        _trunc_norm(self.head.weight)

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = paddle.load(pretrained, map_location='cpu')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                    k.split('.')[0] in pretrained_layers
                    or pretrained_layers[0] is '*'
                )
                if need_init:
                    if 'pos_embed' in k and v.size() != model_dict[k].size():
                        size_pretrained = v.size()
                        size_new = model_dict[k].size()

                        ntok_new = size_new[1]
                        ntok_new -= 1

                        posemb_tok, posemb_grid = v[:, :1], v[0, 1:]

                        gs_old = int(paddle.sqrt(len(posemb_grid)))
                        gs_new = int(paddle.sqrt(ntok_new))

                        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                        posemb_grid = paddle.ndimage.zoom(
                            posemb_grid, zoom, order=1
                        )
                        posemb_grid = posemb_grid.reshape(1, gs_new ** 2, -1)
                        v = paddle.to_tensor(
                            paddle.concat([posemb_tok, posemb_grid], axis=1)
                        )

                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)

    def forward_features(self, x):
        for i in range(self.num_stage):
            x, cls_tokens = self.stages[i](x)

        if self.cls_token is not None:
            x = self.norm(cls_tokens)
            x = paddle.squeeze(x)
        else:
            #'b c h w -> b (h w) c'
            B, C, H, W = x.shape
            x = paddle.transpose(x, [0, 2, 3, 1])
            x = paddle.reshape(x, [B, H*W, C])
            x = self.norm(x)
            x = paddle.mean(x, axis=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x

def build_cvt(config):
    model = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=config.MODEL.NUM_CLASSES,
        num_stage=config.MODEL.NUM_STAGES,
        patch_size=config.MODEL.PATCH_SIZE,
        patch_stride=config.MODEL.PATCH_STRIDE,
        patch_padding=config.MODEL.PATCH_PADDING,
        embed_dim=config.MODEL.DIM_EMBED,
        depth=config.MODEL.DEPTH,
        num_heads=config.MODEL.NUM_HEADS,
        drop_rate=config.MODEL.DROP_RATE,
        attn_drop_rate=config.MODEL.ATTN_DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        with_cls_token=config.MODEL.CLS_TOKEN
    )
    return model






























