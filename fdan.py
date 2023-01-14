import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation, groups=groups)

def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        if c1.shape[2] >= 7 and c1.shape[3] >= 7: 
            v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        else:
            v_max = F.adaptive_max_pool2d(input=c1,output_size=(1,1))
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)
        
        return x * m

class RFDB(nn.Module):
    def __init__(self, in_channels, plus=False, cfg=None):
        super(RFDB, self).__init__()
        self.act = activation(cfg.MODEL.ACTIVATION)
        self.esa = ESA(in_channels, nn.Conv2d)
        self.plus = plus
        self.reduction_ratio = 0.5
        self.in_channels = in_channels

        self.c0_e = conv_layer(in_channels=int(in_channels * (self.reduction_ratio ** 1)), out_channels=int(in_channels * (self.reduction_ratio ** 1)), kernel_size=1)
        self.c0_f = conv_layer(in_channels=int(in_channels * (self.reduction_ratio ** 1)), out_channels=int(in_channels * (self.reduction_ratio ** 1)), kernel_size=3)

        self.c1_e = conv_layer(in_channels=int(in_channels * (self.reduction_ratio ** 2)), out_channels=int(in_channels * (self.reduction_ratio ** 2)), kernel_size=1)
        self.c1_f = conv_layer(in_channels=int(in_channels * (self.reduction_ratio ** 2)), out_channels=int(in_channels * (self.reduction_ratio ** 2)), kernel_size=3)

        self.c2_e = conv_layer(in_channels=int(in_channels * (self.reduction_ratio ** 3)), out_channels=int(in_channels * (self.reduction_ratio ** 3)), kernel_size=1)
        self.c2_f = conv_layer(in_channels=int(in_channels * (self.reduction_ratio ** 3)), out_channels=int(in_channels * (self.reduction_ratio ** 3)), kernel_size=3)

    def forward(self, input):
        feature_l0 = input

        feature_l0_split = torch.split(feature_l0, [int(self.in_channels * (self.reduction_ratio ** 1)), int(self.in_channels * (self.reduction_ratio ** 1))], dim=1)
        feature_l0_direct = feature_l0_split[0]
        feature_l0_deep = feature_l0_split[1]
        feature_l0_deep_1 = self.act(self.c0_e(feature_l0_deep))
        feature_l0_output = (feature_l0_direct - feature_l0_deep_1) if self.plus == False else (feature_l0_direct + feature_l0_deep_1)
        feature_l1 = self.act(self.c0_f(feature_l0_deep_1))

        feature_l1_split = torch.split(feature_l1, [int(self.in_channels * (self.reduction_ratio ** 2)), int(self.in_channels * (self.reduction_ratio ** 2))], dim=1)
        feature_l1_direct = feature_l1_split[0]
        feature_l1_deep = feature_l1_split[1]
        feature_l1_deep_1 = self.act(self.c1_e(feature_l1_deep))
        feature_l1_output = (feature_l1_direct - feature_l1_deep_1) if self.plus == False else (feature_l1_direct + feature_l1_deep_1)
        feature_l2 = self.act(self.c1_f(feature_l1_deep_1))

        feature_l2_split = torch.split(feature_l2, [int(self.in_channels * (self.reduction_ratio ** 3)), int(self.in_channels * (self.reduction_ratio ** 3))], dim=1)
        feature_l2_direct = feature_l2_split[0]
        feature_l2_deep = feature_l2_split[1]
        feature_l2_deep_1 = self.act(self.c2_e(feature_l2_deep))
        feature_l2_output = (feature_l2_direct - feature_l2_deep_1) if self.plus == False else (feature_l2_direct + feature_l2_deep_1)
        feature_l3 = self.act(self.c2_f(feature_l2_deep_1))

        out = torch.cat([feature_l0_output, feature_l1_output, feature_l2_output, feature_l3], dim=1)
        out_fused = self.esa(out) 

        return out_fused

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

class Net(nn.Module):
    def __init__(self, cfg=None):
        super(Net, self).__init__()
        self.cfg = cfg
        in_nc=3
        nf=48 if not hasattr(cfg.MODEL, 'CHANNEL') else cfg.MODEL.CHANNEL
        num_modules=6
        out_nc=3
        scale=cfg.MODEL.SCALE
        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)
        self.act = activation(cfg.MODEL.ACTIVATION)

        self.B1 = RFDB(in_channels=nf, plus=cfg.MODEL.PLUS, cfg=cfg)
        self.B2 = RFDB(in_channels=nf, plus=cfg.MODEL.PLUS, cfg=cfg)
        self.B3 = RFDB(in_channels=nf, plus=cfg.MODEL.PLUS, cfg=cfg)
        self.B4 = RFDB(in_channels=nf, plus=cfg.MODEL.PLUS, cfg=cfg)
        self.B5 = RFDB(in_channels=nf, plus=cfg.MODEL.PLUS, cfg=cfg)
        self.B6 = RFDB(in_channels=nf, plus=cfg.MODEL.PLUS, cfg=cfg)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=scale)
        self.scale_idx = 0

    def forward(self, input=None, gt=None, training=None, mode='training'):
        # Init bags
        scalars = {}
        tensors = {}
        outputs = {}

        # Inference
        if type(input) == dict:
            lr_sdr = input['lr_sdr']
        else:
            lr_sdr = input

        out_fea = self.act(self.fea_conv(lr_sdr))
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)
        
        if type(input) == dict:
            outputs['hr_hdr'] = output
            
            # Backward
            if training:
                # Calculate scalar
                if hasattr(self.cfg.MODEL, 'LOSS'):
                    if self.cfg.MODEL.LOSS == 'l1':
                        loss = torch.nn.functional.l1_loss(output, gt['hr_hdr'])
                    elif self.cfg.MODEL.LOSS == 'l2':
                        loss = torch.nn.functional.mse_loss(output, gt['hr_hdr'])
                else:
                    loss = torch.nn.functional.l1_loss(output, gt['hr_hdr'])

                # Visualization
                scalars.update({'loss': loss.item()})
                
                # Backward
                loss.backward()
        
            return outputs, scalars, tensors
        else:
            return output