# encoding: utf-8

import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a

from einops import rearrange, repeat
from .Transformer import Transformer
import torch.nn.functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num):
        super(ClassBlock, self).__init__()

        add_block = []
        add_block += [nn.Conv2d(input_dim, input_dim//4, kernel_size=1)]
        add_block += [nn.BatchNorm2d(input_dim//4)]
        add_block += [nn.Conv2d(input_dim//4, input_dim//16, kernel_size=1)]
        add_block += [nn.BatchNorm2d(input_dim//16)]
        add_block = nn.Sequential(*add_block)

        classifier = []
        classifier += [nn.Linear(input_dim//16, 2)]
        classifier = nn.Sequential(*classifier)

        self.add_block = add_block
        self.classifier = classifier
        self.gap = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x = self.add_block(x)
        x = self.gap(x).squeeze()
        x = self.classifier(x)

        return x

def _make_layer(block, inplanes, planes, blocks, stride=2):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice, bot_depth, tfc_depth, in_dim):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        # self.part_num = cfg.CLUSTERING.PART_NUM
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.gap3 = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        self.HAT = HAT(
            img_size=(16, 8),
            patch_size=1,
            in_dim=in_dim,
            poi_dim=2048,
            tfc_depth=tfc_depth,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            )

        self.backbone_planes = 2048
        f_dim1, f_dim2, f_dim3 = in_dim

        self.bottleneck_b = nn.BatchNorm1d(self.backbone_planes)
        self.bottleneck_b.bias.requires_grad_(False)  # no shift
        self.classifier_b = nn.Linear(self.backbone_planes, self.num_classes, bias=False)

        self.bottleneck_t1 = nn.BatchNorm1d(f_dim1)
        self.bottleneck_t1.bias.requires_grad_(False)  # no shift
        self.classifier_t1 = nn.Linear(f_dim1, self.num_classes, bias=False)

        self.bottleneck_t2 = nn.BatchNorm1d(f_dim1+f_dim2)
        self.bottleneck_t2.bias.requires_grad_(False)  # no shift
        self.classifier_t2 = nn.Linear(f_dim1+f_dim2, self.num_classes, bias=False)

        self.bottleneck_t3 = nn.BatchNorm1d(self.backbone_planes)
        self.bottleneck_t3.bias.requires_grad_(False)  # no shift
        self.classifier_t3 = nn.Linear(self.backbone_planes, self.num_classes, bias=False)

        self.bottleneck_b.apply(weights_init_kaiming)
        self.classifier_b.apply(weights_init_classifier)

        self.bottleneck_t1.apply(weights_init_kaiming)
        self.classifier_t1.apply(weights_init_classifier)

        self.bottleneck_t2.apply(weights_init_kaiming)
        self.classifier_t2.apply(weights_init_classifier)

        self.bottleneck_t3.apply(weights_init_kaiming)
        self.classifier_t3.apply(weights_init_classifier)

        self.non_linear1 = _make_layer(Bottleneck, f_dim1, f_dim1//4, bot_depth, stride=1)
        self.non_linear2 = _make_layer(Bottleneck, f_dim2, f_dim2//4, bot_depth, stride=1)
        self.non_linear3 = _make_layer(Bottleneck, f_dim3, f_dim3//4, bot_depth, stride=1)

        self.max_pool1 = nn.AdaptiveMaxPool2d((16, 8))
        self.max_pool2 = nn.AdaptiveMaxPool2d((16, 8))

    def forward(self, x):

        ############ CNN Baseline
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        # x = self.relu(x)    # add missed relu
        x = self.base.maxpool(x)

        x1 = self.base.layer1(x)
        x2 = self.base.layer2(x1)
        x3 = self.base.layer3(x2)
        x = self.base.layer4(x3)

        gap_b = self.gap1(x).squeeze()

        x_t_1 = self.max_pool1(self.non_linear1(x1))
        x_t_2 = self.max_pool2(self.non_linear2(x2))
        x_t_3 = self.non_linear3(x3)

        x_mid_1, x_mid_2, x_mid_3 = self.HAT(x_t_1, x_t_2, x_t_3)

        feat_b = self.bottleneck_b(gap_b)
        feat_t1 = self.bottleneck_t1(x_mid_1)
        feat_t2 = self.bottleneck_t2(x_mid_2)
        feat_t3 = self.bottleneck_t3(x_mid_3)

        if self.training:
            cls_score_b = self.classifier_b(feat_b)
            cls_score_t1 = self.classifier_t1(feat_t1)
            cls_score_t2 = self.classifier_t2(feat_t2)
            cls_score_t3 = self.classifier_t3(feat_t3)

            return [cls_score_b, cls_score_t3, cls_score_t1, cls_score_t2], [gap_b, x_mid_3, x_mid_1, x_mid_2]  # global feature for triplet loss
        else:
            return torch.cat((feat_b, feat_t3), dim=1)

    # def load_param(self, trained_path):
    #     param_dict = torch.load(trained_path)
    #     for i in param_dict:
    #         if 'classifier' in i:
    #             continue
    #         self.state_dict()[i].copy_(param_dict[i])

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for k,v in param_dict.state_dict().items():
            if 'classifier' in k:
                continue
            self.state_dict()[k].copy_(param_dict.state_dict()[k])

class TFC(nn.Module):
    def __init__(self, in_channel, out_channel, img_size, num_patch, p_size, emb_dropout, T_depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super(TFC, self).__init__()

        height, width = img_size

        self.p_size = p_size

        self.patch_to_embedding = nn.Linear(in_channel, out_channel)
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channel))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, out_channel))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(out_channel, T_depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

        self.NeA = Bottleneck(out_channel, out_channel//4)

    def forward(self, x, mask=None):

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p_size, p2=self.p_size)
        x = self.patch_to_embedding(x)
        b, n, _ = x.size()
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x_mid = x[:, 0]
        x_mid = self.to_latent(x_mid)
        x = rearrange(x[:, 1:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.p_size, p2=self.p_size, h=16, w=8)
        x = self.NeA(x)

        return x, x_mid

class HAT(nn.Module):
    def __init__(self, img_size, patch_size, in_dim, poi_dim, heads, mlp_dim, tfc_depth, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super(HAT, self).__init__()

        ###################### parameters
        T_depth1, T_depth2, T_depth3 = tfc_depth
        inc1, inc2, inc3 = in_dim

        self.TFC_S1 = TFC(in_channel=inc1, out_channel=inc1, img_size=[16, 8], num_patch=128, p_size=1, emb_dropout=0.1, T_depth=T_depth1,
                          heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=0.1)
        self.TFC_S2 = TFC(in_channel=(inc1+inc2), out_channel=(inc1+inc2), img_size=[16, 8], num_patch=128, p_size=1, emb_dropout=0.1, T_depth=T_depth2,
                          heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=0.1)
        self.TFC_S3 = TFC(in_channel=(inc1+inc2+inc3), out_channel=poi_dim, img_size=[16, 8], num_patch=128, p_size=1, emb_dropout=0.1, T_depth=T_depth3,
                          heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=0.1)


    def forward(self, x1, x2, x3, mask=None):

        x1, x_mid_1 = self.TFC_S1(x1)
        x2, x_mid_2 = self.TFC_S2(torch.cat((x2 ,x1), dim=1))
        x3, x_mid_3 = self.TFC_S3(torch.cat((x3 ,x2), dim=1))

        return x_mid_1, x_mid_2, x_mid_3
