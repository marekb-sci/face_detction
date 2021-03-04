# -*- coding: utf-8 -*-
import torchvision
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import misc as misc_nn_ops
import torchvision.models
from torchvision.models.mobilenet import ConvBNReLU, _make_divisible, InvertedResidual
from collections import OrderedDict

from torchvision.models.utils import load_state_dict_from_url

from torch import nn

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def resnet_fpn_backbone(backbone_name, pretrained):
    """copied from """
    backbone = torchvision.models.resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    # freeze layers
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)

class MyMobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MyMobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        # conv1
        self.conv1 = ConvBNReLU(3, input_channel, stride=2)
        # layer1
        t, c, n, s = inverted_residual_setting[0]
        output_channel = _make_divisible(c * width_mult, round_nearest)
        features = []
        for i in range(n):
            stride = s if i == 0 else 1
            features.append(block(input_channel, output_channel, stride, expand_ratio=t))
            input_channel = output_channel
        self.layer1 = nn.Sequential(*features)
        # layer 2
        t, c, n, s = inverted_residual_setting[1]
        output_channel = _make_divisible(c * width_mult, round_nearest)
        features = []
        for i in range(n):
            stride = s if i == 0 else 1
            features.append(block(input_channel, output_channel, stride, expand_ratio=t))
            input_channel = output_channel
        self.layer2 = nn.Sequential(*features)
        # layer 3
        t, c, n, s = inverted_residual_setting[2]
        output_channel = _make_divisible(c * width_mult, round_nearest)
        features = []
        for i in range(n):
            stride = s if i == 0 else 1
            features.append(block(input_channel, output_channel, stride, expand_ratio=t))
            input_channel = output_channel
        self.layer3 = nn.Sequential(*features)
        # layer 4
        t, c, n, s = inverted_residual_setting[3]
        output_channel = _make_divisible(c * width_mult, round_nearest)
        features = []
        for i in range(n):
            stride = s if i == 0 else 1
            features.append(block(input_channel, output_channel, stride, expand_ratio=t))
            input_channel = output_channel
        self.layer4 = nn.Sequential(*features)
        # layer 5
        t, c, n, s = inverted_residual_setting[4]
        output_channel = _make_divisible(c * width_mult, round_nearest)
        features = []
        for i in range(n):
            stride = s if i == 0 else 1
            features.append(block(input_channel, output_channel, stride, expand_ratio=t))
            input_channel = output_channel
        self.layer5 = nn.Sequential(*features)
        # layer 6
        t, c, n, s = inverted_residual_setting[5]
        output_channel = _make_divisible(c * width_mult, round_nearest)
        features = []
        for i in range(n):
            stride = s if i == 0 else 1
            features.append(block(input_channel, output_channel, stride, expand_ratio=t))
            input_channel = output_channel
        self.layer6 = nn.Sequential(*features)
        # layer 7
        t, c, n, s = inverted_residual_setting[6]
        output_channel = _make_divisible(c * width_mult, round_nearest)
        features = []
        for i in range(n):
            stride = s if i == 0 else 1
            features.append(block(input_channel, output_channel, stride, expand_ratio=t))
            input_channel = output_channel
        self.layer7 = nn.Sequential(*features)
        self.conv_final = ConvBNReLU(input_channel, self.last_channel, kernel_size=1)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.conv_final(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

def my_mobilenet_v2(pretrained=False, progress=True):

    model = MyMobileNetV2()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        rename_dict = {'features.0.': 'conv1.',
                       'features.1.': 'layer1.0.',
                       'features.2.': 'layer2.0.',
                       'features.3.': 'layer2.1.',
                       'features.4.': 'layer3.0.',
                       'features.5.': 'layer3.1.',
                       'features.6.': 'layer3.2.',
                       'features.7.': 'layer4.0.',
                       'features.8.': 'layer4.1.',
                       'features.9.': 'layer4.2.',
                       'features.10.': 'layer4.3.',
                       'features.11.': 'layer5.0.',
                       'features.12.': 'layer5.1.',
                       'features.13.': 'layer5.2.',
                       'features.14.': 'layer6.0.',
                       'features.15.': 'layer6.1.',
                       'features.16.': 'layer6.2.',
                       'features.17.': 'layer7.0.',
                       'features.18.': 'conv_final.'
                       }
        def replace(x):
            for k, v in rename_dict.items():
                if k in x:
                    return x.replace(k, v)
            return x
        renamed_state_dict = OrderedDict([(replace(k), v) for k, v in state_dict.items()])
        model.load_state_dict(renamed_state_dict)
    return model

def mobilenetV2_fpn_backbone(pretrained):
    backbone = MyMobileNetV2()

    return_layers = {'layer2': '0', 'layer3': '1', 'layer5': '2', 'conv_final': '3'}

    in_channels_list = [
        24,
        32,
        96,
        1280,
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)


class WithFPN(nn.Module):
    def __init__(self, backbone, in_channels_list, out_channels,
                 names_translator=None):
        super(WithFPN, self).__init__()
        self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels
        self.names_translator = names_translator

    def forward(self, x):
        x = self.body(x)
        if self.names_translator is not None:
            x = OrderedDict([(self.names_translator.get(k, k), v) for k, v in x.items()])
        x = self.fpn(x)
        return x


def get_FasterRCNN_on_resnet50():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_FasterRCNN_on_mobilenet():

    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = torchvision.models.detection.FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model

def get_FasterRCNN_on_mobilenet_fpn():

    model = get_FasterRCNN_on_resnet50()
    model.backbone = mobilenetV2_fpn_backbone(True)

    return model

if __name__ == '__main__':
    mnet = torchvision.models.mobilenet.mobilenet_v2(pretrained=True).eval()
    mmnet = my_mobilenet_v2(pretrained=True).eval()

    import torch
    test_in = torch.rand(1, 3, 224, 224)
    o1 = mnet(test_in)
    o2 = mmnet(test_in)
    print('mobilenet correct:', torch.allclose(o1, o2))

    mfpn = mobilenetV2_fpn_backbone(True)
    rfpn = resnet_fpn_backbone('resnet50', True)

    f1 = rfpn(test_in)
    f2 = mfpn(test_in)
    print('\nRESNET FPN SIZES:')
    for k, v in f1.items():
        print(k, v.size())

    print('\nMOBILENET FPN SIZES:')
    for k, v in f2.items():
        print(k, v.size())

# mnet = torchvision.models.mobilenet.MobileNetV2()
# {i: mnet.features[:i+1](test_in).size() for i in range(len(mnet.features))}
# {0: torch.Size([1, 32, 112, 112]),
#  1: torch.Size([1, 16, 112, 112]),
#  2: torch.Size([1, 24, 56, 56]),
#  3: torch.Size([1, 24, 56, 56]),
#  4: torch.Size([1, 32, 28, 28]),
#  5: torch.Size([1, 32, 28, 28]),
#  6: torch.Size([1, 32, 28, 28]),
#  7: torch.Size([1, 64, 14, 14]),
#  8: torch.Size([1, 64, 14, 14]),
#  9: torch.Size([1, 64, 14, 14]),
#  10: torch.Size([1, 64, 14, 14]),
#  11: torch.Size([1, 96, 14, 14]),
#  12: torch.Size([1, 96, 14, 14]),
#  13: torch.Size([1, 96, 14, 14]),
#  14: torch.Size([1, 160, 7, 7]),
#  15: torch.Size([1, 160, 7, 7]),
#  16: torch.Size([1, 160, 7, 7]),
#  17: torch.Size([1, 320, 7, 7]),
#  18: torch.Size([1, 1280, 7, 7])}

# fpn output, resnet50
# {'0': torch.Size([1, 256, 56, 56]),
#  '1': torch.Size([1, 256, 28, 28]),
#  '2': torch.Size([1, 256, 14, 14]),
#  '3': torch.Size([1, 256, 7, 7]),
#  'pool': torch.Size([1, 256, 4, 4])}