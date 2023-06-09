import argparse
import coremltools as ct
import glob
import numpy as np
import os
import torch
from torch import nn
from torch.utils.mobile_optimizer import optimize_for_mobile
from gfpgan import GFPGANer

def make_GFPGANer(version, upscale, bg_upsampler=None):
    if version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    elif version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    elif version == '1.4':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    else:
        raise ValueError(f'Wrong model version {version}.')

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    return restorer

class GFPGANerWrapper(nn.Module):
    def __init__(self):
        super(GFPGANerWrapper, self).__init__()
        self.model = make_GFPGANer("1.3", 1).gfpgan

    def forward(self, x):
        res = self.model(x, return_rgb=False, weight=0.5)[0]
        return res


def main(args):
    torch_model = GFPGANerWrapper().eval()
    print("MODEL: ")
    print(torch_model)

    # Trace the model with random data.
    example_input = torch.rand(1, 3, 512, 512)
    traced_model = torch.jit.trace(torch_model, example_input)

    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.ImageType(name="input", shape=example_input.shape)],
        outputs=[ct.ImageType(name="output")]
    )

    print(model)

    # # Save the converted model.
    model.save("newmodel.mlpackage")
    pass

def parse_args():
    """Convert GFPGAN to CoreML.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--version', type=str, default='1.3', help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
