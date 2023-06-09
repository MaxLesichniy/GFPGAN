import argparse
import os
import torch
import torchvision
from torch import nn
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.backends._coreml.preprocess import (
    CompileSpec,
    TensorSpec,
    CoreMLComputeUnit,
)
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

    return (restorer, model_name)

class GFPGANWrapper(nn.Module):
    def __init__(self, model):
        super(GFPGANWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        res = self.model(x, return_rgb=False, weight=0.5)[0]
        return res

def main(args):
    gfpgan, model_name = make_GFPGANer("1.3", 1)
    torch_model = GFPGANWrapper(gfpgan.gfpgan).eval()
    print("MODEL: ")
    print(torch_model)

    # Trace the model with random data.
    example_input = torch.rand(1, 3, 512, 512)
    traced_model = torch.jit.trace(torch_model, example_input)
    # out = traced_model(example_input)

    optimized_model = optimize_for_mobile(traced_model)
    optimized_model._save_for_lite_interpreter(f"{model_name}.pt")

    # optimized_model = optimize_for_mobile(traced_model, backend="metal")
    # print(torch.jit.export_opnames(optimized_model))
    # optimized_model._save_for_lite_interpreter(f"{model_name}_metal.pt")

    # mlmodel = torch._C._jit_to_backend("coreml", traced_model, spec())
    # mlmodel._save_for_lite_interpreter(f"{model_name}_coreml.ptl")
    pass

def spec():
    return {
        "forward": CompileSpec(
            inputs=(
                TensorSpec(
                    shape=[1, 3, 512, 512],
                ),
            ),
            outputs=(
                TensorSpec(
                    shape=[1, 3, 512, 512],
                ),
            ),
            backend=CoreMLComputeUnit.ALL,
            allow_low_precision=True,
        ),
    }

def parse_args():
    """Export GFPGAN to Torch Script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--version', type=str, default='1.3', help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
