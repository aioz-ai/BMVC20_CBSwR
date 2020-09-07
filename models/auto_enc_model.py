from .inception_v1 import inception_v1_encoder, inception_v1_decoder
import torch.nn as nn
import os
import torch
import hickle
import torchvision.models as models
from utils import freeze_module


class AutoEncoderModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoderModel, self).__init__()
        self.encoder = encoder
        # freeze_module(self.encoder)
        self.decoder = decoder

    def forward(self, inp):
        latent = self.encoder(inp)
        recon = self.decoder(latent)
        return recon


def build_auto_enc_model(model_type, pretrained=True):
    if model_type == 'vision':
        encoder = models.googlenet(pretrained)
        encoder = nn.Sequential(*list(encoder.children())[:-2])
    else:
        encoder = inception_v1_encoder()
        if pretrained:
            base_model_weights_path = 'models/googlenet.h5'
            if os.path.exists(base_model_weights_path):
                encoder.load_state_dict(
                    {k: torch.from_numpy(v).cuda() for k, v in hickle.load(base_model_weights_path).items()})
            encoder = nn.Sequential(*list(encoder.children())[:-1])
    decoder = inception_v1_decoder()
    model = AutoEncoderModel(encoder, decoder)
    return model
