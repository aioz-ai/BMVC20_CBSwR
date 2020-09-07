import os
import hickle
import torch
from torch import nn as nn
from models import inception_v1_encoder, inception_v1_decoder
from models.util_layers import Flatten, Normalize
import torchvision
from utils import freeze_module
from models.decoder import Decoder

from utils import freeze_module


class DualModel(nn.Module):
    """
    Dual model which perform both metric learning and clustering
    """
    def __init__(self, base_model, decoder, low_dim, n_cluster, freeze=False):
        super(DualModel, self).__init__()
        # Backbone model
        self.base_model = base_model
        self.flatten = nn.Sequential(nn.AvgPool2d((7, 7), (1, 1), ceil_mode=True), Flatten())
        # Reconstruction
        self.decoder = decoder
        # Feature extractor
        self.feat_ext = get_feature_block(1024, low_dim)
        # Metric learning branch
        self.l2norm = Normalize(2)
        # Clustering branch
        self.clustering = get_cluster_block(low_dim, n_cluster)
        if freeze:
            freeze_module(self.decoder)
        self.mode = 'normal'

    def forward(self, input):
        """
        input --> emb --> feats --->  l2norm --> metric_out
                                          `.
                                            `-->  cluster_out
        """
        if self.mode == 'pool':
            return self._forward_pooling(input)
        else:
            return self._forward_normal(input)

    def _forward_pooling(self, input):
        emb7x7 = self.base_model(input)
        emb = self.flatten(emb7x7)
        feats = self.feat_ext(emb)
        metric_out = self.l2norm(feats)
        # cluster_out = self.clustering(self.l2norm(emb))
        cluster_out = self.clustering(metric_out)
        return metric_out, self.l2norm(emb), self.l2norm(cluster_out)

    def _forward_normal(self, input):
        emb7x7 = self.base_model(input)
        emb = self.flatten(emb7x7)
        feats = self.feat_ext(emb)
        metric_out = self.l2norm(feats)
        cluster_out = self.clustering(metric_out)
        # cluster_out = self.clustering(self.l2norm(emb))
        return metric_out, self.l2norm(cluster_out), emb7x7

    def metric_repr(self, input):
        return self.l2norm(input)


def build_dual_model(model_type='vision', pretrained=True, low_dim=128, n_cluster=100, freeze=True):
    if model_type == 'vision':
        encoder = torchvision.models.googlenet(pretrained)
        encoder = nn.Sequential(*list(encoder.children())[:-2])
    else:
        encoder = inception_v1_encoder()
        if pretrained:
            base_model_weights_path = 'models/googlenet.h5'
            if os.path.exists(base_model_weights_path):
                encoder.load_state_dict(
                    {k: torch.from_numpy(v).cuda() for k, v in hickle.load(base_model_weights_path).items()})
            encoder = nn.Sequential(*list(encoder.children())[:-1])
    # decoder = inception_v1_decoder()
    decoder = Decoder()
    model = DualModel(encoder, decoder, low_dim, n_cluster)
    return model


def get_feature_block(input_size, output_size):
    block = [
        # nn.BatchNorm1d(1024),
        # nn.Dropout(0.6),
        nn.Linear(input_size, output_size)
    ]
    block = nn.Sequential(*block)
    return block


def get_cluster_block(input_size, output_size):
    return nn.Sequential(
        # nn.ReLU(),
        nn.BatchNorm1d(input_size),
        nn.Dropout(0.6),
        # nn.Linear(input_size, 512, bias=False),
        # nn.ReLU(),
        # nn.Dropout(),
        # nn.BatchNorm1d(512),
        nn.Linear(input_size, output_size),
    )
