import torch

from auxiliary_scripts.backbones.resnet import ResnetBackbone

import auxiliary_scripts.models as models

BACKBONE_DICT = {'Resnet': ResnetBackbone}

def save_model(model, filepath):

    save = {
        'backbone_name': model.backbone.__name__,
        'backbone_cfg': model.backbone.cfg,
        'classifier_cfg': model.classifier.cfg,
        'backbone_state': model.backbone.state_dict(),
        'classifier_state': model.classifier.state_dict()
    }

    torch.save(save, filepath)

def load_model(filepath, device='cuda', train=False):
    state = torch.load(filepath)

    backbone = BACKBONE_DICT[state['backbone_name']](state['backbone_cfg'])

    backbone.load_state_dict(state['backbone_state'])

    classifier_head = models.ClassificationHead(state['classifier_cfg'])
    classifier_head.load_state_dict(state['classifier_state'])
    
    model = models.Classifier(backbone=backbone, classifier=classifier_head)

    return model.to(device).train(train)