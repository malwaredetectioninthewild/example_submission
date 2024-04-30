from torch import nn
import numpy as np

import torch

import auxiliary_scripts.utils as utils
import auxiliary_scripts.data_utils as dutils


class ClassificationHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.feat_dim = self.cfg['feat_dim']
        self.num_out = self.cfg.get('num_out', 2)

        self.classifier = nn.Sequential(*[nn.Linear(self.feat_dim, self.feat_dim), 
                                            nn.BatchNorm1d(self.feat_dim),
                                            nn.ReLU(),
                                            nn.Linear(self.feat_dim, self.num_out)])
        
    def forward(self, x):
        return self.classifier(x)


class Classifier(nn.Module):
    def __init__(self, backbone, classifier):
                 
        super().__init__()

        self.backbone = backbone
        self.classifier = classifier

        if hasattr(self.backbone, 'embedding'):
            self.dtype = torch.long
        else:
            self.dtype = torch.float

        self.forward = self.forward_w_logits # default forward function

    def forward_w_logits(self, x): # return logits

        x = self.backbone(x)
        logits = self.classifier(x)
        return logits
    
    def forward_w_probs(self, x):
        logits = self.forward_w_logits(x)

        if self.classifier.num_out == 1:
            return torch.sigmoid(logits)
        
        if self.classifier.num_out == 2:
            return torch.nn.functional.softmax(logits, dim=1) 
        
    def forward_w_reprs(self, x):
        feats = self.backbone(x)
        out = self.classifier(feats)
        return feats, out
            
    def forward_w_thresholds(self, x, threshold):

        x = self.forward_w_penult(x)
        x = x.clip(min=0, max=threshold)
        logits = self.classifier.classifier[-1](x)

        return logits
    
    def forward_w_penult(self, x):
        x = self.backbone(x)
        x = self.classifier.classifier[:-1](x)
        return x

    def predict_proba(self, feats, temperature=1, batch_size=1024):

        device = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
        loader = dutils.get_loader(dutils.ManualData(feats, dtype=self.dtype, device=device), shuffle=False, batch_size=batch_size, device=device)

        self.eval()

        all_preds = []

        for x  in loader:
            x = x.to(device, dtype=self.dtype)

            with torch.no_grad():
                logits = self.forward(x)
                logits = logits/temperature
                    
            if self.classifier.num_out == 1:
                preds = torch.sigmoid(logits).squeeze().cpu().detach().numpy()
                preds = np.column_stack((1-preds, preds)).reshape(len(x), -1)
            else:
                preds = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().detach().numpy().reshape(len(x), -1)

            all_preds.append(preds)
        
        all_preds = np.concatenate(all_preds)        
        return all_preds

    def get_output(self, feats, out_type='backbone', threshold=None, temperature=1, batch_size=4096):
        device = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
        loader = dutils.get_loader(dutils.ManualData(feats, dtype=self.dtype, device=device), shuffle=False, batch_size=batch_size, device=device)

        self.eval()

        all_outs = []

        for x  in loader:
            x = x.to(device, dtype=self.dtype)
            
            with torch.no_grad():
                if out_type == 'penult':
                    out = self.forward_w_penult(x).cpu().detach().numpy().reshape(len(x), -1)
                
                elif out_type == 'logits':
                    if threshold is None:
                        out = self.forward_w_logits(x).cpu().detach().numpy().reshape(len(x), -1)
                    else:
                        out = self.forward_w_thresholds(x, threshold).cpu().detach().numpy().reshape(len(x), -1)

                elif out_type == 'backbone':
                    out = self.backbone(x).cpu().detach().numpy().reshape(len(x), -1)
                
                elif out_type == 'proba':
                    logits = self.forward_w_logits(x)
                    logits = logits/temperature
                    if self.classifier.num_out == 1:
                        out = torch.sigmoid(logits).squeeze().cpu().detach().numpy()
                        out = np.column_stack((1-out, out)).reshape(len(x), -1)
                    else:
                        out = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().detach().numpy().reshape(len(x), -1)

            all_outs.append(out)
        
        all_outs = np.vstack(all_outs)        
        return all_outs


def get_classifier(backbone_name, cfg):
    # cfg = {'num_out':2, 'in_dim': 100, 'num_tokens': 1000, 'regularization_coeff': 0.5, 'capacity_coeff': 2, 'maxlen': 100}
    # classifier = models.get_classifier('Resnet', cfg)

    backbone = utils.BACKBONE_DICT[backbone_name](cfg)
    classification_head = ClassificationHead({**cfg, **{'feat_dim': backbone.feat_dim}})
    classifier = Classifier(backbone, classification_head)
    return classifier