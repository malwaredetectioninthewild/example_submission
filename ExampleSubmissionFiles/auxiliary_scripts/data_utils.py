import torch

class ManualData(torch.utils.data.Dataset):
    def __init__(self, data, labels=None, soft_labels=None, use_soft=False, families=None, transforms=None, dtype=torch.float, device='cpu'):
        self.device = device
        self.data = torch.from_numpy(data).to(device, dtype=dtype)

        if labels is not None:
            self.labels = torch.from_numpy(labels).to(device, dtype=torch.long)
        else:
            self.labels = None
        
        if soft_labels is not None:
            self.soft_labels = torch.from_numpy(soft_labels).to(device, dtype=torch.float)
        else:
            self.soft_labels = None

        self.use_soft = use_soft
            
        self.families = families
        self.transforms = transforms

        self.printed = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        if self.transforms is not None:
            if not self.printed:
                print('adding noise!!!')
                self.printed = True

            data = self.transforms(data)

        ret = [data]

        if self.soft_labels is not None and self.use_soft:
            ret.append(self.soft_labels[idx])

        elif self.labels is not None:
            ret.append(self.labels[idx])

        if self.families is not None:
            ret.append(self.families[idx])

        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)        

def get_loader(dataset, shuffle=False, batch_size=128, underrep_multiplier=0, device='cpu'):
    if device == 'cpu':
        num_workers = 4
    else:
        num_workers = 0
    
    # to oversample the underrepresented class
    if underrep_multiplier > 0:
        labels = dataset.labels
        un_labels, label_counts = torch.unique(labels, return_counts=True)
        l_to_idx = {int(l):ii for ii,l in enumerate(un_labels)}
        min_label_count = torch.argmin(label_counts)
        label_counts[min_label_count] = label_counts[min_label_count] / underrep_multiplier
        labels_weights = 1. / label_counts
        weights = labels_weights[[l_to_idx[int(l)] for l in labels]]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)