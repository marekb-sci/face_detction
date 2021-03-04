# -*- coding: utf-8 -*-

import functools
import torch
import torchvision
import json
import numpy as np
import matplotlib.pyplot as plt
import models



# %%
config = {
    'backbone': 'resnet50', #allowed 'resnet50' or 'mobilenetV2'
    'validation_part': 0.2, # size of validation dataset
    'num_epochs': 200,
    'train_backbone': False,
    'batch_size': 8,
    'device': None,
    'COCO_dataset': 'path to train2017 directory from COCO dataset',
    'COCO_annotations': 'path to instances_train2017.json file from COCO dataset' 
    }


# %%

# load a model pre-trained pre-trained on COCO
if config['backbone'] == 'resnet50':
    model = models.get_FasterRCNN_on_resnet50()
elif config['backbone'] == 'mobilenetV2':
    model = models.get_FasterRCNN_on_mobilenet_fpn()
else:
    raise ValueError(f'incorrect model name {config["backbone"]}')

transform = torchvision.transforms.ToTensor()

# %%

class TrainingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, ds, face_index_file):
        self.ds = ds
        self.face_index = self.load_face_index(face_index_file)
        self.img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
            ])

    @staticmethod
    def load_face_index(face_index_file):
        face_index = []
        with open(face_index_file) as f:
            for line in f:
                entry_data = json.loads(line.strip())
                bboxes = torch.as_tensor([instance['bbox'] for instance in entry_data[1:]], dtype=torch.float32)
                entry = {'image_idx': entry_data[0]['img_idx'],
                         'bboxes': bboxes}
                face_index.append(entry)
        return face_index



    def __getitem__(self, idx):
        entry_data = self.face_index[idx]
        img, raw_labels = self.ds[entry_data['image_idx']]

        bboxes = entry_data['bboxes']
        img = self.img_transform(img)

        labels = {
            "boxes": bboxes,
            "labels": torch.ones(len(bboxes), dtype=torch.int64),
            }
        return img, labels

    def __len__(self):
        return len(self.face_index)

# %%
coco_dataset = torchvision.datasets.CocoDetection(config['COCO_dataset'],
                                                  config['COCO_annotations'])

face_dataset = TrainingFaceDataset(coco_dataset,
                                   'data/faces_10001.txt')

# img, labels = face_dataset[0]


# %%
N_train = int(len(face_dataset) * (1 - config['validation_part']) )

dataset_train = torch.utils.data.Subset(face_dataset,
                                        list(range(N_train))
                                        )

dataset_test = torch.utils.data.Subset(face_dataset,
                                      list(range(N_train, config['N']))
                                      )

# %%
def collate_fn(batch):
    return tuple(zip(*batch))

# %%

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=config['batch_size'], shuffle=True, #num_workers=1,
    collate_fn=collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False,# num_workers=1,
    collate_fn=collate_fn)


#%%

if config['device'] is None:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = config['device']

# construct an optimizer
if not config['train_backbone']:
    for p in model.backbone.parameters():
        p.requires_grad =  False
params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.Adam(params, lr=0.005)
# optimizer = torch.optim.SGD(params, lr=0.005,
#                             momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                step_size=3,
#                                                gamma=0.1)
#%%
from pathlib import Path
outputdirs = {'test': Path('outputs/test_new'),
              'train': Path('outputs/train_new'),
              'weights': Path('outputs/weights_new')
              }

for path in outputdirs.values():
    path.mkdir(exist_ok=True, parents=True)

# %%
import engine

history = []
for epoch in range(config['num_epochs']):
    print('EPOCH', epoch)
    # train for one epoch, printing every 10 iterations
    loss = engine.train_one_epoch(model, optimizer, data_loader, device, epoch)
    history.extend(loss)

torch.save(model.state_dict(), outputdirs['weights'] / f'weights_{config["N"]}_{config["num_epochs"]}')

history_dict = {}
for k in history[0].keys():
    history_dict[k] = [loss_dict[k] for loss_dict in history]
history_dict['total'] = np.sum([np.array(v) for v in history_dict.values()], axis=0)


# %%
import matplotlib
try:
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic("matplotlib Qt5")
except:
    print('unable to start matplotlib Qt5')

fig, axes = plt.subplots(len(history_dict), sharex=True)
for i, (k, v) in enumerate(history_dict.items()):
    axes[i].plot(v)

# %%
def visualize_boxes(img, detections, target=None, nmax=5, color='g'):
    fig, ax = plt.subplots()
    plt.imshow(img.permute(1,2,0))

    for x1, y1, x2, y2 in detections['boxes'].detach().cpu().numpy()[:nmax]:
        bbox = matplotlib.patches.Rectangle((x1, y1), x2-x1, y2-y1,
              linewidth=2, edgecolor='r', facecolor='none', alpha=0.7)
        ax.add_patch(bbox)

    if target is not None:
        for x1, y1, x2, y2 in target['boxes'].detach().cpu().numpy():
            bbox = matplotlib.patches.Rectangle((x1, y1), x2-x1, y2-y1,
                  linewidth=0.8, edgecolor='g', facecolor='none', alpha=0.7)
            ax.add_patch(bbox)

# %%

pil_transform = torchvision.transforms.ToPILImage()
model.eval()

datasets = {'test': dataset_test, 'train': dataset_train }

for label in ['test', 'train']:
    outputdir = outputdirs[label]
    dataset = datasets[label]
    for idx in range(len(dataset)):
        print(label, idx)
        img, target = dataset[idx]
        with torch.no_grad():
            preds = model([img])[0]
        visualize_boxes(img, preds, target)
        plt.savefig(f'{outputdir}/{idx}.png', dpi=150)
        plt.clf()
        plt.close('all')