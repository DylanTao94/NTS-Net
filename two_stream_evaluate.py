import os
import torch.utils.data
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import PROPOSAL_NUM, SAVE_FREQ, LR, WD, save_dir, RGB_MODEL_PATH, FLOW_MODEL_PATH
from core import model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

start_epoch = 0
save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)

import os
import argparse
import torch.utils.data
import datasets


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch NTS-Net Action Recognition')
parser.add_argument('--data', metavar='DIR',default='./datasets/haa500_basketball_flows',
                    help='path to dataset')
parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
                    help='path to datset setting files')
parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
                    choices=["rgb", "flow"],
                    help='modality: rgb | flow')
parser.add_argument('--dataset', '-d', default='haa500_basketball',
                    choices=["ucf101", "hmdb51", "haa500_basketball"],
                    help='dataset: ucf101 | hmdb51 | haa500_basketball')
parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 50)')
parser.add_argument('--iter_size', default=1, type=int,
                    metavar='I', help='iter size as in Caffe to reduce memory usage (default: 5)')
parser.add_argument('--frame_rate', default=1, type=int,
                    metavar='N', help='length of sampled video frames (default: 1)')
parser.add_argument('--new_width', default=340, type=int,
                    metavar='N', help='resize width (default: 340)')
parser.add_argument('--new_height', default=256, type=int,
                    metavar='N', help='resize height (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[100, 200], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print_freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--save_freq', default=1, type=int,
                    metavar='N', help='save frequency (default: 1)')
parser.add_argument('--resume', default='./checkpoints', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

global args
args = parser.parse_args()

# data loading
val_setting_file = "val_%s_split%d.txt" % ('flow', args.split)
val_split_file = os.path.join(args.settings, args.dataset, val_setting_file)

print("Loading data from {}".format(args.data))

val_dataset = datasets.__dict__[args.dataset](root=args.data,
                                              source=val_split_file,
                                              phase="val",
                                              modality="both",
                                              is_color=True,
                                              new_width=args.new_width,
                                              new_height=args.new_height)

print('{} rgb samples found, {} flow samples found.'.format(len(val_dataset), len(val_dataset)*2))

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)


# load model
print("loading models ... ")
rgb_nts_net = model.attention_net(topN=PROPOSAL_NUM)
# rgb_nts_net.load_state_dict(torch.load(RGB_MODEL_PATH))
flow_nts_net = model.attention_net(topN=PROPOSAL_NUM)
# flow_nts_net.load_state_dict(torch.load(FLOW_MODEL_PATH))
rgb_nts_net = rgb_nts_net.to(device)
flow_nts_net = flow_nts_net.to(device)

# define loss function (criterion)
criterion = torch.nn.CrossEntropyLoss().to(device)
rgb_nts_net.eval()
flow_nts_net.eval()



# evaluate on val
loss = 0
correct = 0
n_sample = 0
for i, (input, target) in enumerate(val_loader):
    with torch.no_grad():
        batch_size = len(target)
        frame = input[0].float().to(device)
        flow_x = input[1].float().to(device)
        flow_y = input[2].float().to(device)

        label = target.to(device)

        _, concat_logits_rgb, _, _, _ = rgb_nts_net(frame)
        _, concat_logits_x, _, _, _ = flow_nts_net(flow_x)
        _, concat_logits_y, _, _, _ = flow_nts_net(flow_y)

        # calculate loss
        concat_loss_rgb = criterion(concat_logits_rgb, label)
        concat_loss_x = criterion(concat_logits_x, label)
        concat_loss_y = criterion(concat_logits_y, label)

        concat_logits = (concat_logits_rgb + concat_logits_x + concat_logits_y) / 3
        # calculate accuracy
        _, concat_predict = torch.max(concat_logits, 1)

        n_sample += batch_size
        correct += torch.sum(concat_predict.data == label.data)
        loss += concat_loss_rgb.item() * batch_size
        loss += concat_loss_x.item() * batch_size
        loss += concat_loss_y.item() * batch_size

        label = target.to(device)

accuracy = float(correct) / n_sample
loss = loss / n_sample
print(
    'loss: {:.3f} and acc: {:.3f}'.format(
        loss,
        accuracy))