import os
import torch.utils.data
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import PROPOSAL_NUM, SAVE_FREQ, LR, WD, save_dir
from core import model as nts_net
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

import video_transforms
import datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch NTS-Net Action Recognition')
parser.add_argument('--data', metavar='DIR',default='./datasets/haa500_basketball_frames',
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
train_setting_file = "train_%s_split%d.txt" % (args.modality, args.split)
train_split_file = os.path.join(args.settings, args.dataset, train_setting_file)
val_setting_file = "val_%s_split%d.txt" % (args.modality, args.split)
val_split_file = os.path.join(args.settings, args.dataset, val_setting_file)
if not os.path.exists(train_split_file) or not os.path.exists(val_split_file):
    print("No split file exists in %s directory. Preprocess the dataset first" % (args.settings))
print("Loading data from {}".format(args.data))
train_dataset = datasets.__dict__[args.dataset](root=args.data,
                                                source=train_split_file,
                                                phase="train",
                                                modality=args.modality,
                                                is_color= args.modality == 'rgb',
                                                frame_rate=args.frame_rate,
                                                new_width=args.new_width,
                                                new_height=args.new_height)
val_dataset = datasets.__dict__[args.dataset](root=args.data,
                                              source=val_split_file,
                                              phase="val",
                                              modality=args.modality,
                                              is_color= args.modality == 'flow',
                                              frame_rate=args.frame_rate,
                                              new_width=args.new_width,
                                              new_height=args.new_height)

print('{} samples found, {} train samples and {} test samples.'.format(len(val_dataset)+len(train_dataset),
                                                                       len(train_dataset),
                                                                       len(val_dataset)))

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)


# create model
print("Building model ... ")
model = nts_net.attention_net(topN=PROPOSAL_NUM)

# define loss function (criterion)
criterion = torch.nn.CrossEntropyLoss().to(device)

if not os.path.exists(args.resume):
    os.makedirs(args.resume)
print("Saving everything to directory %s." % (args.resume))

# define optimizers
raw_parameters = list(model.pretrained_model.parameters())
part_parameters = list(model.proposal_net.parameters())
concat_parameters = list(model.concat_net.parameters())
partcls_parameters = list(model.partcls_net.parameters())

raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)
concat_optimizer = torch.optim.SGD(concat_parameters, lr=LR, momentum=0.9, weight_decay=WD)
part_optimizer = torch.optim.SGD(part_parameters, lr=LR, momentum=0.9, weight_decay=WD)
partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=LR, momentum=0.9, weight_decay=WD)
schedulers = [MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(concat_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(part_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(partcls_optimizer, milestones=[60, 100], gamma=0.1)]

model = model.to(device)

for epoch in range(start_epoch, args.epochs):
    # begin training
    model.train()
    for i, (input, target) in enumerate(train_loader):
        batch_size = len(target)
        if args.modality == "rgb":
            frame = input[0].float().to(device)
            label = target.to(device)
            raw_logits, concat_logits, part_logits, _, top_n_prob = model(frame)
        elif args.modality == "flow":
            flow_x = input[0].float().to(device)
            flow_y = input[1].float().to(device)
            label = target.to(device)
            raw_logits_x, concat_logits_x, part_logits_x, _, top_n_prob_x = model(flow_x)
            raw_logits_y, concat_logits_y, part_logits_y, _, top_n_prob_y = model(flow_y)
            raw_logits, concat_logits, part_logits, top_n_prob = (
            raw_logits_x + raw_logits_y, concat_logits_x + concat_logits_y, part_logits_x + part_logits_y,
            top_n_prob_x + top_n_prob_y)

        # Teacher Loss 来帮助计算 Navigator Loss 的临时shape
        # 与partcls_loss仅仅改了一个shape
        part_loss = nts_net.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                    label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size, PROPOSAL_NUM)
        # Feature detector loss
        raw_loss = criterion(raw_logits, label)
        # Scrutinizing loss
        concat_loss = criterion(concat_logits, label)
        # Navigator loss
        rank_loss = nts_net.ranking_loss(top_n_prob, part_loss)
        # Teacher loss
        partcls_loss = criterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                 label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))
        total_loss = raw_loss + rank_loss + concat_loss + partcls_loss

        raw_optimizer.zero_grad()
        part_optimizer.zero_grad()
        concat_optimizer.zero_grad()
        partcls_optimizer.zero_grad()

        total_loss.backward()
        raw_optimizer.step()
        part_optimizer.step()
        concat_optimizer.step()
        partcls_optimizer.step()
        for scheduler in schedulers:
            scheduler.step

        # evaluate in the middle
        if i % args.print_freq == 0:
            # eval train test
            train_loss, train_accuracy = evaluate(model, train_loader, args.modality, "train")
            # eval val test
            val_loss, val_accuracy = evaluate(model, val_loader, args.modality, "val")

    # save model
    if epoch % args.save_freq == 0:
	    # save model
        model_state_dict = model.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_accuracy,
            'val_loss': val_loss,
            'val_acc': val_accuracy,
            'model_state_dict': model_state_dict},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

print('finishing training')

def evaluate(model, dataloader, modality, phase):
    model.eval()
    loss = 0
    correct = 0
    n_sample = 0
    for i, (input, target) in enumerate(dataloader):
        with torch.no_grad():
            if modality == "rgb":
                frame = input[0].float().to(device)
                label = target.to(device)
                raw_logits, concat_logits, part_logits, _, top_n_prob = model(frame)
            elif modality == "flow":
                flow_x = input[0].float().to(device)
                flow_y = input[1].float().to(device)
                label = target.to(device)
                raw_logits_x, concat_logits_x, part_logits_x, _, top_n_prob_x = model(flow_x)
                raw_logits_y, concat_logits_y, part_logits_y, _, top_n_prob_y = model(flow_y)
                raw_logits, concat_logits, part_logits, top_n_prob = (raw_logits_x+raw_logits_y, concat_logits_x+concat_logits_y, part_logits_x+part_logits_y, top_n_prob_x+top_n_prob_y)
            elif modality == "both":
                frame = input[0].float().to(device)
                flow_x = input[1].float().to(device)
                flow_y = input[2].float().to(device)
                label = target.to(device)
                raw_logits, concat_logits, part_logits, _, top_n_prob = model(frame) + model(flow_x) + model(flow_y)
            # calculate loss
            concat_loss = criterion(concat_logits, label)
            # calculate accuracy
            _, concat_predict = torch.max(concat_logits, 1)
            n_sample += batch_size
            correct += torch.sum(concat_predict.data == label.data)
            loss += concat_loss.item() * batch_size

    accuracy = float(correct) / n_sample
    loss = loss / n_sample
    print(
        'epoch:{} - {} loss: {:.3f} and {} acc: {:.3f} total samples: {}'.format(
            epoch,
            phase,
            loss,
            phase,
            accuracy,
            n_sample))
    return loss, accuracy
