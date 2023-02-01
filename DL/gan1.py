import os.path as osp
import argparse
import pdb
import gzip
import pickle
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE, VGAE, APPNP, GCNConv
from torch_geometric.nn.models import InnerProductDecoder
import torch_geometric.transforms as T
cuda = True if torch.cuda.is_available() else False
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--model', type=str, default='VGNAE')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--channels', type=int, default=784)
parser.add_argument('--scaling_factor', type=float, default=1.8)
parser.add_argument('--training_rate', type=float, default=0.85) 
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(args.channels, args.channels//2, normalize=False),
            *block(args.channels//2, args.channels//4),
            *block(args.channels//4, args.channels//2),
            nn.Linear(args.channels//2, args.channels),
            nn.Tanh()

        )

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.channels, args.channels//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.channels//2,args.channels//4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.channels//4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        validity = self.model(x)

        return validity.view(-1)

if args.dataset in ['Cora', 'CiteSeer', 'PubMed']:
    dataset = Planetoid(path, args.dataset, 'public')
if args.dataset in ['cs', 'physics']:
    dataset = Coauthor(path, args.dataset, 'public')
if args.dataset in ['computers', 'photo']:
    dataset = Amazon(path, args.dataset, 'public')


data = dataset[0]
data = T.NormalizeFeatures()(data)
if args.dataset == 'Cora':
    dataset_name = 'cora'
elif args.dataset == 'CiteSeer':
    dataset_name ='citeseer'
else:
    dataset_name = 'pubmed'

with gzip.open(dataset_name+str(args.channels)+'_embedding.pkl','rb') as f:
    # pickle.dump(val_edges,f)
    real_data = pickle.load(f)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
channels = args.channels
train_rate = args.training_rate
val_ratio = (1-args.training_rate) / 3
test_ratio = (1-args.training_rate) / 3 * 2
data = train_test_split_edges(data.to(dev), val_ratio=val_ratio, test_ratio=test_ratio)

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
wandb.init(project="train_rec", entity="faith0918")
wandb.run.name = str(args.dataset)+"GAN"
wandb.config={
    "model":"GAN"
}
# ----------
#  Training
# ----------
Decoder = InnerProductDecoder()
def test(z, pos_edge_index: Tensor, neg_edge_index: Tensor):
        from sklearn.metrics import average_precision_score, roc_auc_score
        
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = Decoder(z, pos_edge_index, True)
        neg_pred = Decoder(z, neg_edge_index, True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)
# def test(pos_edge_index, neg_edge_index, plot_his=0):
#     generator.eval()
#     pdb.set_trace()
#     with torch.no_grad():
#         z = generator(data.x)
#     return GAE.test(z, [pos_edge_index, neg_edge_index.cuda()])
for epoch in range(args.n_epochs):
    # Adversarial ground truths
    valid = torch.ones(data.num_nodes).cuda()
    fake = torch.zeros(data.num_nodes).cuda()


    # -----------------
    #  Train Generator
    # -----------------

    
    # for _ in range(5):
    optimizer_G.zero_grad()
    # Sample noise as generator input
    z = torch.randn([data.num_nodes, args.channels]).cuda()

    # Generate a batch of images
    gen_data = generator(z)

    # Loss measures generator's ability to fool the discriminator
    g_loss = adversarial_loss(discriminator(gen_data), valid)

    g_loss.backward()
    optimizer_G.step()
    # for _ in range(5):
    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(real_data), valid)
    fake_loss = adversarial_loss(discriminator(gen_data.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2

    d_loss.backward()
    optimizer_D.step()
    generator.eval()
    with torch.no_grad():
        z = generator(data.x)
    auc, ap = test(z, data.test_pos_edge_index, data.test_neg_edge_index)
    real_auc, real_ap = test(real_data, data.test_pos_edge_index, data.test_neg_edge_index)
    wandb.log({"D loss":d_loss, "auc":auc, "ap":ap,"G loss":g_loss})
    print(epoch,"epoch", " AUC:", auc, " AP:",ap)
    print(epoch,"epoch", " real AUC:", real_auc, " real AP:",real_ap)
    