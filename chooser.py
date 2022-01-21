# train a network that chooses among inputs
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
#generate training data
N_train = 1000
P = 7
N = int(1.3 * N_train)
X = torch.randn(N,P)
N_objects_in_sample = torch.randint(5, 15, (N,))
data = []
labels = []

def select_heuristic(input_tensor):
    return input_tensor.max(1)[0].argmax()

for n_obj in N_objects_in_sample:
    data.append(X[torch.randperm(N)[:n_obj]])
    labels.append(select_heuristic(data[-1]))

#split to train and test
N_test = N - N_train
test = data[:N_test]
test_labels = labels[:N_test]
data = data[N_test:]
labels = labels[N_test:]

class Chooser(nn.Module):
    def __init__(self, input_dim):
        super(Chooser, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128),
                                 nn.Tanh(),
                                 nn.Linear(128, 64),
                                 nn.Tanh(),
                                 nn.Linear(64,16),
                                 nn.Tanh(),
                                 nn.Linear(16, 1)
                                )

    def subsum_by_end_index(self, arr, end_inds):
        b = arr.cumsum(1)
        c = b.gather(1, end_inds) #select relevant terms
        c = torch.cat([torch.zeros(1, 1), c], dim=-1) #start the sum with zeros
        res = c.unsqueeze(2) - c.unsqueeze(1)
        return torch.diagonal(res, offset=-1, dim1=1, dim2=2)

    def exp_subsum_by_end_index(self, arr, end_inds):
        b = torch.logcumsumexp(arr, dim=1)
        c = b.gather(1, end_inds) #select relevant terms
        c = torch.cat([torch.zeros(1, 1), c], dim=-1) #start the sum with zeros
        res = c.unsqueeze(2) - c.unsqueeze(1)
        return torch.diagonal(res, offset=-1, dim1=1, dim2=2)

    def softplus(self, x):
        return torch.log(1 + torch.exp(x))

    def forward(self, x):
        sizes = torch.tensor([e.shape[0] for e in x])
        x_flatten = torch.cat(x, dim=0)
        res_flatten = self.net(x_flatten)
        #res_flatten = chooser.net(x_flatten)
        
        #softmax by hand
        b = torch.max(res_flatten)
        res_flatten = res_flatten - b #stabilize with simple "baseline"
        #asd = res_flatten.clone().detach().T
        res_flatten = res_flatten.exp().T
        end_inds = sizes.cumsum(0)-1
        end_inds = end_inds.unsqueeze(0)
        denominator = self.subsum_by_end_index(res_flatten, end_inds) #problem: cumsum is instable upon difference: sometimes it does not recognize difference
        #kek = self.exp_subsum_by_end_index(asd, end_inds)
        #import ipdb; ipdb.set_trace()
        #denominator = chooser.subsum_by_end_index(res_flatten, end_inds)
        denominator = denominator.repeat_interleave(sizes)
        
        res_flatten = res_flatten.squeeze(0)
        res_flatten = res_flatten / denominator
        
        return res_flatten, sizes

bs = 8
chooser = Chooser(P)
opt = torch.optim.Adam(chooser.parameters(), lr = 0.0001)
losses = []
for ii in range(5000):
    inds = torch.randperm(N_train)[:bs]
    x_batch = [data[ind] for ind in inds]
    y_batch = torch.stack([labels[ind] for ind in inds])
    #x = x_batch
    opt.zero_grad()
    flat_probs, sizes = chooser(x_batch)
    # if torch.any(torch.isinf(flat_probs)):
    #     import ipdb; ipdb.set_trace()
    #     flat_probs, sizes = chooser.forward(x_batch)

    start_inds = torch.cat((torch.zeros([1]),sizes.cumsum(0)[:-1]))
    start_inds = start_inds.int()

    #categorical loss by hand
    select_inds = start_inds + y_batch
    loss = -flat_probs[select_inds].log().sum()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    print(loss.item())

plt.plot(losses)

#test accuracy (slow)
with torch.no_grad():
    flat, sizes = chooser(test)
ind = 0
preds = torch.zeros(N_test)
randoms = torch.zeros(N_test)
for ii, size in enumerate(sizes):
    #ii, size = 0, sizes[0]
    preds[ii] = flat[ind:(ind+size)].argmax()
    randoms[ii] = torch.randint(0, size, (1,))
    ind += size
    

test_labels = torch.tensor(test_labels)
print("test accuracy", (preds == test_labels).float().mean())

(1/sizes).mean()
(randoms == test_labels).float().mean()
