import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


from model import VPR_model
from dataloader import BerlinDataset
from loss import AverageMeter

model = VPR_model(num_clusters=16, dim=512)
model = model.cuda()

print(model)

load_model = torch.load('./컴퓨터비전/NetVLAD/pittsburgh_checkpoint.pth.tar')
model.load_state_dict(load_model['state_dict'])

train_dataset = BerlinDataset(condition="train")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,shuffle=True,num_workers=0)   

epochs = 20
global_batch_size = 8
lr = 0.00001
momentum = 0.9
weightDecay = 0.001
losses = AverageMeter()
best_loss = 100.0
margin = 0.1 

criterion = nn.TripletMarginLoss(margin=margin**0.5, p=2, reduction='sum').cuda()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weightDecay)

model.train()

for epoch in tqdm(range(epochs)):
    for batch_idx, (train_image,train_label) in enumerate(train_loader) :
        output_train = model.encoder(train_image.squeeze().cuda())
        output_train = model.pool(output_train)
        triplet_loss = criterion(output_train[0].reshape(1,-1),output_train[1].reshape(1,-1),output_train[2].reshape(1,-1))

        if batch_idx == 0 :
            optimizer.zero_grad()

        triplet_loss.backward(retain_graph=True)
        losses.update(triplet_loss.item())

        if (batch_idx +1) % global_batch_size == 0 :
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.grad /= global_batch_size
                
                optimizer.step()
                optimizer.zero_grad()

        if batch_idx % 20 == 0 and batch_idx != 0:
            print('epoch : {}, batch_idx  : {}, triplet_loss : {}'.format(epoch,batch_idx,losses.avg))

    if best_loss > losses.avg :
        best_save_name = 'best_model.pt'
        best_path = F"./컴퓨터비전/NetVLAD/ckpt/{best_save_name}" 
        torch.save(model.state_dict(), best_path)
        
    model_save_name = 'model_{:02d}.pt'.format(epoch)
    path = F"./컴퓨터비전/NetVLAD/ckpt/{model_save_name}" 
    torch.save(model.state_dict(), path)
#