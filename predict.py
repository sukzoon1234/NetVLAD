import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import faiss
import json

#
from dataloader import BerlinDataset
from model import VPR_model

model = VPR_model(num_clusters=16, dim=512)
model = model.cuda()

print(model)

load_model = torch.load('./컴퓨터비전/NetVLAD/ckpt/best_model.pt')
model.load_state_dict(load_model)


#Extract NetVLAD descriptors from Reference and Query
cluster_dataset = BerlinDataset(condition="cluster")
cluster_loader = torch.utils.data.DataLoader(cluster_dataset, batch_size=1,shuffle=False,num_workers=0) 

train_feature_list = list()

model.eval()

with torch.no_grad():
  for batch_idx, train_image in tqdm(enumerate(cluster_loader)) :
    output_train = model.encoder(train_image.cuda())
    output_train = model.pool(output_train)
    train_feature_list.append(output_train.squeeze().detach().cpu().numpy())

train_feature_list = np.array(train_feature_list)

test_dataset = BerlinDataset(condition="test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,shuffle=False,num_workers=0) 

test_feature_list = list()

with torch.no_grad():
  for batch_idx, test_image in tqdm(enumerate(test_loader)) :
    output_test = model.encoder(test_image.cuda())
    output_test = model.pool(output_test)
    test_feature_list.append(output_test.squeeze().detach().cpu().numpy())

test_feature_list = np.array(test_feature_list)


#Predict the top-20 highest probability reference indices using fassi
n_values = [1,5,10,20]

faiss_index = faiss.IndexFlatL2(train_feature_list.shape[1])
faiss_index.add(train_feature_list)
_, predictions = faiss_index.search(test_feature_list, max(n_values))


# make  'submission.json'
file_path = "./submit.json"

data = {}
data['Query'] = list()

for i in range(len(predictions)) :
  data_t = [("id",i),("positive",predictions[i].tolist())]
  data_t = dict(data_t)
  data['Query'].append(data_t)
  
with open(file_path, 'w') as outfile:
    json.dump(data, outfile, indent=4)