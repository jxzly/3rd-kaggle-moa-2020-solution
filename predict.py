# daishu prediction

import pandas as pd
import numpy as np
import warnings
import gc,os
from time import time
import datetime,random
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import QuantileTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset, DataLoader,RandomSampler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import argparse

def Parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--input_dir',
                      default='./data', help='input data path of dataset')
    args.add_argument('--new_test_data_path',
                      default='./data/test_features.csv', help='input data path of dataset')
    args.add_argument('--output_dir',
                      default='./prediction', help='input data path of dataset')
    args = args.parse_args()
    return args

args = Parse_args()

warnings.simplefilter('ignore')

ncompo_genes = 80
ncompo_cells = 10

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

Seed_everything(seed=42)

def Metric(labels,preds):
    labels = np.array(labels)
    preds = np.array(preds)
    metric = 0
    for i in range(labels.shape[1]):
        metric += (-np.mean(labels[:,i]*np.log(np.maximum(preds[:,i],1e-15))+(1-labels[:,i])*np.log(np.maximum(1-preds[:,i],1e-15))))
    return metric/labels.shape[1]

files = ['%s/test_features.csv'%args.input_dir,
         '%s/train_targets_scored.csv'%args.input_dir,
         '%s/train_features.csv'%args.input_dir,
         '%s/train_targets_nonscored.csv'%args.input_dir,
         '%s/train_drug.csv'%args.input_dir,
         '%s/sample_submission.csv'%args.input_dir]

test = pd.read_csv(files[0])
train_target = pd.read_csv(files[1])
train = pd.read_csv(files[2])
train_nonscored = pd.read_csv(files[3])
train_drug = pd.read_csv(files[4])

new_test = pd.read_csv(args.new_test_data_path)

drop_cols = ['g-513', 'g-370', 'g-707', 'g-300', 'g-130', 'g-375', 'g-161',
       'g-191', 'g-376', 'g-176', 'g-477', 'g-719', 'g-449', 'g-204',
       'g-595', 'g-310', 'g-276', 'g-399', 'g-438', 'g-537', 'g-582',
       'g-608', 'g-56', 'g-579', 'g-45', 'g-252', 'g-12', 'g-343',
       'g-737', 'g-571', 'g-555', 'g-506', 'g-299', 'g-715', 'g-239',
       'g-654', 'g-746', 'g-436', 'g-650', 'g-326', 'g-630', 'g-465',
       'g-487', 'g-290', 'g-714', 'g-452', 'g-227', 'g-170', 'g-520',
       'g-467']+['g-54', 'g-87', 'g-111', 'g-184', 'g-237', 'g-302', 'g-305',
       'g-313', 'g-348', 'g-399', 'g-450', 'g-453', 'g-461', 'g-490',
       'g-497', 'g-550', 'g-555', 'g-584', 'g-592', 'g-682', 'g-692',
       'g-707', 'g-748', 'g-751']
drop_cols = list(set(drop_cols))
train.drop(drop_cols,axis=1,inplace=True)
test.drop(drop_cols,axis=1,inplace=True)
new_test.drop(drop_cols,axis=1,inplace=True)

genes = [col for col in train.columns if col.startswith("g-")]
cells = [col for col in train.columns if col.startswith("c-")]

features = genes + cells
targets = [col for col in train_target if col!='sig_id']
nonscored_targets = [col for col in train_nonscored if col!='sig_id']


ori_train = train.copy()
ctl_train = train.loc[train['cp_type']=='ctl_vehicle'].append(test.loc[test['cp_type']=='ctl_vehicle']).reset_index(drop=True)

def Feature(df):
    transformers = {}
    for col in tqdm(genes+cells):
        transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution='normal')
        transformer.fit(df[:train.shape[0]][col].values.reshape(-1,1))#transformer.fit(df[col].values.reshape(-1,1))
        df[col] = transformer.transform(df[col].values.reshape(-1,1)).reshape(1,-1)[0]
        transformers[col] = transformer
    gene_pca = PCA(n_components = ncompo_genes,
                    random_state = 42).fit(df[:train.shape[0]+test.shape[0]][genes])
    pca_genes = gene_pca.transform(df[genes])
    cell_pca = PCA(n_components = ncompo_cells,
                    random_state = 42).fit(df[:train.shape[0]+test.shape[0]][cells])
    pca_cells = cell_pca.transform(df[cells])
    pca_genes = pd.DataFrame(pca_genes, columns = [f"pca_g-{i}" for i in range(ncompo_genes)])
    pca_cells = pd.DataFrame(pca_cells, columns = [f"pca_c-{i}" for i in range(ncompo_cells)])
    df = pd.concat([df, pca_genes, pca_cells], axis = 1)

    for col in ['cp_time','cp_dose']:
        tmp = pd.get_dummies(df[col],prefix=col)
        df = pd.concat([df,tmp],axis=1)
        df.drop([col],axis=1,inplace=True)
    return df,transformers,gene_pca,cell_pca

tt = train.append(test).append(new_test).reset_index(drop=True)
tt,transformers,gene_pca,cell_pca = Feature(tt)
train = tt[:train.shape[0]]
test = tt[train.shape[0]:train.shape[0]+test.shape[0]].reset_index(drop=True)
new_test = tt[train.shape[0]+test.shape[0]:].reset_index(drop=True)

if 1:
    train_target = train_target.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    train_nonscored = train_nonscored.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    train_drug = train_drug.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    ori_train = ori_train.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    train = train.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)

def Ctl_augment(train,target,train_nonscored):
    aug_trains = []
    aug_targets = []
    for _ in range(3):
        train1 = train.copy()
        target1 = target.copy()
        nonscore_target1 = train_nonscored.copy()
        ctl1 = ctl_train.sample(train1.shape[0],replace=True).reset_index(drop=True)
        ctl2 = ctl_train.sample(train1.shape[0],replace=True).reset_index(drop=True)

        ctl3 = ctl_train.sample(train1.shape[0],replace=True).reset_index(drop=True)
        ctl4 = ctl_train.sample(train1.shape[0],replace=True).reset_index(drop=True)
        mask_index1 = list(np.random.choice(ctl3.index.tolist(),int(ctl3.shape[0]*0.4),replace=False))
        ctl3.loc[mask_index1,genes+cells] = 0.0
        ctl4.loc[mask_index1,genes+cells] = 0.0

        ctl5 = ctl_train.sample(train1.shape[0],replace=True).reset_index(drop=True)
        ctl6 = ctl_train.sample(train1.shape[0],replace=True).reset_index(drop=True)
        mask_index2 = list(np.random.choice(list(set(ctl5.index)-set(mask_index1)),int(ctl5.shape[0]*0.3),replace=False))
        ctl5.loc[mask_index1+mask_index2,genes+cells] = 0.0
        ctl6.loc[mask_index1+mask_index2,genes+cells] = 0.0

        train1[genes+cells] = train1[genes+cells].values + ctl1[genes+cells].values - ctl2[genes+cells].values \
                              + ctl3[genes+cells].values - ctl4[genes+cells].values + ctl5[genes+cells].values - ctl6[genes+cells].values# * np.random.rand(train1.shape[0]).reshape(-1,1)
        aug_train = train1.merge(target1,how='left',on='sig_id')
        aug_train = aug_train.merge(nonscore_target1,how='left',on='sig_id')
        aug_trains.append(aug_train[['cp_time','cp_dose']+genes+cells])
        aug_targets.append(aug_train[targets+nonscored_targets])
    df = pd.concat(aug_trains).reset_index(drop=True)
    target = pd.concat(aug_targets).reset_index(drop=True)
    for col in tqdm(genes+cells):
        df[col] = transformers[col].transform(df[col].values.reshape(-1,1)).reshape(1,-1)[0]
    pca_genes = gene_pca.transform(df[genes])
    pca_cells = cell_pca.transform(df[cells])
    pca_genes = pd.DataFrame(pca_genes, columns = [f"pca_g-{i}" for i in range(ncompo_genes)])
    pca_cells = pd.DataFrame(pca_cells, columns = [f"pca_c-{i}" for i in range(ncompo_cells)])
    df = pd.concat([df, pca_genes, pca_cells], axis = 1)
    for col in ['cp_time','cp_dose']:
        tmp = pd.get_dummies(df[col],prefix=col)
        df = pd.concat([df,tmp],axis=1)
        df.drop([col],axis=1,inplace=True)
    xs = df[train_cols].values
    ys = target[targets].values
    ys1 = target[nonscored_targets].values
    return xs,ys,ys1

import torch
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing / 2
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2619422201258426)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2619422201258426)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size, 402))
    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        y = self.dense3(x)
        y1 = self.dense4(x)
        return y,y1


class GBN(nn.Module):
    def __init__(self,inp,vbs=128,momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp,momentum=momentum)
        self.vbs = vbs
    def forward(self,x):
        chunk = torch.chunk(x,max(1,x.size(0)//self.vbs),0)
        res = [self.bn(y) for y in chunk ]
        return torch.cat(res,0)

class GLU(nn.Module):
    def __init__(self,inp_dim,out_dim,fc=None,vbs=128):
        super().__init__()
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(inp_dim,out_dim*2)
        self.bn = GBN(out_dim*2,vbs=vbs)
        self.od = out_dim
        self.dropout = nn.Dropout(0.2619422201258426)
    def forward(self,x):
        x = self.dropout(self.bn(F.leaky_relu((self.fc(x)))))
        return x[:,:self.od]*torch.sigmoid(x[:,self.od:])


class FeatureTransformer(nn.Module):
    def __init__(self,inp_dim,out_dim,shared,n_ind,vbs=128):
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared:
            self.shared.append(GLU(inp_dim,out_dim,shared[0],vbs=vbs))
            first= False
            for fc in shared[1:]:
                self.shared.append(GLU(out_dim,out_dim,fc,vbs=vbs))
        else:
            self.shared = None
        self.independ = nn.ModuleList()
        if first:
            if shared:
                self.independ.append(GLU(inp_dim,out_dim,vbs=vbs))
            else:
                self.independ.append(GLU(out_dim,out_dim,vbs=vbs))
        for x in range(first, n_ind):
            self.independ.append(GLU(out_dim,out_dim,vbs=vbs))
        self.scale = torch.sqrt(torch.tensor([.5],device=device))
        self.dropout = nn.Dropout(0.2619422201258426)
        self.bn = nn.BatchNorm1d(out_dim)
        self.fc = nn.Linear(inp_dim,out_dim)
    def forward(self,x):
        if self.shared:
            x = self.dropout(self.bn(F.leaky_relu(self.shared[0](x))))
            for glu in self.shared[1:]:
                glu_x = self.dropout(glu(x))
                x = torch.add(x, glu_x)
                x = x*self.scale
        else:
            x = self.dropout(self.bn(F.leaky_relu(self.fc(x))))
        for glu in self.independ:
            glu_x = self.dropout(glu(x))
            x = torch.add(x, glu_x)
            x = x*self.scale
        return x

class AttentionTransformer(nn.Module):
    def __init__(self,inp_dim,out_dim,relax,vbs=128):
        super().__init__()
        self.fc = nn.Linear(inp_dim,out_dim)
        self.bn = GBN(out_dim,vbs=vbs)
        self.r = torch.tensor([relax],device=device)
    def forward(self,a,priors):
        a = self.bn(self.fc(a))
        mask = torch.sigmoid(a*priors)
        priors =priors*(self.r-mask)
        return mask,priors

class DecisionStep(nn.Module):
    def __init__(self,inp_dim,n_d,n_a,shared,n_ind,relax,vbs=128):
        super().__init__()
        self.fea_tran = FeatureTransformer(inp_dim,n_d+n_a,shared,n_ind,vbs)
        self.atten_tran = AttentionTransformer(n_a,inp_dim,relax,vbs)
    def forward(self,x,a,priors):
        mask,priors = self.atten_tran(a,priors)
        loss = ((-1)*mask*torch.log(mask+1e-10)).mean()
        x = self.fea_tran(x*mask)#x*mask
        return x,loss,priors

class TabNet(nn.Module):
    def __init__(self,inp_dim,final_out_dim,n_d=64,n_a=64,n_shared=2,n_ind=2,n_steps=5,relax=1.2,vbs=128):
        super().__init__()
        if n_shared>0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim,2*(n_d+n_a)))
            for x in range(n_shared-1):
                self.shared.append(nn.Linear(n_d+n_a,2*(n_d+n_a)))
        else:
            self.shared=None
        self.first_step = FeatureTransformer(inp_dim,n_d+n_a,self.shared,n_ind)
        self.steps = nn.ModuleList()
        for x in range(n_steps-1):
            self.steps.append(DecisionStep(inp_dim,n_d,n_a,self.shared,n_ind,relax,vbs))
        self.fc = Model(n_d,final_out_dim,1500)
        self.bn = nn.BatchNorm1d(inp_dim)
        self.n_d = n_d
    def forward(self,x):
        x = self.bn(x)
        x_a = self.first_step(x)[:,self.n_d:]
        loss = torch.zeros(1).to(x.device)
        out = torch.zeros(x.size(0),self.n_d).to(x.device)
        priors = torch.ones(x.shape).to(x.device)
        for step in self.steps:
            x_te,l,priors = step(x,x_a,priors)
            out += F.relu(x_te[:,:self.n_d])
            x_a = x_te[:,self.n_d:]
            loss += l
        return self.fc(out)

class Self_Attention(nn.Module):
    def __init__(self,hidden_size,num_attention_heads,attention_probs_dropout_prob):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                    "The hidden size (%d) is not a multiple of the number of attention "
                    "heads (%d)" % (hidden_size, n_head)
                )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,hidden_states):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size)**0.5

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = context_layer
        return outputs

class Attention_dnn(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size0,hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size0))
        self.batch_norm1 = nn.BatchNorm1d(hidden_size0)
        self.dropout1 = nn.Dropout(0.2619422201258426)

        self.att_dense1 = nn.utils.weight_norm(nn.Linear(1, 64))

        self.self1 = Self_Attention(64, num_attention_heads, attention_probs_dropout_prob)

        self.batch_norm2 = nn.BatchNorm1d(hidden_size0)
        self.dropout2 = nn.Dropout(0.2619422201258426)

        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size0, hidden_size))
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2619422201258426)

        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(0.2619422201258426)

        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
        self.dense5 = nn.utils.weight_norm(nn.Linear(hidden_size, 402))

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.dropout1(self.batch_norm1(F.leaky_relu(self.dense1(x))))
        ori_x = x
        x = x.view(x.shape[0],x.shape[1],1)

        x = self.att_dense1(x)
        x = self.self1(x)
        x = torch.max(x,dim=-1)[0]
        x = x + ori_x#torch.cat([x,ori_x],dim=1)

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)

        x = F.leaky_relu(self.dense3(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)

        y = self.dense4(x)
        y1 = self.dense5(x)
        return y,y1

class Dnn(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Dnn, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2619422201258426)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2619422201258426)

        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(0.2619422201258426)

        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
        self.dense5 = nn.utils.weight_norm(nn.Linear(hidden_size, 402))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.leaky_relu(self.dense3(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        y = self.dense4(x)
        y1 = self.dense5(x)
        return y,y1

device = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 23

def predict(features, test, mn,  folds=5, seed=6):
    sub = test[['sig_id']]
    for t in targets:
        sub[t] = 0

    preds = []
    test_X = test[features].values
    test_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(test_X)),batch_size=128,shuffle=False)

    for fold, (trn_ind, val_ind) in enumerate(MultilabelStratifiedKFold(n_splits = folds, shuffle=True, random_state=seed)\
                                              .split(train, train_target[targets])):
        if mn == 'tabnet':
            model = TabNet(len(features),len(targets),n_d=128,n_a=256,n_shared=1,n_ind=1,n_steps=3,relax=2.,vbs=128)
        elif mn == 'attention_dnn':
            model = Attention_dnn(len(features),len(targets),256,1500,2,0.3)
        else:
            model = Dnn(len(features),len(targets),1500)
        model.to(device)
        state_dict = torch.load('./model/model_%s_seed%s_fold%s.ckpt'%(mn,seed,fold), torch.device("cuda" if torch.cuda.is_available() else "cpu") )
        model.load_state_dict(state_dict)
        model.eval()
        t_preds = []
        for data in tqdm(test_data_loader):
            x = data[0].to(device)
            with torch.no_grad():
                outputs,_ = model(x)
            t_preds.extend(list(outputs.sigmoid().cpu().detach().numpy()))
        preds.append(t_preds)
    sub[targets] = np.array(preds).mean(axis=0)
    return sub

train_cols = [col for col in train.columns if col not in ['sig_id','cp_type']]

print('daishu attention_dnn predict begin')

seed = 6
Seed_everything(seed)
sub = predict(train_cols,new_test,mn='attention_dnn',seed=seed)
outputs = []
'''for seed in [66,666]:
    Seed_everything(seed)
    outputs.append(predict(train_cols,new_test,mn='attention_dnn',seed=seed))
'''
print('daishu attention_dnn predict end')

print('daishu tabnet predict begin')

for seed in [8]:
    Seed_everything(seed)
    outputs.append(predict(train_cols,new_test,mn='tabnet',seed=seed))

print('daishu tabnet predict end')

print('daishu dnn predict begin')

for seed in [9]:
    Seed_everything(seed)
    outputs.append(predict(train_cols,new_test,mn='dnn',seed=seed))

print('daishu dnn predict end')

for i,output in enumerate(outputs):
    sub[targets] += output[targets]
sub[targets] /= (1+len(outputs))

sub.loc[new_test['cp_type']=='ctl_vehicle',targets] = 0.0
sub.to_csv('%s/daishu_submission.csv'%args.output_dir,index=False)


# shiji1 predict

ncompo_genes = 80
ncompo_cells = 10

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

Seed_everything(seed=42)

def Metric(labels,preds):
    labels = np.array(labels)
    preds = np.array(preds)
    metric = 0
    for i in range(labels.shape[1]):
        metric += (-np.mean(labels[:,i]*np.log(np.maximum(preds[:,i],1e-15))+(1-labels[:,i])*np.log(np.maximum(1-preds[:,i],1e-15))))
    return metric/labels.shape[1]

files = ['%s/test_features.csv'%args.input_dir,
         '%s/train_targets_scored.csv'%args.input_dir,
         '%s/train_features.csv'%args.input_dir,
         '%s/train_targets_nonscored.csv'%args.input_dir,
         '%s/train_drug.csv'%args.input_dir,
         '%s/sample_submission.csv'%args.input_dir]

test = pd.read_csv(files[0])
train_target = pd.read_csv(files[1])
train = pd.read_csv(files[2])
train_nonscored = pd.read_csv(files[3])
train_drug = pd.read_csv(files[4])

new_test = pd.read_csv(args.new_test_data_path)

genes = [col for col in train.columns if col.startswith("g-")]
cells = [col for col in train.columns if col.startswith("c-")]

features = genes + cells
targets = [col for col in train_target if col!='sig_id']
targets_ns=[col for col in train_nonscored if col!='sig_id']

train_target=pd.merge(train_target,train_nonscored,on='sig_id')

ori_train = train.copy()
ctl_train = train.loc[train['cp_type']=='ctl_vehicle'].append(test.loc[test['cp_type']=='ctl_vehicle']).reset_index(drop=True)
ctl_train2 = train.loc[train['cp_type']=='ctl_vehicle'].reset_index(drop=True)

ori_test = test.copy()
ctl_test = test.loc[test['cp_type']=='ctl_vehicle'].reset_index(drop=True)


def Feature(df):
    transformers={}
    for col in tqdm(genes+cells):
        transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution='normal')
        transformer.fit(df[:train.shape[0]][col].values.reshape(-1,1))
        df[col] = transformer.transform(df[col].values.reshape(-1,1)).reshape(1,-1)[0]
        transformers[col]=transformer
    gene_pca = PCA(n_components = ncompo_genes,
                    random_state = 42).fit(df[genes])
    pca_genes = gene_pca.transform(df[genes])
    cell_pca = PCA(n_components = ncompo_cells,
                    random_state = 42).fit(df[cells])
    pca_cells = cell_pca.transform(df[cells])
    pca_genes = pd.DataFrame(pca_genes, columns = [f"pca_g-{i}" for i in range(ncompo_genes)])
    pca_cells = pd.DataFrame(pca_cells, columns = [f"pca_c-{i}" for i in range(ncompo_cells)])
    df = pd.concat([df, pca_genes, pca_cells], axis = 1)
    for col in ['cp_time','cp_dose']:
        tmp = pd.get_dummies(df[col],prefix=col)
        df = pd.concat([df,tmp],axis=1)
        df.drop([col],axis=1,inplace=True)
    return df,transformers,gene_pca,cell_pca

tt = train.append(test).append(new_test).reset_index(drop=True)
tt,transformers,gene_pca,cell_pca = Feature(tt)
train = tt[:train.shape[0]]
test = tt[train.shape[0]:train.shape[0]+test.shape[0]].reset_index(drop=True)
new_test = tt[train.shape[0]+test.shape[0]:].reset_index(drop=True)

if 1:
    train_target = train_target.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    train_nonscored = train_nonscored.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    train_drug = train_drug.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    ori_train = ori_train.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    train = train.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)

class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2619422201258426)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(0.2619422201258426)
        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2619422201258426)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = F.leaky_relu(self.dense4(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def predict(features, test,  folds=5, seed=1,lr=1/90.0/3.5*3,weight_decay=1e-5/3):
    sub = test[['sig_id']]
    for t in targets:
        sub[t] = 0.0
    preds = []
    test_X = test[features].values
    test_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(test_X)),batch_size=1024,shuffle=False)

    for fold, (trn_ind, val_ind) in enumerate(MultilabelStratifiedKFold(n_splits = folds, shuffle=True, random_state=seed)\
                                              .split(train, train_target[targets])):
        model = Model(len(features),len(targets),1500)
        model.to(device)

        state_dict = torch.load('./model/model_dnn_final_fold%s'%fold+'_'+str(seed)+'.ckpt', torch.device("cuda" if torch.cuda.is_available() else "cpu") )
        model.load_state_dict(state_dict)
        model.eval()

        t_preds = []
        for data in tqdm(test_data_loader):
            x = data[0].to(device)
            with torch.no_grad():
                outputs = model(x)
            t_preds.extend(list(outputs.sigmoid().cpu().detach().numpy()))
        preds.append(t_preds)
    sub[targets] = np.array(preds).mean(axis=0)
    return sub

train_cols = [col for col in train.columns if col not in ['sig_id','cp_type']]

print('shiji1 predict begin')
Seed_everything(0)
sub = predict(train_cols,new_test,seed=0,lr=1/90.0/3.5*7,weight_decay=1e-5/9.5)

outputs = []
for seed in [1,2,3]:
    Seed_everything(seed)
    outputs.append(predict(train_cols,new_test,seed=seed,lr=1/90.0/3.5*7,weight_decay=1e-5/9.5))
print('shiji1 predict end')

for output in outputs:
    sub[targets] += output[targets]
sub[targets] /= (1+len(outputs))

sub.loc[new_test['cp_type']=='ctl_vehicle',targets] = 0.0
sub.to_csv('%s/shiji_submission1.csv'%args.output_dir,index=False)

# shiji2 predict

ncompo_genes = 600
ncompo_cells = 50

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

Seed_everything(seed=42)

def Metric(labels,preds):
    labels = np.array(labels)
    preds = np.array(preds)
    metric = 0
    for i in range(labels.shape[1]):
        metric += (-np.mean(labels[:,i]*np.log(np.maximum(preds[:,i],1e-15))+(1-labels[:,i])*np.log(np.maximum(1-preds[:,i],1e-15))))
    return metric/labels.shape[1]

files = ['%s/test_features.csv'%args.input_dir,
         '%s/train_targets_scored.csv'%args.input_dir,
         '%s/train_features.csv'%args.input_dir,
         '%s/train_targets_nonscored.csv'%args.input_dir,
         '%s/train_drug.csv'%args.input_dir,
         '%s/sample_submission.csv'%args.input_dir]

test = pd.read_csv(files[0])
train_target = pd.read_csv(files[1])
train = pd.read_csv(files[2])
train_nonscored = pd.read_csv(files[3])
train_drug = pd.read_csv(files[4])

new_test = pd.read_csv(args.new_test_data_path)

genes = [col for col in train.columns if col.startswith("g-")]
cells = [col for col in train.columns if col.startswith("c-")]

features = genes + cells
targets = [col for col in train_target if col!='sig_id']

ori_train = train.copy()
ctl_train = train.loc[train['cp_type']=='ctl_vehicle'].append(test.loc[test['cp_type']=='ctl_vehicle']).reset_index(drop=True)
ctl_train2 = train.loc[train['cp_type']=='ctl_vehicle'].reset_index(drop=True)

ori_test = test.copy()
ctl_test = test.loc[test['cp_type']=='ctl_vehicle'].reset_index(drop=True)

def Feature(df):
    transformers={}
    for col in tqdm(genes+cells):
        transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution='normal')
        transformer.fit(df[:train.shape[0]][col].values.reshape(-1,1))
        df[col] = transformer.transform(df[col].values.reshape(-1,1)).reshape(1,-1)[0]
        transformers[col]=transformer
    gene_pca = PCA(n_components = ncompo_genes,
                    random_state = 42).fit(df[:train.shape[0]+test.shape[0]][genes])
    pca_genes = gene_pca.transform(df[genes])
    cell_pca = PCA(n_components = ncompo_cells,
                    random_state = 42).fit(df[:train.shape[0]+test.shape[0]][cells])
    pca_cells = cell_pca.transform(df[cells])
    pca_genes = pd.DataFrame(pca_genes, columns = [f"pca_g-{i}" for i in range(ncompo_genes)])
    pca_cells = pd.DataFrame(pca_cells, columns = [f"pca_c-{i}" for i in range(ncompo_cells)])
    df = pd.concat([df, pca_genes, pca_cells], axis = 1)

    nor_var_col = [col for col in df.columns if col in ['sig_id','cp_type','cp_time','cp_dose'] or '_gt_' in col or '_lt_' in col]

    var_thresh = VarianceThreshold(0.8)
    var_cols = [col for col in df.columns if col not in ['sig_id','cp_type','cp_time','cp_dose'] and '_gt_' not in col and '_lt_' not in col]
    var_transformer = var_thresh.fit(df[:train.shape[0]+test.shape[0]][var_cols])
    var_data = var_transformer.transform(df[var_cols])
    df = pd.concat([df[nor_var_col],pd.DataFrame(var_data)],axis=1)
    for col in ['cp_time','cp_dose']:
        tmp = pd.get_dummies(df[col],prefix=col)
        df = pd.concat([df,tmp],axis=1)
        df.drop([col],axis=1,inplace=True)
    return df,transformers,gene_pca,cell_pca,var_thresh

tt = train.append(test).append(new_test).reset_index(drop=True)

tt,transformers,gene_pca,cell_pca,var_thresh = Feature(tt)
train = tt[:train.shape[0]]
test = tt[train.shape[0]:train.shape[0]+test.shape[0]].reset_index(drop=True)
new_test = tt[train.shape[0]+test.shape[0]:].reset_index(drop=True)
if 1:
    train_target = train_target.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    train_nonscored = train_nonscored.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    train_drug = train_drug.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    ori_train = ori_train.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    train = train.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)



class resnetModel(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size,ispretrain=False):
        super(resnetModel, self).__init__()
        self.ispretrain=ispretrain
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))


        self.batch_norm2 = nn.BatchNorm1d(num_features+hidden_size)
        self.dropout2 = nn.Dropout(0.2619422201258426)
        self.dense2 = nn.utils.weight_norm(nn.Linear(num_features+hidden_size, hidden_size))
        self.batch_norm20 = nn.BatchNorm1d(hidden_size)
        self.dropout20 = nn.Dropout(0.2619422201258426)
        self.dense20 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))


        self.batch_norm3 = nn.BatchNorm1d(2*hidden_size)
        self.dropout3 = nn.Dropout(0.2619422201258426)
        self.dense3 = nn.utils.weight_norm(nn.Linear(2*hidden_size, hidden_size))
        self.batch_norm30 = nn.BatchNorm1d(hidden_size)
        self.dropout30 = nn.Dropout(0.2619422201258426)
        self.dense30 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        #self.batch_norm6 = nn.BatchNorm1d(2*hidden_size)
        #self.dropout6 = nn.Dropout(0.2619422201258426)
        #self.dense6 = nn.utils.weight_norm(nn.Linear(2*hidden_size, hidden_size))
        #self.batch_norm60 = nn.BatchNorm1d(hidden_size)
        #self.dropout60 = nn.Dropout(0.2619422201258426)
        #self.dense60 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))


        self.batch_norm4 = nn.BatchNorm1d(2*hidden_size)
        self.dropout4 = nn.Dropout(0.2619422201258426)
        if self.ispretrain:
          self.dense4 = nn.utils.weight_norm(nn.Linear(2*hidden_size, num_targets))
        else:
          self.dense5 = nn.utils.weight_norm(nn.Linear(2*hidden_size, num_targets))

    def forward(self, x):
        x1 = self.batch_norm1(x)
        x1 = F.leaky_relu(self.dense1(x1))
        x = torch.cat([x,x1],1)

        x2 = self.batch_norm2(x)
        x2 = self.dropout2(x2)
        x2 = F.leaky_relu(self.dense2(x2))
        x2 = self.batch_norm20(x2)
        x2 = self.dropout20(x2)
        x2 = F.leaky_relu(self.dense20(x2))
        x = torch.cat([x1,x2],1)

        x3 = self.batch_norm3(x)
        x3 = self.dropout3(x3)
        x3 = F.leaky_relu(self.dense3(x3))
        x3 = self.batch_norm30(x3)
        x3 = self.dropout30(x3)
        x3 = F.leaky_relu(self.dense30(x3))
        x3 = torch.cat([x2,x3],1)

        #x4 = self.batch_norm3(x)
        #x4 = self.dropout3(x4)
        #x4 = F.leaky_relu(self.dense3(x4))
        #x4 = self.batch_norm30(x4)
        #x4 = self.dropout30(x4)
        #x4 = F.leaky_relu(self.dense30(x4))
        #x4 = torch.cat([x3,x4],1)

        x4 = self.batch_norm4(x3)
        x4 = self.dropout4(x4)
        if self.ispretrain:
          x4 = self.dense4(x4)
        else:
          x4 = self.dense5(x4)
        return x4

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def predict(features, test,  folds=5, seed=1,lr=1/90.0/3.5*3,weight_decay=1e-5/3):
    sub = test[['sig_id']]
    for t in targets:
        sub[t] = 0.0
    preds = []
    test_X = test[features].values
    test_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(test_X)),batch_size=1024,shuffle=False)
    for fold, (trn_ind, val_ind) in enumerate(MultilabelStratifiedKFold(n_splits = folds, shuffle=True, random_state=seed)\
                                              .split(train, train_target[targets])):
        model = resnetModel(len(features),len(targets),1500)
        model.to(device)
        state_dict = torch.load('./model/model_resnet2_fold%s'%fold+'_'+str(seed)+'.ckpt', torch.device("cuda" if torch.cuda.is_available() else "cpu") )
        model.load_state_dict(state_dict)
        model.eval()

        t_preds = []
        for data in tqdm(test_data_loader):
            x = data[0].to(device)
            with torch.no_grad():
                outputs = model(x)
            t_preds.extend(list(outputs.sigmoid().cpu().detach().numpy()))
        preds.append(t_preds)

    sub[targets] = np.array(preds).mean(axis=0)
    return sub



train_cols = [col for col in train.columns if col not in ['sig_id','cp_type']]

print('shiji2 predict begin')

Seed_everything(0)
sub = predict(train_cols,new_test,seed=0,lr=1/90.0/2,weight_decay=1e-5/2.7)

outputs = []
for seed in [1,2,3]:
    Seed_everything(seed)
    outputs.append(predict(train_cols,new_test,seed=seed,lr=1/90.0/2,weight_decay=1e-5/2.7))

print('shiji2 predict end')

for output in outputs:
    sub[targets] += output[targets]
sub[targets] /= (1+len(outputs))
sub.loc[new_test['cp_type']=='ctl_vehicle',targets] = 0.0
sub.to_csv('%s/shiji_submission2.csv'%args.output_dir,index=False)

#shiji3 predict


ncompo_genes = 600
ncompo_cells = 50

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

Seed_everything(seed=42)

def Metric(labels,preds):
    labels = np.array(labels)
    preds = np.array(preds)
    metric = 0
    for i in range(labels.shape[1]):
        metric += (-np.mean(labels[:,i]*np.log(np.maximum(preds[:,i],1e-15))+(1-labels[:,i])*np.log(np.maximum(1-preds[:,i],1e-15))))
    return metric/labels.shape[1]

files = ['%s/test_features.csv'%args.input_dir,
         '%s/train_targets_scored.csv'%args.input_dir,
         '%s/train_features.csv'%args.input_dir,
         '%s/train_targets_nonscored.csv'%args.input_dir,
         '%s/train_drug.csv'%args.input_dir,
         '%s/sample_submission.csv'%args.input_dir]

test = pd.read_csv(files[0])
train_target = pd.read_csv(files[1])
train = pd.read_csv(files[2])
train_nonscored = pd.read_csv(files[3])
train_drug = pd.read_csv(files[4])

new_test = pd.read_csv(args.new_test_data_path)

genes = [col for col in train.columns if col.startswith("g-")]
cells = [col for col in train.columns if col.startswith("c-")]

features = genes + cells
targets = [col for col in train_target if col!='sig_id']
targets_ns=[col for col in train_nonscored if col!='sig_id']

train_target=pd.merge(train_target,train_nonscored,on='sig_id')

ori_train = train.copy()
ctl_train = train.loc[train['cp_type']=='ctl_vehicle'].append(test.loc[test['cp_type']=='ctl_vehicle']).reset_index(drop=True)
ctl_train2 = train.loc[train['cp_type']=='ctl_vehicle'].reset_index(drop=True)

ori_test = test.copy()
ctl_test = test.loc[test['cp_type']=='ctl_vehicle'].reset_index(drop=True)

def Feature(df):
    transformers={}
    for col in tqdm(genes+cells):
        transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution='normal')
        transformer.fit(df[:train.shape[0]][col].values.reshape(-1,1))
        df[col] = transformer.transform(df[col].values.reshape(-1,1)).reshape(1,-1)[0]
        transformers[col]=transformer
    gene_pca = PCA(n_components = ncompo_genes,
                    random_state = 42).fit(df[:train.shape[0]+test.shape[0]][genes])
    pca_genes = gene_pca.transform(df[genes])
    cell_pca = PCA(n_components = ncompo_cells,
                    random_state = 42).fit(df[:train.shape[0]+test.shape[0]][cells])
    pca_cells = cell_pca.transform(df[cells])
    pca_genes = pd.DataFrame(pca_genes, columns = [f"pca_g-{i}" for i in range(ncompo_genes)])
    pca_cells = pd.DataFrame(pca_cells, columns = [f"pca_c-{i}" for i in range(ncompo_cells)])
    df = pd.concat([df, pca_genes, pca_cells], axis = 1)

    nor_var_col = [col for col in df.columns if col in ['sig_id','cp_type','cp_time','cp_dose'] or '_gt_' in col or '_lt_' in col]

    var_thresh = VarianceThreshold(0.8)
    var_cols = [col for col in df.columns if col not in ['sig_id','cp_type','cp_time','cp_dose'] and '_gt_' not in col and '_lt_' not in col]
    var_transformer = var_thresh.fit(df[:train.shape[0]+test.shape[0]][var_cols])
    var_data = var_transformer.transform(df[var_cols])
    df = pd.concat([df[nor_var_col],pd.DataFrame(var_data)],axis=1)
    for col in ['cp_time','cp_dose']:
        tmp = pd.get_dummies(df[col],prefix=col)
        df = pd.concat([df,tmp],axis=1)
        df.drop([col],axis=1,inplace=True)
    return df,transformers,gene_pca,cell_pca,var_thresh

tt = train.append(test).append(new_test).reset_index(drop=True)
tt,transformers,gene_pca,cell_pca,var_thresh = Feature(tt)
train = tt[:train.shape[0]]
test = tt[train.shape[0]:train.shape[0]+test.shape[0]].reset_index(drop=True)
new_test = tt[train.shape[0]+test.shape[0]:].reset_index(drop=True)
if 1:
    train_target = train_target.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    train_nonscored = train_nonscored.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    train_drug = train_drug.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    ori_train = ori_train.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
    train = train.loc[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)

class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.2619422201258426)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(0.2619422201258426)
        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2619422201258426)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = F.leaky_relu(self.dense4(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def predict(features, test,  folds=5, seed=1,lr=1/90.0/3.5*3,weight_decay=1e-5/3):
    sub = test[['sig_id']]
    for t in targets:
        sub[t] = 0.0
    preds = []
    test_X = test[features].values
    test_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(test_X)),batch_size=1024,shuffle=False)
    for fold, (trn_ind, val_ind) in enumerate(MultilabelStratifiedKFold(n_splits = folds, shuffle=True, random_state=seed)\
                                              .split(train, train_target[targets])):

        model = Model(len(features),len(targets),1500)
        model.to(device)

        state_dict = torch.load('./model/model_dnn2_final_fold%s'%fold+'_'+str(seed)+'.ckpt', torch.device("cuda" if torch.cuda.is_available() else "cpu") )
        model.load_state_dict(state_dict)
        model.eval()

        t_preds = []
        for data in tqdm(test_data_loader):
            x = data[0].to(device)
            with torch.no_grad():
                outputs = model(x)
            t_preds.extend(list(outputs.sigmoid().cpu().detach().numpy()))
        preds.append(t_preds)

    sub[targets] = np.array(preds).mean(axis=0)
    return sub

train_cols = [col for col in train.columns if col not in ['sig_id','cp_type']]

print('shiji3 predict begin')

Seed_everything(0)
sub = predict(train_cols,new_test,seed=0,lr=1/90.0/3.5*8,weight_decay=1e-6)

outputs = []
for seed in [1,2,3]:
    Seed_everything(seed)
    outputs.append(predict(train_cols,new_test,seed=seed,lr=1/90.0/3.5*8,weight_decay=1e-6))

print('shiji3 predict end')

for output in outputs:
    sub[targets] += output[targets]
sub[targets] /= (1+len(outputs))

sub.loc[test['cp_type']=='ctl_vehicle',targets] = 0.0
sub.to_csv('%s/shiji_submission3.csv'%args.output_dir,index=False)

# blend 

daishu_sub = pd.read_csv('%s/daishu_submission.csv'%args.output_dir)
shiji_sub1 = pd.read_csv('%s/shiji_submission1.csv'%args.output_dir)
shiji_sub2 = pd.read_csv('%s/shiji_submission2.csv'%args.output_dir)
shiji_sub3 = pd.read_csv('%s/shiji_submission3.csv'%args.output_dir)
shiji_sub = shiji_sub1.copy()

targets = [col for col in daishu_sub.columns if col!='sig_id']

shiji_sub[targets] = (shiji_sub1[targets].values+shiji_sub2[targets].values+shiji_sub3[targets].values) / 3
daishu_sub[targets] = daishu_sub[targets].values * 0.7 + shiji_sub[targets].values * 0.3

daishu_sub.to_csv('%s/submission.csv'%args.output_dir,index=False)