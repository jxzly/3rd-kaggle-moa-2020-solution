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
sub = pd.read_csv(files[5])

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

tt = train.append(test).reset_index(drop=True)
tt,transformers,gene_pca,cell_pca = Feature(tt)
train = tt[:train.shape[0]]
test = tt[train.shape[0]:].reset_index(drop=True)

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
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
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
def train_and_predict(features, sub, aug, mn,  folds=5, seed=6):
    oof = train[['sig_id']]
    for t in targets:
        oof[t] = 0.0
    preds = []
    test_X = test[features].values
    test_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(test_X)),batch_size=128,shuffle=False)
    eval_train_loss = 0
    for fold, (trn_ind, val_ind) in enumerate(MultilabelStratifiedKFold(n_splits = folds, shuffle=True, random_state=seed)\
                                              .split(train, train_target[targets])):
        train_X = train.loc[trn_ind,features].values
        eval_train_X = train_X.copy()
        train_Y = train_target.loc[trn_ind,targets].values
        train_Y1 = train_nonscored.loc[trn_ind,nonscored_targets].values
        eval_train_Y = train_Y.copy()
        eval_train_Y1 = train_Y1.copy()
        if aug:
            aug_X,aug_Y,aug_Y1 = Ctl_augment(ori_train.loc[trn_ind],train_target.loc[trn_ind],train_nonscored.loc[trn_ind])
            train_X = np.concatenate([train_X,aug_X],axis=0)
            train_Y = np.concatenate([train_Y,aug_Y],axis=0)
            train_Y1 = np.concatenate([train_Y1,aug_Y1],axis=0)
            del aug_X,aug_Y,aug_Y1
        valid_X = train.loc[val_ind,features].values
        valid_Y = train_target.loc[val_ind,targets].values

        eval_train_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(eval_train_X),torch.Tensor(eval_train_Y)),batch_size=128,shuffle=False, drop_last=False)
        train_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(train_X),torch.Tensor(train_Y),torch.Tensor(train_Y1)),batch_size=128,shuffle=True, drop_last=True)
        valid_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(valid_X),torch.Tensor(valid_Y)),batch_size=1024,shuffle=False)

        if mn == 'tabnet':
            model = TabNet(len(features),len(targets),n_d=128,n_a=256,n_shared=1,n_ind=1,n_steps=3,relax=2.,vbs=128)
            optimizer = torch.optim.Adam(model.parameters(),betas=(0.9, 0.99), lr=1e-3, weight_decay=1.00e-5/5,eps=1e-5)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1/90.0/3, epochs=EPOCHS, steps_per_epoch=len(train_data_loader))
        elif mn == 'attention_dnn':
            model = Attention_dnn(len(features),len(targets),256,1500,2,0.3)
            optimizer = torch.optim.Adam(model.parameters(),betas=(0.9, 0.99), lr=1e-3, weight_decay=1.00e-5/4.75,eps=1e-5)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1/90.0/4, epochs=EPOCHS, steps_per_epoch=len(train_data_loader))
        else:
            model = Dnn(len(features),len(targets),1500)
            optimizer = torch.optim.Adam(model.parameters(),betas=(0.9, 0.99), lr=1e-3, weight_decay=1.00e-5/6,eps=1e-5)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1/90.0/3.5*3, epochs=EPOCHS, steps_per_epoch=len(train_data_loader))
        model.to(device)

        loss_fn = nn.BCEWithLogitsLoss()
        loss_tr = SmoothBCEwLogits(smoothing =0.001)
        best_valid_metric = 1e9
        not_improve_epochs = 0

        for epoch in range(EPOCHS):
            if epoch > 0 and aug:
                aug_X,aug_Y,aug_Y1 = Ctl_augment(ori_train.loc[trn_ind],train_target.loc[trn_ind],train_nonscored.loc[trn_ind])
                train_X = np.concatenate([eval_train_X,aug_X],axis=0)
                train_Y = np.concatenate([eval_train_Y,aug_Y],axis=0)
                train_Y1 = np.concatenate([eval_train_Y1,aug_Y1],axis=0)
                train_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(train_X),torch.Tensor(train_Y),torch.Tensor(train_Y1)),batch_size=128,shuffle=True, drop_last=True)
                del aug_X,aug_Y,aug_Y1
            if epoch > 19:
                train_data_loader = DataLoader(dataset=TensorDataset(torch.Tensor(train_X),torch.Tensor(train_Y),torch.Tensor(train_Y1)),batch_size=512,shuffle=True, drop_last=True)
            # train
            train_loss = 0.0
            train_num = 0
            for data in (train_data_loader):
                optimizer.zero_grad()
                x,y,y1 = [d.to(device) for d in data]
                outputs,outputs1 = model(x)
                loss1 = loss_tr(outputs, y)
                loss2 = loss_fn(outputs1, y1)
                loss = loss1*0.5 + loss2*0.5
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_num += x.shape[0]
                train_loss += (loss1.item()*x.shape[0])

            train_loss /= train_num
            # eval
            model.eval()
            valid_loss = 0.0
            valid_num = 0
            valid_preds = []
            for data in (valid_data_loader):
                x,y = [d.to(device) for d in data]
                outputs,_ = model(x)
                valid_preds.extend(list(outputs.sigmoid().cpu().detach().numpy()))
                loss = loss_fn(outputs, y)
                valid_num += x.shape[0]
                valid_loss += (loss.item()*x.shape[0])
            valid_loss /= valid_num
            valid_mean = np.mean(valid_preds)
            t_preds = []
            for data in (test_data_loader):
                x = data[0].to(device)
                with torch.no_grad():
                    outputs,_ = model(x)
                t_preds.extend(list(outputs.sigmoid().cpu().detach().numpy()))
            pred_mean = np.mean(t_preds)
            if valid_loss < best_valid_metric:
                torch.save(model.state_dict(),'./model/model_%s_seed%s_fold%s.ckpt'%(mn,seed,fold))
                not_improve_epochs = 0
                best_valid_metric = valid_loss
                print('[epoch %s] lr: %.6f, train_loss: %.6f, valid_metric: %.6f, valid_mean:%.6f, pred_mean:%.6f'%(epoch,optimizer.param_groups[0]['lr'],train_loss,valid_loss,valid_mean,pred_mean))
            else:
                not_improve_epochs += 1
                print('[epoch %s] lr: %.6f, train_loss: %.6f, valid_metric: %.6f, valid_mean:%.6f, pred_mean:%.6f, NIE +1 ---> %s'%(epoch,optimizer.param_groups[0]['lr'],train_loss,valid_loss,valid_mean,pred_mean,not_improve_epochs))
                if not_improve_epochs >= 50:
                    break
            model.train()
        state_dict = torch.load('./model/model_%s_seed%s_fold%s.ckpt'%(mn,seed,fold), torch.device("cuda" if torch.cuda.is_available() else "cpu") )
        model.load_state_dict(state_dict)
        model.eval()
        train_preds = []
        for data in (eval_train_data_loader):
            x = data[0].to(device)
            with torch.no_grad():
                outputs,_ = model(x)
            train_preds.extend(list(outputs.sigmoid().cpu().detach().numpy()))
        train_loss = Metric(eval_train_Y,train_preds)
        eval_train_loss += train_loss
        print('eval_train_loss:',train_loss)
        valid_preds = []
        for data in tqdm(valid_data_loader):
            x,y = [d.to(device) for d in data]
            with torch.no_grad():
                outputs,_ = model(x)
            valid_preds.extend(list(outputs.cpu().detach().numpy()))
        oof.loc[val_ind,targets] = 1 / (1+np.exp(-np.array(valid_preds)))
        t_preds = []
        for data in tqdm(test_data_loader):
            x = data[0].to(device)
            with torch.no_grad():
                outputs,_ = model(x)
            t_preds.extend(list(outputs.sigmoid().cpu().detach().numpy()))
        print(np.mean(1 / (1+np.exp(-np.array(valid_preds)))),np.mean(t_preds))
        preds.append(t_preds)
        del train_X,train_Y,valid_X,valid_Y,train_data_loader,valid_data_loader

    sub[targets] = np.array(preds).mean(axis=0)
    print('eval_train_loss:',eval_train_loss/folds,oof[targets].mean().mean(),sub[targets].mean().mean())
    print('valid_metric:%.6f'%Metric(train_target[targets].values,oof[targets].values))
    return oof,sub

train_cols = [col for col in train.columns if col not in ['sig_id','cp_type']]

seed = 6
Seed_everything(seed)
oof,sub = train_and_predict(train_cols,sub.copy(),aug=True,mn='attention_dnn',seed=seed)
outputs = []
for seed in [66,666]:
    Seed_everything(seed)
    outputs.append(train_and_predict(train_cols,sub.copy(),aug=True,mn='attention_dnn',seed=seed))

for seed in [8,88,888]:
    Seed_everything(seed)
    outputs.append(train_and_predict(train_cols,sub.copy(),aug=True,mn='tabnet',seed=seed))
for seed in [9,99,999]:
    Seed_everything(seed)
    outputs.append(train_and_predict(train_cols,sub.copy(),aug=True,mn='dnn',seed=seed))

for i,output in enumerate(outputs):
    oof[targets] += output[0][targets]
    sub[targets] += output[1][targets]
oof[targets] /= (1+len(outputs))
sub[targets] /= (1+len(outputs))

valid_metric = Metric(train_target[targets].values,oof[targets].values)

print('oof mean:%.6f,sub mean:%.6f,valid metric:%.6f'%(oof[targets].mean().mean(),sub[targets].mean().mean(),valid_metric))
sub.loc[test['cp_type']=='ctl_vehicle',targets] = 0.0
sub.to_csv('./daishu_submission.csv',index=False)
