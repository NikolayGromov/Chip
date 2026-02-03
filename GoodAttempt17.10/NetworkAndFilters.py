import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import matplotlib.ticker as ticker
from scipy import stats
import scipy.io
#%matplotlib inline


MAXLEN = 20000

Hs = 7

def removeArts(signal):
    s = signal.copy()
    mean = np.mean(s)
    normS = normalization(s)
    for i in range(100, len(signal)):
        if (normS[i])>Hs:#abs(normS[i])>5:
            #print(i)
            s[i] = s[i-100]
  
    return s

def normalization(s):
    return np.array((s-s.mean())/s.std())

def gausFilter(signal, sigma = 35):
    min1 = np.min(signal)
    gf = gaussian_filter(signal, sigma)
    min2 = np.min(gf)
    return gf*min1/min2

ca3 =  np.loadtxt("VisibleAnswer.txt")#("Amp100_CA3")



normS = normalization(ca3)

if (normS[normS > 0]).max() < np.abs(normS[normS < 0]).max():
    ca3 *= -1
    normS *= -1
    

indices = np.where((normS>Hs) == True)[0]  # все индексы максимумов
plt.figure(figsize=(16, 9))
plt.plot(normS, label="input")
plt.plot([0, len(normS)], [Hs, Hs], label="Hs")
plt.legend()
plt.savefig("Hs.png")
last_max_index = indices[-1]#1000
print(indices)
print(len(indices))
print("argmax", last_max_index)

plt.figure(figsize=(16, 9))
plt.plot(ca3[last_max_index-100:last_max_index+100], label="input")
plt.legend()
plt.savefig("RawInputSignalPart.png")

plt.figure(figsize=(16, 9))
plt.plot(removeArts(ca3[last_max_index-1000:last_max_index+2000]), label="input")
plt.legend()
plt.savefig("RemovedArtefact.png")

plt.figure(figsize=(16, 9))
plt.plot(normalization(removeArts(ca3[last_max_index-1000:last_max_index+2000])), label="input")
plt.legend()
plt.savefig("RemovedArtefactAndNorm.png")

plt.figure(figsize=(16, 9))
plt.plot(gausFilter(normalization(removeArts(ca3[last_max_index-1000:last_max_index+2000])), sigma=5), label="input")
plt.legend()
plt.savefig("RemovedArtefactAndNormAndFilter.png")

signal_test_ca3 = gausFilter(normalization(removeArts(ca3[last_max_index-1000:last_max_index+2000]))) / 10
print(signal_test_ca3.shape)

plt.figure(figsize=(16, 9))
plt.plot(signal_test_ca3, label="input")
plt.legend()
plt.savefig("NetworkInput.png")


h_params_loss = [1, 1000, 200, 0.2, 2, "interval"]


def create_inout_sequences(input_data, target_data, tw, hop, batch_size=64):
    inout_seq = []
    L = len(input_data)
    for i in range(0, L - max(batch_size,train_window), hop):
        train_seq = input_data[i:i+tw]
        train_label = target_data[i:i+tw]
        inout_seq.append((torch.FloatTensor(train_seq), torch.FloatTensor(train_label)))
    return inout_seq

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, 
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(2*hidden_layer_size, output_size)


    def forward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # layer 2
        x = self.dropout(lstm_out)
        predictions = self.linear_2(x)
        return predictions
    
def GetWeitsWin(concat_len):
    y_data = stats.norm.pdf(np.arange(0, concat_len, 1), 0, concat_len/5)
    y_data2 = stats.norm.pdf(np.arange(-concat_len, 0, 1), 0, concat_len/5)

    weights = np.concatenate((np.expand_dims(y_data, axis=1), np.expand_dims(y_data2, axis=1)), axis=1)
    weights = weights / np.expand_dims(weights.sum(axis=1), axis=1)
    return weights

def SmoothSignal(signal, win, hop):
    #now support only win // hop = 2
    sm_signal = []
    weights = GetWeitsWin(hop)
    sm_signal.append(signal[:hop])
    for w in range(0, len(signal) - win, win):
        left_signal = signal[w+(win-hop):w+win] * weights[:, 0]
        right_signal = signal[w+win:w+win+hop] * weights[:, 1]
           
        sm_signal.append(left_signal + right_signal)
    #sm_signal.append(signal[w+win+hop: w+2*win])
        
    return np.array(sm_signal).flatten()

def GetPredSignal(model_lstm, win, hop, test_dataloader):
    preds = []
    model_lstm.eval()
    for idx, (x, y) in enumerate(tqdm(test_dataloader)):


            x = x[..., None]#.to('cuda')

            out = model_lstm(x)
            
            
            preds.append(out.detach().cpu().numpy().flatten())
    preds = np.array(preds).flatten()
    sm_preds = SmoothSignal(preds, win, hop)
    return sm_preds

def getAnswer(signal, maxlen=MAXLEN):
    left = right = np.argmin(signal)
    mean = np.mean(signal)
    while signal[left]<mean and left>0:
        left-=1
    while signal[right]<mean and right<maxlen:
        right+=1
    return [left, right]

batch_size, train_window, n_hidden, dropout, n_layers, mode = h_params_loss
hop_len = train_window // 2

    
test_inout_seq = create_inout_sequences(signal_test_ca3, np.zeros_like(signal_test_ca3), train_window, hop_len)#signal_test_ca1, train_window, hop_len)

test_dataloader = DataLoader(test_inout_seq, batch_size=batch_size, shuffle=False, drop_last=True)

model_lstm = LSTMModel(input_size=1, hidden_layer_size=n_hidden, num_layers=n_layers, output_size=1, dropout=dropout)
model_lstm = model_lstm#.to('cuda')
model_lstm.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device('cpu')))

sm_preds = GetPredSignal(model_lstm, train_window, hop_len, test_dataloader)
print(len(test_dataloader))
print(sm_preds.shape)
print("It computed")

#bord = getAnswer(sm_preds)
#tr_bord = getAnswer(signal_test_ca1)

plt.figure(figsize=(16, 9))

#plt.plot(signal_test_ca1, label="true")
plt.plot(sm_preds, label="predicted")
#plt.plot([bord[0]]*2, [-5, 2], color="r", linewidth = 3, label="Pred bord")
#plt.plot([bord[1]]*2, [-5, 2], color="r", linewidth = 3)
#plt.plot([tr_bord[0]]*2, [-5, 2], color="g", linewidth = 3, label="True bord")
#plt.plot([tr_bord[1]]*2, [-5, 2], color="g", linewidth = 3)


plt.legend()
plt.savefig("Predicted.png")
#plt.show()

ca1 = np.concatenate([np.zeros(2016), sm_preds[500:1500][::16], np.zeros(2017)]) / 10
ca1 = np.clip((-ca1), a_min=0, a_max=None)
def Convert(x, old_low=0, old_high=3.3, new_low=0, new_high=4095):
    interval_0_1 = (x - old_low) / (old_high - old_low)
    scaled = new_low + (new_high - new_low) * interval_0_1
    return scaled.astype(int)
    
ca1_converted = (Convert(ca1) * 2.8).astype(int) # change 1 to any multiplier
print("len", (ca1_converted > 0).sum())
print("uniq", np.unique(ca1_converted))

print("Max", np.max(ca1_converted))

np.savetxt("Digits.txt", ca1_converted, fmt='%i', newline=", ")

import os

text = ""
with open("Digits.txt", 'r') as filehandle:
    text = filehandle.read()
    text = "{" + text[:-2] + "};"
with open("Digits.txt", 'w') as filehandle:
    filehandle.write(text)
