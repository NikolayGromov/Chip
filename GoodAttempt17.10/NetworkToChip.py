import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import matplotlib.ticker as ticker
from scipy import stats
#%matplotlib inline

n_signals = 9

ca1 = []
ca3 = []
for i in range(n_signals):
    ca1.append(np.load("data_intersec/ca1_3_" + str(i) + ".npy"))
    ca3.append(np.load("data_intersec/ca3_3_" + str(i) + ".npy"))   
    
ca1 = np.array(ca1)
ca3 = np.array(ca3)

def gausFilter(signal, sigma = 35):
    min1 = np.min(signal)
    gf = gaussian_filter(signal, sigma)
    min2 = np.min(gf)
    return gf*min1/min2

for i in range(n_signals):
    ca1[i] = gausFilter(ca1[i])
    ca3[i] = gausFilter(ca3[i])    
    
train_ca1 = ca1[:-1].copy()
test_ca1 = ca1[-1:].copy()

train_ca3 = ca3[:-1].copy()
test_ca3 = ca3[-1:].copy()

for i in range(len(train_ca1)):
    train_ca1[i] = (train_ca1[i] - train_ca1[i].mean()) / train_ca1[i].std()
    train_ca3[i] = (train_ca3[i] - train_ca3[i].mean()) / train_ca3[i].std() 
    
# for i in range(len(test_ca1)):
#     test_ca1[i] = (test_ca1[i] - test_ca1[i].mean()) / test_ca1[i].std()
#     test_ca3[i] = (test_ca3[i] - test_ca3[i].mean()) / test_ca3[i].std()
    

test_ca1 = (test_ca1 - test_ca1.mean()) / test_ca1.std()
test_ca3 = (test_ca3 - test_ca3.mean()) / test_ca3.std()


MAXLEN = 12000

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

MAXLEN = 12000
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

train_inout_seq = []
for i in range(len(train_ca1)):
    train_inout_seq += create_inout_sequences(train_ca3[i], train_ca1[i], train_window, hop_len)
    

test_inout_seq = []
for i in range(len(test_ca1)):
    test_inout_seq += create_inout_sequences(test_ca3[i], test_ca1[i], train_window, hop_len)

train_dataloader = DataLoader(train_inout_seq, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_inout_seq, batch_size=batch_size, shuffle=False, drop_last=True)

model_lstm = LSTMModel(input_size=1, hidden_layer_size=n_hidden, num_layers=n_layers, output_size=1, dropout=dropout)
model_lstm = model_lstm#.to('cuda')
model_lstm.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device('cpu')))

sm_preds = GetPredSignal(model_lstm, train_window, hop_len, test_dataloader)

print("It computed")

bord = getAnswer(sm_preds)
tr_bord = getAnswer(test_ca1[0])

plt.figure(figsize=(16, 9))

plt.plot(test_ca1[0], label="true")
plt.plot(sm_preds, label="predicted")
plt.plot([bord[0]]*2, [-5, 2], color="r", linewidth = 3, label="Pred bord")
plt.plot([bord[1]]*2, [-5, 2], color="r", linewidth = 3)
plt.plot([tr_bord[0]]*2, [-5, 2], color="g", linewidth = 3, label="True bord")
plt.plot([tr_bord[1]]*2, [-5, 2], color="g", linewidth = 3)


plt.legend()
plt.show()
