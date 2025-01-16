import math
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Hyperparameters
num_items = 9557
batch_size = 32
d_model = 64
dim_feedforward = 512
num_tran_block = 2
nhead = 2
edropout = 0.25
tdropout = 0.2
num_epochs = 10
learning_rate = 0.001
max_len = 20 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=d_model, max_len=max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class LocalEncoder(nn.Module):
    def __init__(self, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, tdropout = tdropout, num_layers=num_tran_block):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=tdropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

    def forward(self, x, padding_mask):
        x = self.transformer_encoder(src=x, src_key_padding_mask=padding_mask)  # (batch, seq, d_model)
        output = x[:,-1,:]
        s_L = output.squeeze(1)  # (batch, d_model)
        return s_L
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, tdropout = tdropout, num_layers=num_tran_block):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout=tdropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)


    def forward(self, x, mask, padding_mask):
        x = self.transformer_decoder(tgt=x, memory=x, tgt_mask=mask, tgt_key_padding_mask=None)
        output = x[:,-1,:]
        s_D = output.squeeze(1)
        return s_D


class MyModel(nn.Module):
    def __init__(self, d_model=d_model, num_items=num_items,max_len=max_len, edropout=edropout,pad_token=0):
        super().__init__()

        self.pad_token = pad_token
        self.num_items = num_items
        self.embedding = nn.Embedding(self.num_items, d_model, padding_idx=pad_token)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.e_dropout = nn.Dropout(edropout)
        self.local_encoder = LocalEncoder()
        self.decoder =  DecoderBlock()
        self.W5 = nn.Linear(2*d_model, d_model, bias=False)

    def generate_causal_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def generate_key_padding_mask(self, data):
        mask = (data.clone().detach() == self.pad_token)
        return mask

    def forward(self, encoder_input, decoder_input, device):  # x = (batch, seq)
        batch = len(encoder_input)
        causal_mask = self.generate_causal_mask(decoder_input.size(1)).to(device)
        encoder_padding_mask = self.generate_key_padding_mask(encoder_input).to(device)
        decoder_padding_mask = self.generate_key_padding_mask(decoder_input).to(device)
        encoder_input = self.embedding(encoder_input) * math.sqrt(d_model)  # (batch, seq, d_model)
        encoder_input = self.pos_encoder(encoder_input)
        encoder_input = self.e_dropout(encoder_input)

        decoder_input = self.embedding(decoder_input) * math.sqrt(d_model)
        decoder_input = self.pos_encoder(decoder_input)
        decoder_input = self.e_dropout(decoder_input)

        s_L, s_D = self.local_encoder(encoder_input, encoder_padding_mask), self.decoder(decoder_input, causal_mask, decoder_padding_mask)
        s_cat = torch.cat((s_L, s_D), dim=1)  # (batch, 2*d_model)
        sT = self.W5(s_cat).unsqueeze(-1)  # (batch, d_model, 1)
        M = self.embedding(torch.arange(self.num_items).unsqueeze(0).repeat(batch, 1).to(device)) # (batch, num_items, d_model)
        y = torch.matmul(M, sT).squeeze()
        return y
        
    
class SessionDataset(Dataset):
    def __init__(self, sessions, max_len=max_len):
        self.sessions = sessions
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        session = self.sessions[idx]
        data = session[:-1]
        label = session[-1]
        encoder_input = None
        decoder_input = None
        if len(data) < self.max_len:
            encoder_input = [0] * (self.max_len - len(data)) + data
            decoder_input = [0] * (self.max_len - len(data)) + data
        else:
            encoder_input = data[-self.max_len:]
            decoder_input = data[-self.max_len:]
        return torch.tensor(encoder_input), torch.tensor(decoder_input),torch.tensor(label)
    
def train(model, train_loader, epoch, num_epochs, criterion, optimizer, device):
    torch.cuda.empty_cache()
    model.train()
    total_loss = 0
    batch_iterator = tqdm(train_loader, desc=f'Processing epoch {epoch+1}/{num_epochs}', total = len(train_loader))
    for encoder_input, decoder_input, label in batch_iterator:
        encoder_input, decoder_input, label = encoder_input.to(device), decoder_input.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(encoder_input, decoder_input, device)
        loss = criterion(output, label)
        batch_iterator.set_postfix({f'loss': f'{loss.item():6.3f}'})
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for encoder_input, decoder_input, label in val_loader:
            encoder_input, decoder_input, label = encoder_input.to(device), decoder_input.to(device), label.to(device)
            output = model(encoder_input, decoder_input, device)
            loss = criterion(output, label)
            total_loss += loss.item()
    return total_loss / len(val_loader)

if __name__ == '__main__':
    with open('dataset/yoochoose_64_9556/train_64.txt', 'rb') as f:
        sessions = pickle.load(f)
    
    dataset = SessionDataset(sessions, max_len)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel(d_model=d_model, num_items=num_items,max_len=max_len, edropout=edropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, epoch, num_epochs, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Save model
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'weight/model_b32_t2_1cmask_0pmask.pt')