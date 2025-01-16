import torch
import pickle
import torch.optim as optim
from torch.utils.data import DataLoader
from train import SessionDataset, MyModel, num_items, batch_size, d_model, dim_feedforward, num_tran_block, nhead, edropout, tdropout, num_epochs, learning_rate, max_len

torch.set_printoptions(threshold=10000)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

PATH = 'weight/model_b32_t2_1cmask_0pmask.pt'

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()

with open('dataset/yoochoose_64_9556/test_64.txt', 'rb') as f:
    sessions = pickle.load(f)

test_dataset = SessionDataset(sessions, max_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

num_samples = 0
metric_ks=[20, 10, 5]
hits = [0] * len(metric_ks)
sum_mrrs = [0] * len(metric_ks)

with torch.no_grad():
    for encoder_input, decoder_input, label in test_loader:
        encoder_input, decoder_input, label = encoder_input.to(device), decoder_input.to(device), label.to(device)
        output = model(encoder_input, decoder_input, device)
        probabilities = torch.softmax(output, dim=-1)
        batch_size = label.size(0)
        num_samples += batch_size

        for idx in range(len(metric_ks)):
            top_items = torch.topk(probabilities, metric_ks[idx], dim=-1).indices
            tgt2 = label.unsqueeze(-1).to(device)
            result = torch.any(tgt2 == top_items, dim=-1)
            hits[idx] += torch.sum(result).item()

            items_rank = torch.arange(1, top_items.size(1) + 1).repeat(batch_size, 1).to(device) # (batch, 20)
            rank = torch.where(tgt2 == top_items, items_rank, float('inf'))
            inv_rank = torch.where(rank != float('inf'), 1.0 / rank.float(), torch.tensor(0.0))
            sum_mrrs[idx] += inv_rank.sum(dim=1).sum().item()

for idx in range(len(metric_ks)):
    hit_rate = hits[idx] * 100 / num_samples
    mrr_rate = sum_mrrs[idx] * 100 / num_samples
    print(f"HR@{metric_ks[idx]}: {hit_rate}%")
    print(f"MRR@{metric_ks[idx]}: {mrr_rate}%")