from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
from train import MyModel, num_items, batch_size, d_model, dim_feedforward, num_tran_block, nhead, edropout, tdropout, num_epochs, learning_rate, max_len

torch.manual_seed(42)

# Tạo FastAPI app
app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

PATH = 'weight/model_b32_t2.pt'

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()

# Class để định nghĩa đầu vào và đầu ra của API
class PredictionRequest(BaseModel):
    clicked_item_ids: list[int]
    topk: int

class PredictionResponse(BaseModel):
    recommended_item_ids: list[int]

def make_input(data, max_len = max_len):
    encoder_input = None
    decoder_input = None
    if len(data) < max_len:
        encoder_input = data + [0] * (max_len - len(data))
        decoder_input = [0] * (max_len - len(data)) + data
    else:
        encoder_input = data[-max_len:]
        decoder_input = data[-max_len:]
    return torch.tensor(encoder_input).unsqueeze(0), torch.tensor(decoder_input).unsqueeze(0)

# Định nghĩa route API cho dự đoán
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    clicked_item_ids = request.clicked_item_ids
    topk = request.topk

    if len(clicked_item_ids) == 0:
        raise HTTPException(status_code=400, detail="No items provided for prediction")

    print("Clicked item id: ", clicked_item_ids)

    # Chuẩn bị dữ liệu đầu vào cho mô hình PyTorch
    encoder_input, decoder_input = make_input(clicked_item_ids) # [1, N]
    encoder_input, decoder_input = encoder_input.to(device), decoder_input.to(device)
    # Thực hiện dự đoán với mô hình
    with torch.no_grad():
        output = model(encoder_input, decoder_input, device)
        probabilities = torch.softmax(output, dim=-1)
        recommended_item_ids = torch.topk(probabilities, topk, dim=-1).indices # (batch, 20)
        recommended_item_ids = recommended_item_ids.squeeze().tolist()
        print("Recommend item id: ", recommended_item_ids)
    # Trả về kết quả
    return PredictionResponse(recommended_item_ids=recommended_item_ids)

