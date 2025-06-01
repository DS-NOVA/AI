import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from train_dataset import build_training_dataset  # 같은 폴더니까 그냥 import 가능
import numpy as np

# 클래스 정의
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)

if __name__ == "__main__":
    # 데이터 불러오기
    X_train, X_test, Y_train, Y_test = build_training_dataset()

    # PyTorch 텐서로 변환
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = TensorDataset(
        torch.tensor(X_train).float(),
        torch.tensor(Y_train).float()
    )
    test_ds = TensorDataset(
        torch.tensor(X_test).float(),
        torch.tensor(Y_test).float()
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # 모델 정의
    output_dim = Y_train.shape[1]
    model = MultiLabelClassifier(output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    # 학습 루프
    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"[{epoch+1}] Loss: {epoch_loss / len(train_loader):.4f}")

    # 평가
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test).float().to(device)).cpu().numpy()
        y_pred_bin = (y_pred > 0.5).astype(int)

    print("\n[Classification Report]")
    print(classification_report(Y_test, y_pred_bin))

    print("\n[Multilabel Confusion Matrix]")
    print(multilabel_confusion_matrix(Y_test, y_pred_bin))
