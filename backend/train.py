import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import FeedForwardNet
import os

def train_loop(model, dataloader, optimizer, criterion, device):
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Dummy data for illustration
    X = torch.randn(1000, 4)
    y = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = FeedForwardNet(input_dim=4, hidden_dim=args.hidden_dim, output_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loop(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs} done.")

    # Save model in checkpoints/
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model.pt")
    print("✅ Model saved to checkpoints/model.pt")

if __name__ == "__main__":
    main()
