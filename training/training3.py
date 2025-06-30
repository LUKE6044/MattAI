import torch
import torch.nn as nn
import os

chars = list("abcdefghijklmnopqrstuvwxyz ,.!?\n")
stoi = {ch:i for i,ch in enumerate(chars)}

def encode(text):
    return [stoi.get(c, 0) for c in text.lower()]

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.rnn = nn.RNN(16, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    data = torch.tensor(encode(text))
    inputs = data[:-1].unsqueeze(0)
    targets = data[1:].unsqueeze(0)
    return inputs, targets

def train_until_converge(model, inputs, targets, max_epochs=1000, patience=20, lr=0.01, checkpoint_path="model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs, targets = inputs.to(device), targets.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    
    # Carica checkpoint se esiste
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Checkpoint caricato da {checkpoint_path}")
    
    model.train()
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, len(stoi)), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        print(f"Epoch {epoch}, Loss: {current_loss:.6f}")
        
        # Check miglioramento
        if current_loss < best_loss - 1e-6:
            best_loss = current_loss
            epochs_no_improve = 0
            # Salva checkpoint
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint salvato con loss {best_loss:.6f}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping dopo {epoch} epoche senza miglioramenti.")
            break

# Uso esempio
if __name__ == "__main__":
    model = CharRNN(len(stoi), 64)
    inputs, targets = load_dataset("../datasets/dataset.txt")
    train_until_converge(model, inputs, targets)
