import torch
import torch.nn as nn
import os
import signal
import sys

# Disabilita cuDNN (utile per la riproducibilità e per evitare errori)
torch.backends.cudnn.enabled = False

# Dizionario caratteri
chars = list("abcdefghijklmnopqrstuvwxyz ,.!?\n")
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# Codifica / Decodifica
def encode(text):
    return [stoi.get(c, 0) for c in text.lower()]

def decode(indices):
    return ''.join([itos.get(i, '?') for i in indices])

# RNN
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.rnn = nn.RNN(16, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out)

# Caricamento dataset con suddivisione in finestre
SEQ_LEN = 1000

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    data = torch.tensor(encode(text), dtype=torch.long)
    inputs, targets = [], []
    for i in range(0, len(data) - SEQ_LEN - 1, SEQ_LEN):
        inputs.append(data[i:i + SEQ_LEN])
        targets.append(data[i + 1:i + SEQ_LEN + 1])
    return torch.stack(inputs), torch.stack(targets)

# Salvataggio su CTRL+C
global_model = None
global_checkpoint_path = "model.pth"

def handle_interrupt(sig, frame):
    print("\n🔁 Interruzione rilevata, salvataggio in corso...")
    if global_model:
        torch.save(global_model.state_dict(), global_checkpoint_path)
        print(f"✅ Modello salvato in {global_checkpoint_path}")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

# Training
def train(model, inputs, targets, epochs=50, lr=0.01, checkpoint_path="model.pth"):
    global global_model, global_checkpoint_path
    global_model = model
    global_checkpoint_path = checkpoint_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs, targets = inputs.to(device), targets.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"📦 Checkpoint caricato da {checkpoint_path}")

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i in range(inputs.size(0)):
            input_batch = inputs[i].unsqueeze(0)
            target_batch = targets[i].unsqueeze(0)

            optimizer.zero_grad()
            output = model(input_batch)
            loss = criterion(output.view(-1, len(stoi)), target_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"🌀 Epoch {epoch}, Loss: {total_loss:.4f}")
        if epoch % 5 == 0 and epoch != 0:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Salvataggio automatico in {checkpoint_path}")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"🏁 Fine training. Modello salvato in {checkpoint_path}")

# Generazione
def generate_text(model, start_str, length=200):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    input_seq = torch.tensor(encode(start_str)).unsqueeze(0).long().to(device)
    generated = list(input_seq[0].cpu().numpy())

    with torch.no_grad():
        for _ in range(length):
            outputs = model(input_seq)
            last_logits = outputs[0, -1]
            probs = torch.softmax(last_logits, dim=0)
            next_char = torch.multinomial(probs, 1).item()
            generated.append(next_char)
            input_seq = torch.tensor(generated[-SEQ_LEN:]).unsqueeze(0).long().to(device)

    return decode(generated)

# Main
if __name__ == "__main__":
    model = CharRNN(len(stoi), 64)
    inputs, targets = load_dataset("../datasets/italy.txt")
    train(model, inputs, targets)

    print("\n--- Generazione testo ---")
    print(generate_text(model, "ciao, ", 300))
