import torch
import torch.nn as nn
import os
import signal
import sys

torch.backends.cudnn.enabled = False

chars = list("abcdefghijklmnopqrstuvwxyz ,.!?\n")
stoi = {ch: i for i, ch in enumerate(chars)}

def encode(text):
    return [stoi.get(c, 0) for c in text.lower()]

def decode(indices):
    return ''.join([chars[i] for i in indices])

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)
        self.rnn = nn.RNN(16, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x).contiguous()
        out, _ = self.rnn(x)
        return self.fc(out)

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    data = torch.tensor(encode(text))
    inputs = data[:-1].unsqueeze(0).long()
    targets = data[1:].unsqueeze(0).long()
    return inputs, targets

# 🧠 Variabile globale per accesso da handler
global_model = None
global_checkpoint_path = "model.pth"

# ⛑️ Salva su CTRL+C
def handle_interrupt(sig, frame):
    print("\n🔁 Interruzione rilevata, salvataggio in corso...")
    if global_model:
        torch.save(global_model.state_dict(), global_checkpoint_path)
        print(f"✅ Modello salvato in {global_checkpoint_path}")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_interrupt)

def train(model, inputs, targets, epochs=1000, lr=0.01, checkpoint_path="model.pth"):
    global global_model, global_checkpoint_path
    global_model = model
    global_checkpoint_path = checkpoint_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs, targets = inputs.to(device).contiguous(), targets.to(device).contiguous()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"📦 Checkpoint caricato da {checkpoint_path}")

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, len(stoi)), targets.view(-1))
        loss.backward()
        optimizer.step()

        print(f"🌀 Epoch {epoch}, Loss: {loss.item():.6f}")

        # Salva ogni 5 epoche
        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Salvataggio automatico in {checkpoint_path}")

    # Salva finale
    torch.save(model.state_dict(), checkpoint_path)
    print(f"🏁 Fine training. Modello salvato in {checkpoint_path}")

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
            next_char = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_char)
            input_seq = torch.tensor(generated[-len(input_seq[0]):]).unsqueeze(0).long().to(device)

    return decode(generated)

if __name__ == "__main__":
    model = CharRNN(len(stoi), 64)
    inputs, targets = load_dataset("dataset5.txt")
    train(model, inputs, targets)

    print("\n--- Generazione testo ---")
    print(generate_text(model, "ciao, ", 300))
