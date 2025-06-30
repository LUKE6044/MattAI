import torch
import torch.nn as nn

# Vocabolario semplice (lettere minuscole + punteggiatura)
chars = list("abcdefghijklmnopqrstuvwxyz ,.!?\n")
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

def encode(text):
    return [stoi.get(c, 0) for c in text.lower()]

def decode(indices):
    return ''.join([itos[i] for i in indices])

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 16)  # da token a embedding
        self.rnn = nn.RNN(16, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out)  # output per ogni token
        return out

# Carica il testo da file
with open("dataset.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

data = torch.tensor(encode(text))

# input: da 0 a -2
inputs = data[:-1].unsqueeze(0)  # batch dimension 1
# target: da 1 a -1
targets = data[1:].unsqueeze(0)

print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")

model = CharRNN(len(stoi), 64).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

inputs = inputs.cuda()
targets = targets.cuda()

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)  # [1, seq_len, vocab_size]
    loss = criterion(outputs.view(-1, len(stoi)), targets.view(-1))
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
def generate_text(model, start_str, length=100):
    model.eval()
    input_seq = torch.tensor(encode(start_str)).unsqueeze(0).cuda()  # batch 1
    generated = list(input_seq[0].cpu().numpy())
    
    with torch.no_grad():
        for _ in range(length):
            outputs = model(input_seq)  # [1, seq_len, vocab_size]
            last_logits = outputs[0, -1]  # ultimo token predetto
            probs = torch.softmax(last_logits, dim=0)
            next_char = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_char)
            input_seq = torch.tensor(generated[-len(input_seq[0]):]).unsqueeze(0).cuda()
    
    return decode(generated)
print(generate_text(model, "ciao, ", 200))
