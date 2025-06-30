import torch
import torch.nn as nn
import torch.optim as optim

# Definiamo la RNN
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hidden = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# Esempio super minimale per capire struttura

input_size = 10   # dimensione input (es: one-hot)
hidden_size = 20  # neuroni nascosti
output_size = 10  # dimensione output

model = SimpleRNN(input_size, hidden_size, output_size).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dummy dati (da sostituire con dataset reale)
# 5 batch, sequenze lunghe 3, input_size dimensioni
inputs = torch.randn(5, 3, input_size).cuda()
labels = torch.tensor([1, 2, 3, 4, 0]).cuda()

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
