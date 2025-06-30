import torch
import training4
import warnings
import torch
import torch.nn as nn
warnings.filterwarnings("ignore", category=UserWarning, message=".*weights_only=False.*")
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Crea modello
    model = training4.CharRNN(len(training4.stoi), 64).to(device)
    
    # Carica pesi
    model.load_state_dict(torch.load("../model.pth", map_location=device))
    model.eval()
    
    
    testo_generato = training4.generate_text(model, "Nel Seicent", 1000)
    print(testo_generato)

if __name__ == "__main__":
    main()
