import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class NICECouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim//2)
        ).to(device)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        y1 = x1
        y2 = x2 + self.net(x1)
        return torch.cat([y1, y2], dim=1)

    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=1)
        x1 = y1
        x2 = y2 - self.net(y1)
        return torch.cat([x1, x2], dim=1)

class ExtendedSympNet(nn.Module):
    def __init__(self, latent_dim, active_dim=4, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.active_dim = active_dim
        self.latent_dim = latent_dim

        self.H_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        ).to(device)

        self.S = nn.Parameter(torch.zeros(active_dim, active_dim, device=device))
        torch.nn.init.normal_(self.S, 0, 0.1)

        self.dt_q = nn.Parameter(torch.randn(1, device=device) * 0.1 + 0.5)
        self.dt_p = nn.Parameter(torch.randn(1, device=device) * 0.1 + 0.5)
        self.alpha = nn.Parameter(torch.tensor(0.0001, device=device))

    def forward(self, z, dt=0.1):
        z = z.clone().requires_grad_(True)
        z_active = z[:, :self.active_dim]
        z_aux = z[:, self.active_dim:]

        z1 = z_active[:, :2].requires_grad_(True)
        z2 = z_active[:, 2:].requires_grad_(True)

        z_combined = torch.cat([z1, z2, z_aux], dim=1)
        H = self.H_net(z_combined).sum()

        dHdz1 = torch.autograd.grad(H, z1, retain_graph=True, create_graph=True)[0]
        dHdz2 = torch.autograd.grad(H, z2, retain_graph=True, create_graph=True)[0]

        S = self.S - self.S.t()
        dz1 = dHdz2 * self.dt_q + self.alpha * (z_active @ S.t())[:, :2]
        dz2 = -dHdz1 * self.dt_p + self.alpha * (z_active @ S)[:, 2:]

        z_active_new = z_active + dt * torch.cat([dz1, dz2], dim=1)
        return torch.cat([z_active_new, z_aux], dim=1)

    def enforce_symplecticity(self):
        with torch.no_grad():
            self.S.data = 0.5 * (self.S - self.S.t())
            self.dt_q.data.abs_()
            self.dt_p.data.abs_()

class PNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.transformer = NICECouplingLayer(4, 125)
        self.sympNet = ExtendedSympNet(4, dropout=dropout)
        self.lowestLoss = float('inf')
        
    def forward(self, x):
        theta = self.transformer(x)
        phi = self.sympNet(theta)
        return self.transformer.inverse(phi)

    def predict(self, x, steps):
        trajectory = []
        current = x.clone()
        for _ in range(steps):
            current = self.forward(current)
            trajectory.append(current.detach())
        return torch.stack(trajectory)

def load_data():
    # Load training data
    with open('../data/train.txt', 'r') as f:
        train_data = torch.tensor([list(map(float, line.split())) for line in f if line.strip()]).float()
    
    # Load testing data
    with open('../data/test.txt', 'r') as f:
        test_data = torch.tensor([list(map(float, line.split())) for line in f if line.strip()]).float()
    
    # Prepare training tensors
    X_train = torch.cat([train_data[:1200, :2], train_data[:1200, 2:]], dim=1).to(device)
    y_train = torch.cat([train_data[1:1201, :2], train_data[1:1201, 2:]], dim=1).to(device)
    
    # Prepare test tensors
    X_test = torch.cat([test_data[:, :2], test_data[:, 2:]], dim=1).to(device)
    
    return X_train, y_train, X_test

def train(model, X_train, y_train, X_test, epochs=200000, lr=0.0001):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5)
    
    best_loss = float('inf')
    batch_size = 256
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Shuffle data each epoch
        perm = torch.randperm(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]
        
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size].clone().requires_grad_(True)
            y_batch = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            pred = model(X_batch)
            
            loss = F.mse_loss(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Enforce symplecticity
        if epoch % 10 == 0:
            model.sympNet.enforce_symplecticity()
        
        # Evaluation
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = evaluate(model, X_test)
                scheduler.step(test_loss)
                
                if test_loss < best_loss:
                    best_loss = test_loss
                    model.lowestLoss = best_loss
                    torch.save(model.state_dict(), 'best_model.pt')
                    print(f"New best model saved with test loss: {best_loss:.6f}")
            
            print(f"Epoch {epoch} | Train Loss: {epoch_loss/(len(X_train)/batch_size):.6f} | Test Loss: {test_loss:.6f}")
            
            # Clear memory
            torch.cuda.empty_cache()

def evaluate(model, X_test, steps=300):
    initial = X_test[0:1].clone()
    ground_truth = X_test[:steps+1]
    
    predicted = model.predict(initial, steps)
    
    mse = F.mse_loss(predicted, ground_truth[1:steps+1])
    return mse.item()

if __name__ == "__main__":
    # Load and prepare data
    X_train, y_train, X_test = load_data()
    
    # Initialize model
    model = PNN().to(device)
    print("Model initialized on", next(model.parameters()).device)
    
    # Train
    print("Starting training...")
    train(model, X_train, y_train, X_test)
    print("Training complete!")
    
    # Final evaluation
    final_loss = evaluate(model, X_test)
    print(f"Final test loss: {final_loss:.6f}")
