import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class NICECouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim // 2)
        )

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
    def __init__(self, latent_dim, active_dim=4, hidden_dim=256):
        super().__init__()
        self.active_dim = active_dim
        self.latent_dim = latent_dim

        # Enhanced Hamiltonian network
        self.H_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Physics-informed parameters with constraints
        self.S = nn.Parameter(torch.zeros(active_dim, active_dim))
        torch.nn.init.orthogonal_(self.S)
        self.register_parameter('S', self.S)
        
        self.dt_q = nn.Parameter(torch.tensor(0.1))
        self.dt_p = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def forward(self, z, dt=0.1):
        z_active = z[:, :self.active_dim]
        z_aux = z[:, self.active_dim:]

        with torch.enable_grad():
            z_combined = torch.cat([z_active, z_aux], dim=1).requires_grad_(True)
            H = self.H_net(z_combined).sum()
            
            dHdz = torch.autograd.grad(H, z_combined, create_graph=True)[0]
            dHdz1 = dHdz[:, :2]
            dHdz2 = dHdz[:, 2:4]

        # Enforce symplectic structure
        S = 0.5 * (self.S - self.S.t())
        
        dz1 = dHdz2 * self.dt_q + self.alpha * (z_active @ S.t())[:, :2]
        dz2 = -dHdz1 * self.dt_p + self.alpha * (z_active @ S)[:, 2:]

        # Symplectic Euler integration
        z_active_new = z_active + dt * torch.cat([dz1, dz2], dim=1)
        return torch.cat([z_active_new, z_aux], dim=1)

class EnhancedPNN(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.transformer = NICECouplingLayer(4, hidden_dim)
        self.sympNet = ExtendedSympNet(4, hidden_dim=hidden_dim)
        
        # Physics-guided initialization
        with torch.no_grad():
            for layer in self.transformer.net:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.normal_(0, 0.02)
            self.sympNet.H_net[-1].weight.data *= 0.1

    def forward(self, x):
        theta = self.transformer(x)
        phi = self.sympNet(theta)
        return self.transformer.inverse(phi)

    def predict(self, x0, steps, dt=0.1):
        trajectory = [x0]
        current = x0.clone()
        for _ in range(steps):
            current = self(current) * dt + current
            trajectory.append(current)
        return torch.stack(trajectory)

def train_pnn(model, X_train, y_train, epochs=1000, lr=0.001):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            
            # Forward pass with gradient scaling
            with torch.cuda.amp.autocast():
                pred = model(batch_X)
                loss = F.mse_loss(pred, batch_y)
                
                total_loss = loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        scheduler.step()
        epoch_loss /= len(loader)
        
        # Validation check
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_train[:1024])
                val_loss = F.mse_loss(val_pred, y_train[:1024]).item()
                
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'best_pnn_model.pth')
                
            print(f"Epoch {epoch} | Train Loss: {epoch_loss:.4e} | Val Loss: {val_loss:.4e}")
    
    model.load_state_dict(torch.load('best_pnn_model.pth'))
    return model

# Enhanced evaluation with phase space visualization
def evaluate_and_plot(model, X_test, steps=300):
    model.eval()
    initial_state = X_test[0:1].to(device)
    
    with torch.no_grad():
        pred_traj = model.predict(initial_state, steps)
    
    true_traj = X_test[:steps+1]
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Position space
    axs[0,0].plot(true_traj[:, 2].cpu(), true_traj[:, 3].cpu(), 'b-', label='True')
    axs[0,0].plot(pred_traj[:, 0, 2].cpu(), pred_traj[:, 0, 3].cpu(), 'r--', label='Pred')
    axs[0,0].set_title('Position Space')
    
    # Momentum space
    axs[0,1].plot(true_traj[:, 0].cpu(), true_traj[:, 1].cpu(), 'b-')
    axs[0,1].plot(pred_traj[:, 0, 0].cpu(), pred_traj[:, 0, 1].cpu(), 'r--')
    axs[0,1].set_title('Momentum Space')
    
    # Energy conservation
    true_energy = (true_traj[:, :2].norm(dim=1) - 1/(true_traj[:, 2:].norm(dim=1)+1e-8)).cpu()
    pred_energy = (pred_traj[:, 0, :2].norm(dim=1) - 1/(pred_traj[:, 0, 2:].norm(dim=1)+1e-8)).cpu()
    axs[1,0].plot(true_energy, 'b-', label='True')
    axs[1,0].plot(pred_energy, 'r--', label='Pred')
    axs[1,0].set_title('System Energy')
    
    # Error progression
    errors = [(pred_traj[i,0] - true_traj[i]).norm().item() for i in range(len(pred_traj))]
    axs[1,1].semilogy(errors)
    axs[1,1].set_title('Prediction Error')
    
    plt.tight_layout()
    plt.savefig('pnn_analysis.png')
    plt.show()
    
    return np.mean(errors)

if __name__ == "__main__":
    # Data loading with normalization
    data = np.loadtxt('../data/train.txt')
    X_train = torch.tensor(data[:-1], dtype=torch.float32)
    y_train = torch.tensor(data[1:], dtype=torch.float32)
    
    model = EnhancedPNN(hidden_dim=256).to(device)
    model = train_pnn(model, X_train.to(device), y_train.to(device), epochs=10000, lr=0.001)
    
    # Load and normalize test data
    test_data = np.loadtxt('../data/test.txt')
    X_test = torch.tensor(test_data, dtype=torch.float32)
    avg_error = evaluate_and_plot(model, X_test.to(device))
    print(f"Average prediction error: {avg_error:.4e}")
