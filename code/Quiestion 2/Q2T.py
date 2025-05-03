import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NICECouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.m1 = nn.Sequential(
            nn.Linear(dim//2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim//2)
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        y1 = x1 
        y2 = x2 + self.m1(x1)
        return torch.cat([y1, y2], dim=1)

    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=1)
        x1 = y1
        x2 = y2 - self.m1(y1)
        return torch.cat([x1, x2], dim=1)

class ExtendedSympNet(nn.Module):
    def __init__(self, latent_dim, active_dim=4, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.active_dim = active_dim
        self.latent_dim = latent_dim

        # Hamiltonian network with residual connections
        self.H_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            self._make_residual_block(hidden_dim, dropout),
            self._make_residual_block(hidden_dim, dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Skew-symmetric matrix parameterization
        self.W = nn.Parameter(torch.randn(active_dim, active_dim, device=device) * 0.01
        self.register_buffer('S', None)  # Will be computed as W - W.T
        
        # Learnable parameters with proper initialization
        self.dt_q = nn.Parameter(torch.randn(1, device=device) * 0.1 + 0.5)
        self.dt_p = nn.Parameter(torch.randn(1, device=device) * 0.1 + 0.5)
        self.alpha = nn.Parameter(torch.tensor(0.01, device=device))

    def _make_residual_block(self, hidden_dim, dropout):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

    def verlet_step(self, z, dt):
        self.S = self.W - self.W.t()  # Ensure skew-symmetry
        
        z = z.clone().requires_grad_(True)
        z_active = z[:, :self.active_dim]
        q = z_active[:, :2].requires_grad_(True)
        p = z_active[:, 2:].requires_grad_(True)
        z_aux = z[:, self.active_dim:]

        # Compute Hamiltonian and gradients
        z_combined = torch.cat([q, p, z_aux], dim=1)
        H = self.H_net(z_combined).squeeze(-1)
        
        dHdq = torch.autograd.grad(H, q, grad_outputs=torch.ones_like(H), create_graph=True)[0]
        dHdp = torch.autograd.grad(H, p, grad_outputs=torch.ones_like(H), create_graph=True)[0]

        # Symplectic updates with learned parameters
        # Half-step for momentum
        p_half = p - (dt/2) * self.dt_p * dHdq + (dt/2) * self.alpha * (z_active @ self.S)[:, 2:]
        
        # Full-step for position
        q_full = q + dt * self.dt_q * dHdp + dt * self.alpha * (z_active @ self.S.t())[:, :2]
        
        # Another half-step for momentum
        z_active_half = torch.cat([q_full, p_half], dim=1)
        z_combined_half = torch.cat([q_full, p_half, z_aux], dim=1)
        H_half = self.H_net(z_combined_half).squeeze(-1)
        dHdq_half = torch.autograd.grad(H_half, q_full, grad_outputs=torch.ones_like(H_half), create_graph=True)[0]
        
        p_full = p_half - (dt/2) * self.dt_p * dHdq_half + (dt/2) * self.alpha * (z_active_half @ self.S)[:, 2:]

        z_active_final = torch.cat([q_full, p_full], dim=1)
        return torch.cat([z_active_final, z_aux], dim=1)

    def suzuki_4th_order(self, z, dt):
        # Coefficients for symmetric composition
        s = (4 ** (1/3))
        a1 = 1/(2*(2 - s))
        a2 = (1 - s)/(2*(2 - s))
        
        z = self.verlet_step(z, a1*dt)
        z = self.verlet_step(z, a1*dt)
        z = self.verlet_step(z, a2*dt)
        z = self.verlet_step(z, a1*dt)
        z = self.verlet_step(z, a1*dt)
        return z

    def forward(self, z, dt=0.1):
        return self.suzuki_4th_order(z, dt)

class PNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = NICECouplingLayer(4, 128)
        self.sympNet = ExtendedSympNet(4, hidden_dim=256)
        self.best_loss = float('inf')

    def forward(self, x):
        x = x.clone().requires_grad_(True)
        theta = self.transformer(x)
        phi = self.sympNet(theta)
        return self.transformer.inverse(phi)

    @torch.inference_mode()
    def predict(self, x, steps):
        trajectory = [x[0].clone()]
        current = x.clone()
        
        for _ in range(steps):
            current = self.forward(current)
            trajectory.append(current[0].clone())
            
        return torch.stack(trajectory)

def load_data():
    # Load and normalize data with feature-wise standardization
    def process_data(lines):
        data = torch.tensor([list(map(float, line.strip().split())) for line in lines if line.strip()])
        means = data.mean(dim=0)
        stds = data.std(dim=0)
        return (data - means) / (stds + 1e-8)

    with open('../../data/train.txt', 'r') as f:
        train_data = process_data(f.readlines()[:1200])
        
    with open('../../data/test.txt', 'r') as f:
        test_data = process_data(f.readlines()[:300])

    # Create shifted targets for training
    return (train_data[:-1], train_data[1:], test_data)

def train(model, X_train, y_train, X_test, epochs=100000, lr=0.001):
    model = model.to(device)
    X_train, y_train, X_test = X_train.to(device), y_train.to(device), X_test.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )
    
    best_loss = float('inf')
    no_improve = 0
    batch_size = 128
    
    for epoch in (pbar := tqdm(range(epochs))):
        model.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0
        
        # Curriculum learning: gradually increase prediction horizon
        unroll_steps = 1 + (epoch // 500)
        
        for i in range(0, len(X_train), batch_size):
            optimizer.zero_grad()
            
            # Multi-step unrolling for better temporal consistency
            current = X_train[perm[i:i+batch_size]]
            loss = 0
            
            for step in range(unroll_steps):
                pred = model(current)
                loss += F.mse_loss(pred, y_train[perm[i:i+batch_size] if step == 0 else current)
                current = pred.detach().requires_grad_(True)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        # Validation and early stopping
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model.predict(X_test[:1], 299)
                test_loss = F.mse_loss(test_pred, X_test[:300]).item()
                
                scheduler.step(test_loss)
                
                if test_loss < best_loss:
                    best_loss = test_loss
                    no_improve = 0
                    torch.save(model.state_dict(), "best_model.pth")
                    
                    # Visualization
                    plot(X_test[:300].cpu(), test_pred.cpu(), 
                         save_path=f'trajectory_epoch_{epoch}.png')
                else:
                    no_improve += 50
                    if no_improve >= 500:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            pbar.set_description(f"Epoch {epoch} | "
                                 f"Train Loss: {total_loss/len(X_train):.3e} | "
                                 f"Test Loss: {test_loss:.3e} | "
                                 f"LR: {optimizer.param_groups[0]['lr']:.2e}")

def plot(ground_truth, predicted, save_path='trajectory.png'):
    plt.figure(figsize=(10, 8))
    
    # Phase space plots
    for i, (label, dims) in enumerate(zip(
        ['Phase Space (q)', 'Phase Space (p)'],
        [(0, 1), (2, 3)]
    )):
        plt.subplot(2, 1, i+1)
        plt.plot(ground_truth[:, dims[0]], ground_truth[:, dims[1]], 'b-', label='Truth')
        plt.plot(predicted[:, dims[0]], predicted[:, dims[1]], 'r--', label='Predicted')
        plt.xlabel(f'Dimension {dims[0]+1}')
        plt.ylabel(f'Dimension {dims[1]+1}')
        plt.title(label)
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    X_train, y_train, X_test = load_data()
    model = PNN()
    train(model, X_train, y_train, X_test)
