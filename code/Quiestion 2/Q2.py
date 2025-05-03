import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NICECouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.m1 = nn.Sequential(
            nn.Linear(dim//2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
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
    def __init__(self, latent_dim, active_dim=4, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.active_dim = active_dim
        self.latent_dim = latent_dim

        self.H_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

        self.S = nn.Parameter(torch.zeros(active_dim, active_dim, device=device))
        self.dt_q = nn.Parameter(torch.randn(1, device=device) * 0.1 + 0.5)
        self.dt_p = nn.Parameter(torch.randn(1, device=device) * 0.1 + 0.5)
        self.alpha = nn.Parameter(torch.tensor(0.01, device=device))

#    def forward(self, z, dt=0.1):
#        # Maintain gradient flow
#        z = z.clone().requires_grad_(True)
#        z_active = z[:, :self.active_dim]
#
#        # Split components with gradient tracking
#        z1 = z_active[:, :2].requires_grad_(True)
#        z2 = z_active[:, 2:].requires_grad_(True)
#        z_aux = z[:, self.active_dim:]
#
#
#        # Compute Hamiltonian
#        z_combined = torch.cat([z1, z2, z_aux], dim=1)
#        H = self.H_net(z_combined).squeeze(-1)
#
#
#        # Calculate gradients
#        dHdz1 = torch.autograd.grad(H, z1, grad_outputs=torch.ones_like(H), create_graph=True)[0]
#        dHdz2 = torch.autograd.grad(H, z2, grad_outputs=torch.ones_like(H), create_graph=True)[0]
#
#
#        # Symplectic update
#        S = self.S - self.S.t()
#        dz1 = dHdz2 * self.dt_q + self.alpha * (z_active @ S.t())[:, :2]
#        dz2 = -dHdz1 * self.dt_p + self.alpha * (z_active @ S)[:, 2:]
#
#        z_active_new = z_active + dt * torch.cat([dz1, dz2], dim=1)
#
#
#        return torch.cat([z_active_new, z_aux], dim=1)


    def verlet_step(self,z, dt, H_net, S, alpha, active_dim, dt_q, dt_p):
        z = z.clone().requires_grad_(True)
        z_active = z[:, :active_dim]
        z1 = z_active[:, :2].requires_grad_(True)
        z2 = z_active[:, 2:].requires_grad_(True)
        z_aux = z[:, active_dim:]

        # Compute Hamiltonian and gradients
        z_combined = torch.cat([z1, z2, z_aux], dim=1)
        H = H_net(z_combined).squeeze(-1)
        dHdz1 = torch.autograd.grad(H, z1, grad_outputs=torch.ones_like(H), create_graph=True)[0]
        dHdz2 = torch.autograd.grad(H, z2, grad_outputs=torch.ones_like(H), create_graph=True)[0]

        # Half-step for z2 (p)
        dz2_half = -dHdz1 * (dt/2) * dt_p + alpha * (z_active @ S)[:, 2:] * (dt/2)
        z2_new = z2 + dz2_half

        # Full-step for z1 (q)
        dz1_full = dHdz2 * dt * dt_q + alpha * (z_active @ S.t())[:, :2] * dt
        z1_new = z1 + dz1_full

        # Another half-step for z2 (p)
        z_active_half = torch.cat([z1_new, z2_new], dim=1)
        z_combined_half = torch.cat([z1_new, z2_new, z_aux], dim=1)
        H_half = H_net(z_combined_half).squeeze(-1)
        dHdz1_half = torch.autograd.grad(H_half, z1_new, grad_outputs=torch.ones_like(H_half), create_graph=True)[0]
        dz2_half2 = -dHdz1_half * (dt/2) * dt_p + alpha * (z_active_half @ S)[:, 2:] * (dt/2)
        z2_final = z2_new + dz2_half2

        z_active_final = torch.cat([z1_new, z2_final], dim=1)
        return torch.cat([z_active_final, z_aux], dim=1)


    def suzuki_4th_order(self,z, dt, H_net, S, alpha, active_dim, dt_q, dt_p):
        # Suzuki's coefficients
        a1 = 1.0 / (4 - 4**(1/3))
        a2 = a1
        a3 = 1 - 4 * a1
        a4 = a1
        a5 = a1

        # Apply composition
        z = self.verlet_step(z, a1 * dt, H_net, S, alpha, active_dim, dt_q, dt_p)
        z = self.verlet_step(z, a2 * dt, H_net, S, alpha, active_dim, dt_q, dt_p)
        z = self.verlet_step(z, a3 * dt, H_net, S, alpha, active_dim, dt_q, dt_p)
        z = self.verlet_step(z, a4 * dt, H_net, S, alpha, active_dim, dt_q, dt_p)
        z = self.verlet_step(z, a5 * dt, H_net, S, alpha, active_dim, dt_q, dt_p)

        return z

    def forward(self, z, dt=0.1):
        return self.suzuki_4th_order(
            z, dt, 
            self.H_net, self.S, self.alpha, 
            self.active_dim, self.dt_q, self.dt_p
        )


    def enforce_symplecticity(self):
        with torch.no_grad():
            self.S.data = 0.5 * (self.S - self.S.t())
            self.dt_q.data.abs_()
            self.dt_p.data.abs_()

class PNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = NICECouplingLayer(4, 64)
        self.sympNet = ExtendedSympNet(4)
        self.lowestLoss = float('inf')

    def forward(self, x):
        x = x.clone().requires_grad_(True)
        theta = self.transformer.forward(x)
        phi = self.sympNet(theta)
        return self.transformer.inverse(phi)


    def predict(self, x, steps):
        trajectory = [x.detach()[0]]
        current = x.clone()
        
        for _ in range(steps):
            with torch.enable_grad():  # Enable gradients temporarily
                current = current.requires_grad_(True)
                next_step = self.forward(current)
            trajectory.append(next_step.detach()[0])
            current = next_step.detach()
            
        return torch.stack(trajectory)

def load_data():
    # Load training data
    with open('../../data/train.txt', 'r') as f:
        train_lines = [list(map(float, line.strip().split())) for line in f if line.strip()]
    
    # Load testing data
    with open('../../data/test.txt', 'r') as f:
        test_lines = [list(map(float, line.strip().split())) for line in f if line.strip()]

    # Process data
    train_data = torch.tensor(train_lines[:1200], dtype=torch.float32)
    trainP_data = torch.tensor(train_lines[1:1201], dtype=torch.float32)
    test_data = torch.tensor(test_lines[:300], dtype=torch.float32)
    
    train_data = torch.nn.functional.normalize(train_data, p=2, dim=1)
    test_data = torch.nn.functional.normalize(test_data, p=2, dim=1)
    trainP_data = torch.nn.functional.normalize(trainP_data, p=2, dim=1)

    return train_data, trainP_data, test_data

def train(model, X_train, y_train, X_test, epochs=500000, lr=0.0005):

    model = model.to(device)
    X_train, y_train, X_test = X_train.to(device), y_train.to(device), X_test.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,              # or your specific learning rate
        betas=(0.9, 0.999),   # default AdamW momentum settings
        eps=1e-8              # numerical stability
    )

    best_loss = float('inf')
    batch_size = 512

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=0.00001 # or whatever lower bound you want
    )
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Shuffle data
        perm = torch.randperm(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]
        
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size].requires_grad_(True)
            y_batch = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = F.mse_loss(pred, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Enforce symplecticity
        if epoch % 100 == 0:
            model.sympNet.enforce_symplecticity()

        # Validation
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = evaluate(model, X_test)
                scheduler.step(test_loss)

                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(model.state_dict(), "best_model.pth")
                    pred_trajectory = model.predict(X_test[0:1].clone().to(device), 299)
                    mse_T = F.mse_loss(pred_trajectory,X_test[:300],reduction='none').mean(dim=1)
                    plotMSE(mse_T,save_path=f'MSE_{epoch}.png')
                    plot(X_test[:300], pred_trajectory, save_path=f'trajectory_epoch_{epoch}.png')
                    print(f"New best model saved. Test Loss: {test_loss:.6f}")
                
            print(f"Epoch {epoch} | Train Loss: {epoch_loss:.6f} | Test Loss: {test_loss:.6f} | LR: {lr}")


def evaluate(model, X_test, steps=300):
    model.eval()
    initial = X_test[0:1].clone().to(device)
    ground_truth = X_test[:steps+1].cpu()
    
    with torch.no_grad():
        # Enable gradients temporarily for physics calculations
        with torch.enable_grad():
            predicted = model.predict(initial, steps-1).cpu()
    
    return F.mse_loss(predicted, ground_truth).sum()


def plot(ground_truth, predicted_trajectory, save_path='test.png', show=True):
    if os.path.exists(save_path):
        os.remove(save_path)
    plt.figure(figsize=(8, 6))

    # Ground truth trajectory
    plt.plot(
        ground_truth[:, 2].cpu().numpy(), ground_truth[:, 3].cpu().numpy(),
        'b-', label='Ground Truth', linewidth=2
    )

    # Predicted trajectory
    plt.plot(
        predicted_trajectory[:, 2].cpu().detach().numpy(), predicted_trajectory[:, 3].cpu().detach().numpy(),
        'r--', label='PNN Prediction', linewidth=2
    )

    # Start point
    plt.scatter(
        ground_truth[0, 2].cpu().numpy(), ground_truth[0, 3].cpu().numpy(),
        c='green', s=100, label='Start Point'
    )

    plt.xlabel('Position $x_1$', fontsize=14)
    plt.ylabel('Position $x_2$', fontsize=14)
    plt.title('Charged Particle Trajectory Prediction', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.axis('equal')  # Maintains spatial proportions

    plt.tight_layout()
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plotMSE(loss, save_path='test.png', show=True):
    """Plot the ground truth vs predicted trajectory"""
    if os.path.exists(save_path):
        os.remove(save_path)
        
    plt.figure(figsize=(8, 6))
    
    t = torch.arange(len(loss))

    plt.plot(t.numpy(),loss.cpu().numpy())
    plt.xlabel("Step(0.1 Seconds)")
    plt.ylabel("MSE Loss")
    plt.title("MSE Vs Time")

    
    try:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    if show:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    X_train, y_train, X_test = load_data()
    model = PNN().to(device)
    X_train, y_train, X_test = X_train.to(device), y_train.to(device), X_test.to(device)
    train(model, X_train, y_train, X_test)
