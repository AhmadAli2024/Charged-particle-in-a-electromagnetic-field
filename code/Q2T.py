import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
    def __init__(self, latent_dim, active_dim=4, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.active_dim = active_dim
        self.latent_dim = latent_dim

        self.H_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),  
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),  
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.S = nn.Parameter(torch.zeros(active_dim, active_dim, device=device))
        self.dt_q = nn.Parameter(torch.randn(1, device=device) * 0.1 + 0.5)
        self.dt_p = nn.Parameter(torch.randn(1, device=device) * 0.1 + 0.5)
        self.alpha = nn.Parameter(torch.tensor(0.01, device=device))

    def forward(self, z, dt=0.1):
        # Maintain gradient flow
        z = z.clone().requires_grad_(True)
        z_active = z[:, :self.active_dim]

        # Split components with gradient tracking
        z1 = z_active[:, :2].requires_grad_(True)
        z2 = z_active[:, 2:].requires_grad_(True)
        z_aux = z[:, self.active_dim:]


        # Compute Hamiltonian
        z_combined = torch.cat([z1, z2, z_aux], dim=1)
        H = self.H_net(z_combined).squeeze(-1)


        # Calculate gradients
        dHdz1 = torch.autograd.grad(H, z1, grad_outputs=torch.ones_like(H), create_graph=True)[0]
        dHdz2 = torch.autograd.grad(H, z2, grad_outputs=torch.ones_like(H), create_graph=True)[0]


        # Symplectic update
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
    def __init__(self):
        super().__init__()
        self.transformer = NICECouplingLayer(4, 128)
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
    with open('../data/train.txt', 'r') as f:
        train_lines = [list(map(float, line.strip().split())) for line in f if line.strip()]
    
    # Load testing data
    with open('../data/test.txt', 'r') as f:
        test_lines = [list(map(float, line.strip().split())) for line in f if line.strip()]

    # Process data
    train_data = torch.tensor(train_lines[:1200], dtype=torch.float32)
    test_data = torch.tensor(test_lines, dtype=torch.float32)

    return train_data, torch.tensor(train_lines[1:1201], dtype=torch.float32), test_data

def train(model, X_train, y_train, X_test, epochs=200000, lr=0.0003):
    model = model.to(device)
    X_train, y_train, X_test = X_train.to(device), y_train.to(device), X_test.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    best_loss = float('inf')
    batch_size = 128
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Shuffle data
        #perm = torch.randperm(len(X_train))
        #X_train, y_train = X_train[perm], y_train[perm]
        
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
        if epoch % 50 == 0:
            model.sympNet.enforce_symplecticity()
        
        # Validation
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = evaluate(model, X_test)
                
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(model.state_dict(), 'best_model.pth')
                    plot(X_test,model.predict(X_test[0:1].clone().to(device),300))
                    
            print(f"Epoch {epoch} | Train Loss: {epoch_loss:.6f} | Test Loss: {test_loss:.6f}")


def evaluate(model, X_test, steps=300):
    model.eval()
    initial = X_test[0:1].clone().to(device)
    ground_truth = X_test[:steps+1].cpu()
    
    with torch.no_grad():
        # Enable gradients temporarily for physics calculations
        with torch.enable_grad():
            predicted = model.predict(initial, steps-1).cpu()
    
    return F.mse_loss(predicted, ground_truth[1:steps+1]).sum()

import matplotlib.pyplot as plt

def plot(ground_truth, predicted_trajectory, save_path='test.png', show=True):
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

if __name__ == "__main__":
    X_train, y_train, X_test = load_data()
    model = PNN().to(device)
    X_train, y_train, X_test = X_train.to(device), y_train.to(device), X_test.to(device)
    train(model, X_train, y_train, X_test)
