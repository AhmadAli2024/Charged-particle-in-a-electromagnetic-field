import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Get GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extended Symp Net Modules
class SigmaBlock(nn.Module):
    def __init__(self, d, c_dim, l, nonlinearity=torch.tanh):
        super().__init__()
        self.K1 = nn.Linear(d, l, bias=False)
        self.K2 = nn.Linear(c_dim, l, bias=False)
        self.b = nn.Parameter(torch.zeros(l))
        self.a = nn.Parameter(torch.ones(l))
        self.nonlinearity = nonlinearity
        self.output_proj = nn.Linear(l, d, bias=False)

    def forward(self, x, c):
        # x: (batch, d), c: (batch, c_dim)
        h = self.K1(x) + self.K2(c) + self.b  # (batch, l)
        h = self.nonlinearity(h)             # (batch, l)
        h = self.a * h                       # (batch, l)
        out = self.output_proj(h)            # (batch, d)
        return out

class EUp(nn.Module):
    def __init__(self, d, c_dim, l, nonlinearity=torch.tanh):
        super().__init__()
        self.sigma = SigmaBlock(d, c_dim, l, nonlinearity)

    def forward(self, p, q, c):
        delta = self.sigma(q, c)
        return p + delta, q, c

class ELow(nn.Module):
    def __init__(self, d, c_dim, l, nonlinearity=torch.tanh):
        super().__init__()
        self.sigma = SigmaBlock(d, c_dim, l, nonlinearity)

    def forward(self, p, q, c):
        delta = self.sigma(p, c)
        return p, q + delta, c

class ExtendedSympNet(nn.Module):
    def __init__(self, d, c_dim, l, depth, nonlinearity=nn.Tanh()):
        super().__init__()
        layers = []
        for i in range(depth):
            if i % 2 == 0:
                layers.append(EUp(d, c_dim, l, nonlinearity))
            else:
                layers.append(ELow(d, c_dim, l, nonlinearity))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # Split input into p, q, c
        d = (x.shape[1] - self.layers[0].sigma.K2.in_features) // 2
        p, q, c = x[:, :d], x[:, d:2*d], x[:, 2*d:]

        for layer in self.layers:
            p, q, c = layer(p, q, c)

        return torch.cat([p, q, c], dim=1)

# Transformer to change the coordanients of the system 
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


# Complete Poission Neural Network
class PNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = NICECouplingLayer(4, 64)
        self.sympNet = ExtendedSympNet(d=2, c_dim=0, l=128, depth=6)
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


def train(model, X_train, y_train, X_test, epochs=500000, lr=0.01):

    model = model.to(device)
    X_train, y_train, X_test = X_train.to(device), y_train.to(device), X_test.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,              # or your specific learning rate
        betas=(0.9, 0.999),   # default AdamW momentum settings
        weight_decay=1e-3,
        eps=1e-8              # numerical stability
    )

    best_loss = float('inf')
    no_improve = 0
    batch_size = 128

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.7,
    patience=10,
    min_lr=1e-6
    )
    
    for epoch in (pbar := tqdm(range(epochs))):
        model.train()
        epoch_loss = 0
        
        # Shuffle data
        perm = torch.randperm(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]
        
        for i in range(0, len(X_train), batch_size):
            # Get batches
            X_batch = X_train[i:i+batch_size].requires_grad_(True)
            y_batch = y_train[i:i+batch_size]
            
            # Zero out the gradients
            optimizer.zero_grad()

            # Forward 
            pred = model(X_batch)

            # Get loss
            loss = F.mse_loss(pred, y_batch)
            
            loss.backward()

            # clip gradients for stable learning
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
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
                
            pbar.set_description(f"Epoch {epoch} | "
                                 f"Train Loss: {epoch_loss:.6f} | "
                                 f"Test Loss: {test_loss:.3e} | "
                                 f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                                 f"PB: {best_loss:.6f}")


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
        #print(f"Plot saved to {save_path}")
    except Exception as e:
        pass
        #print(f"Error saving plot: {e}")
    
    if show:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    X_train, y_train, X_test = load_data()
    model = PNN().to(device)
    X_train, y_train, X_test = X_train.to(device), y_train.to(device), X_test.to(device)
    train(model, X_train, y_train, X_test)
