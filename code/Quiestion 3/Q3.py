import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PINN(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4)
        )

        # Learnable parameters with positive constraint using softplus
        self.lambda1 = nn.Parameter(torch.tensor(0.5))
        self.lambda2 = nn.Parameter(torch.tensor(0.5))
        self.lambda3 = nn.Parameter(torch.tensor(0.5))
        self.lambda4 = nn.Parameter(torch.tensor(0.5))

    def computeDerivative(self, X, dt=0.1):
        # make X require grad
        X = X.requires_grad_()
        # Get the Hamiltonian
        H = self.model(X)
        # Get the derivative of the Hamiltonian
        dH = torch.autograd.grad(H,X,grad_outputs=torch.ones_like(H),create_graph=True)[0] 
        # Split derivatives
        dH_dq = dH[:,2:]
        dH_dp = dH[:,:2]
        return torch.cat([-dH_dq,dH_dp],dim=1) # Return in terms of dt

    def rk4StepForward(self, X, dt):
        k1 = self.computeDerivative(X)
        k2 = self.computeDerivative(X + 0.5 * dt * k1)
        k3 = self.computeDerivative(X + 0.5 * dt * k2)
        k4 = self.computeDerivative(X + dt * k3)

        return X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rk4StepBackwards(self, X, dt):
        k1 = self.computeDerivative(X)
        k2 = self.computeDerivative(X - 0.5 * dt * k1)
        k3 = self.computeDerivative(X - 0.5 * dt * k2)
        k4 = self.computeDerivative(X - dt * k3)

        return X - (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def risidualLoss(self,X,Y,dt):
        # Step forward and back
        XPred = self.rk4StepForward(X,dt)
        YPred = self.rk4StepBackwards(Y,dt)
        # split the position from velocity for better accuracy
        Xp = XPred[:,:2]
        Xq = XPred[:,2:]
        Yp = YPred[:,:2]
        Yq = YPred[:,2:]
        # Get all the losses
        XPredLoss = self.lambda1 * F.mse_loss(Xp,Y[:,:2]) + self.lambda2 * F.mse_loss(Xq,Y[:,2:])
        YPredLoss = self.lambda3 * F.mse_loss(Yp,X[:,:2]) + self.lambda4 * F.mse_loss(Yq,X[:,2:])
        return YPredLoss + XPredLoss

    def forward(self,X,dt):
        return self.rk4StepForward(X,dt)

    def predict(self, x, steps):
        trajectory = [x.detach()[0]]
        current = x.clone()
        
        for _ in range(steps):
            with torch.enable_grad():  # Enable gradients temporarily
                current = current.requires_grad_(True)
                next_step = self.forward(current,0.1)
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
            X_batch = X_train[i:i+batch_size].requires_grad_(True)
            y_batch = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            loss = model.risidualLoss(X_batch,y_batch,0.1)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss
        
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
    model = PINN().to(device)
    X_train, y_train, X_test = X_train.to(device), y_train.to(device), X_test.to(device)
    train(model, X_train, y_train, X_test)

