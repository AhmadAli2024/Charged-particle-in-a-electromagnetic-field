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

# Global variables
sigma_x = None
sigma_p = None

class PINN(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Lambda 1 being m and Lambda 2 being q
        self.register_parameter('log_lambda1', nn.Parameter(torch.log(torch.tensor(0.5))))
        self.register_parameter('log_lambda2', nn.Parameter(torch.log(torch.tensor(0.5))))

    @property
    def lambda1(self):
        return torch.exp(self.log_lambda1)

    @property
    def lambda2(self):
        return torch.exp(self.log_lambda2)

    def dynamicMatrix(self, X):
        batch_size = X.shape[0]
        x3, x4 = X[:, 2], X[:, 3]
        a = (x3 ** 2 + x4 ** 2) ** (1/2)  # shape: (batch_size,)
        b = self.lambda1 # This coresponds to 1/m
        c = self.lambda2 * b # This corespons to q/m^2 

        zeros = torch.zeros(batch_size, device=X.device)

        row0 = torch.stack([zeros, c * a, -b.expand_as(a), zeros], dim=1)
        row1 = torch.stack([-c * a, zeros, zeros, -b.expand_as(a)], dim=1)
        row2 = torch.stack([b.expand_as(a), zeros, zeros, zeros], dim=1)
        row3 = torch.stack([zeros, b.expand_as(a), zeros, zeros], dim=1)

        dynamic = torch.stack([row0, row1, row2, row3], dim=1)  # shape: (batch_size, 4, 4)
        return dynamic



    def computeDerivative(self, X, dt=0.1):
        # make X require grad
        X = X.requires_grad_()
        # Get the Hamiltonian
        H = self.model(X).squeeze()
        # Get the derivative of the Hamiltonian
        dH = torch.autograd.grad(H,X,grad_outputs=torch.ones_like(H),create_graph=True)[0] 
        dH_pos = dH[:,:2]  
        dH_neg = -dH[:,2:]  

        return torch.cat((dH_pos, dH_neg), dim=1)

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

    def risidualFunc(self,X,dt=0.1):
        # make X require grad
        X = X.requires_grad_()
        # Get the Hamiltonian
        H = self.model(X).squeeze()
        # Get the derivative of the Hamiltonian
        dH = torch.autograd.grad(H,X,grad_outputs=torch.ones_like(H),create_graph=True)[0] 
        dynamic = self.dynamicMatrix(X)
        return torch.bmm(dynamic, dH.unsqueeze(2)).squeeze(2)

    def Rrk4StepForward(self, X, dt=0.1):
        k1 = self.risidualFunc(X)
        k2 = self.risidualFunc(X + 0.5 * dt * k1)
        k3 = self.risidualFunc(X + 0.5 * dt * k2)
        k4 = self.risidualFunc(X + dt * k3)

        return X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def Rrk4StepBackwards(self, X, dt=0.1):
        k1 = self.risidualFunc(X)
        k2 = self.risidualFunc(X - 0.5 * dt * k1)
        k3 = self.risidualFunc(X - 0.5 * dt * k2)
        k4 = self.risidualFunc(X - dt * k3)

        return X - (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def risidualLoss(self,X,Y,dt):
        # Actual Loss
        XPredA = self.rk4StepForward(X,dt)
        YPredA = self.rk4StepBackwards(Y,dt)
        XpA = XPredA[:,:2]
        XqA = XPredA[:,2:]
        YpA = YPredA[:,:2]
        YqA = YPredA[:,2:]
        XPredLossA = F.mse_loss(XpA,Y[:,:2]) + F.mse_loss(XqA,Y[:,2:])
        YPredLossA = F.mse_loss(YpA,X[:,:2]) + F.mse_loss(YqA,X[:,2:])
        # Step forward and back
        XPred = self.Rrk4StepForward(X,dt)
        YPred = self.Rrk4StepBackwards(Y,dt)
        # split the position from velocity for better accuracy
        Xp = XPred[:,:2]
        Xq = XPred[:,2:]
        Yp = YPred[:,:2]
        Yq = YPred[:,2:]
        # Get all the losses
        XPredLoss = F.mse_loss(Xp,Y[:,:2]) + F.mse_loss(Xq,Y[:,2:])
        YPredLoss = F.mse_loss(Yp,X[:,:2]) + F.mse_loss(Yq,X[:,2:])
        return YPredLoss + XPredLoss + XPredLossA + YPredLossA

    def forward(self,X,dt):
        return self.rk4StepForward(X,dt)

    def learnedForward(self,X,dt):
        return self.Rrk4StepForward(X,dt)

    def predict(self, x, steps):
        trajectory = [x.detach()[0]]
        current = x.clone()
        
        for _ in range(steps):
            with torch.enable_grad():  # Enable gradients temporarily
                current = current.requires_grad_(True)
                next_step = self.learnedForward(current,0.1)
            trajectory.append(next_step.detach()[0])
            current = next_step.detach()
            
        return torch.stack(trajectory)


def load_data(
    train_path: str = '../../data/train.txt',
    test_path:  str = '../../data/test.txt',
    train_size: int = 1200,
    test_size:  int = 300,
    eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global sigma_x
    global sigma_p
    # Read raw lines
    with open(train_path, 'r') as f:
        raw_train = [list(map(float, line.split())) for line in f if line.strip()]
    with open(test_path, 'r') as f:
        raw_test  = [list(map(float, line.split())) for line in f if line.strip()]

    # Convert to tensors and slice
    train_data  = torch.tensor(raw_train[:train_size], dtype=torch.float32)
    trainP_data = torch.tensor(raw_train[1:train_size+1], dtype=torch.float32)
    test_data   = torch.tensor(raw_test[:test_size],  dtype=torch.float32)

    # Compute per-feature mean and std on training set
    #mean = train_data.mean(dim=0, keepdim=True)
    #std  = train_data.std(dim=0, keepdim=True) + eps

    #sigma_x = mean[0,:2]
    #sigma_p = mean[0,2:]

    # Apply normalization
    #train_data  = (train_data  - mean) / std
    #trainP_data = (trainP_data - mean) / std
    #test_data   = (test_data   - mean) / std

    return train_data, trainP_data, test_data

def train(model, X_train, y_train, X_test, epochs=500000, lr=0.001):
    global sigma_p
    global sigma_x

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
        if epoch % 50 == 0:
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
                    #lambda1_norm = model.lambda1.item()
                    #lambda2_norm = model.lambda2.item()
                    #lambda1_phys = lambda1_norm * (sigma_p**2 / sigma_x**2).mean().item()
                    #lambda2_phys = lambda2_norm * (sigma_p   / sigma_x).mean().item()
                
            pbar.set_description(f"Epoch {epoch} | "
                                 f"Train Loss: {epoch_loss:.6f} | "
                                 f"Test Loss: {test_loss:.3e} | "
                                 f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                                 f"m: {1/model.lambda1:.6f} | "
                                 f"q: {model.lambda2*((1/model.lambda1)**2):.6f}")


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

