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
            nn.ReLU(),
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
    def __init__(self, latent_dim, active_dim=4, hidden_dim=64,dropout=0.3):
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
        )

        self.S = nn.Parameter(torch.zeros(self.active_dim, self.active_dim))
        torch.nn.init.normal_(self.S, 0, 0.1)

        self.dt_q = nn.Parameter(torch.randn(1) * 0.1 + 0.5)
        self.dt_p = nn.Parameter(torch.randn(1) * 0.1 + 0.5)

        self.alpha = nn.Parameter(torch.tensor(0.0001))

    def forward(self, z, dt=0.1):
        # Ensure we maintain the computation graph
        z = z.clone().requires_grad_(True)
        z_active = z[:, :self.active_dim]  # z1, z2
        z_aux = z[:, self.active_dim:]     # auxiliary vars

        z1 = z_active[:, :2].requires_grad_(True)
        z2 = z_active[:, 2:].requires_grad_(True)

        z_combined = torch.cat([z1, z2, z_aux], dim=1)
        H = self.H_net(z_combined).sum()

        # Compute partial derivatives ∂H/∂z1 and ∂H/∂z2
        dHdz1 = torch.autograd.grad(H, z1, retain_graph=True, create_graph=True)[0]
        dHdz2 = torch.autograd.grad(H, z2, retain_graph=True, create_graph=True)[0]

        # Enforce skew-symmetric structure on S
        S = self.S - self.S.t()

        dz1 = dHdz2 * self.dt_q + self.alpha * (z_active @ S.t())[:, :2]
        dz2 = -dHdz1 * self.dt_p + self.alpha * (z_active @ S)[:, 2:]

        z_active_new = z_active + dt * torch.cat([dz1, dz2], dim=1)
        z_new = torch.cat([z_active_new, z_aux], dim=1)

        return z_new



    def enforce_symplecticity(self):
        with torch.no_grad():
            self.S.data = 0.5 * (self.S - self.S.t())
            self.dt_q.data.abs_()
            self.dt_p.data.abs_()
        self.S.to(device)


class PNN(nn.Module):
    def __init__(self,dropout=0.3):
        super().__init__()
        self.transformer = NICECouplingLayer(4, 125)
        self.sympNet = ExtendedSympNet(4,dropout=0.3)
        self.lowestLoss = 1e20

    def forward(self, x):
        theta = self.transformer.forward(x)
        phi = self.sympNet.forward(theta)
        thetaI = self.transformer.inverse(phi)
        return thetaI

    def predict(self, x, steps):
        trajectory = [x]  # Keep full tensor on GPU
        current = x.clone()

        for _ in range(steps):
            with torch.enable_grad():
                current = current.requires_grad_(True)
                next_step = self.forward(current)
            trajectory.append(next_step)  # Keep on GPU
            current = next_step.detach()  # Still on GPU
        return torch.cat(trajectory)  # Stack on GPU

    #def predict(self, x, steps):
    #    trajectory = [x.detach().cpu()]
    #    x = x.clone().requires_grad_(True)
    #    
    #    for _ in range(steps):
    #        with torch.set_grad_enabled(True):
    #            x = self.forward(x)
    #            trajectory.append(x.detach().cpu())
    #    
    #    return torch.stack(trajectory).squeeze(1)

def train_pnn(model, X_train, y_train, test=None, epochs=100, lr=0.0001):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5)
    
    best_test_mse = float('inf')
    
    for epoch in range(epochs):
        # Use all data with batching instead of sampling
        batch_size = 256
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            pred_y = model.forward(X_batch)
            
            pred_p, pred_q = pred_y[:, :2], pred_y[:, 2:]
            true_p, true_q = y_batch[:, :2], y_batch[:, 2:]
            
            loss_p = F.mse_loss(pred_p, true_p)
            loss_q = F.mse_loss(pred_q, true_q)
            mse_loss = loss_p + loss_q
            
            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Enforce symplecticity more frequently
        if epoch % 10 == 0:
            model.sympNet.enforce_symplecticity()
        
        # Evaluation
        if epoch % 50 == 0 or epoch < 1000 and epoch % 10 == 0:
            with torch.no_grad():
                test_mse, _ = evaluate_and_plot(model, test)
                scheduler.step(test_mse)
                
                if test_mse < best_test_mse:
                    best_test_mse = test_mse
                    torch.save(model.state_dict(), 'best_model.pt')
                    
                print(f"Epoch {epoch}, Train Loss: {mse_loss.item():.6f}, Test MSE: {test_mse:.6f}, Best: {best_test_mse:.6f}")
    
    return model


# Evaluation function with proper plotting
def evaluate_and_plot(model, X_test, steps=300):
    model.eval()
    
    # Get initial point for trajectory prediction
    initial_state = X_test[0:1].clone().detach().requires_grad_(True).to(device)
    # Ground truth trajectory from test data
    ground_truth = X_test[:steps+1]
    
    # Generate multi-step prediction
    predicted_trajectory = model.predict(initial_state, steps)
    
    # Plot position components (assumed to be last two dimensions)
    #plt.figure(figsize=(12, 8))
    
    # Ground truth (blue)
    #plt.plot(ground_truth[:, 2].cpu().numpy(), ground_truth[:, 3].cpu().numpy(), 
    #         'b-', label='Ground Truth', linewidth=2)
    
    # Prediction (red)
    #plt.plot(predicted_trajectory[:, 2].cpu().detach().numpy(), predicted_trajectory[:, 3].cpu().detach().numpy(), 
    #         'r--', label='PNN Prediction', linewidth=2)
    
    # Highlight start point
    #plt.scatter(ground_truth[0, 2].cpu().numpy(), ground_truth[0, 3].cpu().numpy(), 
    #           c='green', s=100, label='Start Point')
    
    #plt.xlabel('Position x1', fontsize=14)
    #plt.ylabel('Position x2', fontsize=14)
    #plt.title('Charged Particle Trajectory Prediction', fontsize=16)
    #plt.legend(fontsize=12)
    #plt.grid(True)
    #plt.savefig('hello.png')
    #plt.show()
    
    # Calculate and plot MSE over time
    mse_over_time = []
    for t in range(min(len(ground_truth)-1, len(predicted_trajectory)-1)):
        mse = F.mse_loss(predicted_trajectory[t], ground_truth[t]).item()
        mse_over_time.append(mse)
    
    #plt.figure(figsize=(10, 6))
    #plt.semilogy(mse_over_time)
    #plt.xlabel('Time Step', fontsize=14)
    #plt.ylabel('Log MSE', fontsize=14)
    #plt.title('Prediction Error Over Time', fontsize=16)
    #plt.grid(True)
    #plt.savefig('prediction_error.png')
    
    # Calculate average MSE
    avg_mse = sum(mse_over_time) / len(mse_over_time)
    print(f"avg: {avg_mse}, total: {sum(mse_over_time)}")


    if model.lowestLoss > avg_mse:
        model.lowestLoss = avg_mse
        torch.save(pnn.state_dict(), 'Q2Model.pt')
        print("saved")

    model.train()
    
    return avg_mse, predicted_trajectory

if __name__ == "__main__":
    # Load training data
    with open('../data/train.txt', 'r') as f:
        data = [list(map(float, line.strip().split())) for line in f if line.strip()]
    tensor_data = torch.tensor(data, dtype=torch.float32).T
    train_p = tensor_data[:2, :1200].T  # momentum (v1, v2)
    train_q = tensor_data[2:, :1200].T  # position (x1, x2)
    target_p = tensor_data[:2, 1:1201].T
    target_q = tensor_data[2:, 1:1201].T

    # Load testing data
    with open('../data/test.txt', 'r') as f:
        data = [list(map(float, line.strip().split())) for line in f if line.strip()]
    tensor_data = torch.tensor(data, dtype=torch.float32).T
    test_p = tensor_data[:2, :].T
    test_q = tensor_data[2:, :].T

    # Create training and target tensors
    X_train = torch.cat([train_p, train_q], dim=1)
    y_train = torch.cat([target_p, target_q], dim=1)

    # Create test tensors
    X_test = torch.cat([test_p, test_q], dim=1)
    y_test = X_test[1:].clone()  # The target is the next state in the sequence

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    # Initialize PNN model
    pnn = PNN().to(device)
    #pnn.load_state_dict(torch.load('Q2Model1.pt'))
    #pnn.eval()  # important for inference 
    # Train the model
    print("Starting training...")
    pnn = train_pnn(pnn, X_train, y_train,test = X_test, epochs=200000, lr=0.00001)
    print("Training complete!")
    
    # Evaluate and visualize results
    print("Evaluating model...")
    avg_mse, predicted_trajectory = evaluate_and_plot(pnn, X_test)
    
    print("Done!")
