import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PINNS(nn.Module):
    def __init__(self, hidden_dim, q=4):
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

    def forward(self, X):
        return self.model(X)

    def loss(self, X, y, pred, dt=0.1):
        # Current state (t)
        v_current = X[:, :2]
        x_current = X[:, 2:]
        
        # Next state (t+dt) - ground truth
        v_next_true = y[:, :2]
        x_next_true = y[:, 2:]
        
        # Predicted next state (t+dt)
        v_next_pred = pred[:, :2]
        x_next_pred = pred[:, 2:]
        
        # Compute Coulomb force using current position
        x_norm = torch.norm(x_current, dim=1, keepdim=True) + 1e-8
        coulomb_term = x_current / (x_norm**3)
        
        # Compute derivatives using current and next state
        dvdt = (v_next_pred - v_current) / dt
        dxdt = (x_next_pred - x_current) / dt
        
        # Physics residuals
        Rv = coulomb_term - dvdt  # F = ma residual
        Rx = v_current - dxdt     # dx/dt = v residual
        
        # Loss components with softplus to ensure positivity
        lambda1 = F.softplus(self.lambda1)
        lambda2 = F.softplus(self.lambda2)
        lambda3 = F.softplus(self.lambda3)
        lambda4 = F.softplus(self.lambda4)
        
        physics_loss = lambda1 * torch.mean(Rv**2) + lambda2 * torch.mean(Rx**2)
        data_loss = lambda3 * F.mse_loss(v_next_pred, v_next_true) + lambda4 * F.mse_loss(x_next_pred, x_next_true)
        
        total_loss = physics_loss + data_loss
        return total_loss

    def train_pinn(self, X_train, y_train, epochs=10000, lr=0.001, dt=0.1, batch_size=64):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        training_losses = []
        
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred_y = self.forward(batch_X)
                loss = self.loss(batch_X, batch_y, pred_y, dt=dt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(loader)
            training_losses.append(epoch_loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")
        
        return training_losses

    def predict(self, x0, steps, dt=0.1):
        trajectory = [x0]
        current_state = x0.clone()
        for _ in range(steps):
            next_state = self.forward(current_state)
            trajectory.append(next_state)
            current_state = next_state.detach()  # Detach to prevent gradient accumulation
        return torch.stack(trajectory)

def evaluate_and_plot(model, X_test, steps=300):
    initial_state = X_test[0:1].to(device)
    
    # Generate predictions
    with torch.no_grad():
        predicted = model.predict(initial_state, steps)
    
    # Ground truth (assuming X_test contains sequential states)
    ground_truth = X_test[:steps+1]
    
    # Plot trajectories
    plt.figure(figsize=(12, 6))
    plt.plot(ground_truth[:, 2].cpu(), ground_truth[:, 3].cpu(), 'b-', label='Ground Truth')
    plt.plot(predicted[:, 0, 2].cpu(), predicted[:, 0, 3].cpu(), 'r--', label='Prediction')
    plt.scatter(ground_truth[0, 2].cpu(), ground_truth[0, 3].cpu(), c='g', s=100, label='Start')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Trajectory Comparison')
    plt.savefig('trajectory.png')
    plt.show()
    
    # Calculate MSE
    mse = F.mse_loss(predicted[:len(ground_truth), 0], ground_truth).item()
    print(f"Trajectory MSE: {mse:.4e}")
    return mse

if __name__ == "__main__":
    # Load data
    try:
        train_data = np.loadtxt('../data/train.txt')
        test_data = np.loadtxt('../data/test.txt')
        
        # Convert to PyTorch tensors
        X_train = torch.tensor(train_data[:-1], dtype=torch.float32)
        y_train = torch.tensor(train_data[1:], dtype=torch.float32)
        X_test = torch.tensor(test_data, dtype=torch.float32)
        
    except FileNotFoundError:
        print("Generating synthetic data...")
        t = torch.linspace(0, 20, 2000)
        x = torch.stack([
            torch.sin(t) + 0.1*torch.randn(len(t)),
            torch.cos(t) + 0.1*torch.randn(len(t)),
            torch.sin(t) * 0.5,
            torch.cos(t) * 0.5
        ], dim=1)
        X_train = x[:-1000]
        y_train = x[1:-999]
        X_test = x[-1000:]
    
    # Move data to device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test = X_test.to(device)

    # Initialize and train model
    pinn = PINNS(hidden_dim=128).to(device)
    losses = pinn.train_pinn(X_train, y_train, epochs=10000, lr=0.001)
    
    # Plot training loss
    plt.semilogy(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.savefig('training_loss.png')
    plt.show()
    
    # Evaluate
    test_mse = evaluate_and_plot(pinn, X_test)
    torch.save(pinn.state_dict(), 'charged_particle_pinn.pth')
    print(f"Test MSE: {test_mse:.4e}")
