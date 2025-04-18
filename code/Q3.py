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

        # The neural network
        self.H_net = nn.Sequential(
            nn.Linear(4 , hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4)
        )

        # 4 learnable parameters
        self.lambda1 = nn.Parameter(torch.randn(1) * 0.1 + 0.5).to(device)
        self.lambda2 = nn.Parameter(torch.randn(1) * 0.1 + 0.5).to(device)
        self.lambda3 = nn.Parameter(torch.randn(1) * 0.1 + 0.5).to(device)
        self.lambda4 = nn.Parameter(torch.randn(1) * 0.1 + 0.5).to(device)

        # Load IRK weights from file
        file_path = f'../data/Utilities/IRK_weights/Butcher_IRK{q}.txt'
        tmp = torch.from_numpy(np.loadtxt(file_path, ndmin=2).astype(np.float32)).to(device)

        # Extract and reshape the weights
        weights = tmp[:q**2 + q].reshape(q + 1, q)
        self.IRK_alpha = weights[:-1, :].to(device)  # shape (q, q)
        self.IRK_beta = weights[-1:, :].to(device)   # shape (1, q)
        self.IRK_times = tmp[q**2 + q:].to(device)   # shape (q,)

    def forward(self, X0, dt=0.1):
        batch_size = X0.shape[0]

        # Fix: Ensure X0_expanded is correctly shaped for repeat
        X0_expanded = X0.squeeze(1).expand(-1, self.IRK_alpha.shape[0], -1)  # (B, q, 4)

        X_input = X0_expanded.view(-1, 4).to(device)

        H_out = self.H_net(X_input)  # (B * q, 4)

        H_out = H_out.view(batch_size, self.IRK_alpha.shape[0], 4).to(device)

        stage_sum = torch.einsum('bqi,ij->bqj', H_out, self.IRK_alpha).to(device)  # (B, q, 4)

        X_stages = X0.unsqueeze(1) + dt * stage_sum  # (B, q, 4)

        final_sum = torch.einsum('bqi,ij->bqj', H_out, self.IRK_beta).to(device)  # (B, 1, 4)
        X_pred = X0 + dt * final_sum.squeeze(1)  # (B, 4)

        return X_pred

    def compute_residual_loss(self, X0, dt):
        B = X0.shape[0]
        q = self.IRK_alpha.shape[0]

        X0_expanded = X0.unsqueeze(1).expand(-1, q, -1).to(device)  # (B, q, 4)
        X0_expanded.requires_grad_(True)

        input_stages = X0_expanded.reshape(-1, 4).to(device)  # (B * q, 4)

        H = self.H_net(input_stages)  # (B * q, 4)
        H = H.view(B, q, 4).to(device)

        H_pred = X0.unsqueeze(1) + dt * torch.matmul(self.IRK_alpha, H)  # (B, q, 4)

        H_pred_flat = H_pred.view(-1, 4)
        H_new = self.H_net(H_pred_flat).view(B, q, 4).to(device)

        X1_pred = X0 + dt * torch.matmul(self.IRK_beta, H_new).squeeze(1)  # (B, 4)

        residual = X1_pred - (X0 + dt * torch.matmul(self.IRK_beta, H).squeeze(1))

        loss = torch.mean(residual ** 2)
        print(f"Residual Loss: {loss.item()}")  # Debugging step to check residual loss

        return loss

    def predict(self, x, steps):
        trajectory = [x]
        for _ in range(steps):
            x = x.clone().detach().requires_grad_(True).to(device)
            x = self.forward(x)
            trajectory.append(x)
        return torch.stack(trajectory).squeeze(1)

# Evaluation function with proper plotting
def evaluate_and_plot(model, X_test, steps=300):
    model.eval()
    
    # Get initial point for trajectory prediction
    initial_state = X_test[0:1].clone().detach().requires_grad_(True).to(device)
    # Ground truth trajectory from test data
    ground_truth = X_test[:steps+1].to(device)
    
    # Generate multi-step prediction
    predicted_trajectory = model.predict(initial_state, steps)
    
    # Plot position components (assumed to be last two dimensions)
    plt.figure(figsize=(12, 8))
    
    # Ground truth (blue)
    plt.plot(ground_truth[:, 2].cpu().numpy(), ground_truth[:, 3].cpu().numpy(), 
             'b-', label='Ground Truth', linewidth=2)
    
    # Prediction (red)
    plt.plot(predicted_trajectory[:, 2].cpu().detach().numpy(), predicted_trajectory[:, 3].cpu().detach().numpy(), 
             'r--', label='PNN Prediction', linewidth=2)
    
    # Highlight start point
    plt.scatter(ground_truth[0, 2].cpu().numpy(), ground_truth[0, 3].cpu().numpy(), 
               c='green', s=100, label='Start Point')
    
    plt.xlabel('Position x1', fontsize=14)
    plt.ylabel('Position x2', fontsize=14)
    plt.title('Charged Particle Trajectory Prediction', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig('hello.png')
    plt.show()
    
    # Calculate and plot MSE over time
    mse_over_time = []
    for t in range(min(len(ground_truth)-1, len(predicted_trajectory)-1)):
        mse = F.mse_loss(predicted_trajectory[t], ground_truth[t]).item()
        mse_over_time.append(mse)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(mse_over_time)
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Log MSE', fontsize=14)
    plt.title('Prediction Error Over Time', fontsize=16)
    plt.grid(True)
    plt.savefig('prediction_error.png')
    
    # Calculate average MSE
    avg_mse = sum(mse_over_time) / len(mse_over_time)
    print(f"Average MSE over {len(mse_over_time)} time steps: {avg_mse:.6f}")
    
    return avg_mse, predicted_trajectory

if __name__ == "__main__":
    # Load training data
    with open('../data/train.txt', 'r') as f:
        data = [list(map(float, line.strip().split())) for line in f if line.strip()]
    tensor_data = torch.tensor(data, dtype=torch.float32).T.to(device)
    train_p = tensor_data[:2, :1200].T  # momentum (v1, v2)
    train_q = tensor_data[2:, :1200].T  # position (x1, x2)
    target_p = tensor_data[:2, 1:1201].T
    target_q = tensor_data[2:, 1:1201].T

    # Load testing data
    with open('../data/test.txt', 'r') as f:
        data = [list(map(float, line.strip().split())) for line in f if line.strip()]
    tensor_data = torch.tensor(data, dtype=torch.float32).T.to(device)
    test_p = tensor_data[:2, :].T
    test_q = tensor_data[2:, :].T

    # Create training and target tensors
    X_train = torch.cat([train_p, train_q], dim=1).to(device)
    y_train = torch.cat([target_p, target_q], dim=1).to(device)

    # Create test tensors
    X_test = torch.cat([test_p, test_q], dim=1).to(device)
    y_test = X_test[1:].clone().to(device)  # The target is the next state in the sequence

    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize PNN model
    pnn = PINNS(128).to(device)

    # Hyperparameters
    epochs = 100
    batch_size = 64
    learning_rate = 1e-3
    dt = 0.1  # timestep for IRK

    # DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(pnn.parameters(), lr=learning_rate)

    # Training loop
    print("Starting training...")

    for epoch in range(epochs):
        pnn.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            loss = pnn.compute_residual_loss(X_batch, dt)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # Evaluate model
    avg_mse, predicted_trajectory = evaluate_and_plot(pnn, X_test)
    print(f"Average MSE: {avg_mse}")
