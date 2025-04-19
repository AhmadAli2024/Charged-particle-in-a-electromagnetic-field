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

        # Deeper neural network with more hidden layers
        self.model = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4)
        )

        # 4 learnable parameters for the Hamiltonian system
        self.lambda1 = nn.Parameter(torch.randn(1) * 0.1 + 0.5)
        self.lambda2 = nn.Parameter(torch.randn(1) * 0.1 + 0.5)
        self.lambda3 = nn.Parameter(torch.randn(1) * 0.1 + 0.5)
        self.lambda4 = nn.Parameter(torch.randn(1) * 0.1 + 0.5)

        # Load IRK weights from file
        file_path = f'../data/Utilities/IRK_weights/Butcher_IRK{q}.txt'
        try:
            tmp = torch.from_numpy(np.loadtxt(file_path, ndmin=2).astype(np.float32))
            
            # Extract and reshape the weights
            weights = tmp[:q**2 + q].reshape(q + 1, q)
            self.IRK_alpha = weights[:-1, :].to(device)  # shape (q, q)
            self.IRK_beta = weights[-1:, :].to(device)   # shape (1, q)
            self.IRK_times = tmp[q**2 + q:].to(device)   # shape (q,)
        except FileNotFoundError:
            print(f"Warning: IRK weights file not found at {file_path}")
            print("Initializing with default values")
            # Initialize with some default values (for Gauss-Legendre IRK4)
            self.IRK_alpha = torch.tensor([[0.25, -0.0625], [0.25, 0.25]]).to(device)
            self.IRK_beta = torch.tensor([[0.5, 0.5]]).to(device)
            self.IRK_times = torch.tensor([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6]).to(device)

    def forward(self, X0, dt=0.1):
        return self.model(X0)

    def loss(self, true, pred, dt=0.1):
        Rv = true/(100*(true.pow(2).sum(dim=2)).pow(3/2,dim=1)) - (pred[:2]*dt)
        Rx = true[:2]-(pred[2:]*dt)
        Rl = lambda1*Rv + lambda2*Rx
        L = lambda3*((pred[:2]-true[:2]).pow(2,dim=1)) + lambda4((pred[2:]-true[2:]).pow(2,dim=1))
        return Rl+L

    def train_pinn(self, X_train, y_train, epochs=100, lr=0.001):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            perm = torch.randperm(X_train.size(0))
            X_train = X_train[perm]
            y_train = y_train[perm]
            X_train = X_train[:64]
            y_train = y_train[:64]

            pred_y = self.model.forward(X_train)


            self.loss(y_train,pred_y).backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            optimizer.step()

            # Optional symplecticity enforcement
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {mse_loss:.12f}")

        return self.model

    def predict(self, x, steps):
        """
        Predict a trajectory for multiple steps ahead
        
        Args:
            x: Initial state tensor of shape (1, 4)
            steps: Number of steps to predict
            
        Returns:
            Trajectory tensor of shape (steps+1, 4)
        """
        trajectory = [x]
        current_x = x
        
        for _ in range(steps):
            current_x = self.forward(current_x)
            trajectory.append(current_x)
            
        return torch.cat(trajectory, dim=0)


# Evaluation function with proper plotting
def evaluate_and_plot(model, X_test, steps=300):
    model.eval()
    
    # Get initial point for trajectory prediction
    initial_state = X_test[0:1].clone().detach().requires_grad_(True).to(device)
    
    # Ground truth trajectory from test data
    ground_truth = X_test[:steps+1]
    
    # Generate multi-step prediction
    with torch.no_grad():
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
    plt.savefig('trajectory_prediction.png')
    plt.show()
    
    # Calculate and plot MSE over time
    mse_over_time = []
    for t in range(min(len(ground_truth), len(predicted_trajectory))):
        mse = F.mse_loss(predicted_trajectory[t:t+1], ground_truth[t:t+1]).item()
        mse_over_time.append(mse)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(mse_over_time)
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Log MSE', fontsize=14)
    plt.title('Prediction Error Over Time', fontsize=16)
    plt.grid(True)
    plt.savefig('prediction_error.png')
    plt.show()
    
    # Calculate average MSE
    avg_mse = sum(mse_over_time) / len(mse_over_time)
    print(f"Average MSE over {len(mse_over_time)} time steps: {avg_mse:.6f}")
    
    return avg_mse, predicted_trajectory

if __name__ == "__main__":
    # Load training data
    try:
        with open('../data/train.txt', 'r') as f:
            data = [list(map(float, line.strip().split())) for line in f if line.strip()]
        tensor_data = torch.tensor(data, dtype=torch.float32).T
        train_p = tensor_data[:2, :1200].T  # momentum (p1, p2)
        train_q = tensor_data[2:, :1200].T  # position (q1, q2)
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
        
    except FileNotFoundError:
        print("Warning: Data files not found. Creating synthetic data for testing.")
        # Create synthetic data for testing if files are not found
        # Simple harmonic oscillator
        t = torch.linspace(0, 30, 3000)
        p = torch.sin(t).unsqueeze(1)
        q = torch.cos(t).unsqueeze(1)
        
        # Create 2D version
        p2 = torch.sin(t + 0.5).unsqueeze(1)
        q2 = torch.cos(t + 0.5).unsqueeze(1)
        
        # Combine into training and test data
        X_full = torch.cat([p, p2, q, q2], dim=1)
        
        # Split into train and test
        X_train = X_full[:2000]
        y_train = X_full[1:2001]
        X_test = X_full[2000:]
    
    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize PNN model
    pinn = PINNS(hidden_dim=128).to(device)

    # Hyperparameters
    epochs = 500  # Increased from 100
    batch_size = 64
    learning_rate = 0.1
    dt = 0.1  # timestep for IRK

    # Optimizer
    optimizer = torch.optim.Adam(pinn.parameters(), lr=learning_rate)

    # Training loop
    print("Starting training...")
    training_losses = []
    pinn = pinn.train_pinn(X_train,y_train)


    print("Training complete!")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.semilogy(training_losses)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Log Loss', fontsize=14)
    plt.title('Training Loss', fontsize=16)
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    # Evaluate and visualize results
    print("Evaluating model...")
    avg_mse, predicted_trajectory = evaluate_and_plot(pnn, X_test)
    
    # Save the model
    torch.save(pnn.state_dict(), 'pinns_model.pth')
    
    print("Done!")
