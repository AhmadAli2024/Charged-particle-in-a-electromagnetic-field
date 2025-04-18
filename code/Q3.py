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
        self.H_net = nn.Sequential(
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

    def hamiltonian_vector_field(self, x):
        """
        Compute the Hamiltonian vector field.
        For a Hamiltonian system: dx/dt = J ∇H(x)
        where J is the symplectic matrix and H is the Hamiltonian function.
        
        Args:
            x: Input tensor of shape (batch_size, 4) representing [p1, p2, q1, q2]
        
        Returns:
            Vector field of shape (batch_size, 4)
        """
        # Split input into momentum (p) and position (q)
        p = x[:, :2]  # (p1, p2)
        q = x[:, 2:]  # (q1, q2)
        
        # Apply the H_net to get the output
        H_out = self.H_net(x)
        
        # Apply the symplectic structure
        # For a Hamiltonian system with p = (p1, p2) and q = (q1, q2):
        # dp/dt = -∂H/∂q
        # dq/dt = ∂H/∂p
        
        # Scale the outputs using the learnable parameters
        dp_dt = -self.lambda1 * H_out[:, 2:] - self.lambda2 * q
        dq_dt = self.lambda3 * H_out[:, :2] + self.lambda4 * p
        
        # Combine to form the vector field
        vector_field = torch.cat([dp_dt, dq_dt], dim=1)
        
        return vector_field

    def forward(self, X0, dt=0.1):
        """
        Perform one time step using the IRK method
        
        Args:
            X0: Initial state tensor of shape (batch_size, 4)
            dt: Time step size
            
        Returns:
            X1: Next state tensor of shape (batch_size, 4)
        """
        batch_size = X0.shape[0]
        q = self.IRK_alpha.shape[0]  # Number of stages in IRK method
        
        # Initialize stage vectors (K values in IRK method)
        K = torch.zeros(batch_size, q, 4, device=X0.device)
        
        # Fixed-point iteration to solve for the stages
        for _ in range(10):  # Usually 5-10 iterations are sufficient
            # Current estimate of stage points
            X_stages = X0.unsqueeze(1) + dt * torch.einsum('bqi,ij->bqj', K, self.IRK_alpha)
            
            # Compute vector field at stage points
            X_stages_flat = X_stages.reshape(-1, 4)
            vector_field = self.hamiltonian_vector_field(X_stages_flat).reshape(batch_size, q, 4)
            
            # Update stages
            K = vector_field
            
        # Compute final step using the converged stages
        X1 = X0 + dt * torch.einsum('bqi,ij->bj', K, self.IRK_beta)
        
        return X1

    def compute_residual_loss(self, X0, X1, dt):
        """
        Compute the physics-informed residual loss
        
        Args:
            X0: Initial state tensor of shape (batch_size, 4)
            X1: Target state tensor of shape (batch_size, 4)
            dt: Time step size
            
        Returns:
            Loss value
        """
        # Predict the next state
        X1_pred = self.forward(X0, dt)
        
        # Data loss - how well our prediction matches the target
        data_loss = F.mse_loss(X1_pred, X1)
        
        # Physics loss - conservation of Hamiltonian (energy)
        # In a perfect Hamiltonian system, energy should be conserved
        batch_size = X0.shape[0]
        q = self.IRK_alpha.shape[0]
        
        # Compute stages and their vector fields
        K = torch.zeros(batch_size, q, 4, device=X0.device)
        
        # One iteration to get rough estimates of the stages
        X_stages = X0.unsqueeze(1) + dt * torch.einsum('bqi,ij->bqj', K, self.IRK_alpha)
        X_stages_flat = X_stages.reshape(-1, 4)
        
        # Compute vector field and reshape
        vector_field = self.hamiltonian_vector_field(X_stages_flat).reshape(batch_size, q, 4)
        
        # The residual is how well the stages satisfy the IRK equations
        stage_sum = torch.einsum('bqi,ij->bqj', vector_field, self.IRK_alpha)
        X_stages_new = X0.unsqueeze(1) + dt * stage_sum
        
        # Physics-based residual
        physics_loss = F.mse_loss(X_stages_new, X_stages)
        
        # Total loss is a weighted sum of data and physics losses
        total_loss = data_loss + 0.1 * physics_loss
        
        return total_loss

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
    pnn = PINNS(hidden_dim=128).to(device)

    # Hyperparameters
    epochs = 500  # Increased from 100
    batch_size = 64
    learning_rate = 1e-3
    dt = 0.1  # timestep for IRK

    # DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(pnn.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

    # Training loop
    print("Starting training...")
    training_losses = []

    for epoch in range(1, epochs + 1):
        pnn.train()
        total_loss = 0.0

        for batch_idx, (X0, X1) in enumerate(train_loader):
            X0, X1 = X0.to(device), X1.to(device)

            optimizer.zero_grad()
            loss = pnn.compute_residual_loss(X0, X1, dt)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(pnn.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        training_losses.append(avg_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        if epoch % 10 == 0:
            print(f"[Epoch {epoch}/{epochs}] Loss: {avg_loss:.6f} | λ1={pnn.lambda1.item():.3f} λ2={pnn.lambda2.item():.3f} λ3={pnn.lambda3.item():.3f} λ4={pnn.lambda4.item():.3f}")
            
        # Early stopping check
        if epoch > 50 and avg_loss < 1e-6:
            print(f"Converged at epoch {epoch} with loss {avg_loss:.6f}")
            break

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
