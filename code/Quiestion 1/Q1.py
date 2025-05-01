import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Standard symplectic matrix J
J = torch.tensor([[0, 1], [-1, 0]], dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Symplectic Modules ---
class LinearUp(nn.Module):
    """
    Symplectic linear module: y = [I, S; 0, I] @ x + b
    where S is symmetric.
    """
    def __init__(self, dim, bias=True):
        super().__init__()
        self.dim = dim  # half-dimension d
        # S: symmetric matrix parameter
        self.S = nn.Parameter(torch.zeros(dim, dim))
        if bias:
            self.b = nn.Parameter(torch.zeros(2*dim))
        else:
            self.register_parameter('b', None)
    
    def forward(self, x):
        # x: (batch, 2d) split into (p, q)
        p, q = torch.split(x, self.dim, dim=1)
        # y1 = p + S @ q; y2 = q
        # Ensure S is symmetric
        S_symmetric = 0.5 * (self.S + self.S.t())
        y1 = p + q @ S_symmetric
        y2 = q
        y = torch.cat([y1, y2], dim=1)
        return y + (self.b if self.b is not None else 0)

class LinearLow(nn.Module):
    """
    Symplectic linear module: y = [I, 0; S, I] @ x + b
    """
    def __init__(self, dim, bias=True):
        super().__init__()
        self.dim = dim
        self.S = nn.Parameter(torch.zeros(dim, dim))
        if bias:
            self.b = nn.Parameter(torch.zeros(2*dim))
        else:
            self.register_parameter('b', None)
    
    def forward(self, x):
        p, q = torch.split(x, self.dim, dim=1)
        # Ensure S is symmetric
        S_symmetric = 0.5 * (self.S + self.S.t())
        y1 = p
        y2 = q + p @ S_symmetric
        y = torch.cat([y1, y2], dim=1)
        return y + (self.b if self.b is not None else 0)

class ActivationUp(nn.Module):
    """
    Symplectic activation module: y = [I, diag(a) * sigma(); 0, I] @ x
    """
    def __init__(self, dim, act=nn.Tanh()):
        super().__init__()
        self.dim = dim
        self.act = act
        self.a = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        p, q = torch.split(x, self.dim, dim=1)
        y1 = p + self.act(q) * self.a
        y2 = q
        return torch.cat([y1, y2], dim=1)

class ActivationLow(nn.Module):
    """
    Symplectic activation module: y = [I, 0; diag(a) * sigma(), I] @ x
    """
    def __init__(self, dim, act=nn.Tanh()):
        super().__init__()
        self.dim = dim
        self.act = act
        self.a = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        p, q = torch.split(x, self.dim, dim=1)
        y1 = p
        y2 = q + self.act(p) * self.a
        return torch.cat([y1, y2], dim=1)

class GradientUp(nn.Module):
    """
    Symplectic gradient module: y = [I, K^T diag(a) sigma(Kx+b); 0, I] @ x
    """
    def __init__(self, dim, hidden):
        super().__init__()
        self.dim = dim
        self.K = nn.Parameter(torch.randn(hidden, dim) / np.sqrt(hidden))  # Xavier init
        self.b = nn.Parameter(torch.zeros(hidden))
        self.a = nn.Parameter(torch.ones(hidden))
        self.act = nn.Tanh()
    
    def forward(self, x):
        p, q = torch.split(x, self.dim, dim=1)
        # nonlinear gradient
        g = self.act(q @ self.K.t() + self.b) * self.a  # Fixed: should be q not p
        y1 = p + g @ self.K
        y2 = q
        return torch.cat([y1, y2], dim=1)

class GradientLow(nn.Module):
    """
    Symplectic gradient module: y = [I, 0; K^T diag(a) sigma(Kx+b), I] @ x
    """
    def __init__(self, dim, hidden):
        super().__init__()
        self.dim = dim
        self.K = nn.Parameter(torch.randn(hidden, dim) / np.sqrt(hidden))  # Xavier init
        self.b = nn.Parameter(torch.zeros(hidden))
        self.a = nn.Parameter(torch.ones(hidden))
        self.act = nn.Tanh()
    
    def forward(self, x):
        p, q = torch.split(x, self.dim, dim=1)
        g = self.act(p @ self.K.t() + self.b) * self.a  # Fixed: should be p not q
        y1 = p
        y2 = q + g @ self.K
        return torch.cat([y1, y2], dim=1)

# --- SympNet Composition ---
class SympNet(nn.Module):
    """
    Symplectic Neural Network composed of alternating modules.
    Example structure: [LinearUp, ActivationLow, LinearLow, ActivationUp] x N
    """
    def __init__(self, dim, hidden_dim=64, num_blocks=4):
        super().__init__()
        self.dim = dim
        
        # Build network with clearer structure
        layers = []
        for _ in range(num_blocks):
            layers.extend([
                LinearUp(dim), 
                ActivationLow(dim),
                GradientUp(dim, hidden_dim),
                LinearLow(dim),
                GradientLow(dim, hidden_dim),
                ActivationUp(dim)
            ])
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (batch, 2*dim) = [p, q]
        output = self.net(x)
        return torch.split(output, self.dim, dim=1)  # Fixed split dimension
    
    def predict(self, x, steps):
        """Generate a trajectory for a given number of steps"""
        trajectory = [x.detach().clone()]  # Store initial state as full state vector
        current = x.clone()
        
        for _ in range(steps):
            # Forward pass with appropriate gradient handling
            current = current.detach().requires_grad_(True)
            next_p, next_q = self.forward(current)
            next_state = torch.cat([next_p, next_q], dim=1)
            
            # Store full state
            trajectory.append(next_state.detach().clone())
            current = next_state
            
        return torch.cat(trajectory, dim=0)  # Return as batch

def load_data(train_path='../../data/train.txt', test_path='../../data/test.txt'):
    """Load and preprocess data with error handling"""
    try:
        # Load training data
        with open(train_path, 'r') as f:
            train_lines = [list(map(float, line.strip().split())) for line in f if line.strip()]
        
        # Load testing data
        with open(test_path, 'r') as f:
            test_lines = [list(map(float, line.strip().split())) for line in f if line.strip()]
        
        # Process data
        train_data = torch.tensor(train_lines[:1200], dtype=torch.float32)
        trainP_data = torch.tensor(train_lines[1:1201], dtype=torch.float32)
        test_data = torch.tensor(test_lines[:300], dtype=torch.float32)
        
        # Normalize data
        train_data = F.normalize(train_data, p=2, dim=1)
        test_data = F.normalize(test_data, p=2, dim=1)
        trainP_data = F.normalize(trainP_data, p=2, dim=1)
        
        return train_data, trainP_data, test_data
    
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(f"Please check that the data files exist at: {train_path} and {test_path}")
        raise

def evaluate(model, X_test, steps=300):
    """Evaluate model on test data"""
    model.eval()
    initial = X_test[0:1].clone().to(device)
    ground_truth = X_test[:steps+1].cpu()
    
    with torch.no_grad():
        predicted = model.predict(initial, steps).cpu()
    
    # Match dimensions for comparison
    pred_len = min(len(predicted), len(ground_truth))
    return F.mse_loss(predicted[:pred_len], ground_truth[:pred_len])

def plot(ground_truth, predicted_trajectory, save_path='test.png', show=True):
    """Plot the ground truth vs predicted trajectory"""
    if os.path.exists(save_path):
        os.remove(save_path)
        
    plt.figure(figsize=(8, 6))
    
    # Extract position variables (assuming they're indices 2 and 3)
    # Check dimensions to make sure we're plotting correctly
    if len(predicted_trajectory.shape) == 2:
        # Direct trajectory plot - get positions
        gt_positions = ground_truth[:, 2:4].cpu().numpy()
        pred_positions = predicted_trajectory[:, 2:4].cpu().detach().numpy()
    else:
        # Handle case where trajectory might be batched differently
        gt_positions = ground_truth[:, 2:4].cpu().numpy()
        pred_positions = predicted_trajectory[:, 2:4].cpu().detach().numpy()
    
    # Ground truth trajectory
    plt.plot(
        gt_positions[:, 0], gt_positions[:, 1],
        'b-', label='Ground Truth', linewidth=2
    )
    
    # Predicted trajectory
    plt.plot(
        pred_positions[:, 0], pred_positions[:, 1],
        'r--', label='PNN Prediction', linewidth=2
    )
    
    # Start point
    plt.scatter(
        gt_positions[0, 0], gt_positions[0, 1],
        c='green', s=100, label='Start Point'
    )
    
    plt.xlabel('Position $x_1$', fontsize=14)
    plt.ylabel('Position $x_2$', fontsize=14)
    plt.title('Charged Particle Trajectory Prediction', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.axis('equal')  # Maintains spatial proportions
    plt.tight_layout()
    
    try:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
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


def train(model, X_train, y_train, X_test, epochs=5000000, lr=0.001, batch_size=512, 
          save_path='best_model.pth', patience=10):
    """Train the model with improved techniques"""
    model = model.to(device)
    X_train, y_train, X_test = X_train.to(device), y_train.to(device), X_test.to(device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Early stopping variables
    best_loss = float('inf')
    no_improvement = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Shuffle data
        perm = torch.randperm(len(X_train))
        X_train_shuffled, y_train_shuffled = X_train[perm], y_train[perm]
        
        # Batch training
        for i in range(0, len(X_train_shuffled), batch_size):
            X_batch = X_train_shuffled[i:i+batch_size].requires_grad_(True)
            y_batch = y_train_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_p, pred_q = model(X_batch)
            
            # Calculate loss
            p_loss = F.mse_loss(pred_p, y_batch[:, :model.dim])
            q_loss = F.mse_loss(pred_q, y_batch[:, model.dim:])
            loss = p_loss + q_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation every 100 epochs
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = evaluate(model, X_test)
                
                # Update scheduler
                scheduler.step(test_loss)
                
                # Save model if improved
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(model.state_dict(), save_path)
                    no_improvement = 0
                    
                    # Plot trajectory
                    pred_trajectory = model.predict(X_test[0:1].clone().to(device), 299)
                    mse_T = F.mse_loss(pred_trajectory,X_test[:300],reduction='none').mean(dim=1)
                    plotMSE(mse_T,save_path=f'MSE_{epoch}.png')
                    plot(X_test[:300], pred_trajectory, save_path=f'trajectory_epoch_{epoch}.png')
                    print(f"New best model saved. Test Loss: {test_loss:.6f}")
                #else:
                    #no_improvement += 1
            
            # Print training progress
            print(f"Epoch {epoch} | Train Loss: {epoch_loss:.6f} | Test Loss: {test_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Configure paths (adjust these)
    train_path = '../data/train.txt'
    test_path = '../data/test.txt'
    
    try:
        # Load data
        X_train, y_train, X_test = load_data(train_path, test_path)
        
        # Create model
        model = SympNet(dim=2, hidden_dim=128, num_blocks=4)
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train model
        train(model, X_train, y_train, X_test, epochs=100000, patience=20)
        
        # Final evaluation
        best_model = SympNet(dim=2, hidden_dim=128, num_blocks=4)
        best_model.load_state_dict(torch.load('best_model.pth'))
        best_model = best_model.to(device)
        
        final_loss = evaluate(best_model, X_test.to(device))
        print(f"Final test loss: {final_loss:.6f}")
        
        # Final trajectory visualization
        pred_trajectory = best_model.predict(X_test[0:1].clone().to(device), 299)
        plot(X_test[:300], pred_trajectory, save_path='final_trajectory.png')
        
    except Exception as e:
        print(f"Error during execution: {e}")
