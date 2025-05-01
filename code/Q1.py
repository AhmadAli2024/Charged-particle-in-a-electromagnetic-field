import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Symplectic Euler-based SympNet
class SympNet(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.f_q = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 2)
        )
        self.g_p = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, dt=0.1):
        p = x[:,:2]
        q = x[:,2:]
        p_new = p - dt * self.f_q(q)
        q_new = q + dt * self.g_p(p_new)
        return p_new, q_new

    def predict(self, x, steps):
        trajectory = [x]
        current = x.clone()
        
        for _ in range(steps):
            next_step = self.forward(current)
            trajectory.append(next_step[0])
            current = next_step
            
        return torch.stack(trajectory)


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

def load_data():
    # Load training data
    with open('../data/train.txt', 'r') as f:
        train_lines = [list(map(float, line.strip().split())) for line in f if line.strip()]
    
    # Load testing data
    with open('../data/test.txt', 'r') as f:
        test_lines = [list(map(float, line.strip().split())) for line in f if line.strip()]

    # Process data
    train_data = torch.tensor(train_lines[:1200], dtype=torch.float32)
    trainP_data = torch.tensor(train_lines[1:1201], dtype=torch.float32)
    test_data = torch.tensor(test_lines[:300], dtype=torch.float32)
    
    train_data = torch.nn.functional.normalize(train_data, p=2, dim=1)
    test_data = torch.nn.functional.normalize(test_data, p=2, dim=1)
    trainP_data = torch.nn.functional.normalize(trainP_data, p=2, dim=1)

    return train_data, trainP_data, test_data


def evaluate(model, X_test, steps=300):
    model.eval()
    initial = X_test[0:1].clone().to(device)
    ground_truth = X_test[:steps+1].cpu()
    
    with torch.no_grad():
        # Enable gradients temporarily for physics calculations
        with torch.enable_grad():
            predicted = model.predict(initial, steps-1).cpu()
    
    return F.mse_loss(predicted, ground_truth).sum()


def train(model, X_train, y_train, X_test, epochs=500000, lr=0.002):
    model = model.to(device)
    X_train, y_train, X_test = X_train.to(device), y_train.to(device), X_test.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,              # or your specific learning rate
        weight_decay=1e-5,    # small weight decay for stability
        betas=(0.9, 0.999),   # default AdamW momentum settings
        eps=1e-8              # numerical stability
    )

    best_loss = float('inf')
    batch_size = 512 
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Shuffle data
        perm = torch.randperm(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]
        
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size].requires_grad_(True)
            y_batch = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            pred_q, pred_p = model(X_batch, dt=0.1)
            loss = F.mse_loss(pred_q, y_batch[:,2:]) + F.mse_loss(pred_p, y_batch[:,:2])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 2000 == 0:
            if lr >= 0.0005:
                lr*=0.8

        # Validation
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = evaluate(model, X_test)
                
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(model.state_dict(), 'best_model.pth')
                    plot(X_test,model.predict(X_test[0:1].clone().to(device),299))
                    print("plotted")
                    
            print(f"Epoch {epoch} | Train Loss: {epoch_loss:.6f} | Test Loss: {test_loss:.6f} | LR: {lr}")



if __name__ == "__main__":
    X_train, y_train, X_test = load_data()
    model = SympNet().to(device)
    X_train, y_train, X_test = X_train.to(device), y_train.to(device), X_test.to(device)
    train(model, X_train, y_train, X_test)

