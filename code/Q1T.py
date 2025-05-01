import torch
import torch.nn as nn
import torch.nn.functional as F

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
        y1 = p + q @ self.S.t()
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
        y1 = p
        y2 = q + p @ self.S.t()
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
        self.K = nn.Parameter(torch.randn(hidden, dim))
        self.b = nn.Parameter(torch.zeros(hidden))
        self.a = nn.Parameter(torch.ones(hidden))
        self.act = nn.Tanh()

    def forward(self, x):
        p, q = torch.split(x, self.dim, dim=1)
        # nonlinear gradient
        g = self.act(p @ self.K.t() + self.b) * self.a
        y1 = p
        y2 = q + g @ self.K
        return torch.cat([y1, y2], dim=1)

class GradientLow(nn.Module):
    """
    Symplectic gradient module: y = [I, 0; K^T diag(a) sigma(Kx+b), I] @ x
    """
    def __init__(self, dim, hidden):
        super().__init__()
        self.dim = dim
        self.K = nn.Parameter(torch.randn(hidden, dim))
        self.b = nn.Parameter(torch.zeros(hidden))
        self.a = nn.Parameter(torch.ones(hidden))
        self.act = nn.Tanh()

    def forward(self, x):
        p, q = torch.split(x, self.dim, dim=1)
        g = self.act(q @ self.K.t() + self.b) * self.a
        y1 = p + g @ self.K
        y2 = q
        return torch.cat([y1, y2], dim=1)

# --- SympNet Composition ---

class SympNet(nn.Module):
    """
    Original Symplectic Neural Network composed of alternating modules.
    Example structure: [LinearUp, ActivationLow, LinearLow, ActivationUp] x N
    """
    def __init__(self, dim, depth=4, hidden=64):
        super().__init__()
        layers = []
        for i in range(depth):
            layers += [LinearUp(dim), ActivationLow(dim), GradientUp(dim,128),
                       LinearLow(dim),GradientLow(dim,128), ActivationUp(dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 2*dim) = [p, q]
        return torch.split(self.net(x),2,dim=1)

    def predict(self, x, steps):
        trajectory = [x.detach()[0]]
        current = x.clone()
        
        for _ in range(steps):
            with torch.enable_grad():  # Enable gradients temporarily
                current = current.requires_grad_(True)
                next_stepP, next_stepQ = self.forward(current)
            trajectory.append(next_stepQ.detach())
            current = torch.cat((next_stepP,next_stepQ),dim=1)
            
        return torch.stack(trajectory)


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
            pred_q, pred_p = model(X_batch)
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
                print(test_loss)
                
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(model.state_dict(), 'best_model.pth')
                    plot(X_test,model.predict(X_test[0:1].clone().to(device),299))
                    print("plotted")
                    
            print(f"Epoch {epoch} | Train Loss: {epoch_loss:.6f} | Test Loss: {test_loss:.6f} | LR: {lr}")



if __name__ == "__main__":
    X_train, y_train, X_test = load_data()
    model = SympNet(2,128).to(device)
    X_train, y_train, X_test = X_train.to(device), y_train.to(device), X_test.to(device)
    train(model, X_train, y_train, X_test)


