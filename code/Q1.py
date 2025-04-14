import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Plotting functions
def plot_trajectories(pred_qs, true_qs, title="Trajectory Comparison"):
    plt.figure(figsize=(6, 6))
    plt.plot(true_qs[:, 0], true_qs[:, 1], label='Ground Truth', linewidth=2)
    plt.plot(pred_qs[:, 0], pred_qs[:, 1], '--', label='SympNet Prediction', linewidth=2)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def plot_mse(pred_qs, true_qs, title="MSE over Time"):
    mse = ((pred_qs - true_qs)**2).mean(dim=1)  # shape: (n_steps,)
    plt.figure(figsize=(6, 4))
    plt.plot(mse.numpy(), label='Position MSE')
    plt.xlabel('Time Step')
    plt.ylabel('MSE')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

# Load training data
with open('train.txt', 'r') as f:
    data = [list(map(float, line.strip().split())) for line in f if line.strip()]
tensor_data = torch.tensor(data, dtype=torch.float32).T
train_p = tensor_data[:2, :1200].T  # momentum (v1, v2)
train_q = tensor_data[2:, :1200].T  # position (x1, x2)
target_p = tensor_data[:2, 1:1201].T
target_q = tensor_data[2:, 1:1201].T

# Load testing data
with open('test.txt', 'r') as f:
    data = [list(map(float, line.strip().split())) for line in f if line.strip()]
tensor_data = torch.tensor(data, dtype=torch.float32).T
test_p = tensor_data[:2, :].T
test_q = tensor_data[2:, :].T

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

    def forward(self, q, p, dt=0.1):
        p_new = p - dt * self.f_q(q)
        q_new = q + dt * self.g_p(p_new)
        return q_new, p_new

# Initialize model and optimizer
model = SympNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop
for epoch in range(500):
    pred_q, pred_p = model(train_q, train_p, dt=0.1)
    loss = F.mse_loss(pred_q, target_q) + F.mse_loss(pred_p, target_p)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# Recurrent rollout on test set
def rollout(model, q0, p0, steps, dt=0.1):
    qs = [q0]
    ps = [p0]
    for _ in range(steps):
        q_next, p_next = model(qs[-1], ps[-1], dt=dt)
        qs.append(q_next)
        ps.append(p_next)
    return torch.stack(qs[1:], dim=0), torch.stack(ps[1:], dim=0)

# Starting from test initial state (t=1200)
init_q = test_q[0].unsqueeze(0)  # shape (1, 2)
init_p = test_p[0].unsqueeze(0)  # shape (1, 2)

pred_qs, pred_ps = rollout(model, init_q, init_p, steps=299, dt=0.1)
true_qs = test_q[1:300]  # 299 points from t=1201 to t=1500

# Plot results
plot_trajectories(pred_qs.detach().squeeze(), true_qs)
plot_mse(pred_qs.detach().squeeze(), true_qs)

# Pint final MSE
final_mse = F.mse_loss(pred_qs.detach().squeeze(), true_qs)
print(f"Final rollout MSE: {final_mse.item():.6f}")
