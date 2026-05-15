import torch
import numpy as np
from pcam_model import PCAMModel, build_default_R
from data import make_patterns

seed = 42
K, N = 16, 64
X = make_patterns(K=K, N=N, seed=seed)
R = build_default_R(N=N, seed=seed)
model = PCAMModel(X, R)

H = model.hessian(X[0])
H = 0.5 * (H + H.T)

H_t = torch.tensor(H, dtype=torch.float64)

# We want to minimize the condition number of D^{1/2} H D^{1/2}
# log_pi is the unconstrained parameter
log_pi = torch.zeros(N, dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam([log_pi], lr=0.1)

best_spread = float('inf')

for i in range(1000):
    optimizer.zero_grad()
    
    # Clip and normalise pi
    pi = torch.exp(log_pi)
    pi = torch.clamp(pi, 0.1, 10.0)
    pi = pi / pi.mean()
    
    pi_sqrt = torch.sqrt(pi)
    
    S = (pi_sqrt.unsqueeze(1) * H_t) * pi_sqrt.unsqueeze(0)
    S = 0.5 * (S + S.T)
    
    # Differentiable eigenvalues
    eigs = torch.linalg.eigvalsh(S)
    
    # Spread
    spread = eigs[-1] / eigs[0]
    
    if spread.item() < best_spread:
        best_spread = spread.item()
        
    if i % 100 == 0:
        print(f"Iter {i}: spread = {spread.item():.4f}")
        
    spread.backward()
    optimizer.step()

print(f"\nFinal optimal spread: {best_spread:.4f}")
