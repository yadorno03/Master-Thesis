import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from NeuralNets import make_score_network


class ReturnStandardization(nn.Module):
    """Fixed per-asset affine map between returns r and the DFM's coordinates y.

        y = (r - mean) / scale

    The map is the identity until fit, which takes the mean and standard deviation
    of each asset from training returns only. Chen et al. (2025) standardize each
    stock's returns in the same way before training on market data. The buffers
    follow the model's dtype and device and are saved with its state.
    """

    def __init__(self, d):
        super().__init__()
        self.register_buffer('mean', torch.zeros(d))
        self.register_buffer('scale', torch.ones(d))

    @torch.no_grad()
    def fit(self, returns):
        """Estimate the mean and the (unbiased) standard deviation of each asset in float64."""
        values = torch.as_tensor(returns).detach().cpu().double()
        if values.ndim != 2 or values.shape[1] != len(self.mean) or len(values) < 2 \
                or not torch.isfinite(values).all():
            raise ValueError(f'Standardization needs at least two finite rows of {len(self.mean)} returns')
        scale = values.std(0)
        if not (scale > 0).all():
            raise ValueError('Every asset needs a positive standard deviation on the training rows')
        self.mean.copy_(values.mean(0))
        self.scale.copy_(scale)

    def standardize(self, returns):
        """Model coordinates of returns, in the dtype of the buffers."""
        return (returns - self.mean) / self.scale

    def to_returns(self, y):
        """Map model coordinates back to returns, as a NumPy array in the dtype of y.

        The affine map is computed in float64 on the CPU, then rounded once.
        """
        values = torch.as_tensor(y).detach().cpu()
        returns = self.mean.detach().cpu().double() + self.scale.detach().cpu().double() * values.double()
        return returns.to(values.dtype).numpy()

    def portfolio_event(self, weights, threshold):
        """The return event w'r <= C in model coordinates: (scale * w)'y <= C - w'mean.

        Returns the model-coordinate weights (float64 NumPy) and threshold (float).
        """
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(self.mean),) or not np.isfinite(weights).all() or not np.isfinite(threshold):
            raise ValueError(f'The portfolio needs {len(self.mean)} finite weights and a finite threshold')
        mean = self.mean.detach().cpu().double().numpy()
        scale = self.scale.detach().cpu().double().numpy()
        return scale * weights, float(threshold - weights @ mean)


class UnconditionalDFM(nn.Module):
    """
    Wrapper class for the Unconditional Diffusion Factor Model (DFM).
    Handles the Ornstein-Uhlenbeck (O-U) forward process and computes the
    Denoising Score Matching loss of Chen et al. (2025).

    Attributes:
        d (int): Asset return dimension.
        k (int): Latent factor dimension.
        sigma_max (float): Maximum boundary for idiosyncratic noise.
        t0 (float): Small positive early-stopping time to prevent score explosion near t=0.
        T (float): Terminal time horizon.
        factor_score_network (FactorScoreNetwork): Underlying score network,
            built by name: 'mlp' (Chen et al., 2025) or 'gaussian_mlp' (Gaussian part + MLP).
        standardization (ReturnStandardization): Map from returns to the model
            coordinates y in which the diffusion, its score and its samples live.
            Identity unless fit_dfm standardizes the training returns.

    loss_fn and train_dfm keep the uniform-time objective of Chen et al. (2025).
    The notebook instead trains with synthetic_benchmark.unconditional.fit_dfm,
    which uses log-spaced times and noise-variance weighting.
    """
    def __init__(self, d, k, sigma_max=1.0, t0=1e-4, T=1.0, hidden_dims=[128, 128], score_network='mlp'):
        super().__init__()
        self.d = d
        self.k = k
        self.sigma_max = sigma_max
        self.t0 = t0
        self.T = T
        self.factor_score_network = make_score_network(score_network, d, k, sigma_max, hidden_dims)
        self.standardization = ReturnStandardization(d)

    def q_sample(self, r0, t, z):
        """
        Samples from the transition kernel q(R_t | R_0 = r0) of the forward O-U process:
            R_t = alpha_t * r0 + sqrt(h_t) * z

        Args:
            r0 (Tensor): Initial asset returns, in model coordinates, of shape (B, d).
            t (Tensor): Diffusion times of shape (B, 1).
            z (Tensor): Standard Gaussian noise of shape (B, d).

        Returns:
            Tensor: Noised returns at time t of shape (B, d).
        """
        alpha_t = torch.exp(-t / 2.0)
        h_t = -torch.expm1(-t)
        return alpha_t * r0 + torch.sqrt(h_t) * z

    def loss_fn(self, r0):
        """
        Computes uniformly time-weighted DSM: mean of squared Euclidean errors.

        Time is uniform on [t0, T]; no noise-variance loss weighting is applied.

        Args:
            r0 (Tensor): Batch of clean returns, in model coordinates, of shape (B, d).

        Returns:
            Tensor: Scalar loss value.
        """
        B = r0.size(0)

        t = torch.rand(B, 1, device=r0.device, dtype=r0.dtype) * (self.T - self.t0) + self.t0

        z = torch.randn_like(r0)

        r_t = self.q_sample(r0, t, z)

        score_pred = self.factor_score_network(r_t, t)

        h_t = -torch.expm1(-t)

        target_score = -z / torch.sqrt(h_t)

        loss = (score_pred - target_score).square().sum(dim=-1).mean()

        return loss

def train_dfm(dataset, k, epochs=5, lr=1e-3, batch_size=64, sigma_max=1.0, t0=1e-4, T=1.0, hidden_dims=[128, 128], device='cpu', dtype=torch.float32):
    """
    Uniform-time training loop for DFM (MLP); the notebook uses fit_dfm instead.

    Args:
        dataset (Tensor or np.ndarray): Asset return dataset of shape (N, d).
        k (int): Number of latent factors.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        sigma_max (float): Max noise variance.
        t0 (float): Lower time boundary.
        T (float): Upper time boundary (horizon).
        hidden_dims (list): Hidden layer sizes of MLP g_zeta.
        device (str or torch.device): Training device.
        dtype (torch.dtype): Model and data precision; float32 by default,
            or float64. Input data is converted to this precision.

    Returns:
        UnconditionalDFM: The trained DFM model.
        list: History of epoch average losses.
    """
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    if not isinstance(dataset, (torch.Tensor, np.ndarray)):
        raise ValueError("dataset must be a Tensor or numpy array")
    dataset = torch.as_tensor(dataset)
    if dataset.is_complex() or dataset.dtype == torch.bool:
        raise ValueError("dataset must be real-valued")
    dataset = dataset.detach().to(dtype=dtype)
    if dataset.ndim != 2 or min(dataset.shape) < 1 or not torch.isfinite(dataset).all():
        raise ValueError("dataset must be finite with nonempty shape [N, d]")

    N, d = dataset.shape
    print(f"Initializing DFM training: Sample size N={N}, Assets d={d}, Factors k={k}, Epochs={epochs}")

    # Create DataLoader
    tensor_dataset = TensorDataset(dataset)
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Initialize wrapper model
    model = UnconditionalDFM(d=d, k=k, sigma_max=sigma_max, t0=t0, T=T, hidden_dims=hidden_dims)
    model.to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        samples = 0

        for batch_x, in dataloader:
            batch_x = batch_x.to(device=device, dtype=dtype)

            optimizer.zero_grad()
            loss = model.loss_fn(batch_x)
            loss.backward()
            optimizer.step()

            # Retract V to orthonormal columns after each optimizer step.
            model.factor_score_network.project_orthogonal()

            # loss_fn averages samples; weight partial batches by their size.
            epoch_loss += loss.item() * batch_x.shape[0]
            samples += batch_x.shape[0]

        avg_loss = epoch_loss / samples
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} | Average DSM Loss: {avg_loss:.6f}")

    return model, loss_history
