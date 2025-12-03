# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.func import jvp # Efficiently compute total derivatives
import numpy as np

# --- Hyperparameters ---
BATCH_SIZE = 256
EPOCHS = 500
LEARNING_RATE = 1e-4
IMAGE_SIZE = 28
CHANNELS = 1 # 1 for MNIST/Fashion
DATASET = 'fashion' # or 'mnist'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
# --- Define the scalar function f(t) ---
def f_scalar_function(t):
    # In standard Flow Matching, the forward path is often x_t = (1 - t) * x0 + t * x1 
    # The derivative of the coefficient of x_t w.r.t. t is often taken.
    # We will use f(t) = t for demonstration as it has a non-zero derivative.
    return t

def compute_flow_matching_loss(model, x0, t, x1, cond):
    """
    Computes the customized Flow Matching loss based on the provided derivative constraint.
    """
    """
    Computes the customized PINN loss with corrected PyTorch broadcasting (B, 1, 1, 1).
    Loss: || u - (V - f * du/dt) / (df/dt) ||^2
    x(0) -> image distribution
    x(1) -> noise distribution
    f(t) * u(x(t), t) = integral of v(x(t), t) from 0 to t
    Then f(1) * u(x(1), 1) = x1 - x0
    x0 = x1 - f(1) * u(x(1), 1)
    """

    # --- 1. Generate Flow Variables ---
    # Linear Interpolation Path (FM standard)
    t_reshape = t.view(-1, 1, 1, 1)
    # we adopt the notation in https://kexue.fm/archives/10958
    # x0∼p0(x0) is image distribution we want to learn -> t = 0
    # x1~p1(x1) is an analytical distribution that's easy to sample -> t = 1
    xt =  (1 - t_reshape) * x0 + t_reshape * x1 
    # take the derivative of xt to t
    V_target = x1 - x0 # Shape: (B, C, H, W)

    # Optional : a learnt v(xt, t)
    # There are two places where we need to use v(xt, t)
    # The first order term and the second order term
    # V_pred = model.forward_v(xt, t)
    # learn_v must be True if use learned v

    # --- 2. Compute u(t) and its total derivative du/dt using jvp ---
    
    # Model input is the noisy image xt and the time t.
    # The derivative is taken w.r.t the time t.
    
    # Concatenate xt and t into a single tensor for jvp, then pass to model
    # To use jvp, the model must accept a tuple of inputs: model((xt, t))
    # Let's wrap the network call for jvp compatibility

    # Tangent vector: derivative of (xt, t) w.r.t. t is (d(xt)/dt, dt/dt)
    # dt/dt is 1
    # for dx/dt, either we use V_pred, or we can use V_target
    tangent_xt = V_target
    tangent_t = torch.ones_like(t)
    
    # u is the network output (the predicted velocity field)
    u, du_dt = jvp(lambda xt, t : model.forward_u(xt, t, cond), (xt, t), (tangent_xt, tangent_t))
    
    # --- 3. Compute f(t) and its derivative df/dt using jvp ---
    f, df_dt = jvp(f_scalar_function, (t,), (tangent_t,))

    # --- 4. Loss Computation with Detachment ---
    
    # Detach to prevent second-order gradients and stabilize training
    du_dt_detached = du_dt.detach()
    f_detached = f.detach()
    df_dt_detached = df_dt.detach()

    # Reshape detached scalars (B) to match image size (B, C, H, W)
    B, C, H, W = x0.shape
    f_reshaped = f_detached.view(B, 1, 1, 1).expand_as(x0)
    df_dt_reshaped = df_dt_detached.view(B, 1, 1, 1).expand_as(x0)

    # df_dt_reshaped should >> 0
    RHS_denominator = df_dt_reshaped
    # We can use V_target or V_pred here
    # RHS_numerator = V_target - f * (du/dt)_detached
    RHS_numerator = V_target - f_reshaped * du_dt_detached
    
    # RHS_expression = (V - f * du/dt) / (df/dt)
    RHS_expression = RHS_numerator / RHS_denominator
    
    # The L2 Loss
    loss_u = torch.mean((u - RHS_expression)**2)
    # VARIANT : we can also use a learnt V_target (also called a auxiliary v-pred target)
    # loss_v = torch.mean((V_target - V_pred)**2)
    loss = loss_u
    return loss

# %%
def sample_images(model, num_samples, device=DEVICE, seed=0):
    model.eval()
    # 1. Initialize with noise (t=0)
    x0 = torch.randn(num_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, generator=torch.Generator().manual_seed(seed)).to(device)
    cond = torch.randint(0, 9, (num_samples,), generator=torch.Generator().manual_seed(seed)).to(device)
    u = model.forward_u(x0, torch.ones(num_samples, 1).to(device), cond)
    x1 = x0 - f_scalar_function(torch.ones(1).to(device)) * u
    # Normalize images back to 0-1 range for viewing/saving
    final_samples = (x1.clamp(-1, 1) + 1) / 2
    # Example: print shape of generated images
    # print(f"Generated samples shape: {final_samples.shape}")
    return final_samples

def sample_images_ode_solver(model, num_samples, num_steps=50, device=DEVICE, seed=0):
    model.eval()
    # 1. Initialize with noise at t=1
    x = torch.randn(num_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, generator=torch.Generator().manual_seed(seed)).to(device)
    cond = torch.randint(0, 9, (num_samples,), generator=torch.Generator().manual_seed(seed)).to(device)
    dt = 1.0 / num_steps
    
    # 2. Iterate backward in time (from t=1 down to t=0)
    for i in range(num_steps):
        t = 1.0 - i * dt
        t_tensor = torch.full((num_samples, 1), t).to(device)
        
        # Get the predicted vector field v(x, t)
        # The ODE is dx/dt = -v(x, t)
        v_t = model.forward_v(x, t_tensor, cond)
        
        # Euler step: x(t - dt) = x(t) + (dx/dt) * (-dt)
        # x_new = x_old + (-v_t) * (-dt)  -> x_new = x_old + v_t * dt
        x = x - v_t * dt
    
    # Normalize images back to 0-1 range for viewing/saving
    final_samples = (x.clamp(-1, 1) + 1) / 2
    return final_samples

# %%
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def display_image_grid(
    images: torch.Tensor,
    nrow: int = 4,
    title: str = "Image Grid",
    figsize: tuple = (6, 6)
) -> None:
    """
    Displays a grid of images from a PyTorch tensor batch.

    Args:
        images (torch.Tensor): The input tensor of shape (B, C, H, W).
        nrow (int): The number of images to display per row in the grid. Defaults to 4.
        title (str): The title for the plot.
        figsize (tuple): The size of the Matplotlib figure.
    """
    # Ensure the tensor is on the CPU and converted to float if necessary
    images = images.cpu().float()

    # 1. Use make_grid to create a single grid tensor
    grid_tensor = torchvision.utils.make_grid(images, nrow=nrow, normalize=True)
    
    # 2. Convert to NumPy array and transpose dimensions for Matplotlib
    # PyTorch: (C, H, W) -> Matplotlib: (H, W, C)
    grid_np = grid_tensor.permute(1, 2, 0).numpy()

    # 3. Display the image grid using Matplotlib
    plt.figure(figsize=figsize)
    
    # Check if it's a single-channel (grayscale) image to use cmap='gray' 
    # and remove the channel dimension with .squeeze()
    if grid_np.shape[2] == 1:
        plt.imshow(grid_np.squeeze(), cmap='gray')
    else:
        plt.imshow(grid_np) # For RGB/multi-channel images
        
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"output/{title}.png")
    plt.close()

# %%
def train_model(model, train_loader, optimizer, epochs, device):
    model.train()
    
    print(f"\n--- Starting Training on {DATASET.upper()} with Custom PINN Loss ---")
    
    for epoch in range(epochs):
        print("--- Starting Evaluation ---")
        display_image_grid(
            images=sample_images(model, num_samples=16, seed=0),
            nrow=4,
            title=f"{epoch}-onestep",
            figsize=(5, 5)
        )
        if model.learn_v:
            display_image_grid(
                images=sample_images_ode_solver(model, num_samples=16, seed=0),
                nrow=4,
                title=f"{epoch}-odesolver",
                figsize=(5, 5)
            )
        total_loss = 0
        for data, cond in train_loader:
            x0 = data.to(device)
            B = x0.shape[0]
            
            # --- 1. Setup Inputs ---
            t = torch.rand(B, 1).to(device)
            x1 = torch.randn_like(x0).to(device)
            cond = cond.to(device)
            t.requires_grad_(True)
            
            # --- 2. Calculate Loss and Backprop ---
            optimizer.zero_grad()
            loss = compute_flow_matching_loss(model, x0, t, x1, cond)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.5f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'output/fm_pinn_model_epoch_{epoch + 1}.pth')

    print("--- Training Complete ---")
    torch.save(model.state_dict(), 'fm_pinn_model_final.pth')

# %%
# --- Main Execution Block ---
if __name__ == '__main__':
    import os
    os.makedirs("output", exist_ok=True)
    # 1. Data Loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
    ])
    
    if DATASET == 'mnist':
        train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif DATASET == 'fashion':
        train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 2. Model Initialization
    from model import DiT
    model = DiT(
        input_size=28,
        patch_size=4,
        in_channels=1,
        hidden_size=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=10,
        learn_v=True).to(DEVICE)
    # model = UNet(1,1).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    
    # 3. Training, Pytorch doesn't support forward AD for flash attention
    with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.MATH]):
        train_model(model, train_loader, optimizer, EPOCHS, DEVICE)

# %%
# # --- Define the scalar function f(t) ---
# def f_scalar_function(t):
#     # In standard Flow Matching, the forward path is often x_t = t*x_0 + (1-t)*z
#     # The derivative of the coefficient of x_t w.r.t. t is often taken.
#     # We will use f(t) = t + 1 for demonstration as it has a non-zero derivative.
#     # NOTE: The choice of f(t) is critical and highly dependent on the ODE setup.
#     return t

# def compute_flow_matching_loss(model : UNet, x0, t, x1):
#     """
#     Computes the customized Flow Matching loss based on the provided derivative constraint.
#     """
#     """
#     Computes the customized PINN loss with corrected PyTorch broadcasting (B, 1, 1, 1).
#     Loss: || u - (V - f * du/dt) / (df/dt) ||^2
#     x(0) -> image distribution
#     x(1) -> noise distribution
#     f(t) * u(x(t), t) = integral of v(x(t), t) from 0 to t
#     Then f(1) * u(x(1), 1) = x1 - x0
#     x0 = x1 - f(1) * u(x(1), 1)
#     """
#     # --- 1. Generate Flow Variables ---
#     # Linear Interpolation Path (FM standard)
#     t_reshape = t.view(-1, 1, 1, 1)
#     # we adopt the notation in https://kexue.fm/archives/10958
#     # x0∼p0(x0) is image distribution we want to learn -> t = 0
#     # x1~p1(x1) is an analytical distribution that's easy to sample -> t = 1
#     xt =  (1 - t_reshape) * x0 + t_reshape * x1 
#     # take the derivative of xt to t
#     V_target = x1 - x0 # Shape: (B, C, H, W)
    
#     # --- 2. Compute u(t) and its total derivative du/dt using jvp ---
    
#     # Model input is the noisy image xt and the time t.
#     # The derivative is taken w.r.t the time t.
    
#     # Concatenate xt and t into a single tensor for jvp, then pass to model
#     # To use jvp, the model must accept a tuple of inputs: model((xt, t))
#     # Let's wrap the network call for jvp compatibility

#     # Tangent vector: derivative of (xt, t) w.r.t. t is (d(xt)/dt, dt/dt)
#     # Since xt is not a simple function of t in the graph, we must treat the inputs as independent for JVP:
#     # d(xt)/dt is 0 in this context, and dt/dt is 1.
#     tangent_xt = V_target.detach()
#     tangent_t = torch.ones_like(t)
    
#     # u is the network output (the predicted velocity field)
#     u, du_dt = jvp(model.forward_u, (xt, t), (tangent_xt, tangent_t))
    
#     # --- 3. Compute f(t) and its derivative df/dt using jvp ---
#     f, df_dt = jvp(f_scalar_function, (t,), (tangent_t,))

#     # --- 4. Loss Computation with Detachment ---
    
#     # Detach to prevent second-order gradients and stabilize training
#     du_dt_detached = du_dt.detach()
#     f_detached = f.detach()
#     df_dt_detached = df_dt.detach()

#     # Reshape detached scalars (B) to match image size (B, C, H, W)
#     B, C, H, W = x0.shape
#     f_reshaped = f_detached.view(B, 1, 1, 1).expand_as(x0)
#     df_dt_reshaped = df_dt_detached.view(B, 1, 1, 1).expand_as(x0)
    
#     # Ensure no division by zero
#     epsilon = 1e-6
#     RHS_denominator = df_dt_reshaped + torch.sign(df_dt_reshaped) * epsilon
    
#     # RHS_numerator = V_target - f * (du/dt)_detached
#     RHS_numerator = V_target - f_reshaped * du_dt_detached
    
#     V_pred = model.forward_v(xt, t)

#     RHS_expression = RHS_numerator / RHS_denominator
    
#     # The L2 Loss
#     #loss = torch.mean((u - RHS_expression)**2)
#     loss_u = torch.mean((u - RHS_expression)**2)
#     # loss_v = torch.mean((V_target - V_pred)**2)
#     loss_v = 0
#     loss = loss_u + loss_v
#     return loss


