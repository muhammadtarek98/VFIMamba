import torch
import torch.nn.functional as F

def forward_warp(flow, img):
    """
    Forward warping function to warp an image using the given flow.

    Args:
        flow (torch.Tensor): Optical flow tensor of shape (B, 2, H, W).
        img (torch.Tensor): Image tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor: Warped image tensor of shape (B, C, H, W).
    """
    B, C, H, W = img.size()
    grid_x, grid_y = torch.meshgrid(
        torch.arange(0, W, device=img.device),
        torch.arange(0, H, device=img.device),
        indexing="xy"
    )
    grid = torch.stack((grid_x, grid_y), dim=0).float()  # Shape: (2, H, W)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # Shape: (B, 2, H, W)

    flow_grid = grid + flow  # Add flow to the grid
    flow_grid[:, 0, :, :] = 2.0 * flow_grid[:, 0, :, :] / max(W - 1, 1) - 1.0
    flow_grid[:, 1, :, :] = 2.0 * flow_grid[:, 1, :, :] / max(H - 1, 1) - 1.0
    flow_grid = flow_grid.permute(0, 2, 3, 1)  # Shape: (B, H, W, 2)

    warped_img = F.grid_sample(img, flow_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped_img
