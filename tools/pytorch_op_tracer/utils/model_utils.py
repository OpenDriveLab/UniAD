"""Model utilities for UniAD"""

import torch

# Try importing UniAD components
try:
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model
    MMDET3D_AVAILABLE = True
except ImportError:
    MMDET3D_AVAILABLE = False
    Config = None
    build_model = None
    load_checkpoint = None


def create_dummy_input(config=None, device='cuda'):
    """Create dummy input for model tracing"""
    batch_size = 1
    
    if MMDET3D_AVAILABLE and config is not None:
        # Create proper input dict for UniAD
        dummy_input = {
            'img': torch.randn(batch_size, 6, 3, 928, 1600).to(device),
            'img_metas': [[{
                'lidar2img': torch.eye(4).unsqueeze(0).repeat(6, 1, 1).numpy(),
                'can_bus': torch.zeros(18).numpy(),
                'scene_token': 'dummy_scene',
                'timestamp': 0.0
            }]]
        }
    else:
        # Fallback for testing without mmdet3d
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    return dummy_input


def load_uniad_model(config_path: str, checkpoint_path: str = None, device: str = 'cuda'):
    """Load UniAD model from config and checkpoint"""
    if not MMDET3D_AVAILABLE:
        raise ImportError("mmdet3d is not available. Please install it to use UniAD models.")
    
    # Load config
    cfg = Config.fromfile(config_path)
    
    # Build model
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = load_checkpoint(model, checkpoint_path, map_location=device)
    
    model = model.to(device)
    model.eval()
    
    return model, cfg