# SPDX-License-Identifier: Apache-2.0
import logging
import os
from typing import Any, Optional

import torch

import gguf
import numpy as np

from .factory import register_algorithm
from .template import AlgorithmTemplate

logger = logging.getLogger(__name__)

@register_algorithm("erase")
class EraseAlgorithm(AlgorithmTemplate):
    """Erase algorithm: h' = h - erase_intensity * proj_{h1}(h)
    
    This algorithm erases the component of hidden state in the direction of h1.
    Formula: h_perp = h - erase_intensity * (h · h1 / ||h1||^2) * h1
    
    The result is the component of h that is orthogonal to h1, scaled by erase_intensity.
    - erase_intensity: scalar between 0.0 and 1.0 controlling the erasure strength
    - Only 2 methods needed: _transform and load_from_path
    - All parameter management is handled by AlgorithmTemplate
    - Payload can be a Tensor (backward compatible) or dict with 'direction' and 'erase_intensity'
    - Supports GGUF and PT file formats
    """

    def set_steer_vector(self, index: int, **kwargs) -> None:
        """Override to handle erase_intensity parameter."""
        payload = kwargs.get("payload")
        erase_intensity = kwargs.get("erase_intensity", 1.0)
        scale_factor = kwargs.get("scale_factor", 1.0)
        
        if payload is None:
            raise ValueError(f"{self.__class__.__name__} requires 'payload' in kwargs")
        
        # Validate erase_intensity
        if not isinstance(erase_intensity, (int, float)) or not (0.0 <= erase_intensity <= 1.0):
            raise ValueError(f"erase_intensity must be a scalar between 0.0 and 1.0, got {erase_intensity}")
        
        # Convert Tensor payload to dict format with erase_intensity
        if isinstance(payload, torch.Tensor):
            payload = {
                "direction": payload * scale_factor,
                "erase_intensity": float(erase_intensity)
            }
        elif isinstance(payload, dict):
            # If already a dict, add erase_intensity and scale_factor
            payload = {
                **payload,
                "erase_intensity": float(erase_intensity),
                "scale_factor": scale_factor
            }
            # Apply scale_factor to direction if present
            if "direction" in payload:
                payload["direction"] = payload["direction"] * scale_factor
        
        self._payloads[index] = payload

    def _transform(self, hidden_state: torch.Tensor, params: torch.Tensor | dict) -> torch.Tensor:
        """Apply erasure: h' = h - erase_intensity * proj_{h1}(h) (with optional normalization).
        
        Args:
            hidden_state: [batch, hidden_dim] or [hidden_dim]
            params: Either:
                - [hidden_dim] Tensor: the direction vector h1 to erase (backward compatible, erase_intensity=1.0)
                - dict: {"direction": [hidden_dim] Tensor, "erase_intensity": float} with optional "scale_factor"
            
        Returns:
            The component of hidden_state orthogonal to params (h1), scaled by erase_intensity
        """
        # Extract direction vector and erase_intensity from params
        if isinstance(params, dict):
            h1 = params["direction"]
            erase_intensity = params.get("erase_intensity", 1.0)
        else:
            # Backward compatibility: treat Tensor as direction with full intensity
            h1 = params
            erase_intensity = 1.0

        # Ensure params is the right shape for computation
        if h1.dim() == 1:
            h1 = h1.unsqueeze(0)  # [1, hidden_dim]
        
        # Compute ||h1||^2
        h1_norm_sq = torch.sum(h1 * h1, dim=-1, keepdim=True)  # [1, 1]
        
        # Compute h · h1 (dot product along last dimension)
        # hidden_state: [batch, hidden_dim], h1: [1, hidden_dim]
        dot_product = torch.sum(hidden_state * h1, dim=-1, keepdim=True)  # [batch, 1]
        
        # Compute projection scalar: (h · h1) / ||h1||^2
        proj_scalar = dot_product / (h1_norm_sq + 1e-8)  # [batch, 1]
        
        # Compute projection vector: proj_scalar * h1
        proj_vector = proj_scalar * h1  # [batch, hidden_dim]
        
        # Compute h_perp = h - erase_intensity * proj_{h1}(h)
        h_perp = hidden_state - erase_intensity * proj_vector
        
        if self.normalize:
            # Preserve original norm
            norm_pre = torch.norm(hidden_state, dim=-1, keepdim=True)
            norm_post = torch.norm(h_perp, dim=-1, keepdim=True)
            return h_perp * norm_pre / (norm_post + 1e-8)
        else:
            return h_perp


    @classmethod
    def load_from_path(cls, path: str, device: str, **kwargs) -> dict:
        """Load Erase direction vector from GGUF file."""
        
        config = kwargs.get("config")
        if config is None:
            raise ValueError("EraseAlgorithm.load_from_path requires 'config' in kwargs")

        file_ext = os.path.splitext(path)[1].lower()

        if file_ext == '.gguf':
            return cls._load_from_gguf(path, device, **kwargs)
        elif file_ext == '.pt':
            return cls._load_from_pt(path, device, **kwargs)
        else:
            raise ValueError(f"EraseAlgorithm only supports .gguf files, got: {file_ext}")
        return cls._load_from_gguf(path, device, **kwargs)

    @classmethod
    def _load_from_pt(cls, path: str, device: str, **kwargs) -> dict:
        """Load Erase direction vector from PT file."""
        
        config = kwargs.get("config")
        dtype = config.adapter_dtype if config is not None else torch.float32
        target_layers = kwargs.get("target_layers")
        if target_layers is None:
            raise ValueError("Loading .pt files requires 'target_layers' in kwargs")
        
        direction = torch.load(path).to(device).to(dtype)
        sv_weights = {layer_idx: direction for layer_idx in target_layers}

        return {"layer_payloads": sv_weights}
        
    @classmethod
    def _load_from_gguf(cls, path: str, device: str, **kwargs) -> dict:
        """Load Erase direction vector from GGUF file."""
        
        config = kwargs.get("config")
        
        reader = gguf.GGUFReader(path)
        
        # Validate file type
        archf = reader.get_field("general.architecture")
        if archf and len(archf.parts):
            arch = str(bytes(archf.parts[-1]), encoding="utf-8", errors="replace")
            if arch != "steervector" and arch != "controlvector":
                # Only log, don't enforce
                # logger.warning(".gguf file with arch %s may not be a steer vector", arch)
                pass

        sv_weights = {}
        for tensor in reader.tensors:
            if not tensor.name.startswith("direction."):
                continue
            try:
                layer = int(tensor.name.split(".")[1])
            except (ValueError, IndexError) as e:
                raise ValueError(f".gguf file has invalid direction field name: {tensor.name}") from e
            
            np_copy = np.array(tensor.data, copy=True)
            sv_weights[layer] = torch.from_numpy(np_copy).to(device).to(config.adapter_dtype if config is not None else torch.float32)
            
        return {"layer_payloads": sv_weights}

