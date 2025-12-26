import torch
import torch.nn as nn

class ExactAAttnBlockStatic(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Bundled 1x1 Conv (QKV) - Weight Shape [192, 64, 1, 1]
        self.qkv = nn.Conv2d(64, 192, 1, bias=True)
        # 2. Depthwise Branch (7x7) - group=64
        self.pe = nn.Conv2d(64, 64, 7, 1, 3, groups=64, bias=True)
        # 3. Final 1x1 Projection
        self.proj = nn.Conv2d(64, 64, 1, bias=True)
        
        self.num_heads = 2
        self.head_dim = 32

    def forward(self, x):
        # --- PHASE 1: THE TOP OF THE GRAPH ---
        qkv = self.qkv(x) # Output: [1, 192, 40, 40]
        
        # Reshape 1
        x1 = qkv.view(1, 192, 1600)
        # Transpose 1 (matches 0, 2, 1 usually)
        x2 = x1.transpose(1, 2)              # [1, 1600, 192]
        # Reshape 2
        x3 = x2.view(1, 1600, 2, 96)
        # TRANSPOSE 2 (The one you specified: 0, 2, 3, 1)
        # This moves the Head dimension (2) to the front and packs QKV at the end
        x4 = x3.permute(0, 2, 3, 1)          # [1, 2, 96, 1600]
        
        # --- PHASE 2: THE SPLIT ---
        # Split axis 2 (the 96 dim) into 32, 32, 32
        q_raw, k_raw, v_raw = torch.split(x4, 32, dim=2)
        
        # --- PHASE 3: ATTENTION ---
        # Reshape to align for MatMul (1600 becomes the last dim for K.T)
        q = q_raw.transpose(2, 3) # [1, 2, 1600, 32]
        k = k_raw.transpose(2, 3) # [1, 2, 1600, 32]
        v = v_raw.transpose(2, 3) # [1, 2, 1600, 32]
        
        attn = (q @ k.transpose(-2, -1)) * 0.1767766952966369
        attn = attn.softmax(dim=-1)
        
        # Transpose BEFORE MatMul (0, 1, 3, 2) as requested earlier
        attn = attn.permute(0, 1, 3, 2)
        x_attn_raw = (v_raw @ attn)
        
        # --- PHASE 4: RECONSTRUCTION ---
        x_attn=x_attn_raw.permute(0,3,1,2)
        x_attn = x_attn.reshape(1, 40, 40, 64).permute(0, 3, 1, 2)
        
        # PE Path branch
        v_img = v.transpose(1, 2).reshape(1, 40, 40, 64).permute(0, 3, 1, 2)
        x_pe = self.pe(v_img)
        
        return self.proj(x_attn + x_pe)

# --- Static Export ---
model = ExactAAttnBlockStatic().eval()
dummy_input = torch.randn(1, 64, 40, 40)

torch.onnx.export(
    model, 
    dummy_input, 
    "YoloV12AttnStandalone.onnx",
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)