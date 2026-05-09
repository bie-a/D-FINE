"""get_params.py - 获取模型参数量和 FLOPs"""
import torch

# 根据 D-FINE 实际的导入方式调整
from src.core import YAMLConfig
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    
    config = YAMLConfig(args.config)
    model = config.model
    model.eval()
    
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params_m:.2f}M")
    
    # 计算 FLOPs
    from thop import profile, clever_format
    dummy_input = torch.randn(1, 3, 640, 640)
    flops, params = profile(model, inputs=(dummy_input,))
    gflops = flops / 1e9
    print(f"FLOPs: {gflops:.1f}G")

if __name__ == "__main__":
    main()