"""
Quantization from Scratch
Interview question: "How does model quantization work?"
"""
import numpy as np

def quantize_to_int8(weights: np.ndarray) -> tuple:
    """
    Quantize FP32 weights to INT8
    
    Process:
    1. Find min/max of weights
    2. Calculate scale factor
    3. Quantize to INT8 range (-128 to 127)
    4. Store scale for dequantization
    
    Returns:
        (quantized_weights, scale, zero_point)
    """
    # Find range
    w_min = np.min(weights)
    w_max = np.max(weights)
    
    # Calculate scale: map [w_min, w_max] to [-128, 127]
    scale = (w_max - w_min) / 255.0
    
    # Zero point: where 0 in INT8 maps to in FP32
    zero_point = -w_min / scale
    
    # Quantize: convert to INT8
    quantized = np.round(weights / scale + zero_point).astype(np.int8)
    quantized = np.clip(quantized, -128, 127)
    
    return quantized, scale, zero_point

def dequantize_from_int8(quantized: np.ndarray, scale: float, 
                         zero_point: float) -> np.ndarray:
    """
    Dequantize INT8 back to FP32
    
    Reverse process of quantization
    """
    return (quantized.astype(np.float32) - zero_point) * scale

def quantize_to_int4(weights: np.ndarray) -> tuple:
    """
    Quantize FP32 weights to INT4 (4-bit)
    More aggressive quantization, more memory savings
    """
    w_min = np.min(weights)
    w_max = np.max(weights)
    
    # INT4 range: -8 to 7
    scale = (w_max - w_min) / 15.0
    zero_point = -w_min / scale
    
    quantized = np.round(weights / scale + zero_point).astype(np.int8)
    quantized = np.clip(quantized, -8, 7)
    
    return quantized, scale, zero_point

def calculate_compression_ratio(original: np.ndarray, quantized: np.ndarray) -> float:
    """Calculate memory compression ratio"""
    original_size = original.nbytes
    quantized_size = quantized.nbytes
    return original_size / quantized_size


# Usage Example
if __name__ == "__main__":
    print("Quantization Example")
    print("=" * 60)
    
    # Original FP32 weights
    np.random.seed(42)
    weights_fp32 = np.random.randn(1000, 1000).astype(np.float32)
    
    print(f"Original weights shape: {weights_fp32.shape}")
    print(f"Original size: {weights_fp32.nbytes / 1024 / 1024:.2f} MB")
    
    # Quantize to INT8
    weights_int8, scale, zero_point = quantize_to_int8(weights_fp32)
    print(f"\nINT8 Quantization:")
    print(f"Quantized size: {weights_int8.nbytes / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {calculate_compression_ratio(weights_fp32, weights_int8):.2f}x")
    
    # Dequantize
    weights_dequant = dequantize_from_int8(weights_int8, scale, zero_point)
    error = np.mean(np.abs(weights_fp32 - weights_dequant))
    print(f"Reconstruction error (MAE): {error:.6f}")
    
    # Quantize to INT4
    weights_int4, scale4, zp4 = quantize_to_int4(weights_fp32)
    print(f"\nINT4 Quantization:")
    print(f"Quantized size: {weights_int4.nbytes / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {calculate_compression_ratio(weights_fp32, weights_int4):.2f}x")
    
    weights_dequant4 = dequantize_from_int8(weights_int4, scale4, zp4)
    error4 = np.mean(np.abs(weights_fp32 - weights_dequant4))
    print(f"Reconstruction error (MAE): {error4:.6f}")

