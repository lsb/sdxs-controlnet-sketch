import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxconverter_common import float16, auto_mixed_precision
from onnxsim import simplify
import numpy as np

text_encoder_file = "sd_text_encoder_fp32.onnx"
text_encoder_fp16_file = "sd_text_encoder_fp16.onnx"
text_encoder_q8_file = "sd_text_encoder_q8.onnx"
sd512_file = "sd512fp32.onnx"
sd512_linear32_file = "sd512fp32linear32.onnx"
sd512_fp16_file = "sd512fp16.onnx"
sd512_q8_file = "sd512q8.onnx"

def sim(model):
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    return model_simp

def fp16(model, no_matmul=False):
    return sim(float16.convert_float_to_float16(
        model,
        min_positive_val=1e-10, max_finite_val=1e3, keep_io_types=True,
        op_block_list=["MatMul"] if no_matmul else None,
    ))

text_encoder = sim(onnx.load(text_encoder_file))
sd512 = sim(onnx.load(sd512_file))

print('loaded')

onnx.save(fp16(text_encoder), text_encoder_fp16_file)
quantize_dynamic(text_encoder, text_encoder_q8_file, weight_type=QuantType.QUInt8)

print('done with text encoder')

onnx.save(fp16(sd512), sd512_fp16_file)

print('done with fp16')

quantize_dynamic(onnx.load(sd512_linear32_file), sd512_q8_file, weight_type=QuantType.QUInt8, op_types_to_quantize=["MatMul"])
onnx.save(sim(onnx.load(sd512_q8_file)), sd512_q8_file)
# onnx.save(
#     auto_mixed_precision.auto_convert_mixed_precision(
#         onnx.load(sd512_q8_file),
#         {"image": np.ones((512, 512), dtype=np.uint8),
#          "prompt_embeds": np.zeros((1,77,768), dtype=np.float32),
#          "conditioning_scale": np.array([0.0], dtype=np.float32),
#          "latents": np.zeros((1,4,512//8,512//8), dtype=np.float32),
#         }, 
#         validate_fn=lambda a,b: True,
#         keep_io_types=True,
#     ),
# sd512_q8_file)
# quantize_dynamic(fp16(sd512, no_matmul=True) , sd512_q8_file, weight_type=QuantType.QUInt8, op_types_to_quantize=["MatMul"])
                 
                 
#                   nodes_to_exclude=[
#     "/decoder/layers/0/Conv",
#     "/decoder/layers/2/conv/0/Conv",
#     "/decoder/layers/2/conv/2/Conv",
#     "/decoder/layers/2/conv/4/Conv",
#     "/decoder/layers/3/conv/0/Conv",
#     "/decoder/layers/3/conv/2/Conv",
#     "/decoder/layers/3/conv/4/Conv",
#     "/decoder/layers/4/conv/0/Conv",
#     "/decoder/layers/4/conv/2/Conv",
#     "/decoder/layers/4/conv/4/Conv",
#     "/decoder/layers/6/Conv",
#     "/decoder/layers/7/conv/0/Conv",
#     "/decoder/layers/7/conv/2/Conv",
#     "/decoder/layers/7/conv/4/Conv",
#     "/decoder/layers/8/conv/0/Conv",
#     "/decoder/layers/8/conv/2/Conv",
#     "/decoder/layers/8/conv/4/Conv",
#     "/decoder/layers/9/conv/0/Conv",
#     "/decoder/layers/9/conv/2/Conv",
#     "/decoder/layers/9/conv/4/Conv",
#     "/decoder/layers/11/Conv",
#     "/decoder/layers/12/conv/0/Conv",
#     "/decoder/layers/12/conv/2/Conv",
#     "/decoder/layers/12/conv/4/Conv",
#     "/decoder/layers/13/conv/0/Conv",
#     "/decoder/layers/13/conv/2/Conv",
#     "/decoder/layers/13/conv/4/Conv",
#     "/decoder/layers/14/conv/0/Conv",
#     "/decoder/layers/14/conv/2/Conv",
#     "/decoder/layers/14/conv/4/Conv",
#     "/decoder/layers/16/Conv",
#     "/decoder/layers/17/conv/0/Conv",
#     "/decoder/layers/17/conv/2/Conv",
#     "/decoder/layers/17/conv/4/Conv",
#     "/decoder/layers/18/Conv",
# ])
