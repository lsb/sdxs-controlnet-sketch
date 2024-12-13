import torch
torch.set_num_threads(16)

torch.set_grad_enabled(False)

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel # special from lsb/diffusers@sdxs-hotpatches!

ctlnetmodelname = "IDKiro/sdxs-512-dreamshaper-sketch"
sdmodelname = "IDKiro/sdxs-512-dreamshaper"


class SDModelPTOnly16(torch.nn.Module):
    def __init__(self):
        super(SDModelPTOnly16, self).__init__()
        controlnet = ControlNetModel.from_pretrained(ctlnetmodelname, torch_dtype=torch.float16)
        sdmodel = StableDiffusionControlNetPipeline.from_pretrained(sdmodelname, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None)
        # controlnet.time_embedding = None
        # sdmodel.unet.time_embedding = None
        sdmodel.set_progress_bar_config(disable=True)
        self.sdmodel = sdmodel
        sdmodel.text_encoder = sdmodel.text_encoder.to(torch.float8_e4m3fn).to(torch.float8_e5m2).to(torch.float32)
        sdmodel.controlnet = sdmodel.controlnet.to(torch.float8_e4m3fn).to(torch.float8_e5m2).to(torch.float16)
        sdmodel.unet = sdmodel.unet.to(torch.float8_e4m3fn).to(torch.float16)
        sdmodel.vae = sdmodel.vae.to(torch.float16)

        self.controlnet = torch.nn.ModuleList([controlnet])
        self.unet = torch.nn.ModuleList([sdmodel.unet])
        self.vae = torch.nn.ModuleList([sdmodel.vae])

    def forward(self, image, prompt_embeds, conditioning_scale, latents):
        return (self.sdmodel(
            prompt_embeds=prompt_embeds.to(torch.float16),
            image=image.expand(1,3,-1,-1).to(torch.float16) / 255.0,
            num_inference_steps=1,
            guidance_scale=0.0,
            controlnet_conditioning_scale=conditioning_scale.to(torch.float16),
            height=512, width=512, latents=latents.to(torch.float16), output_type="pt"
            ).images[0].permute(1,2,0) * 255).to(torch.uint8)

sdmodel_ptonly = SDModelPTOnly16()
sdmodel_latents = torch.zeros((1,4, 512//8, 512//8), dtype=torch.float32, requires_grad=False)

def trace_jit(mod):
    return torch.jit.trace(
        mod,
        example_inputs=(
            torch.ones((512,512), dtype=torch.uint8, requires_grad=False),
            torch.zeros((1,77,768), dtype=torch.float32, requires_grad=False),
            torch.tensor(0.75, dtype=torch.float32, requires_grad=False),
            sdmodel_latents),
    )

sdmodel_ptonly = sdmodel_ptonly.eval()

print("exporting text encoder")

torch.onnx.export(
    sdmodel_ptonly.sdmodel.text_encoder,
    args=(torch.zeros((1,77), dtype=torch.int64),),
    f="sd_text_encoder_fp32.onnx",
    input_names=["input_ids"],
    output_names=["output_embeddings"],
    external_data=False,
    opset_version=20,
    verbose=False,
    dynamo=False,
)

print("trying to export")
# sdtrace = trace_jit(sdmodel_ptonly)
# torch.jit.save(sdtrace, "sd512.pt")
sdtrace = sdmodel_ptonly

torch.onnx.export(
    sdtrace,
    args=(
        torch.ones((512,512), dtype=torch.uint8),
        torch.zeros((1,77,768), dtype=torch.float32),
        torch.tensor(0.75, dtype=torch.float32), sdmodel_latents),
    f=f"sd512fp32.onnx",
    verbose=False,
    opset_version=20,
    input_names=["image", "prompt_embeds", "conditioning_scale", "latents"],
    output_names=["output_image"],
    external_data=False,
    dynamo=False,
)

class Linear32(torch.nn.Module):
    def __init__(self, weight, bias):
        super(Linear32, self).__init__()
        self.weight = torch.nn.Parameter(weight.to(torch.float32), requires_grad=False)
        if bias is not None:
            self.bias = torch.nn.Parameter(bias.to(torch.float32), requires_grad=False)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        return torch.nn.functional.linear(input.to(torch.float32), self.weight, self.bias).to(torch.float16)

def make_linear32(mod):
    named_children = list(mod.named_children())
    for n,m in named_children:
        if isinstance(m, torch.nn.Linear):
            setattr(mod, n, Linear32(m.weight.data, m.bias.data if m.bias is not None else None))
        else:
            make_linear32(mod=m)

make_linear32(sdtrace)

torch.onnx.export(
    sdtrace,
    args=(
        torch.ones((512,512), dtype=torch.uint8),
        torch.zeros((1,77,768), dtype=torch.float32),
        torch.tensor(0.75, dtype=torch.float32), sdmodel_latents),
    f=f"sd512fp32linear32.onnx",
    verbose=False,
    opset_version=20,
    input_names=["image", "prompt_embeds", "conditioning_scale", "latents"],
    output_names=["output_image"],
    external_data=False,
    dynamo=False,
)

exit()


























































































































class SDModelPTOnly(torch.nn.Module):
    def __init__(self):
        super(SDModelPTOnly, self).__init__()
        controlnet = ControlNetModel.from_pretrained(ctlnetmodelname, torch_dtype=torch.float16)
        sdmodel = StableDiffusionControlNetPipeline.from_pretrained(sdmodelname, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None)
        # controlnet.time_embedding = None
        # sdmodel.unet.time_embedding = None
        sdmodel.set_progress_bar_config(disable=True)
        self.sdmodel = sdmodel
        sdmodel.text_encoder = sdmodel.text_encoder.to(torch.float8_e4m3fn).to(torch.float8_e5m2).to(torch.float16)
        sdmodel.controlnet = sdmodel.controlnet.to(torch.float8_e4m3fn).to(torch.float8_e5m2).to(torch.float16)
        sdmodel.unet = sdmodel.unet.to(torch.float8_e4m3fn).to(torch.float16)
        sdmodel.vae = sdmodel.vae.to(torch.float16)

        self.controlnet = torch.nn.ModuleList([controlnet])
        self.unet = torch.nn.ModuleList([sdmodel.unet])
        self.vae = torch.nn.ModuleList([sdmodel.vae])

    def forward(self, image, prompt_embeds, conditioning_scale, latents):
        return (self.sdmodel(
            prompt_embeds=prompt_embeds.to(torch.float16),
            image=image.expand(1,3,-1,-1).to(torch.float16) / 255.0,
            num_inference_steps=1,
            guidance_scale=0.0,
            controlnet_conditioning_scale=conditioning_scale.to(torch.float16),
            height=512, width=512, latents=latents.to(torch.float16), output_type="pt"
            ).images[0].permute(1,2,0) * 255).to(torch.uint8)

class SDModelTextEncoder(torch.nn.Module):
    def __init__(self, text_encoder):
        super(SDModelTextEncoder, self).__init__()
        self.text_encoder = torch.nn.ModuleList([text_encoder])
    
    def forward(self, input_ids):
        return self.text_encoder[0](input_ids)[0].to(torch.float32)

sdmodel_ptonly = SDModelPTOnly()
sdmodel_latents = torch.zeros((1,4, 512//8, 512//8), dtype=torch.float32, requires_grad=False)

sdmodel_text_encoder = SDModelTextEncoder(sdmodel_ptonly.sdmodel.text_encoder)

print("exporting text encoder")
print("shape is", sdmodel_text_encoder(torch.zeros((1,77), dtype=torch.int64)).shape)

# def mangle_compile(mod):
#     return torch.compile(mod, fullgraph=True, mode="reduce-overhead")

# def mangle_torchir(mod):
#     return _trace._export_to_torch_ir(
#         f=mod,
#         args=(torch.ones((512,512), dtype=torch.uint8, requires_grad=False), torch.zeros((1,77,768), dtype=torch.float32, requires_grad=False), torch.tensor(0.75, dtype=torch.float32, requires_grad=False), sdmodel_latents),
#     )

def mangle_jit(mod):
    return torch.jit.trace(
        mod,
        example_inputs=(
            torch.ones((512,512), dtype=torch.uint8, requires_grad=False),
            torch.zeros((1,77,768), dtype=torch.float32, requires_grad=False),
            torch.tensor(0.75, dtype=torch.float32, requires_grad=False),
            sdmodel_latents),
    )

def mangle_export(mod):
    return torch.export.export(
        mod,
        args=(torch.ones((512,512), dtype=torch.uint8, requires_grad=False), torch.zeros((1,77,768), dtype=torch.float32, requires_grad=False), torch.tensor(0.75, dtype=torch.float32, requires_grad=False), sdmodel_latents),
    ).module()

sdmodel_ptonly = sdmodel_ptonly.eval()

print("exporting text encoder")

torch.onnx.export(
    sdmodel_text_encoder,
    args=(torch.zeros((1,77), dtype=torch.int64),),
    f="sd_text_encoder_fp16.onnx",
    input_names=["input_ids"],
    output_names=["output_embeddings"],
    external_data=False,
    opset_version=20,
    verbose=False,
    dynamo=False,
)


print("trying to jit")
sdtorchir = mangle_jit(sdmodel_ptonly)
torch.jit.save(sdtorchir, "sdtorchir-py.pt")
torch.onnx.export(
    sdtorchir,
    args=(
        torch.ones((512,512), dtype=torch.uint8),
        torch.zeros((1,77,768), dtype=torch.float32),
        torch.tensor(0.75, dtype=torch.float32), sdmodel_latents),
    f=f"sd512fp16.onnx",
    verbose=False,
    opset_version=20,
    input_names=["image", "prompt_embeds", "conditioning_scale", "latents"],
    output_names=["output_image"],
    external_data=False,
    # do_constant_folding=False, # so that we don't fold the quantization back!
    dynamo=False,
)

# import onnx
# from onnxruntime.quantization import quantize_dynamic

# quantize_dynamic("sd512.onnx", "sd512q8.onnx")

exit()



















class SDModelPTOnlyQ(torch.nn.Module):
    def __init__(self):
        super(SDModelPTOnlyQ, self).__init__()
        controlnet = ControlNetModel.from_pretrained(ctlnetmodelname, torch_dtype=torch.float16)
        sdmodel = StableDiffusionControlNetPipeline.from_pretrained(sdmodelname, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None)
        # controlnet.time_embedding = None
        # sdmodel.unet.time_embedding = None
        sdmodel.set_progress_bar_config(disable=True)
        self.sdmodel = sdmodel
        sdmodel.controlnet = sdmodel.controlnet.to(torch.float8_e4m3fn).to(torch.float8_e5m2).to(torch.float32)
        sdmodel.unet = sdmodel.unet.to(torch.float8_e4m3fn).to(torch.float32)
        sdmodel.vae = sdmodel.vae.to(torch.float32)
        self.controlnet = torch.nn.ModuleList([controlnet])
        self.unet = torch.nn.ModuleList([sdmodel.unet])
        self.vae = torch.nn.ModuleList([sdmodel.vae])


    def forward(self, image, prompt_embeds, conditioning_scale, latents):
        return (self.sdmodel(
            prompt_embeds=prompt_embeds.to(torch.float32),
            image=image.expand(1,3,-1,-1).to(torch.float32) / 255.0,
            num_inference_steps=1,
            guidance_scale=0.0,
            controlnet_conditioning_scale=conditioning_scale,
            height=512,
            width=512,
            latents=latents.to(torch.float32),
            output_type="pt"
            ).images[0].permute(1,2,0) * 255).to(torch.uint8)

sdmodel_ptonly = SDModelPTOnlyQ()
sdmodel_ptonly.eval()
a_city_prompt_embeds = sdmodel_ptonly.sdmodel.encode_prompt("a city", device="cpu", num_images_per_prompt=1, do_classifier_free_guidance=False)[0]
sdmodel_latents = sdmodel_ptonly.sdmodel.prepare_latents(1, 4, 512, 512, a_city_prompt_embeds.dtype, "cpu", torch.manual_seed(123), None)

sd_ex = torch.export.export_for_training(mod=sdmodel_ptonly, args=(
    torch.ones((512, 512), dtype=torch.uint8),
    torch.zeros((1,77,768), dtype=torch.float32),
    torch.tensor([0.0], dtype=torch.float32),
    torch.zeros((1,4,512//8,512//8), dtype=torch.float32),
)).module()

from torch.ao.quantization.quantizer.xnnpack_quantizer import (XNNPACKQuantizer, get_symmetric_quantization_config)

sym255quantizer = XNNPACKQuantizer()
sym255quantizer.set_global(
    get_symmetric_quantization_config(is_dynamic=True) # weight_qmin=-127, weight_qmax=127)
)

from torch.ao.quantization.quantize_pt2e import (prepare_pt2e, convert_pt2e)

sd_ex_prep = prepare_pt2e(sd_ex, sym255quantizer)

# dynamic quantization means no calibration necessary

sd_ex_cvt = torch.compile(convert_pt2e(sd_ex_prep), fullgraph=True, mode="reduce-overhead")

# sd_ex_cvt_jit = torch.jit.trace(
#     (sd_ex_cvt),
#     example_inputs=(
#         torch.zeros([512,512], dtype=torch.uint8),
#         torch.zeros([1,77,768], dtype=torch.float32) - 0.1,
#         torch.tensor([0.75], dtype=torch.float32),
#         torch.zeros([1,4,64,64], dtype=torch.float32)
#     )
# )

print("sd_ex_cvt_jit", sd_ex_cvt)

sdxs512oldnnx = torch.onnx.export(
    sd_ex_cvt,
    args=(
        torch.zeros([512,512], dtype=torch.uint8),
        torch.zeros([1,77,768], dtype=torch.float32) - 0.1,
        torch.tensor([0.75], dtype=torch.float32),
        torch.zeros([1,4,64,64], dtype=torch.float32)
    ),
    f="sd512q8.onnx",
    verbose=False,
    opset_version=20,
    input_names=["image", "prompt_embeds", "conditioning_scale", "latents"],
    output_names=["output_image"],
    external_data=False,
    dynamo=True,
    # dynamic_axes={"image": {0: "batch", 1: "channel", 2: "height", 3: "width"}},
)

