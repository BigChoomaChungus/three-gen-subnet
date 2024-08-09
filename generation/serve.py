from io import BytesIO

from fastapi import FastAPI, Depends, Form
from fastapi.responses import Response, StreamingResponse
import uvicorn
import argparse
import base64
from time import time

from omegaconf import OmegaConf

from shap_e.diffusion.gaussian_diffusion import create_gaussian_diffusion
from shap_e.models.download import load_model
from shap_e.rendering.mesh import decode_latent_mesh
from shap_e.util.image_util import create_prompt_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10006)
    parser.add_argument("--config", default="configs/text_mv.yaml")
    return parser.parse_args()


args = get_args()
app = FastAPI()


def get_config() -> OmegaConf:
    config = OmegaConf.load(args.config)
    return config


def get_models(config: OmegaConf = Depends(get_config)):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model('shap-e', device=device)
    return model


@app.post("/generate/")
async def generate(
    prompt: str = Form(),
    config: OmegaConf = Depends(get_config),
    model = Depends(get_models),
):
    buffer = await _generate(model, config, prompt)
    buffer = base64.b64encode(buffer.getbuffer()).decode("utf-8")
    return Response(content=buffer, media_type="application/octet-stream")


async def _generate(model, opt: OmegaConf, prompt: str) -> BytesIO:
    start_time = time()
    
    # Create a prompt image (if required by the model)
    prompt_image = create_prompt_image(prompt, device='cuda')
    
    # Generate a 3D object from the text prompt
    latent = model.sample_latent(prompt_image)
    mesh = decode_latent_mesh(latent)
    
    # Save the 3D object as a PLY file in a buffer
    buffer = BytesIO()
    mesh.save_ply(buffer)
    buffer.seek(0)
    
    print(f"[INFO] It took: {(time() - start_time) / 60.0} min")
    return buffer


@app.post("/generate_raw/")
async def generate_raw(
    prompt: str = Form(),
    opt: OmegaConf = Depends(get_config),
    model = Depends(get_models),
):
    buffer = await _generate(model, opt, prompt)
    return Response(content=buffer.getvalue(), media_type="application/octet-stream")


@app.post("/generate_model/")
async def generate_model(
    prompt: str = Form(),
    opt: OmegaConf = Depends(get_config),
    model = Depends(get_models),
) -> Response:
    buffer = await _generate(model, opt, prompt)
    return StreamingResponse(buffer, media_type="application/octet-stream")


@app.post("/generate_video/")
async def generate_video(
    prompt: str = Form(),
    video_res: int = Form(1088),
    opt: OmegaConf = Depends(get_config),
    model = Depends(get_models),
):
    start_time = time()
    
    # Create a prompt image (if required by the model)
    prompt_image = create_prompt_image(prompt, device='cuda')
    
    # Generate a 3D object from the text prompt
    latent = model.sample_latent(prompt_image)
    mesh = decode_latent_mesh(latent)
    
    print(f"[INFO] It took: {(time() - start_time) / 60.0} min")
    
    # Render video from the generated mesh
    video_utils = VideoUtils(video_res, video_res, 5, 5, 10, -30, 10)
    buffer = video_utils.render_video(mesh)
    
    return StreamingResponse(buffer, media_type="video/mp4")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port)

