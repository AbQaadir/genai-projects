import torch
from diffusers import StableDiffusion3Pipeline
import gc
import contextlib

def load_pipeline(device, dtype):
    # Load the model with specified precision
    return StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=dtype,
        text_encoder_3=None,
        tokenizer_3=None
    ).to(device)

def generate_and_return_image(pipeline, text, device):
    with torch.no_grad():
        with torch.cuda.amp.autocast() if device == "cuda" else contextlib.nullcontext():
            image = pipeline(
                prompt=text,
                num_inference_steps=40,  # Further reduce inference steps to save memory
                height=512,  # Further reduce the height to save memory
                width=512,   # Further reduce the width to save memory
                guidance_scale=9.0,  # Corrected the parameter name
            ).images[0]
    return image

def generate_image(text):
    # Check the CUDA device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Free up memory
        torch.cuda.empty_cache()
        gc.collect()

        # Load the pipeline
        pipeline = load_pipeline(device, torch.float16 if device == "cuda" else torch.float32)
        
        # Generate and return the image
        image = generate_and_return_image(pipeline, text, device)
        
        # Unload the model from GPU to free up memory
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()
        
        return image
    
    except RuntimeError as e:
        if "out of memory" in str(e) and device == "cuda":
            print("CUDA out of memory. Clearing cache and retrying on CPU with float32...")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                device = "cpu"
                # Reload the pipeline with float32 for CPU inference
                pipeline = load_pipeline(device, torch.float32)
                # Generate and return the image
                image = generate_and_return_image(pipeline, text, device)
                # Unload the model from memory to free up memory
                del pipeline
                torch.cuda.empty_cache()
                gc.collect()
                return image
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("Out of memory even on CPU. Please try reducing the image size or inference steps further.")
                    return None
                else:
                    raise e
        else:
            raise e

# Generate the image
text = "Generate a detailed profile of a fictional Sri Lankan cricket player. Include their full name, age, place of birth, batting and bowling styles, notable achievements, key skills, career highlights, personal background, and personality traits."
image = generate_image(text)
if image:
    # Save the image if generated successfully
    image.save("output.png")
else:
    print("Failed to generate image due to memory constraints.")
