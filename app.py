from diffusers import StableDiffusionPipeline
from moviepy import ImageSequenceClip
import gradio as gr
import os

# Initialize Stable Diffusion pipeline
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipeline.to("cuda")  # Use GPU for faster processing

# Directory to save generated images
output_folder = "generated_images"
os.makedirs(output_folder, exist_ok=True)

# Generate images from text prompts
def generate_images(prompt, num_frames=10):
    images = []
    for i in range(num_frames):
        # Modify the prompt slightly for each frame (optional)
        frame_prompt = f"{prompt}, frame {i+1}"
        image = pipeline(frame_prompt).images[0]
        image_path = os.path.join(output_folder, f"frame_{i+1}.png")
        image.save(image_path)
        images.append(image_path)
    return images

# Create a video from the generated images
def create_video(images, output_video="output.mp4"):
    video_clip = ImageSequenceClip(images, fps=24)
    video_clip.write_videofile(output_video, codec="libx264")
    return output_video

# Gradio function to process text input and generate a video
def process_text_to_video(prompt):
    images = generate_images(prompt)
    video_path = create_video(images)
    return video_path

# Gradio interface
interface = gr.Interface(
    fn=process_text_to_video,
    inputs="text",
    outputs="video",
    title="Text-to-Video Generator",
    description="Enter text prompts to generate videos using Stable Diffusion."
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
