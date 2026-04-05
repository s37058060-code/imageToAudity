import os
from io import BytesIO
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# Load NSFW undress model (use a strong uncensored one; download weights or use HF cache)
# For Render, pre-load or use a lighter model to avoid memory issues
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
pipe.to("cpu")  # Render free has limited GPU; switch to cuda if you upgrade instance

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PORT = int(os.getenv("PORT", 8080))

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    image_bytes = await file.download_as_bytearray()
    input_img = Image.open(BytesIO(image_bytes)).resize((512, 512))
    
    prompt = "fully nude woman, completely naked, explicit, detailed realistic skin, high quality, nsfw"
    negative_prompt = "clothes, clothing, underwear, dressed, bra, panties"
    
    result = pipe(prompt=prompt, negative_prompt=negative_prompt, image=input_img, controlnet_conditioning_scale=0.7, num_inference_steps=20).images[0]
    
    output = BytesIO()
    result.save(output, format="PNG")
    output.seek(0)
    
    await update.message.reply_photo(photo=output)

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    # For webhook (recommended for Render Web Service)
    webhook_url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME', 'your-app.onrender.com')}/"
    app.run_webhook(listen="0.0.0.0", port=PORT, url_path="", webhook_url=webhook_url)
    
    # Alternative: if polling, use app.run_polling() but free tier may sleep

if __name__ == "__main__":
    main()
