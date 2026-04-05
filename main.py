import os
from io import BytesIO
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

app = Flask(__name__)

# Load the uncensored nude/undress model (works on CPU for free Render; use GPU instance for speed)
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=torch.float16,
    safety_checker=None  # Disable safety to allow full nude
)
pipe.to("cpu")  # Change to "cuda" if you have GPU

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PORT = int(os.getenv("PORT", 10000))

application = Application.builder().token(TOKEN).build()

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # User uploads any image → bot turns it into nude
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    image_bytes = await file.download_as_bytearray()
    
    input_img = Image.open(BytesIO(image_bytes)).convert("RGB").resize((512, 512))
    
    # Strong nude prompt - fully explicit, no clothes
    prompt = "completely nude woman, fully naked, explicit nsfw, detailed realistic skin texture, perfect anatomy, high quality, seductive pose"
    negative_prompt = "clothes, clothing, dress, underwear, bra, panties, bikini, covered, censored, blurry"
    
    # Generate the nude image
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=input_img,
        controlnet_conditioning_scale=0.75,
        num_inference_steps=25,
        guidance_scale=7.5
    ).images[0]
    
    # Send back the nude picture
    output = BytesIO()
    result.save(output, format="PNG")
    output.seek(0)
    
    await update.message.reply_photo(
        photo=output,
        caption="Here is your fully nude version 🔥"
    )

application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

@app.route('/', methods=['POST'])
async def webhook():
    update = Update.de_json(request.get_json(force=True), application.bot)
    await application.process_update(update)
    return 'OK', 200

@app.route('/')
def index():
    return "Nude Image Generator Bot is running - Upload any photo to get nude version!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
