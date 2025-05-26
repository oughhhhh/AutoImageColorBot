import os
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from telegram.ext import CommandHandler

from DDColor_wrapper import DDColor
from inference import colorize

import nest_asyncio
nest_asyncio.apply()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я бот, который умеет разукрашивать черно-белые фотографии.\n\n"
        "📷 Просто отправь мне чёрно-белое изображение, и я превращу его в цветное с помощью нейросети 🎨"
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()
    await update.message.reply_text("Обрабатываю изображение...")

    colorized_bytes = DDColor(photo_bytes)
    await update.message.reply_photo(photo=colorized_bytes)

    colorized_bytes = colorize(photo_bytes)
    await update.message.reply_photo(photo=colorized_bytes)


if __name__ == "__main__":
    load_dotenv()
    TOKEN = os.getenv("BOT_TOKEN")

    # start backends
    os.system("sudo docker run -d -p 5000:5000 --gpus=all r8.im/piddnad/ddcolor@sha256:ca494ba129e44e45f661d6ece83c4c98a9a7c774309beca01429b58fce8aa695")


    os.system("clear")
    print("Running bot...")

    # start bot
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()
