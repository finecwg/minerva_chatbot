# telegram.py
from chatting import MinervaQA
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']

minerva_qa = MinervaQA()

async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('Hi! Ask me anything about Minerva University.')

async def handle_message(update: Update, context: CallbackContext) -> None:
    query = update.message.text
    response = minerva_qa.answer_query(query)
    await update.message.reply_text(response)

def main() -> None:
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()

if __name__ == '__main__':
    main()
