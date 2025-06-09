import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
import requests
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

API_URL = "http://127.0.0.1:8000/answer"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def handle_start(message: types.Message) -> None:
    await message.answer(
        text=f"Hello, {message.from_user.full_name}!\n This bot can answer medical quesitons"
    )


@dp.message()
async def handle_message(message: types.Message):
    user_q = message.text
    try:
        resp = requests.post(API_URL, json={"question": user_q})
        resp.raise_for_status()
        data = resp.json()
        reply = f"Answer:\n{data['answer']}\nSources:\n" + "\n".join(data["sources"])
    except Exception as e:
        logging.error(e)
        reply = "Извините, произошла ошибка при получении ответа."
    await message.reply(reply)


async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
