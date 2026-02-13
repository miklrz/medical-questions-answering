import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import requests
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()


def build_feedback_keyboard(request_id: str) -> InlineKeyboardMarkup:
    """Inline keyboard: Полезно / Не полезно."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="👍 Полезно",
                    callback_data=f"fb:{request_id}:1",
                ),
                InlineKeyboardButton(
                    text="👎 Не полезно",
                    callback_data=f"fb:{request_id}:0",
                ),
            ],
        ]
    )


@dp.message(CommandStart())
async def handle_start(message: types.Message) -> None:
    await message.answer(
        text=f"Hello, {message.from_user.full_name}!\nЭтот бот отвечает на медицинские вопросы."
    )


@dp.message(F.text)
async def handle_message(message: types.Message):
    user_q = message.text
    try:
        resp = requests.post(f"{API_URL}/answer", json={"question": user_q}, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        answer = data.get("answer", "")
        confidence = data.get("confidence", 0)
        sources = data.get("sources", [])
        warnings = data.get("warnings", [])
        request_id = data.get("request_id", "")

        parts = [f"**Ответ:**\n{answer}"]
        if warnings:
            parts.append(f"\n⚠️ {chr(10).join(warnings)}")
        parts.append(f"\nУверенность: {confidence:.0%}")
        if sources:
            parts.append("\n**Источники:**\n" + "\n".join(f"• {s[:200]}..." if len(s) > 200 else f"• {s}" for s in sources[:3]))

        reply = "\n".join(parts)
        keyboard = build_feedback_keyboard(request_id) if request_id else None
        await message.reply(reply, reply_markup=keyboard, parse_mode="Markdown")
    except Exception as e:
        logging.error(e)
        await message.reply("Извините, произошла ошибка при получении ответа.")


@dp.callback_query(F.data.startswith("fb:"))
async def handle_feedback(callback: types.CallbackQuery):
    """Handle useful/not useful feedback."""
    try:
        _, request_id, useful_str = callback.data.split(":")
        useful = useful_str == "1"

        requests.post(
            f"{API_URL}/feedback",
            params={"request_id": request_id, "useful": useful},
            timeout=5,
        )

        await callback.answer("Спасибо за оценку!")
        # Remove keyboard after feedback
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception as e:
        logging.error(e)
        await callback.answer("Ошибка при отправке оценки.")


async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
