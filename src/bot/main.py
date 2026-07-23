import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import asyncio
import httpx

from src.bot.config import get_bot_config

config = get_bot_config()
if not config.telegram.token:
    raise RuntimeError("TELEGRAM_TOKEN is required via environment or bot config")

bot = Bot(token=config.telegram.token)
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
    logging.info("Got message")
    user_q = message.text
    thinking = await message.reply("⏳ Обрабатываю запрос...")
    try:
        async with httpx.AsyncClient(timeout=config.api.answer_timeout_seconds) as client:
            logging.info("Sending request to api")
            resp = await client.post(
                f"{config.api.base_url}/answer",
                json={"question": user_q},
            )
            logging.info("Received response")
            resp.raise_for_status()
            data = resp.json()

        answer = data.get("answer", "")
        confidence = data.get("confidence", 0)
        sources = data.get("sources", [])
        warnings = data.get("warnings", [])
        request_id = data.get("request_id", "")
        requires_doctor = data.get("requires_doctor_visit", False)

        parts = [f"*Ответ:*\n{answer}"]

        if requires_doctor:
            parts.append("\n🏥 *Рекомендуется визит к врачу*")

        if warnings:
            parts.append("\n⚠️ " + "\n⚠️ ".join(warnings))

        parts.append(f"\nУверенность: {confidence:.0%}")

        if sources:
            short_sources = [
                f"• {s[:200]}..." if len(s) > 200 else f"• {s}" for s in sources[:3]
            ]
            parts.append("\n*Источники:*\n" + "\n".join(short_sources))

        reply = "\n".join(parts)
        keyboard = build_feedback_keyboard(request_id) if request_id else None
        await thinking.delete()
        await message.reply(
            reply,
            reply_markup=keyboard,
            parse_mode=config.telegram.parse_mode,
        )

    except Exception as e:
        logging.error(f"Error handling message: {e}")
        await thinking.edit_text(
            "Извините, произошла ошибка при получении ответа. Попробуйте позже."
        )


@dp.callback_query(F.data.startswith("fb:"))
async def handle_feedback(callback: types.CallbackQuery):
    """Handle useful/not useful feedback."""
    try:
        parts = callback.data.split(":")
        request_id = parts[1]
        useful = parts[2] == "1"

        async with httpx.AsyncClient(timeout=config.api.feedback_timeout_seconds) as client:
            await client.post(
                f"{config.api.base_url}/feedback",
                params={"request_id": request_id, "useful": useful},
            )

        await callback.answer("Спасибо за оценку!")
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception as e:
        logging.error(f"Feedback error: {e}")
        await callback.answer("Ошибка при отправке оценки.")


async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
