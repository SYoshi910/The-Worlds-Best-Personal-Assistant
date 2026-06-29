import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from bot import bot_app, prep_next_block, scheduler
from cal_helper import build_task_map
from config import DEV_RELOAD
from reclaim import close_client
from webhooks import register_gcal_watch, router as webhook_router, schedule_watch_renewal


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the bot, calendar watch, and schedulers; tear down on shutdown."""
    scheduler.start()

    await bot_app.initialize()
    bot_task = asyncio.create_task(bot_app.start())
    poll_task = asyncio.create_task(bot_app.updater.start_polling())
    print("FastAPI server started & ARIA Bot is polling Telegram (Lifespan Mode)!")

    await register_gcal_watch()
    schedule_watch_renewal(scheduler)
    await build_task_map()
    await prep_next_block()

    yield

    for task in (poll_task, bot_task):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    await bot_app.updater.stop()
    await bot_app.stop()
    await bot_app.shutdown()
    await close_client()
    print("ARIA Bot and FastAPI server shut down cleanly via Lifespan.")


app = FastAPI(lifespan=lifespan)
app.include_router(webhook_router)


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"ARIA_status": "Running"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=DEV_RELOAD)
