# main.py
import uvicorn
from fastapi import FastAPI
from bot import bot_app, prep_next_block, scheduler
import asyncio
from contextlib import asynccontextmanager 

from webhooks import router as webhook_router

from webhooks import auto_register_gcal_watch

@asynccontextmanager
async def lifespan(app: FastAPI):
    # [STARTUP ZONE] Everything here runs automatically on boot
    
    scheduler.start()
    
    await bot_app.initialize()
    asyncio.create_task(bot_app.start())
    asyncio.create_task(bot_app.updater.start_polling())
    print("FastAPI server started & ARIA Bot is polling Telegram (Lifespan Mode)!")
    auto_register_gcal_watch()
    prep_next_block()
    
    yield  # <-- The server stays paused here while app runs live
    
    # [SHUTDOWN ZONE] Runs when you hit Ctrl+C
    await bot_app.updater.stop()
    await bot_app.stop()
    await bot_app.shutdown()
    print("ARIA Bot and FastAPI server shut down cleanly via Lifespan.")

app = FastAPI(lifespan=lifespan)

app.include_router(webhook_router)

@app.get("/")
def read_root():
    return {"ARIA_status": "Running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)