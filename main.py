# main.py
import uvicorn
from fastapi import FastAPI
from bot import build_bot
import asyncio
from contextlib import asynccontextmanager 

from webhooks import router as webhook_router
# 1. NEW: Import your auto-registration tool from webhooks
from webhooks import auto_register_gcal_watch

@asynccontextmanager
async def lifespan(app: FastAPI):
    # [STARTUP ZONE] Everything here runs automatically on boot
    bot_app = build_bot()
    
    await bot_app.initialize()
    asyncio.create_task(bot_app.start())
    asyncio.create_task(bot_app.updater.start_polling())
    print("FastAPI server started & ARIA Bot is polling Telegram (Lifespan Mode)!")
    
    # 2. NEW: Fire the Google API handshake right after the bot starts
    auto_register_gcal_watch()
    
    yield  # <-- The server stays paused here while your app runs live
    
    # [SHUTDOWN ZONE] Runs when you hit Ctrl+C
    await bot_app.updater.stop()
    await bot_app.stop()
    await bot_app.shutdown()
    print("ARIA Bot and FastAPI server shut down cleanly via Lifespan.")

# Inject the lifespan manager into FastAPI
app = FastAPI(lifespan=lifespan)

# Plug in your webhook endpoints from webhooks.py
app.include_router(webhook_router)

@app.get("/")
def read_root():
    return {"ARIA_status": "Running"}

if __name__ == "__main__":
    # Note: Keep Ngrok running on port 5000 to match this!
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)