import uvicorn
from fastapi import FastAPI
from bot import build_bot
import asyncio
from contextlib import asynccontextmanager  # <-- 1. Import this helper tool
#from webhooks import router as webhook_router

# 2. Define a unified Lifespan Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [STARTUP ZONE] Everything before 'yield' runs when the server boots
    bot_app = build_bot()
    
    await bot_app.initialize()
    asyncio.create_task(bot_app.start())
    asyncio.create_task(bot_app.updater.start_polling())
    print("🚀 FastAPI server started & ARIA Bot is polling Telegram (Lifespan Mode)!")
    
    yield  # <-- THE SPLIT: The server stays paused here while your app runs live
    
    # [SHUTDOWN ZONE] Everything after 'yield' runs when you hit Ctrl+C
    await bot_app.updater.stop()
    await bot_app.stop()
    await bot_app.shutdown()
    print("🛑 ARIA Bot and FastAPI server shut down cleanly via Lifespan.")

# 3. Pass the lifespan instructions directly into your FastAPI engine
app = FastAPI(lifespan=lifespan)

#app.include_router(webhook_router)

@app.get("/")
def read_root():
    return {"ARIA_status": "Running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)