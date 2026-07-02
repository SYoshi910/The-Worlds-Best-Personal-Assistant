import asyncio
import atexit
import os
import socket
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

import schedule_cache
from bot import bot_app, prep_next_block, scheduler
from cal_helper import build_task_map
from config import DEV_RELOAD
from embeddings import warmup_embeddings
from reclaim import close_client
from webhooks import register_gcal_watch, router as webhook_router, schedule_watch_renewal

# Dedicated port — OS releases the bind when the process exits (including uvicorn reload).
_INSTANCE_LOCK_PORT = 5001
_LOCK_RETRIES = 15 if DEV_RELOAD else 1
_LOCK_RETRY_SEC = 0.5
_lock_socket: socket.socket | None = None
_lock_held = False


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _lock_peers() -> set[int]:
    """Other live PIDs running this project's main.py (or its uvicorn worker)."""
    root = str(_project_root()).lower()
    me = os.getpid()
    peers: set[int] = set()
    try:
        import psutil
    except ImportError:
        return peers

    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = proc.info["pid"]
            if pid == me:
                continue
            cmdline = proc.info["cmdline"] or []
            cmd = " ".join(cmdline).lower()
            if root not in cmd:
                continue
            if "main.py" in cmd or "main:app" in cmd:
                peers.add(pid)
            elif any("spawn_main" in part for part in cmdline):
                parent = psutil.Process(pid).parent()
                if parent and parent.pid != me:
                    pcmd = " ".join(parent.cmdline() or []).lower()
                    if root in pcmd and "main.py" in pcmd:
                        peers.add(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return peers


def _parent_pids() -> set[int]:
    pids = {os.getpid()}
    try:
        import psutil

        parent = psutil.Process(os.getpid()).parent()
        if parent:
            pids.add(parent.pid)
    except (ImportError, psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return pids


def _lock_error(peers: set[int]) -> RuntimeError:
    peer_list = ", ".join(str(p) for p in sorted(peers)) if peers else "?"
    return RuntimeError(
        f"Another ARIA instance is already running (PID {peer_list}). "
        "Stop it before starting a second copy. "
        f"On Windows: Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*Project*main.py*' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
    )


def _try_bind_lock_port() -> bool:
    global _lock_socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", _INSTANCE_LOCK_PORT))
    except OSError:
        sock.close()
        return False
    sock.listen(1)
    _lock_socket = sock
    return True


def _acquire_instance_lock() -> None:
    """Refuse to start if another live worker is already polling Telegram."""
    global _lock_held
    for attempt in range(_LOCK_RETRIES):
        if _try_bind_lock_port():
            _lock_held = True
            atexit.register(_release_instance_lock)
            peers = _lock_peers() - _parent_pids()
            if peers:
                _release_instance_lock()
                raise _lock_error(peers)
            return
        if attempt < _LOCK_RETRIES - 1:
            time.sleep(_LOCK_RETRY_SEC)

    raise _lock_error(_lock_peers() - _parent_pids())


def _release_instance_lock() -> None:
    global _lock_socket, _lock_held
    if not _lock_held:
        return
    _lock_held = False
    if _lock_socket is not None:
        _lock_socket.close()
        _lock_socket = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the bot, calendar watch, and schedulers; tear down on shutdown."""
    _acquire_instance_lock()
    poll_task = None
    bot_task = None
    try:
        scheduler.start()

        await bot_app.initialize()
        bot_task = asyncio.create_task(bot_app.start())
        poll_task = asyncio.create_task(bot_app.updater.start_polling())
        print("FastAPI server started & ARIA Bot is polling Telegram (Lifespan Mode)!")

        await register_gcal_watch()
        schedule_watch_renewal(scheduler)
        await asyncio.to_thread(warmup_embeddings)
        await build_task_map()
        await schedule_cache.refresh()
        await prep_next_block()

        yield
    finally:
        if poll_task is not None and bot_task is not None:
            for task in (poll_task, bot_task):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            try:
                await bot_app.updater.stop()
            except Exception:
                pass
            try:
                await bot_app.stop()
            except Exception:
                pass
            try:
                await bot_app.shutdown()
            except Exception:
                pass
        try:
            await close_client()
        except Exception:
            pass
        _release_instance_lock()
        print("ARIA Bot and FastAPI server shut down cleanly via Lifespan.")


app = FastAPI(lifespan=lifespan)
app.include_router(webhook_router)


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"ARIA_status": "Running"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=DEV_RELOAD)
