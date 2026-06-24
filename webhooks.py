from fastapi import FastAPI, Request, Header, Response
from config import MY_CUSTOM_TOKEN
from reclaim import check_next_upcoming_task

app = FastAPI()

@app.post("/gcal-webhook")
async def handle_gcal_notification(
    request: Request,
    x_goog_resource_state: str = Header(None),
    x_goog_channel_token: str = Header(None)
):
    if x_goog_resource_state == "sync":
        print("Successfully synchronized webhook channel with Google!")
        return Response(status_code=200)

    if x_goog_channel_token != MY_CUSTOM_TOKEN:
        return Response(status_code=403, content="Unauthorized token")

    if x_goog_resource_state == "exists":
        next_task = check_next_upcoming_task()
        if next_task:
            print(f"Sync Triggered! Next up: {next_task.get('title')}")
            # TODO: Trigger whatever Telegram bot function or automation you want here!

    return Response(status_code=200)