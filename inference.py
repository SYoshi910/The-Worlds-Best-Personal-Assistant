from groq import AsyncGroq
from config import GROQ_TOKEN
from datetime import datetime
import json
from reclaim import get_active_tasks
import parsedatetime as pdt
from zoneinfo import ZoneInfo
from system_prompts import SYSTEM_PROMPT_CAL

client = AsyncGroq(api_key=GROQ_TOKEN)

# Initialize the NLP calendar engine
cal = pdt.Calendar()



def parse_to_iso(natural_date_str: str, base_date: datetime) -> str:
    """Converts 'next tuesday' into '2026-06-30T17:00:00-07:00'"""
    if not natural_date_str:
        return None
        
    # Parse using parsedatetime, anchoring to our base_date
    time_struct, parse_status = cal.parse(natural_date_str, sourceTime=base_date)
    
    # A parse_status > 0 means it successfully found a date/time
    if parse_status > 0:
        parsed = datetime(*time_struct[:6], tzinfo=ZoneInfo("America/Los_Angeles"))
        return parsed.strftime('%Y-%m-%dT%H:%M:%S-07:00')
        
    return None

async def call_llm(message: list, model: str = "llama-3.3-70b-versatile") -> dict:
    # 1. Generate a timezone-aware current time
    current_time = datetime.now(ZoneInfo("America/Los_Angeles"))
    
    formatted_prompt = SYSTEM_PROMPT_CAL.format(
        now=current_time.strftime("%A, %B %d, %Y at %I:%M %p %Z"),
        weekday=current_time.strftime("%A"),
        active_titles=[t["title"] for t in get_active_tasks()] 
    )

    full_messages = [{"role": "system", "content": formatted_prompt}] + message

    response = await client.chat.completions.create(
        model=model,
        messages=full_messages,
        max_tokens=1024,
        temperature=0.3,
    )
    print(full_messages)
    raw = response.choices[0].message.content
   
    print(raw)
    

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            return {"action_required": False, "reply": raw, "calls": [], "reasoning": "parse failed"}

    # 2. Intercept and parse the dates before executing the tools
    if data.get("action_required") and data.get("calls"):
        for call in data["calls"]:
            params = call.get("params", {})
            
            # Convert create_task relative dates
            if call["function"] == "create_task" and "due_date_natural" in params:
                old = params['due_date_natural']
                params["due_date"] = parse_to_iso(params.pop("due_date_natural"), current_time)
                print(f"due date changed from '{old}' to '{params['due_date']}'")
                
            # Convert create_event relative dates
            elif call["function"] == "create_event":
                if "start_time_natural" in params:
                    old_start = params["start_time_natural"]
                    params["start"] = parse_to_iso(params.pop("start_time_natural"), current_time)
                    print(f"start time changed from '{old_start}' to '{params['start']}'")
                if "end_time_natural" in params:
                    old_end = params["end_time_natural"]
                    params["end"] = parse_to_iso(params.pop("end_time_natural"), current_time)
                    print(f"end time changed from '{old_end}' to '{params['end']}'")

    return data

async def transcribe_audio(audio_bytes: bytes) -> str:
    transcription = await client.audio.transcriptions.create(
        file=("voice.ogg", audio_bytes),
        model="whisper-large-v3-turbo",
    )
    return transcription.text

