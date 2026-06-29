import numpy as np
import google.generativeai as genai
import reclaim
from config import GEMINI_TOKEN
import difflib

genai.configure(api_key=GEMINI_TOKEN)

FUNCTION_MAP = {
    "log_work": reclaim.log_work,
    "reschedule_task": reclaim.reschedule_task,
    "create_event": reclaim.create_gcal_event,
    "create_task": reclaim.create_task,
    "extend_task_total": reclaim.extend_task_total,
    "complete_task": reclaim.complete_task,
}

TASK_MAP = {}       # id → title
_task_list = []     # ordered list of ids
_embeddings = None

def dispatch(calls: list):
    results = {}

    for call in calls:
        fn_name = call.get("function")
        params = call.get("params", {})
        alias = call.get("result_alias")

        resolved_params = {}
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
                ref = v[2:-2].strip()
                alias_name, field = ref.split(".")
                resolved_params[k] = results.get(alias_name, {}).get(field)
            else:
                resolved_params[k] = v

        # resolve task_query → task_id before calling function
        if "task_query" in resolved_params:
            task = get_task_by_query(resolved_params.pop("task_query"))
            if not task:
                print(f"⚠️ Could not resolve task, skipping {fn_name}")
                continue
            resolved_params["task_id"] = task["id"]

        fn = FUNCTION_MAP.get(fn_name)
        if not fn:
            print(f"⚠️ Unknown function: {fn_name}")
            continue

        result = fn(**resolved_params)

        if alias:
            results[alias] = result if isinstance(result, dict) else {}

    return results

def build_task_map():
    global TASK_MAP, _task_list, _embeddings

    tasks = reclaim.get_active_tasks()
    TASK_MAP = {t["id"]: t["title"] for t in tasks}
    _task_list = list(TASK_MAP.keys())
    titles = list(TASK_MAP.values())

    result = genai.embed_content(
        model="models/gemini-embedding-2",
        content=titles,
        task_type="retrieval_document"
    )
    _embeddings = np.array(result["embedding"])
    print(f"✅ Task map built: {len(TASK_MAP)} tasks indexed")

def get_task_by_query(query: str) -> dict | None:
    if not TASK_MAP or _embeddings is None:
        build_task_map()
    
    titles = list(TASK_MAP.values())
    ids = list(TASK_MAP.keys())
    
    # 1. try difflib first
    matches = difflib.get_close_matches(query, titles, n=1, cutoff=0.6)
    if matches:
        matched_title = matches[0]
        task_id = ids[titles.index(matched_title)]
        print(f"✅ difflib: '{query}' → '{matched_title}'")
        return reclaim.get_task(task_id)
    
    # 2. fall back to Gemini embeddings
    print(f"⚡ difflib miss, falling back to Gemini for '{query}'")
    result = genai.embed_content(
        model="models/gemini-embedding-2",
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = np.array(result["embedding"])
    
    similarities = np.dot(_embeddings, query_embedding) / (
        np.linalg.norm(_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])
    
    if best_score < 0.5:
        print(f"⚠️ No confident match for '{query}' (best: {best_score:.2f})")
        return None
    
    task_id = ids[best_idx]
    print(f"✅ Gemini: '{query}' → '{TASK_MAP[task_id]}' (score: {best_score:.2f})")
    return reclaim.get_task(task_id)

for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print(m.name)