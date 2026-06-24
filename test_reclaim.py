from reclaim import get_task, extend_task
import difflib
# Your confirmed live working Task ID
TARGET_TASK_ID = "13143367"

def run_local_api_test():
    print("=== 🔍 RUNNING PYTHON GET_TASK TEST ===")
    task_data = get_task(TARGET_TASK_ID)
    
    if not task_data:
        print("❌ Script failed: Could not read from Reclaim API.")
        return
        
    print(f"📝 Task Title:      {task_data.get('title')}")
    print(f"📊 Current Status:  {task_data.get('status')}")
    current_chunks = task_data.get('timeChunksRequired')
    print(f"⏳ Total Chunks:    {current_chunks}")
    
    print("\n=== 🚀 RUNNING PYTHON EXTEND_TASK TEST ===")
    # Let's use Python to add 1 more hour (4 chunks) to whatever it is right now! [cite: 32]
    new_chunk_total = current_chunks + 4
    print(f"Sending request to update total chunks from {current_chunks} ──► {new_chunk_total}...")
    
    success = extend_task(TARGET_TASK_ID, new_chunk_total)
    if success:
        print("🎉 Python code successfully updated your live calendar!")
    else:
        print("❌ Modification failed.")

if __name__ == "__main__":
    print('nice')