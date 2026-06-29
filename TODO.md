# ARIA TODO

## High Priority
- [ ] Verify `log_work` endpoint — Reclaim may not expose `/tasks/{id}/log`
- [ ] Task resolution quality — monitor embedding mismatches; tune thresholds if needed

## Medium Priority
- [ ] Debounce race condition — webhook fires before Reclaim updates; `asyncio.sleep(10)` is a bandaid
- [ ] OCI deployment — ngrok is not production-viable; need HTTPS via nginx + certbot or Cloudflare tunnel
- [ ] Voice messages — `voice_master` + `transcribe_audio` built; needs more testing

## Done (cleanup pass)
- [x] Rollback + cancel intercept with single-slot undo
- [x] Webhook `build_task_map` + reclaim cache invalidation
- [x] `message_master` error handler
- [x] Bool normalization in `inference.py`
- [x] Task map rebuild after writes (incremental upsert + full on composites)
- [x] `missed_work` deterministic router → `reschedule_missed_work`
- [x] `extend_task_instance` via `timeChunksRequired` (not `/planner/extend`)
- [x] Work/personal `eventCategory` on create_task (always ask before create)
- [x] 5-minute amendment window (`update_task` in place; undo still available)

## Backlog
- [ ] Researcher agent — orchestrator + research subagents + Tavily + Gmail
- [ ] Conversation history — pass prior Telegram messages as LLM context
- [ ] ARIA proactive pings — nudges based on deadlines, not just calendar blocks
- [ ] Create task flow — confirm duration/due date interactively before `create_task`
- [ ] Graph memory architecture
- [ ] Freemium productization
