# ARIA TODO

## 🔴 High Priority
- [ ] Fix task resolution — replace difflib with Gemini embeddings (cosine similarity) — cal_helper.py get_task_by_query() is built, fix model name (run list_models() to confirm)
- [ ] Rebuild task map on calendar webhook fire (currently only builds at startup)
- [ ] Add error handler to `message_master` so telegram exceptions don't crash silently
- [ ] Add rollback feature — 20s buffer before dispatch fires, "cancel" intercept in message_master
- [ ] Verify log_work endpoint — Reclaim may not expose /tasks/{id}/log
- [ ] Verify extend_task_instance endpoint — /planner/extend/{id} unconfirmed

## 🟡 Medium Priority
- [ ] Stale task map refresh — rebuild after any write operation (complete_task, create_task)
- [ ] Debounce race condition — webhook fires before Reclaim updates, asyncio.sleep(3) is a bandaid
- [ ] `prep_next_block` — handle case where no upcoming events exist (crashes on empty list)
- [ ] OCI deployment — ngrok is not production-viable, need HTTPS via nginx + certbot or Cloudflare tunnel
- [ ] Voice messages — voice_master + transcribe_audio built, needs testing
- [ ] Validate dispatch output before execution — deterministic checks (past times, unknown functions)
- [ ] Normalize action_required and clarification_required to actual bools after LLM parse

## 🟢 Backlog
- [ ] Researcher agent — orchestrator (Antigravity/Gemini) + research subagents (Llama 8B) + adversarial + citation + Tavily + email via Gmail
- [ ] Conversation history — pass prior Telegram messages as context to LLM
- [ ] ARIA proactive pings — nudges based on deadlines and priorities, not just calendar blocks
- [ ] Create task flow — confirm + gather duration/due date interactively before calling create_task
- [ ] Multi-action UX — tell user what actions were taken, not just a generic reply
- [ ] Graph memory architecture — state-conditioned edge weights for long-term context
- [ ] Upgrade task resolution to Gemini Embedding 2 once model name confirmed
- [ ] Freemium productization — if ARIA becomes a real product