# ARIA TODO
- [ ] OCI deployment — ngrok is not production-viable; need HTTPS via nginx + certbot or Cloudflare tunnel
- [ ] Model reasoning tiers — per-model `reasoning` profile (off/low), deterministic complexity signals (clarify/amend/compound), default off + escalate to low on parse/validation failure
- [x] Use Gemma 4's ability to understand pictures to set up the screenshot - task workflow

## Backlog
- [ ] Overdue tasks query — `get_overdue_tasks` read tool (or extend weekly snapshot) so "do I have overdue tasks?" works
- [ ] Researcher agent — orchestrator + research subagents + Tavily + Gmail
- [ ] Conversation history — pass prior Telegram messages as LLM context
- [ ] ARIA proactive pings — nudges based on deadlines, not just calendar blocks
- [ ] Create task flow — confirm duration/due date interactively before `create_task`
- [ ] Projects — group related tasks together under a named project
- [ ] Habits — recurring tasks (Reclaim API permitting); e.g. "daily habit X"
- [ ] Graph memory architecture
- [ ] Freemium productization
