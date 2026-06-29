# ARIA TODO

## 🔴 High Priority
- [ ] Fix task resolution — replace difflib with Gemini embeddings (cosine similarity)
- [ ] Rebuild task map on calendar webhook fire (currently only builds at startup)
- [ ] Add error handler to `message_master` so telegram exceptions don't crash silently
- [ ] Add rollback feature 

## 🟡 Medium Priority
- [ ] Stale task map refresh — rebuild periodically or after any write operation
- [ ] Add conversation memory — ARIA currently has no context of prior messages in a session
- [ ] `prep_next_block` — handle case where no upcoming events exist (crashes on empty list)
- [ ] OCI deployment — ngrok is not production-viable, need HTTPS via nginx + certbot or Cloudflare tunnel

## 🟢 Backlog
- [ ] Researcher agent — orchestrator (Gemini) + research subagents (Llama 8B) + adversarial + citation
- [ ] Conversation history — pass prior Telegram messages as context to LLM
- [ ] ARIA proactive pings — not just calendar blocks, but nudges based on deadlines and priorities
- [ ] Create task flow — confirm + gather duration/due date interactively before calling `create_task`
- [ ] Multi-action UX — tell user what actions were taken, not just a generic reply
- [ ] Graph memory architecture — state-conditioned edge weights for long-term context
- [ ] Freemium productization — if ARIA becomes a real product