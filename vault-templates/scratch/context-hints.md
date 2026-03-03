# Context Hints

This file is injected into your system prompt on every LLM call — conversations and
heartbeats alike. Use it to carry persistent framing forward without repeating yourself.

Write things here that should shape how you think in every session:
- Preferences your owner has expressed ("prefers bullet points over prose")
- Active projects you're mid-way through ("researching knarr network topology")
- Framing that helps you orient fast ("I am the agent for a small team of 3")
- Flags you've set for yourself ("be brief today, owner is busy")

Keep it short. It costs tokens on every call.
Update it when your context shifts. Delete the content when it's stale.

---

*(No hints set yet — write here via: `knowledge_vault action=write path=scratch/context-hints vault=default content="..."`)*
