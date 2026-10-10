# Dev Log

Build-window evidence for judges: everything below was written during HackRU Fall 2026
(Sat Oct 10, 12:00 PM ET → Sun Oct 11, 10:00 AM ET). `git log` is the source of truth.

---

## Sat Oct 10, 2026 (evening)

- **Repo + scaffold.** README (pitch, architecture, evaluation plan), MIT license,
  module skeleton, Colab prep layout. Chose the name *Trailblazer* — it blazes the
  trail from drug to disease through the knowledge graph.
- Design rules locked in: the agent never parses the trial's results section
  ("prior to public results" enforced in code, not by promise); the LLM narrates
  but never scores; UNASSESSABLE is a valid output.
- Next: CT.gov API client, then PrimeKG subset notebook (highest data risk — do it
  while fresh).
