# Ideation — How GoneBy Was Born

*Six weeks, one git warning, and an inversion nobody had made. This is the
story of the idea. For the mechanism, read [OVERVIEW.md](OVERVIEW.md); this
page deliberately won't repeat it.*

---

## It started with orphaned stashes

September 24, 2026. A `hermes update` warning: three autostash entries,
older than seven days — decisions a system made, interrupted, and left in
limbo. Nobody ever came for them.

The cleanup question quietly became the real question: **where does a
working agent's knowledge go when nobody claims it?** Sessions compact,
context windows roll, transcripts expire. The decisions that shaped a
project just... stop existing. Hermes' answer today is `session_search` —
grep over history. Useful, but keyword-shaped. Nobody had a good answer for
*meaning*.

## The abstraction argument

Same session, a few messages in, a principle got laid down that turned out
to be load-bearing:

> *"We should be able to abstract away to defining things into higher tools
> that don't require the original reference... If we can define a system it
> should just do what it needs to do without a long word document — that's
> how systems work, they just get more integrated."*

Read that again as an engineering mandate: **a system's intelligence should
live in its structure, not in prose you have to obey.** That principle
eventually decides GoneBy's most contrarian feature — the capture phase is
deliberately stupid so the rest can afford to be smart.

## The first sketch: pipe it out

> *"Would there be any use to piping context from an LLM project out to a
> vector DB for referential storage... say a project that would compact 3-4
> times over its course? Once the product's done, compress and deduplicate
> the past."*

The honest first response was pushback: raw output between compactions is
mostly noise — retries, dead ends, "let me reconsider" loops. Store the
survivors, not the trace. Filtering at capture time is cheaper.

That pushback was wrong in an interesting way — and the correction is where
the product lives.

## The counter that became the inversion

> *"Actually, vetting the skip-chaotic-results wouldn't be that hard because
> as you are reading through the raw data... you have the shipped MVP, so
> you know what decision was reached. And if what you're reading is not
> relevant in that large log, it gets trashed."*

The room goes quiet for exactly one paragraph. Because the obvious
refinement follows immediately:

> *"What if we went through the full logging of the context backwards.
> Sentence by sentence to eliminate irrelevant content."*

**That's the idea.** The whole product is this inversion, and once you say
it, it sounds inevitable — which is how the best ones arrive.

Walk forward through a project's log and every judgment is a *prediction*:
"will this matter?" Hardest question in the domain, asked at the worst
possible moment — before the answer exists. Walk **backward** from the
shipped product and every judgment is a *verification*: "did this
contribute to what I can already see?" Easiest question in the domain,
asked with the answer key in hand.

And there's a resonance that still feels like a gift: LLMs themselves read
earlier tokens in the full knowledge of later ones — attention is
backward-conditioned the way the walk is. The mechanism mirrors the medium.

The sketch immediately surfaced three problems worth carrying into the
design, because each became a feature:

| Problem (Sept 24) | Feature (Nov 11) |
|---|---|
| You can't load the whole past in one pass — walk segment-wise, carry the kept forward | The backward walk runs as a running filter; content-hashed IDs make every pass idempotent |
| Early "noise" can be the setup for a late breakthrough | Relevance is judged with carried-forward context, and the judge **fails open** — a doubtful call keeps the evidence, never destroys it |
| The same idea appears in every segment | Deduplication happens during the walk, where redundancy is obvious — not at capture, where it isn't |

## Deliberately not an ecosystem bolt-on

Mem0 came up as a base — Apache-2.0, Python, the same Valkey/Qdrant backends
we now ship. It was evaluated and set aside, not because it's bad, but
because tacking a retrospective pass onto someone else's memory pipeline is
the wrong shape: the backward walk is not a feature of a memory system, it
*is* the memory system. Same verdict for building on MemGPT, Cognee,
LangGraph, or Letta — every one of them asks the prediction question. The
verification question had no incumbent.

## Running alongside, never instead

The most Hermes-specific decision, and the one repeated most carefully:
GoneBy is a **general plugin, not a memory provider**. A provider slot is
single-select — claiming it displaces Mem0 or Honcho by definition. GoneBy
claims nothing. It hooks the same lifecycle events, keeps its own encrypted
record of what actually mattered, and at project close hands *your* memory
system additional semantically searchable data: the verified choices that
reached completion. It remembers alongside whatever already remembers for
you. When the day comes that GoneBy should *be* the memory system, Hermes'
`on_pre_compress` seam is purpose-built for exactly that capture point —
that's the upgrade path, not the default.

## The quiet decisions that held

- **The budget was non-negotiable: capture costs zero tokens.** The name
  even started as "the wall" — a zero-token catch-all — and hardened into
  the write-ahead log. Dumb-and-cheap now buys smart-and-once later.
- **Encryption at rest, from day one.** A log of a project's real decisions
  is the most sensitive artifact in the workspace. ChaCha20-Poly1305 per
  entry — the key is the vault, and losing it really means losing the log.
- **"Project-scoped" survived its own challenge.** The claim was checked:
  it was intent, not mechanism. So the fix became the architecture — one
  WAL per project (the backward walk is meaningless against someone else's
  product), shared store, scoped search, cross-project recall explicitly
  opt-in. Ask *"is that real or is that a wish?"* often; it keeps earning
  its adjectives.

## Idea to artifact: 48 days

- **v1.0.0, November 2026** — library, MCP server, Hermes plugin, CLI, full
  capture-policy system (per-project markers, per-gateway kills, opt-in
  mode), 158 tests, and a narrated [explainer video](https://github.com/rahlquist/goneby/releases/download/v1.0.0/goneby-explainer.mp4) on the front page.
- The inversion held up against a real embedding model: related/unrelated
  cosine margin **0.67 / 0.25**, correct top hit, project isolation proven
  against a live vector store — the green unit suite never claimed ranking
  quality, and saying so is part of why the number was worth measuring.
- The two bugs the first live run caught (a sync client in an async path, a
  CLI pointing at a directory) both hid from 115 unit tests. The design
  bias — *verification beats prediction, including about yourself* —
  caught them in minutes. Fitting.

## Why this is exciting

Because the path was simply not taken. Every memory product on the market
asks an agent to guess what it will want to remember — at the moment it can
know least. GoneBy waits, keeps a perfect cheap record, and asks the one
question the shipped artifact can answer: *did you matter?* That's not an
optimization of the existing landscape. It's a different angle on it — the
kind you find by walking backward.

The next projects will make the decisions, compact, forget, and ship. And
for the first time, what actually mattered gets to stick around — verified,
encrypted, and searchable by whatever memory system was already there.

— *r.*, Nov 2026. Idea first argued Sept 24, 2026, in a session titled,
unassumingly, "newmemory."
