# Endlex

A self-hosted, single-user replacement for wandb's metrics tracking and model checkpoint sync. Designed for solo LLM training: cloud GPU instances push metrics and final weights to a home server; all evaluation, inference, and chat experiments happen locally.

Initially built as the observability layer for ArcherChat (a from-scratch nanochat rewrite), but usable for any single-user training workflow that doesn't need wandb's team, sweep, or artifact-quota machinery.

## Status

In planning. See [TECH_PLAN.md](TECH_PLAN.md) for the full design — architecture, storage layout, HTTP API, client contract, performance rules, and roadmap.

## License

TBD.
