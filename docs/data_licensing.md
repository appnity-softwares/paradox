# Data Licensing (U8 step 3)

## Research tick store

- Source: user-provided MT5 tick history export for XAUUSD
  (2024-01-01 → 2026-11-22, 207,336,585 ticks, dataset_hash prefix
  `03e63637f06597a8`), ingested into the local parquet store under
  `data/ticks/`.
- Licence: the export was obtained through the user's own broker
  terminal access. It is used internally for research only; it is NOT
  redistributed with the repository (the parquet store is
  git-ignored).

## Dukascopy second source

- `core/marketdata/dukascopy_ingest.py` downloads free historical tick
  data from Dukascopy's public tick feed for the SAME period and
  stores it under a separate source tag.
- Dukascopy data is provided under the Dukascopy data terms
  (https://www.dukascopy.com — free for research use, no
  redistribution as a standalone dataset). We use it only to
  cross-check our research store (differences reported, never
  republished). Record any licence change here if usage changes.
- Respect the provider's rate limits; the ingest is week-chunked with
  a manifest (`state/dukascopy_manifest.json`) to stay idempotent.

## Firm feed recordings

- Quotes recorded from a firm's server (U8 `FeedRecorder`) belong to
  that firm's feed; they are stored separately under
  `data/feeds/<firm>_<server>/` and used only for feed comparison and
  cost modelling. Not redistributed.
