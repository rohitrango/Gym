# polygon_naming

Two-turn multimodal benchmark. Three 128×128 canvases per task, each
showing one regular n-gon (n in [3, 6]) drawn in one of five primary
colours. The model sees the first two on turn 1 and the third only
after it commits an answer for turn 1. Reward is 1 iff the multiset
of submitted `(sides, colour)` pairs matches the ground truth.

## Requires

`multimodal_simple_agent` — the environment relies on the envelope
protocol (`as_user_messages`) it defines to reveal the second-turn
image mid-episode. Legacy `simple_agent` will not work.

## Regenerate example data

```bash
python resources_servers/polygon_naming/data/generate_data.py \
  --num-rows 5 --seed 0 \
  --output resources_servers/polygon_naming/data/example.jsonl
```

## Smoke test

```bash
gym env start \
  --config resources_servers/polygon_naming/configs/polygon_naming.yaml \
  --model-type openai_model

gym eval run --no-serve \
  --agent polygon_naming_multimodal_simple_agent \
  --input resources_servers/polygon_naming/data/example.jsonl \
  --output results/polygon_naming_rollouts.jsonl \
  --num-repeats 1 \
  --max-output-tokens 4096 \
  --temperature 0.7
```
