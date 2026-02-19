# Description

COBOL compilation and execution benchmark. 499 problems from MultiPL-E (HumanEval + MBPP) adapted for COBOL with GnuCOBOL. The model generates COBOL code, which is compiled and tested against stdin/stdout test cases. Reward is 1.0 if all tests pass, 0.0 otherwise.

## Prerequisites

GnuCOBOL (`cobc`) is **auto-installed** on server startup if not already present:
- **macOS**: installed via Homebrew (requires `brew`)
- **Linux**: built from source into `resources_servers/cobol_compiler/.gnucobol/` (requires `gcc` and `make`; no root needed)

Override the Linux install prefix with `GNUCOBOL_PREFIX=/custom/path`.

## Data

The validation dataset is hosted in the GitLab dataset registry. To download it, add GitLab credentials to `env.yaml` at the NeMo-Gym project root:

```yaml
mlflow_tracking_uri: https://gitlab-master.nvidia.com/api/v4/projects/191584/ml/mlflow
mlflow_tracking_token: <your-gitlab-api-token>
```

Then run:

```bash
ng_prepare_data \
    "+config_paths=[resources_servers/cobol_compiler/configs/cobol_compiler.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
    +output_dirpath=/tmp/cobol_prepare_data \
    +mode=train_preparation \
    +should_download=true \
    +data_source=gitlab
```

This downloads `cobol_multipl_eval.jsonl` (499 problems) to `resources_servers/cobol_compiler/data/`. The dataset uses a structured Chain-of-Thought system prompt with step-by-step reasoning scaffold and I/O parsing patterns.

The `example.jsonl` (5 problems) is included in the repository and does not require downloading.

## Example Usage

Configure your model endpoint in `env.yaml`:

```yaml
policy_base_url: http://localhost:8000/v1   # vLLM, OpenAI-compatible, etc.
policy_api_key: your-key-here
policy_model_name: your-model-name
```

```bash
# Start servers
ng_run "+config_paths=[resources_servers/cobol_compiler/configs/cobol_compiler.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"

# Quick test on 5 examples
ng_collect_rollouts \
    +agent_name=cobol_compiler_simple_agent \
    +input_jsonl_fpath=resources_servers/cobol_compiler/data/example.jsonl \
    +output_jsonl_fpath=results/cobol_rollouts.jsonl \
    +num_repeats=1 \
    "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"

# Full benchmark (499 tasks x 5 repeats)
ng_collect_rollouts \
    +agent_name=cobol_compiler_simple_agent \
    +input_jsonl_fpath=resources_servers/cobol_compiler/data/cobol_multipl_eval.jsonl \
    +output_jsonl_fpath=results/cobol_rollouts_full.jsonl \
    +num_repeats=5 \
    +num_samples_in_parallel=5 \
    "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"

# Compute per-task pass rates
ng_profile \
    +input_jsonl_fpath=resources_servers/cobol_compiler/data/cobol_multipl_eval.jsonl \
    +rollouts_jsonl_fpath=results/cobol_rollouts_full.jsonl \
    +output_jsonl_fpath=results/cobol_profiled.jsonl \
    +pass_threshold=1.0

# View results
ng_viewer +jsonl_fpath=results/cobol_rollouts_full.jsonl
```

Use `openai_model` instead of `vllm_model` if your endpoint supports the OpenAI Responses API (`/v1/responses`).

## Unit Tests

```bash
ng_test +entrypoint=resources_servers/cobol_compiler
```

On systems with long working directory paths (e.g. Lustre mounts), Ray's socket paths may exceed the 107-byte AF_UNIX limit. Set `RAY_TMPDIR=/tmp` to fix this:

```bash
RAY_TMPDIR=/tmp ng_test +entrypoint=resources_servers/cobol_compiler
```

## Licensing Information

Code: Apache 2.0
Data: MIT

Dependencies
- nemo_gym: Apache 2.0
