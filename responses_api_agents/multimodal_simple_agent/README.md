# multimodal_simple_agent

A drop-in extension of `simple_agent` that lets a resources server return
image content — in addition to (or instead of) text — from `/seed_session`
and tool endpoints, by injecting user-role messages into the model's
trajectory.

## Envelope

Any resources-server response may be plain text/JSON (legacy behaviour) or
a JSON envelope with an `as_user_messages` (or `as_user_message`) key:

```json
{
  "function_call_output": "recorded",
  "as_user_messages": [
    {"role": "user", "content": [
      {"type": "input_text",  "text": "Now examine this image."},
      {"type": "input_image", "image_url": "data:image/png;base64,..."}
    ]}
  ]
}
```

- `function_call_output` (optional, string) — text ack that fills the
  paired `function_call_output` in the trajectory. Defaults to `"OK"`.
  Ignored by `/seed_session` (no pending tool call there).
- `as_user_messages` (list) or `as_user_message` (single dict) — user /
  system / developer message dicts appended to the trajectory as
  `NeMoGymEasyInputMessage` items. May carry `input_text` and
  `input_image` content parts; images must be base64 data URLs.

Non-envelope bodies fall through unchanged to a text `function_call_output`,
so existing benchmarks that rely on `simple_agent` keep working when they
swap in `multimodal_simple_agent`.

## Compared to `gymv_agent`

`gymv_agent` transports observations via a typed `GymVStepResponse` off a
dedicated `/step` endpoint, extracts actions from `\boxed{...}` text, and
tracks `reward`/`done` inline. `multimodal_simple_agent` uses the OpenAI
Responses tool-call flow, defers reward computation to `/verify`, and
terminates the loop when the model emits an assistant message with no
function calls — but the observation shape (a list of user messages with
`input_image` content parts) is identical.

## Usage

```yaml
my_env_multimodal_simple_agent:
  responses_api_agents:
    multimodal_simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_env_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      max_steps: 4
```
