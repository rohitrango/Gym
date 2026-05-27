# Description

`text_action_agent` is the Path B action-transport agent for the Gym-V resources
server (see [Doc 2](../../../../../docs/design-docs/doc-2-nemo-gym-game-agent-action-transport.md)).
It runs a multi-turn rollout against a Responses-API model server and the Gym-V
`/step` endpoint, extracting the model's action from the **last** `\boxed{...}`
token in the assistant's plain text rather than from a tool call envelope.

The agent is the side-by-side counterpart of `aviary_agent` (Path A). Both
paths share Gym-V's unified `/step` schema (`tool_calls` for Path A,
`action_string` for Path B) and the `GymVEnvStateEasyInputMessage` channel for
env metadata.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- tenacity: Apache 2.0
