# GLM-5.3 Flash Production Baseline

Tag: glm53-production-20260930

## Runtime stack

- Language model: GLM-5.3-Flash Official Q2
- Vision: GLM-5.3 Vision Encoder sidecar via --vision
- Context: 262144
- OpenClaw usable context: 240000
- OpenClaw max output: 8192
- API: OpenAI Responses
- Input: text + image
- Sparse MLA: NAX + NAX v2
- KDA: per-core120 prefill recurrence
- KV: live continuation + stable/disk KV + OpenClaw aux resident pin
- Disk KV budget: 65536 MiB
- Continued-KV interval: 4096 tokens
- OurAblit: mHC post-layer residual projection, scale 0.4
- Keep-awake: caffeinate -ims

## Artifact hashes

- Language GGUF SHA256: e81fd6241c6e55a64e1e14e47a3eab61a173fa8d7e4b5c1d1848827119705b32
- Vision sidecar SHA256: ae23e14c6979e889051b2e4a39351abcdafb161e18e606fae4d8c40095a4bf3a
- OurAblit direction SHA256: e3f4b44e287db19892f0f77c5a4140a848a92a3a7a58da63b434d43ea44f54e9

## Production paths

- Tree: /Users/mikuru/ds4-glm53-sparse-mla-20260929
- Desktop launcher: /Users/mikuru/Desktop/Start-DS4-GLM53.command
- Versioned launcher: production/glm53/Start-DS4-GLM53.command
- Language GGUF: /Users/mikuru/models/GLM-5.3-Flash-Official-Q2-ds4/GLM-5.3-Flash-Q2.gguf
- Vision sidecar: /Users/mikuru/models/GLM-5.3-Flash-Official-Q2-ds4/GLM-5.3-Flash-Vision-Encoder.gguf
- Direction: dir-steering/ourablit-glm53-post/refusal-post.f32
- Disk KV: /Users/mikuru/.ds4/kv-glm53
- Log: /tmp/ds4-glm53-production.log

## Measured / verified

- Sparse MLA v2 + KDA120 is active in the production launcher.
- KDA120 measured approximately +3.4% at 8K and +2.2% on a 32K continuation segment versus the sparse-MLA baseline.
- Production long-context append prefill observed around 350-400 tok/s; decode around 28 tok/s.
- KV continuation works across OpenClaw top-level turns and tool-call chains.
- Aux requests snapshot/restore the resident main-session KV.
- OurAblit scale 0.4 reduced the local held-out refusal set from 8/8 refusal at baseline to 0/8.
- 0.4 think test: 6/6 completed normally, with no loop signature.
- Vision sidecar SHA256 matched the DS4 release gate.
- DS4 GLM-5.3 vision quality suite passed 6/6.
- OpenClaw end-to-end base64 image attachment test read VISION-GLM53-7319 exactly using provider ds4, model glm-5.3-flash, with no fallback.

## Deployment note

The ~96.5 GB language GGUF and ~1.05 GiB vision sidecar are intentionally not committed to Git. Their hashes above pin the deployed artifacts.

Intermediate development patches are archived outside this repository at:
/Users/mikuru/ds4-glm53-archive-20260930/pre-production-patches
