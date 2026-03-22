---
name: setup-claude-code
description: >-
  Install and configure Claude Code on a remote server accessed via Cursor Remote SSH,
  including SSH reverse tunnel for API access through a local VPN. Use when setting up
  Claude Code, configuring Anthropic API environment, or fixing Claude Code connectivity
  on a remote server.
---

# Setup Claude Code on Remote Server

## Overview

This skill configures Claude Code on a remote server that cannot directly reach the
Anthropic API gateway. It uses an SSH reverse tunnel so API traffic routes through the
user's local machine (which has VPN access).

## Prerequisites

- User must provide: **API key** and **gateway subscription key** (never store these in skill files)
- User's local machine must have VPN access to the API gateway
- Remote server is accessed via Cursor Remote SSH

## Step 1: Install Claude Code

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Add the binary to PATH if prompted:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

Verify installation:

```bash
~/.local/bin/claude --version
```

## Step 2: Configure Environment Variables

Append the following to `~/.bashrc`. **Ask the user for the actual key values** — do not
hardcode them.

```bash
# Claude Code Configuration
export ANTHROPIC_API_KEY="<ask-user>"
export ANTHROPIC_BASE_URL="https://llm-api.amd.com:18443/Anthropic"
export ANTHROPIC_MODEL="Claude-Opus-4.6"
export ANTHROPIC_DEFAULT_OPUS_MODEL="Claude-Opus-4.6"
export ANTHROPIC_DEFAULT_SONNET_MODEL="Claude-Sonnet-4.6"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="Claude-Haiku-4.6"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: <ask-user>"
```

After writing to `~/.bashrc`, also export them in the current session.

### Port in BASE_URL

`18443` is the reverse tunnel port (see Step 3). If the user picks a different port,
update `ANTHROPIC_BASE_URL` to match.

## Step 3: Configure SSH Reverse Tunnel

The remote server cannot resolve `llm-api.amd.com`. The solution:

1. **Add DNS override on remote server** — point the domain to localhost:

```bash
echo "127.0.0.1 llm-api.amd.com" >> /etc/hosts
```

2. **Give user the SSH command with reverse tunnel** — add `-R <port>:llm-api.amd.com:443`
to their existing SSH command. Example:

```
ssh -p <ssh-port> <user>@<host> -R 18443:llm-api.amd.com:443
```

`18443` is an arbitrary high port (1024-65535). The user can change it, but must also
update `ANTHROPIC_BASE_URL` to match.

**How it works:**

```
Remote server :18443 → SSH tunnel → Local machine → VPN → llm-api.amd.com:443
```

TLS works because the domain name in SNI and the certificate both match `llm-api.amd.com`.

For Cursor Remote SSH, the user can also add this to their local `~/.ssh/config`:

```
Host <remote-alias>
    HostName <remote-ip>
    Port <ssh-port>
    User <user>
    RemoteForward 18443 llm-api.amd.com:443
```

## Step 4: Verify

Test the tunnel from the remote server:

```bash
curl -s -X POST "https://llm-api.amd.com:18443/Anthropic/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: dummy" \
  -H "anthropic-version: 2023-06-01" \
  -H "Ocp-Apim-Subscription-Key: <key>" \
  -d '{"model":"Claude-Opus-4.6","max_tokens":50,"messages":[{"role":"user","content":"hi"}]}'
```

If the response contains a valid JSON reply, the setup is complete.

Then test Claude Code itself:

```bash
echo "say hi" | claude --print
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Could not resolve host: llm-api.amd.com` | `/etc/hosts` missing or SSH tunnel not active | Add hosts entry; reconnect SSH with `-R` flag |
| `Connection refused` on port 18443 | SSH tunnel not established | Ensure local SSH command includes `-R 18443:llm-api.amd.com:443` |
| `Deployment of "X" not found` (400) | Wrong model name | Check available models on the gateway; use exact deployment name |
| Connection timeout | Local VPN disconnected | Remind user to check their local machine's VPN connection is active |
| Hangs with no output | Environment variables not set in current session | Run `export` commands directly or `source ~/.bashrc` |
