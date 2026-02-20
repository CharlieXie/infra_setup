# Vast.ai Server Setup Guide

This document describes the steps to configure a fresh vast.ai server instance.
An AI agent can follow these instructions to perform the full setup automatically.

---

## Prerequisites

- A GitHub Personal Access Token (PAT) with repo read access
- The server should have `conda` available at `/opt/miniforge3/bin/conda`

---

## Setup Steps

### 1. Disable Auto-tmux

vast.ai servers auto-attach to a tmux session on login. Disable this behavior:

```bash
touch ~/.no_auto_tmux
```

> This takes effect after restarting the terminal.

---

### 2. Clone the Repository

Clone the repo into `/workspace`:

```bash
cd /workspace
git clone https://<PAT>@github.com/CharlieXie/galaxea_0.git
cd galaxea_0
```

Replace `<PAT>` with the actual GitHub Personal Access Token.

---

### 3. Fetch and Switch to Main Branch

```bash
git fetch origin
git checkout main
```

---

### 4. Create Conda Environment

Install the conda environment from `env.yaml`:

```bash
/opt/miniforge3/bin/conda env create -f env.yaml
```

This may take 10–20 minutes. The environment will be named `py10_g0_train`.

To verify the environment was created successfully:

```bash
/opt/miniforge3/bin/conda env list
/opt/miniforge3/bin/conda run -n py10_g0_train python --version
```

---

### 5. Configure Git Identity

```bash
git config --global user.email "chuanliang.xie@gmail.com"
git config --global user.name "chuanliang"
```

---

### 6. Install rclone

```bash
sudo -v && curl https://rclone.org/install.sh | sudo bash
```

To verify:

```bash
rclone --version
```

---

### 7. Configure rclone Remote (Google Drive)

```bash
rclone config
```

Follow the interactive prompts to add a new remote. Use `n` for a new remote, name it (e.g. `gg1`), select `drive` as the storage type, and authenticate via browser or service account.

To verify the remote works:

```bash
rclone ls <remote_name>:<path>
```

---

## Scripts

### `scripts/watch_and_upload.sh`

Watches for a model checkpoint file to appear in a training run directory, uploads the
entire directory to a remote (Google Drive via rclone), verifies the upload, and then
automatically destroys the vast.ai instance to stop billing.

**Setup:**

1. Edit the configuration variables at the top of the script:
   - `WATCH_DIR` — local path to the training run directory
   - `TARGET_FILE` — checkpoint filename to wait for (e.g. `model_1600.pt`)
   - `REMOTE_DEST` — rclone remote destination (e.g. `gg1:dissert_ntu/models/my_run`)
   - `VAST_API_KEY` — your vast.ai API key (from [vast.ai console](https://cloud.vast.ai/account/))
   - `VAST_INSTANCE_ID` — the instance ID shown in the vast.ai dashboard

2. Run in the background so it persists after logout:

```bash
chmod +x scripts/watch_and_upload.sh
nohup ./scripts/watch_and_upload.sh > watch_and_upload.log 2>&1 &
```

3. Monitor progress:

```bash
tail -f watch_and_upload.log
```

---

## Summary Checklist

- [ ] `touch ~/.no_auto_tmux`
- [ ] Clone repo to `/workspace/galaxea_0` using PAT
- [ ] `git fetch origin && git checkout main`
- [ ] `conda env create -f env.yaml` (env name: `py10_g0_train`)
- [ ] `git config --global user.email` and `user.name`
- [ ] Install rclone via install script
- [ ] Configure rclone remote (`rclone config`)
- [ ] Configure and run `scripts/watch_and_upload.sh` for auto-upload + instance teardown
