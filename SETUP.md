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

## Summary Checklist

- [ ] `touch ~/.no_auto_tmux`
- [ ] Clone repo to `/workspace/galaxea_0` using PAT
- [ ] `git fetch origin && git checkout main`
- [ ] `conda env create -f env.yaml` (env name: `py10_g0_train`)
- [ ] `git config --global user.email` and `user.name`
- [ ] Install rclone via install script
