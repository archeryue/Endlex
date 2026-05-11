# Endlex deploy snippets

The two files here drop Endlex onto a Linux box that already terminates HTTPS
via nginx (Let's Encrypt, your existing domain, etc.).

## One-time setup

```bash
# 1. user + data dir
sudo useradd -r -m -d /var/lib/endlex -s /usr/sbin/nologin endlex
sudo install -d -o endlex -g endlex /var/lib/endlex

# 2. clone + install
sudo -u endlex git clone https://github.com/<you>/Endlex.git /srv/endlex
cd /srv/endlex
sudo -u endlex uv sync --extra server

# 3. environment file
sudo install -d -m 0755 /etc/endlex
sudo install -m 0600 -o endlex deploy/endlex.env.example /etc/endlex/env
sudo $EDITOR /etc/endlex/env   # set ENDLEX_TOKEN at minimum

# 4. systemd unit
sudo install -m 0644 deploy/endlex.service /etc/systemd/system/endlex.service
sudo systemctl daemon-reload
sudo systemctl enable --now endlex
sudo systemctl status endlex

# 5. nginx
sudo $EDITOR /etc/nginx/sites-available/<your-site>   # paste deploy/nginx.conf.snippet inside server { ... }
sudo nginx -t && sudo systemctl reload nginx
```

## On the cloud trainer

```bash
pip install endlex  # or `pip install -e /path/to/Endlex` from a clone
export ENDLEX_URL=https://example.com/endlex
export ENDLEX_TOKEN=<the same token>
```

Then in the trainer:

```python
from endlex import Tracker, upload_checkpoint_async

tracker = Tracker(project="archerchat", name=run_name, config=cfg)
tracker.log({"step": step, "train/loss": loss})

# inside save_checkpoint:
if rank == 0 and os.environ.get("ENDLEX_URL"):
    upload_checkpoint_async(run_name, step, {"model.pt": model_path, "meta.json": meta_path})

tracker.finish()
```
