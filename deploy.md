# ☁️ Deployment Guide: Running Your Bot 24/7

This guide explains how to deploy your AI Trading Bot to a cloud server (VPS) so it runs continuously without your laptop.

## Prerequisite: Rent a VPS (Virtual Private Server)
We recommend a small Linux server. The bot uses minimal resources but needs stability.
- **Provider:** DigitalOcean, Vultr, Linode, or AWS EC2.
- **OS:** Ubuntu 22.04 LTS (Recommended)
- **Specs:** 1 CPU, 1-2GB RAM (Cost: ~$5-10/month)

---

## Step 1: Connect to Your Server
On your computer, open PowerShell or Terminal:
```bash
ssh root@<your-server-ip>
```

## Step 2: Install Docker
Run these commands on the server to install the runtime:
```bash
# Update system
apt-get update && apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```

## Step 3: Upload Your Bot
You can use `scp` (Secure Copy) to send your project files from your local machine to the server.
In a **new** PowerShell window on your PC (navigate to `Forex-bot-main` folder):
```powershell
scp -r . root@<your-server-ip>:/root/bot
```

## Step 4: Run the Bot
Back in your server terminal:
```bash
cd /root/bot

# Build and Start
docker compose up -d --build
```
- `-d`: Runs in "detached" mode (background).
- `--build`: Builds the container.

## Step 5: Verify
1. **Check Status:**
   ```bash
   docker ps
   ```
2. **View Logs:**
   ```bash
   docker logs -f derivatives_bot
   ```
3. **Open Dashboard:**
   Visit `http://<your-server-ip>:8501` in your browser.

---

## ✨ Why This Setup is Best
1.  **Persistence:** Your trained models (`models/`) are saved to the server disk. If you restart the container, you don't lose your brains!
2.  **Auto-Restart:** If the bot crashes or the server reboots, Docker will automatically start it again (`restart: unless-stopped`).
3.  **Isolation:** The bot lives in a clean container, unaffected by server updates.

## 🛠️ Maintenance Commands
- **Stop Bot:** `docker compose down`
- **Update Config:** Edit `config.json` on server, then `docker compose restart`.
- **View Live Logs:** `docker compose logs -f`
