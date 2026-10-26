# B-Side

**A fork of [MediaSage](https://github.com/ecwilsonaz/mediasage) by ecwilsonaz, rebuilt for large libraries, large playlists and local LLMs.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AI-powered playlists and album recommendations for Plex—using only music you actually own.**

> **What B-Side changes.** Upstream MediaSage
> sends the whole candidate list to the model in one prompt and asks for the
> whole playlist back in one JSON answer. With local models that overflows the
> context window, so you end up building playlists 10–20 tracks at a time. This
> fork:
>
> - **Scans the library in batches** sized to the model's context and runs an
>   elimination tournament (best picks from each batch → next round → final
>   ordered pick). Any library size works with an 8k-context model.
> - **Returns pick numbers instead of titles**, so output stays tiny and every
>   pick maps exactly to a library track (no fuzzy-matching misses).
> - **Playlists of any size (1–2000)**; picks are validated and topped up so you
>   always get the size you asked for.
> - **Fixes Ollama**: sends `num_ctx` (Ollama otherwise silently truncates long
>   prompts to its 2–4k default), caps it at 16k by default so large-context
>   models don't exhaust VRAM, uses JSON mode, and detects cut-off output.
> - **Saves big playlists to Plex in chunks** (bulk fetch, 200 tracks per
>   request) instead of one request per track and one giant URL.
> - Streams per-batch progress with heartbeats so long local runs don't time out.
>
> - **Two-model scanning**: batch rounds run on the *Generation* model and the
>   final pick-and-order on the *Analysis* model, so a small fast model can scan
>   while a larger one curates. Selection calls use a strict JSON schema and low
>   temperature (no failed batches in testing vs. intermittent ones with plain
>   JSON mode at the model's default temperature).
>
> - **Exclude seasonal** (default on): keyword filter plus AI screening keeps
>   Christmas and other holiday songs out of playlists.
> - **Library stays current**: background re-sync every 24h (`auto_sync_hours`),
>   a banner when the library is out of date, and syncs that keep the old
>   snapshot usable until the new one is complete.
>
> The rest of this README is upstream's; Docker image names and links there
> refer to the original MediaSage project. To run B-Side in Docker, build it
> from this repository (`docker compose up -d --build`).
>
> Tuning: `OLLAMA_NUM_CTX` (context per call) and `LLM_CONCURRENCY` (parallel
> batch calls). See `config.example.yaml`.

MediaSage is a self-hosted web app that creates playlists and recommends albums by combining LLM intelligence with your Plex library. Every suggestion is guaranteed playable because it only considers music you have.

*Sample Generated Playlist:*
![MediaSage Screenshot](docs/images/screenshot-playlist.png)

*Sample Generated Album Recommendation:*
![MediaSage Screenshot](docs/images/screenshot-album.png)

*Home Screen:*
![MediaSage Screenshot](docs/images/screenshot-home.png)

*Playlist Flow:*
![MediaSage Screenshot](docs/images/screenshot-playlist-start.png)

*Album Flow:*
![MediaSage Screenshot](docs/images/screenshot-album-start.png)

---

## Quick Start

```bash
docker run -d \
  --name mediasage \
  -p 5765:5765 \
  -v mediasage-data:/app/data \
  --restart unless-stopped \
  ghcr.io/ecwilsonaz/mediasage:latest
```

Open **http://localhost:5765** — a setup wizard walks you through connecting Plex, choosing an AI provider, and syncing your library.

You can also pass credentials as environment variables to skip the wizard. See [Configuration](#configuration) for details.

**Requirements:** Docker, a Plex server with music, a [Plex token](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/), and an API key from Google, Anthropic, or OpenAI (or a local model via Ollama).

---

## Contents

- [Why MediaSage?](#why-mediasage)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Development](#development)
- [API Reference](#api-reference)

---

## Why MediaSage?

**Plex users with personal music libraries have few good options for AI playlists.**

Plexamp's built-in Sonic Sage used ChatGPT to generate playlists, but it was designed around Tidal streaming. The AI recommended tracks from an unlimited catalog, and Tidal made them playable. The "limit to library" setting just hid results you didn't own—so if you asked for 25 tracks and only 4 existed in your library, you got a 4-track playlist.

When [Tidal integration ended in October 2024](https://forums.plex.tv/t/tidal-integration-with-plex-ending-october-28-2024/885728), Sonic Sage lost its foundation. Generic tools like ChatGPT have the same problem: they recommend from an infinite catalog with no awareness of what you actually own.

**MediaSage inverts the approach:**

| Filter-Last (Sonic Sage, ChatGPT) | Filter-First (MediaSage) |
|-----------------------------------|-------------------------|
| AI recommends from infinite catalog | AI only sees your library |
| Hide missing tracks after | No missing tracks possible |
| Near-empty playlists | Full playlists, every time |

The result: every track in every playlist exists in your Plex library and plays immediately.

---

## Features

### Playlist Generation

Create playlists two ways:

**Describe what you want** — Natural language prompts like:
- "Melancholy 90s alternative for a rainy day"
- "Upbeat instrumental jazz for a dinner party"
- "Late night electronic, nothing too aggressive"

**Start from a song** — Pick a track you love, then explore musical dimensions: mood, era, instrumentation, genre, production style. Select which qualities you want more of.

### Album Recommendations

Describe a mood or moment, answer two quick questions about your preferences, and get a single perfect album to listen to—with an editorial pitch explaining why it fits.

- **Library mode** — recommends albums you own, ready for instant playback
- **Discovery mode** — suggests albums you don't own yet, based on your taste profile
- **Familiarity control** — choose between comfort picks, hidden gems, or rediscoveries
- **Show Me Another** — regenerate without starting over
- Primary recommendation with a full write-up, plus two secondary picks

### Smart Filtering

Before the AI sees anything, you control the pool:
- **Genres** — Select from your library's actual genre tags
- **Decades** — Filter by era
- **Minimum rating** — Only tracks rated 3+, 4+, etc.
- **Exclude live versions** — Skip concert recordings automatically

Real-time track counts show exactly how your filters narrow results.

### Local Library Cache

MediaSage syncs your Plex library to a local SQLite database. After a one-time sync (~2 min for 18,000 tracks), all library operations—filtering, counting, sending to AI—happen locally in milliseconds instead of waiting on Plex.

- **Setup wizard** walks you through first-run configuration and sync
- **Footer status** shows track count and last sync time
- **Auto-refresh** keeps cache current (syncs if >24h stale)
- **Manual refresh** available anytime

### Multi-Provider Support

Bring your own API key—or run locally:

| Provider | Max Tracks | Typical Cost | Best For |
|----------|------------|--------------|----------|
| **Google Gemini** | ~18,000 | $0.03 – $0.25 | Large libraries, lowest cost |
| **Anthropic Claude** | ~3,500 | $0.15 – $0.25 | Nuanced recommendations |
| **OpenAI GPT** | ~2,300 | $0.05 – $0.10 | Solid all-around |
| **Ollama** ⚗️ | Varies | Free | Privacy, local inference |
| **Custom** ⚗️ | Configurable | Free | Self-hosted, OpenAI-compatible APIs |

⚗️ *Local LLM support is experimental. [Report issues](https://github.com/ecwilsonaz/mediasage/issues).*

> **Free option:** Google Gemini offers a free API tier that's more than enough for personal use — no credit card required. See the [Gemini free credit guide](docs/gemini-free-credit-guide.md) for setup instructions and details.

Estimated cost displays before you generate. MediaSage auto-detects your provider based on which key you configure.

### Play and Save

- **Play Now** — send tracks directly to any Plex device for instant playback
- **Create** a new playlist, **replace** an existing one, or **append** tracks to one
- Device picker shows all active Plex clients with status indicators
- Duplicate detection when appending to existing playlists
- Preview tracks with album art before saving
- Remove tracks you don't want
- Rename the playlist
- See actual token usage and cost

---

## Installation

### Docker Compose (Recommended)

```bash
mkdir mediasage && cd mediasage
curl -O https://raw.githubusercontent.com/ecwilsonaz/mediasage/main/docker-compose.yml
curl -O https://raw.githubusercontent.com/ecwilsonaz/mediasage/main/.env.example
mv .env.example .env
```

Edit `.env`:

```bash
PLEX_URL=http://your-plex-server:32400
PLEX_TOKEN=your-plex-token

# Choose ONE provider:
GEMINI_API_KEY=your-gemini-key
# ANTHROPIC_API_KEY=sk-ant-your-key
# OPENAI_API_KEY=sk-your-key
```

Start:

```bash
docker compose up -d
```

### NAS Platforms

<details>
<summary><strong>Synology (Container Manager)</strong></summary>

**GUI:**
1. **Container Manager** → **Registry** → Search `ghcr.io/ecwilsonaz/mediasage`
2. Download `latest` tag
3. **Container** → **Create**
4. Port: 5765 → 5765
5. Add environment variables: `PLEX_URL`, `PLEX_TOKEN`, `GEMINI_API_KEY`

**Docker Compose:**
```bash
mkdir -p /volume1/docker/mediasage && cd /volume1/docker/mediasage
curl -O https://raw.githubusercontent.com/ecwilsonaz/mediasage/main/docker-compose.yml
curl -O https://raw.githubusercontent.com/ecwilsonaz/mediasage/main/.env.example
mv .env.example .env && nano .env
```
Then in **Container Manager** → **Project** → **Create**, point to `/volume1/docker/mediasage`.

**No Docker?** Some Synology models (especially ARM-based units) don't support Docker/Container Manager. See [Bare Metal](#bare-metal-no-docker) below for running MediaSage directly with Python.

</details>

<details>
<summary><strong>Unraid</strong></summary>

1. **Docker** → **Add Container**
2. Repository: `ghcr.io/ecwilsonaz/mediasage:latest`
3. Port: 5765 → 5765
4. Add variables: `PLEX_URL`, `PLEX_TOKEN`, `GEMINI_API_KEY`

</details>

<details>
<summary><strong>TrueNAS SCALE</strong></summary>

1. **Apps** → **Discover Apps** → **Custom App**
2. Image: `ghcr.io/ecwilsonaz/mediasage`, Tag: `latest`
3. Port: 5765
4. Add environment variables

</details>

<details>
<summary><strong>Portainer</strong></summary>

**Stacks** → **Add Stack**:

```yaml
services:
  mediasage:
    image: ghcr.io/ecwilsonaz/mediasage:latest
    ports:
      - "5765:5765"
    environment:
      - PLEX_URL=http://your-server:32400
      - PLEX_TOKEN=your-token
      - GEMINI_API_KEY=your-key
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

</details>

### Bare Metal (No Docker)

Docker isn't required. MediaSage is Python + FastAPI with no native dependencies, so it runs on any machine with Python 3.11+ — including ARM-based Synology NAS models, Raspberry Pis, or any Linux/macOS/Windows box.

```bash
git clone https://github.com/ecwilsonaz/mediasage.git
cd mediasage
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set your environment variables:

```bash
export PLEX_URL=http://your-plex-server:32400
export PLEX_TOKEN=your-plex-token
export GEMINI_API_KEY=your-gemini-key
```

Start the server:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 5765
```

Access at **http://your-machine-ip:5765**.

<details>
<summary><strong>Running as a background service (systemd)</strong></summary>

To keep MediaSage running after you close your terminal, create a systemd service:

```ini
# /etc/systemd/system/mediasage.service
[Unit]
Description=MediaSage
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/mediasage
EnvironmentFile=/path/to/mediasage/.env
ExecStart=/path/to/mediasage/venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 5765
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable mediasage
sudo systemctl start mediasage
```

</details>

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `PLEX_URL` | Yes | Plex server URL (e.g., `http://192.168.1.100:32400`) |
| `PLEX_TOKEN` | Yes | [Plex authentication token](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/) |
| `GEMINI_API_KEY` | One required | Google Gemini API key |
| `ANTHROPIC_API_KEY` | One required | Anthropic API key |
| `OPENAI_API_KEY` | One required | OpenAI API key |
| `LLM_PROVIDER` | No | Force provider: `gemini`, `anthropic`, `openai`, `ollama`, `custom` |
| `PLEX_MUSIC_LIBRARY` | No | Library name if not "Music" |
| `OLLAMA_URL` | No | Ollama server URL (default: `http://localhost:11434`) |
| `OLLAMA_CONTEXT_WINDOW` | No | Override detected context window for Ollama (default: 32768) |
| `CUSTOM_LLM_URL` | No | Custom OpenAI-compatible API base URL |
| `CUSTOM_LLM_API_KEY` | No | API key for custom provider (if required) |
| `CUSTOM_CONTEXT_WINDOW` | No | Context window size for custom provider (default: 32768) |
| `DISABLE_AUTH` | No | `true` turns off Plex sign-in, so anyone who can reach B-Side uses it as `PLEX_TOKEN`'s account. For emergencies, e.g. if you're locked out. |

### Signing in (multiple users)

Once Plex is set up, **everyone signs in with their own Plex account** before using B-Side. That includes Plex Home members and friends you share your server with.

- **How it works:** the button opens Plex's own sign-in page, so B-Side never sees passwords. Only accounts with **access to your Plex server** can sign in; anyone else is refused.
- **Each person's own Plex:** a signed-in person's playlists are **saved to their own Plex account**. They see their own playlists, their own devices for "Play now", and their own history in B-Side.
- **Plex Home profiles:** members get a **"Who's listening?"** picker (with the profile's PIN if it has one), like the Plex apps. Managed profiles without their own Plex login can be used this way.
- **The server owner is the admin.** Only the owner sees **Settings** (Plex connection, AI models, library sync). Everyone else just makes playlists and album picks.
- **Shared background work:** the library cache and AI models are shared and run as `PLEX_TOKEN`, which must be the server owner's token.
  - Candidate tracks therefore come from the owner's view of the library. A restricted user may occasionally be offered a track they can't play; it's skipped when saving to their Plex.
  - Rating and familiarity filters use the owner's ratings and play counts.
- **Sessions** last 30 days of inactivity and survive restarts (`data/sessions.db`). Use the menu under your name to switch profile or sign out.
- If B-Side is reachable from outside your network, put it behind **HTTPS** (e.g. a reverse proxy) so sign-in cookies are protected in transit. B-Side marks cookies secure when it sees HTTPS.

### Web UI Configuration

You can also configure MediaSage through the **Settings** page in the web UI. Settings entered there are saved to `config.user.yaml` and persist across restarts. Environment variables always take priority over UI-saved settings.

### Advanced: config.yaml

Mount a config file for additional options:

```yaml
plex:
  music_library: "Music"

llm:
  provider: "gemini"
  model_analysis: "gemini-2.5-flash"
  model_generation: "gemini-2.5-flash"
  smart_generation: false  # true = use smarter model for both (higher quality, ~3-5x cost)

defaults:
  track_count: 25
```

### Model Selection

MediaSage uses a two-model strategy by default:

| Role | Purpose | Models Used |
|------|---------|-------------|
| **Analysis** | Interpret prompts, suggest filters, analyze seed tracks | claude-sonnet-4-5 / gpt-4.1 / gemini-2.5-flash |
| **Generation** | Select tracks from filtered list | claude-haiku-4-5 / gpt-4.1-mini / gemini-2.5-flash |

This balances quality with cost. Enable `smart_generation: true` to use the analysis model for everything.

### Local LLM Setup (Experimental)

Run MediaSage with local models for privacy and zero API costs.

<details>
<summary><strong>Ollama</strong></summary>

1. Install [Ollama](https://ollama.ai) and pull a model:
   ```bash
   ollama pull llama3:8b
   ```

2. Configure MediaSage via environment or Settings UI:
   ```bash
   LLM_PROVIDER=ollama
   OLLAMA_URL=http://localhost:11434
   ```

3. Select your model in Settings → the context window is auto-detected.

**Recommended models:** `llama3:8b`, `qwen3:8b`, `mistral` — models with 8K+ context work best.

</details>

<details>
<summary><strong>Custom OpenAI-Compatible API</strong></summary>

For LM Studio, text-generation-webui, vLLM, or any OpenAI-compatible server:

1. Start your server with an OpenAI-compatible endpoint

2. Configure in Settings:
   - **API Base URL:** `http://localhost:5000/v1`
   - **API Key:** If required by your server
   - **Model Name:** The model identifier
   - **Context Window:** Your model's context size

</details>

**Note:** Local models are slower and may produce less accurate results than cloud providers. A 10-minute timeout is used for generation. Models with larger context windows will support more tracks.

---

## How It Works

MediaSage uses a **filter-first architecture** designed for large libraries (50,000+ tracks):

```
┌─────────────────────────────────────────────────────────────────┐
│  1. ANALYZE                                                      │
│     LLM interprets your prompt → suggests genre/decade filters   │
├─────────────────────────────────────────────────────────────────┤
│  2. FILTER                                                       │
│     Plex library narrowed to matching tracks                     │
│     "90s Alternative" → 2,000 tracks                             │
├─────────────────────────────────────────────────────────────────┤
│  3. SAMPLE                                                       │
│     If too large for context, randomly sample                    │
│     Fits within model's token limits                             │
├─────────────────────────────────────────────────────────────────┤
│  4. GENERATE                                                     │
│     Filtered track list + prompt sent to LLM                     │
│     LLM selects best matches from available tracks               │
├─────────────────────────────────────────────────────────────────┤
│  5. MATCH                                                        │
│     Fuzzy matching links LLM selections to library               │
│     Handles minor spelling/formatting differences                │
├─────────────────────────────────────────────────────────────────┤
│  6. SAVE                                                         │
│     Playlist created in Plex                                     │
│     Ready in Plexamp or any Plex client                          │
└─────────────────────────────────────────────────────────────────┘
```

This ensures every track exists in your library while keeping API costs manageable.

---

## Development

### Local Setup

```bash
git clone https://github.com/ecwilsonaz/mediasage.git
cd mediasage
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export PLEX_URL=http://your-plex-server:32400
export PLEX_TOKEN=your-plex-token
export GEMINI_API_KEY=your-key

uvicorn backend.main:app --reload --port 5765
```

### Testing

```bash
pytest tests/ -v
```

### Tech Stack

- **Backend:** Python 3.11+, FastAPI, python-plexapi, rapidfuzz, httpx
- **Frontend:** Vanilla HTML/CSS/JS (no build step)
- **LLM SDKs:** anthropic, openai, google-genai (+ Ollama via REST API)
- **Deployment:** Docker

---

## API Reference

Interactive documentation available at `/docs` when running.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/config` | GET/POST | Get or update configuration |
| `/api/setup/status` | GET | Onboarding checklist state |
| `/api/setup/validate-plex` | POST | Validate Plex credentials |
| `/api/setup/validate-ai` | POST | Validate AI provider credentials |
| `/api/setup/complete` | POST | Mark setup wizard as complete |
| `/api/library/stats` | GET | Library statistics |
| `/api/library/status` | GET | Cache state, track count, sync progress |
| `/api/library/sync` | POST | Trigger background library sync |
| `/api/library/search` | GET | Search library tracks |
| `/api/analyze/prompt` | POST | Analyze natural language prompt |
| `/api/analyze/track` | POST | Analyze a seed track |
| `/api/filter/preview` | POST | Preview filtered track list |
| `/api/generate` | POST | Generate playlist |
| `/api/generate/stream` | POST | Stream playlist generation (SSE) |
| `/api/playlist` | POST | Save playlist to Plex |
| `/api/playlist/update` | POST | Replace or append to a playlist |
| `/api/recommend/albums/preview` | GET | Preview album candidates for filters |
| `/api/recommend/analyze-prompt` | POST | Analyze prompt for genre/decade filters |
| `/api/recommend/questions` | POST | Generate clarifying questions |
| `/api/recommend/generate` | POST | Generate album recommendations |
| `/api/recommend/switch-mode` | POST | Switch library/discovery mode |
| `/api/results` | GET | List saved result history |
| `/api/results/{id}` | GET/DELETE | Get or delete a saved result |
| `/api/plex/clients` | GET | List active Plex clients |
| `/api/plex/playlists` | GET | List existing Plex playlists |
| `/api/play-queue` | POST | Send tracks to a Plex client |
| `/api/art/{rating_key}` | GET | Proxy album art from Plex |
| `/api/ollama/status` | GET | Ollama connection status |
| `/api/ollama/models` | GET | List available Ollama models |
| `/api/ollama/model-info` | GET | Get model details (context window) |

---

## License

MIT
