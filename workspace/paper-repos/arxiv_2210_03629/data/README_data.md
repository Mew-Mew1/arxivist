# Data Setup

This repository uses four benchmarks, none of which are bundled with the
repo. Run `python data/download.py --dataset <name>` for the two that support
automated download (HotpotQA, FEVER); ALFWorld and WebShop require separate
one-time setup described below.

## HotpotQA (`--dataset hotpotqa`)

- Source: https://hotpotqa.github.io/
- File used: `hotpot_dev_distractor_v1.json` (question-only setup, Section 3.1)
- `python data/download.py --dataset hotpotqa` downloads this file into
  `data/hotpotqa/` (path configurable via `configs/config.yaml::data.hotpotqa.data_dir`).
- Publicly available, no license gate.

## FEVER (`--dataset fever`)

- Source: https://fever.ai/dataset/fever.html
- File used: `shared_task_dev.jsonl` (claim-only setup, Section 3.1)
- `python data/download.py --dataset fever` downloads this file into
  `data/fever/` (path configurable via `configs/config.yaml::data.fever.data_dir`).
- Publicly available, no license gate.

## ALFWorld

ALFWorld is **not** downloaded by `data/download.py` -- it is a full text-game
engine with its own asset bundle, installed as a Python package:

```bash
pip install alfworld
export ALFWORLD_DATA=<path-to-store-alfworld-data>
alfworld-download
```

This downloads ~1-2 GB of TextWorld game assets. See
https://github.com/alfworld/alfworld for full instructions and
troubleshooting. Once installed, `src/react_agent/envs/alfworld_env.py`
wraps the installed `alfworld` package automatically; no further repo-level
configuration is required beyond `configs/config.yaml::data.alfworld.data_dir`
pointing at wherever you keep task-instance metadata.

The paper's evaluation set is the 134 unseen ALFWorld evaluation games
(`configs/config.yaml::data.alfworld.n_eval_games`).

## WebShop

WebShop is **not** a static dataset -- it is a Flask web server serving a
1.18M-product simulated shopping site. Set it up separately:

```bash
git clone https://github.com/princeton-nlp/webshop.git
cd webshop
# follow the WebShop repo's own setup.sh / data-download instructions
./setup.sh -d all
python run_dev.py   # starts the server, default http://localhost:3000
```

Once the server is running, point `envs/webshop_env.py`'s `server_url`
argument (or the equivalent config field you wire up) at it. This repo's
`WebShopEnvironment` adapter talks to that server over HTTP; it does not
re-implement WebShop's product catalog or scoring logic.

The paper's evaluation set is 500 test instructions
(`configs/config.yaml::data.webshop.n_eval_instructions`).

## Data directory layout after setup

```
data/
├── README_data.md        (this file)
├── download.py
├── hotpotqa/
│   └── hotpot_dev_distractor_v1.json
├── fever/
│   └── shared_task_dev.jsonl
├── alfworld/              (metadata only; game assets live wherever $ALFWORLD_DATA points)
└── webshop/               (metadata only; the WebShop server runs out-of-repo)
```
