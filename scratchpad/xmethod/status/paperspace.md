# Paperspace (DigitalOcean) as a 4th compute platform: read-only probe (2026-10-04, worker exp-c)

Scope: read-only. Nothing launched, created or modified; no money spent. Credentials read in-process from
~/.paperspace/config.json and sent only as an HTTP Bearer header (never printed, logged, copied or committed).

## Which API / CLI works in 2026
- REST API `https://api.paperspace.com/v1`, `Authorization: Bearer <api_key>`: WORKS (GET /auth/session 200).
  Cloudflare blocks the default Python-urllib User-Agent (error 1010): send a descriptive User-Agent.
- Legacy Core + Gradient APIs (api.paperspace.io, `gradient` CLI incl. `gradient notebooks create --command`):
  RETIRED 15 Jul 2024 (docs banner "Deprecated ... no longer available").
- New CLI `pspace` (curl -fsSL https://paperspace.com/install.sh | sh): macOS / Linux / WSL only (no native
  Windows); covers machines, deployments, projects, networks; no notebook commands. Not installed here (not
  needed: REST was enough).
- Documented v1 resources: auth, machines, machine-availability, deployments, projects, datasets, models,
  templates, startup scripts, storage, teams/users. Notebooks are NOT documented in v1, but GET /v1/notebooks
  answers 200 (the console's endpoint; undocumented).

## Account (GET /auth/session)
- User created 2026-10-03, phone verified; one team = the personal private workspace, owner/admin;
  team.maxMachines = 5. Machines 0, deployments 0, notebooks 0. GET /projects -> 500 (server error).
- Subscription tier: NOT exposed by any endpoint found (/subscription, /billing, /teams/{id} -> 404). A new
  account is on the Free plan by default; confirm in the console (Q1).

## Free machines (Notebooks only, docs pricing page updated 2026-09-01)
- Free plan: Free CPU (C4: 2 vCPU, 4 GB RAM) and Free GPU (M4000: 8 GB GPU, 8 vCPU, 30 GB RAM). Pro adds free
  P4000 / P5000 / RTX4000 / RTX5000 / A4000; Growth adds A5000 / A6000 / A100-80G.
- Free machines run only as Notebooks (web Jupyter) in private workspaces; max auto-shutdown 6 h; "pending" =
  queued for the next free machine (shared pool, no guarantee).
- Concurrency (paperspace.com/gradient/pricing): concurrent jobs (Notebooks, Workflows) Free 1, Pro 3, Growth 10.
  Storage included: Free 5 GB, Pro 15 GB, Growth 50 GB ($0.29/GB/mo over).
- Plans: Pro $8/mo (user) / $12/mo (team), Growth $39/mo.
- Availability now (GET /machine-availability, = regional capacity, not plan entitlement: a Growth-only type also
  reads true; a bogus name reads false): Free-CPU and Free-GPU available in ny2 only (not ca1, ams1).

## Paid CPU machines (docs pricing; hourly, billed while running; vCPU / RAM from the machine-types table)
| type | vCPU | RAM GB | $/h | available now (ny2 / ca1 / ams1) |
|---|---|---|---|---|
| C4 | 2 | 4 | 0.04 | y / y / y |
| C5 | 4 | 8 | 0.08 | y / y / y |
| C6 | 8 | 16 | 0.16 | not queried |
| C7 | 12 | 30 | 0.30 | y / y / y |
| C8 | 16 | 60 | 0.60 | y / y / y |
| C9 | 24 | 120 | 0.90 | y / n / n |
| C10 | 32 | 244 | 1.60 | n / n / n |
(C1 1 vCPU / 0.5 GB $0.0045, C3 2 / 2 $0.018.) Machines also bill disk (50 GB $0.0074/h) and a public IP
($0.0045/h) while they exist; no bandwidth charges. GPUs: M4000 $0.45, P4000 $0.51, RTX4000 $0.56 /h.

## Non-interactive job via API?
- Machines (v1 POST /machines + startup scripts): YES, fully scriptable (create, run a startup script, delete),
  but PAID only (no free machine type for Machines). A C7 (12 vCPU) runs ~ $0.30/h + disk.
- Deployments (containers as a service): scriptable via v1 / pspace, but runs on paid machine types; built for
  serving, not batch jobs.
- Notebooks (the only FREE compute): the documented create-with-command path is retired; v1 notebooks is
  undocumented. A free notebook is a 2-vCPU C4 (or M4000) Jupyter session, 1 at a time, <= 6 h, queued on a
  shared pool: much less than a Kaggle (4 vCPU, 12 h, 5 sessions) or Colab session.
- No free-tier test start was made: the only route is the undocumented POST /notebooks, where a wrong default
  machine type could bill. Not "clearly free".

## Assessment for the study
- Free tier: not useful for our shards (1 x 2 vCPU, 6 h, interactive, queued).
- Paid Machines: a real 4th platform for unattended single-thread shards (1 process per vCPU, R-55): e.g. C7 12
  vCPU at $0.30/h (~$0.025 per vCPU-h). Needs a budget decision and a small launcher (create -> startup script ->
  pull -> delete), plus a speed-factor calibration as host "paperspace|<cpu model>" (R-55 Q2).

## DRAFT: Paperspace Machines launcher (design only; nothing built or run; needs Q2 budget OK)
Goal: run campaign / timing shards on paid C5 (4 vCPU, 8 GB, $0.08/h) or C6 (8 vCPU, 16 GB, $0.16/h), one
single-thread process per vCPU (R-55), driven from the laptop, machine and every billed extra deleted at the end.
Endpoints (v1, Bearer): POST/DELETE /startup-scripts, GET /os-templates, POST /machines, GET /machines/{id},
PATCH /machines/{id}/stop, DELETE /machines/{id}, GET /public-ips, /snapshots, /machine-events (verification).
1. Preflight (refuse on any failure): spend ledger below cap (step 7); GET /machine-availability for the type and
   region; GET /machines and /public-ips show none of ours left over; SSH key present on the account (needed for
   SSH; added once in the console); clean git tree (records stamp the commit, R-35).
2. Startup script (POST /startup-scripts, isRunOnce true, NO secrets): `shutdown -h +<cap_minutes>` hard
   power-off watchdog; install uv; `uv python install 3.12`; create /home/paperspace/xm; touch READY.
3. Create (POST /machines): machineType C5|C6, region with availability, templateId = Ubuntu server OS template
   (GET /os-templates by operatingSystemLabel), diskSize = smallest allowed (50 GB in the pricing table; the API
   enum has 10 values, read it at build time), publicIpType dynamic (released at power-off), startOnCreate true,
   startupScriptId, autoShutdownEnabled true + autoShutdownForce true + autoShutdownTimeout = ceil(cap hours)
   (server-side backstop, hours), takeInitialSnapshot / autoSnapshotEnabled / restorePointEnabled false, name
   xm-ps-<job>. Log machine id + dtCreated in the ledger immediately (before anything else can fail).
4. Run (laptop, SSH as paperspace@<ip>): poll GET /machines/{id} until ready and READY exists; scp the same bundle
   as Kaggle / Colab (cdd_oran, configs, scripts, spec, driver, _bundle_expc lock files); install the pinned lock
   into a 3.12 venv + torch CPU, pin check; `nohup` the shard runner with XM_PLATFORM=paperspace, threads = 1,
   P = nproc (the VM has no cgroup quota; exp_c_timing.cpu_quota falls back to the affinity count), `timeout`.
5. Pull: every ~10 min (in-session loop, no Task Scheduler) rsync/scp res_*.jsonl + logs back (append-only merge
   by key); the run survives a laptop sleep (results stay on the disk until step 6).
6. Teardown (always, in a finally; also a standalone `teardown <job>` command): final pull; PATCH stop; DELETE
   /machines/{id} (docs: deletion removes the machine, its files and snapshots); DELETE the startup script;
   verify GET /machines empty of ours, no static public IP (GET /public-ips), no snapshots; log dtDeleted.
7. Hard spend cap (ledger runs/paperspace/ledger.json): cost = sum over machines of (compute $/h + disk $/h + IP
   $/h) x billed hours, open machines counted to now, hours rounded UP to whole hours (granularity not stated in
   the docs). Launch refused if ledger + projected (rate x cap hours) > cap. Three stops per machine: launcher
   cap, in-VM `shutdown -h`, API force auto-shutdown; a stopped machine still bills its disk, so teardown
   (DELETE) is mandatory and is retried by the next in-session tick until it succeeds.
8. Calibration: a new host type "paperspace|<cpu model>" (R-55 Q2); run the calib block there and pair it with
   the Kaggle reference (results/exp_c/calib/raw/xm-expc-cal-k1) before its shards count as converted.
Costs to watch (docs pricing): disk while the machine EXISTS, stopped included (50 GB $0.0074/h, cap $5/mo);
public IP $0.0045/h (static IPs stay in the account after the machine is deleted until deleted themselves:
use dynamic, verify none left); snapshots / restore points / auto-snapshots (storage; all off); private network
$0.0015/h and shared drives (do not create); leftover startup scripts (free, but delete). Budget example: C6 for
10 h ~ 10 x (0.16 + 0.0074 + 0.0045) = $1.72; C5 ~ $0.92.

## Questions
- Q1 Which plan is the account on (Free / Pro / Growth)? Not exposed via API; please check the console.
- Q2 Is a paid budget for Paperspace Machines wanted (e.g. C7 at $0.30/h)? If yes I build a create / run / pull /
  delete launcher with a hard spend cap; until then nothing is started.
- Q3 ANSWERED (orchestrator, 2026-10-04): no free C4 notebook test (1 x 2 vCPU, not worth it).
- Q4 (for the build, if Q2 = yes) Billing granularity (per hour or per second) and whether the $0.0045/h public
  IP price applies to dynamic IPs are not stated in the docs: the cap assumes whole hours and charges the IP.

Sources: docs.digitalocean.com/products/paperspace/pricing, .../machines/details/machine-types,
.../machines/details/limits, .../notebooks/details/features, .../notebooks/how-to/create-notebooks,
.../reference/paperspace/api-reference/{authentication,machine-types,machine}, .../reference/paperspace/gradient/
commands/notebooks (deprecation), .../reference/paperspace/pspace/install, paperspace.com/gradient/pricing.
