# Dependabot Handling Rulebook (orbion-life)

The single standard for how **every** Dependabot alert and PR is handled across the
org. `homology_api` is the reference implementation; every other repo's pipeline is
templated from it and MUST follow these rules. Goal: the safe majority merges with
**zero human touches**; humans are spent only where they add real security value.

---

## 1. Detection (every repo, always on)

- Dependabot **alerts** + **security updates** + **version updates** enabled.
- The dependency graph MUST actually resolve the repo's real dependencies — a repo
  with a manifest the graph can't parse (PEP 621 `pyproject.toml` with no lockfile,
  deps declared only in a `Dockerfile` `RUN pip install` or a Modal
  `Image.pip_install(...)`) is **blind**, and "0 alerts" is a false negative. Add a
  parseable manifest (`requirements.txt` / lockfile) so the graph populates.
- `.github/dependabot.yml` covers every ecosystem the repo uses (pip / npm /
  github-actions / docker), with patch+minor grouped to cut PR noise.

## 2. Architecture — the trust boundary (every deployable repo)

Two stages, never one:

- **Validate** (`pull_request`, UNPRIVILEGED — no secrets, no OIDC): run CI +
  dependency-review (fail on **High**) + build the image; hand the built image to
  the deploy stage as an artifact. Untrusted dependency code runs ONLY here, where
  there is nothing to steal.
- **Deploy** (`workflow_run`, PRIVILEGED): `docker load` the prebuilt image (never
  executes it), push, run it ONLY inside an isolated **0%-traffic** test revision,
  validate (§4), tear the revision down. NEVER checks out or builds PR code.

Hard rules:
- The privileged stage never runs PR-controlled code while a secret/OIDC token is in
  scope (no pwn-request).
- The PR identity used for merge is re-derived from `gh api commits/{sha}/pulls`,
  never trusted from an artifact.
- The `orbion-ops` app token is used ONLY to read Dependabot alerts
  (`vulnerability_alerts:read`) and to review/merge (`pull_requests:write`). It must
  NOT be a ruleset `always`-bypass actor; checks gate the merge, not a bypass.

## 3. Risk tiering — applies to EVERY Dependabot PR (alert-driven or version bump)

Evaluated after the 0% deploy passes. **Fail closed**: if the alerts API errors,
treat as MANUAL.

| Tier | Conditions | Merge |
|---|---|---|
| **MANUAL** (one informed human review of an already-green PR) | • any updated dep has an open **Critical or High** alert (this includes the security fix *for* a Critical/High — **we pause for a human on these, by policy**) • pip/npm runtime-library **semver-major** • a **secret-handling action** (`azure/login`, `*create-github-app-token*`, `aws-actions/configure-aws-credentials`) | human approves → auto-merge completes |
| **AUTO** (zero human touches when `DEPENDABOT_AUTOMERGE=true` + ruleset configured) | everything else: patch/minor (incl. Med/Low security fixes), GitHub-Actions & Docker version bumps | merges on green checks |

Default (before zero-touch is enabled): **all** tiers wait for one informed approval.

### Alerts with NO available fix
Dependabot opens no PR. The alert sits in the Security tab and is handled manually:
risk-accept (documented, A.8.8, SPO-signed), apply a workaround, or wait for upstream.
These never auto-anything.

## 4. Test-deploy validation — by repo type

The 0% test revision must prove more than "the container started". Minimum per type:

| Repo type | Test-deploy MUST validate |
|---|---|
| **API on ACA** | revision `/health` 200 **and at least one real functional endpoint** returns the expected status for a known cheap request (API health, not just liveness). Stable customer FQDN still 200. |
| **Backend serving a SPA** (covalent) | the API health above **plus** a **CORS-preflight contract check** — `OPTIONS` the functional route with `Origin: https://app.orbion.life` and assert `Access-Control-Allow-Origin` echoes it (proves the frontend's origin is still accepted). |
| **Frontend SPA** (ribbon) | nginx `/_health` 200 + the SPA `index.html` serves, **plus a reachability check** that the backend the SPA is built to call answers a preflight from the SPA's origin (proves FE→BE still works). |
| **Single-process full-stack** (strategy-hub) | the unauth UI route (`GET /login`) returns 200 HTML from the 0% revision (serves both the rendered UI and its backend routes). |
| **Modal** (alphafold-api) | `modal deploy` to a **test environment**, invoke a function via the Modal SDK and assert success, then stop the app. (No HTTP endpoint — the web router was retired.) |
| **Library / research (no deploy)** | no test-deploy gate; pipeline is CI (unit + dependency-review High) + tiered merge only. |

Where a repo has a real integration suite that can run against the live revision FQDN,
run it instead of a bare endpoint probe.

## 5. Merge mechanics & branch protection (every repo)

- Required checks on `main`: `dependabot-ci-green` (Validate job) + `dependabot-test-deploy`
  (deploy status; seeded to success for human PRs so normal development is unblocked).
- Labels created with the built-in token (`issues:write`), not the app token.
- Test revisions named `…--test-dependabot-pr<N>-<run>` so the daily reaper
  (`cleanup-test-revisions.yml`, sweeps `--test-*`) collects any orphans; the deploy
  also self-deactivates on success.
- Zero-touch requires: repo var `DEPENDABOT_AUTOMERGE=true`, ruleset
  `require_code_owner_review:false` + `required_approving_review_count:1` + Dependabot
  as a **pull-request-only** bypass actor (checks still enforced), and "Allow GitHub
  Actions to approve PRs".

## 6. Per-repo customization checklist (when templating)

- [ ] Set `AZURE_RG` / `ACA_APP` / `ACR_NAME` / `IMAGE_NAME` from the repo's own
      `test-deploy.yml` (they differ per repo).
- [ ] Point the unit step at the repo's real unit gate; build with its `Dockerfile`.
- [ ] Set the §4 validation for the repo's type (functional endpoint / FE-BE check /
      Modal invoke / none).
- [ ] Confirm `dependabot.yml` exists with the right ecosystems (add it if missing —
      e.g. `ligand-api-v2`), and that each ecosystem points at the directory holding
      its manifest (e.g. docker at `/docker` if the Dockerfile lives there).
- [ ] **Port the runtime wiring** — copy the env vars, `--secret-volume-mount`, and
      workload-profile/CPU/memory from the repo's existing `test-deploy.yml`
      "create test revision" step into `dependabot-deploy.yml`. The 0% revision must
      BOOT the same way the proven canary does, or the container fails to start and
      the deploy gate is red forever. (Pre-merge requirement, not optional.)
- [ ] Remove `orbion-ops` from the repo's ruleset `always`-bypass (except `stasis`,
      handled separately).

## 7. ISO 27001 mapping

A.8.8 (vuln mgmt + documented risk-acceptance for AUTO tier and no-fix alerts),
A.8.29 (per-PR security testing — unit + dependency-review + real 0% functional test),
A.8.31 (test on 0% revisions; untrusted code never runs with prod creds),
A.8.2 / A.5.15 (least-privilege automation identity).

## 8. Per-repo rollout spec (from recon 2026-06-29)

Resources are `AZURE_RG / ACR_NAME / ACA_APP / IMAGE_NAME`. "Unit gate" = what the
Validate stage runs (repos with no suite fall back to `python -m compileall`).

| Repo | Type | RG / ACR / app / image | Health + functional check | Unit gate | dependabot.yml |
|---|---|---|---|---|---|
| homology_api | api | gcp-migration-rg / gcpmigrationacr / blastlike-api-v2 / blastlike-api | /health; GET /stats=mmseqs2; POST /search bogus-db→400 | pytest test_msa.py | ok — **done (#53)** |
| astra-api | api | gcp-migration-rg / gcpmigrationacr / astra-api / astra-api | /health; GET / →200 | none → compileall | ok |
| preprocess | api | gcp-migration-rg / gcpmigrationacr / preprocess-api / esm-preprocess-api | /health; POST / invalid-seq→400 | none → compileall | ok |
| alphafold-scheduler | api | rg-alphafold-scheduler-de / acralphafoldschedde / alphafold-scheduler / alphafold-scheduler | /health; GET /gpu/status→200 | pytest tests/ | ok |
| ligand-api-v2 | api | orbion-prod / orbionligandacr / ligand-api-v2 / ligand-api-v2 | /health,/ready; GET /v2/info→200 | integration suite (opt-in) | **ADD — missing** |
| covalent | backend+SPA | gcp-migration-rg / gcpmigrationacr / orbion-backend / covalent | /_health; POST /api/auth/login→401; **CORS preflight** Origin app.orbion.life | `node --test` | ok (npm+pip+docker+actions) |
| ribbon | frontend SPA | gcp-migration-rg / gcpmigrationacr / orbion-frontend / orbion-frontend | nginx /_health; **backend preflight reachability** | none (no framework) | ok (npm+docker+actions) |
| strategy-hub | fullstack | rg-strategy-hub-de / acrstrategyhubde / strategy-hub / strategy-hub | GET /login→200 HTML | none → compileall | **FIX target dev→main** |
| alphafold-api | modal | n/a | `modal deploy` test env + SDK invoke + stop | pytest test_modal_app.py | ok |
| quantum-ml-comparator | library | n/a (no deploy) | CI only | pytest | ok — done (#12) |
| white-paper-protein-family | research | n/a (no deploy) | CI only | none | ok |
| astra-models | research | n/a (no deploy) | CI only | none | **FIX target dev→main** |
| stasis | research | n/a (no deploy) | CI only | none | gap — actions only |
| mutagen | research | n/a (no deploy) | CI only | none | **FIX mis-targeted** |
| iso27001-azure-remediation | IaC | n/a (no deploy) | CI only | none | ok (actions only) |

Rollout order (each batch swarm-verified; **prove `homology_api` #53 live first**):
1. FastAPI APIs (astra-api, preprocess, alphafold-scheduler, ligand-api-v2) — homogeneous with #53.
2. covalent (Node + CORS) ↔ ribbon (SPA + reachability) — the cross-repo pair.
3. strategy-hub (single-process) + alphafold-api (Modal smoke).
4. Library/research repos — CI + tiered merge, no deploy.
