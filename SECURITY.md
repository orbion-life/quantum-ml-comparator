# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

When a new minor or major version is released, the previous minor line
receives patch-level security fixes for 3 months, then is considered
end-of-life.

## Reporting a Vulnerability

Please **do not open a public GitHub issue** for security vulnerabilities.

Email `aniruddh.goteti@orbion.life` with:

- A description of the issue and the conditions under which it triggers.
- Steps to reproduce (ideally with a minimal code snippet).
- The version(s) of `quantum-ml-comparator` and its dependencies where
  you observed the issue.
- Your assessment of impact and severity, if you have one.

### What to expect

- **Acknowledgement** within 72 hours of your report.
- **Initial assessment** within 7 days: we confirm the vulnerability,
  estimate severity (CVSS v3.1), and share a tentative timeline.
- **Fix** targeted within 14 days for high or critical severity issues;
  within 30 days for lower severity.
- **Coordinated disclosure**: we credit reporters in the release notes
  (unless you request anonymity) and publish a GitHub Security Advisory
  once the fix is released.

## Scope

In scope:

- Code in the `qmc/` package, `examples/`, and `benchmarks/`.
- CI workflows that run on code from this repository.

Out of scope:

- Vulnerabilities in upstream dependencies (PennyLane, PyTorch,
  scikit-learn, PySCF). Report those to the respective projects.
- Social engineering or physical attacks.
- Denial-of-service on public demo services (none currently provided).

## Preferred Languages

English.
