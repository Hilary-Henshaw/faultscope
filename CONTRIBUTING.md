# Contributing to FaultScope

Thank you for your interest in contributing. This document explains how to set up your
development environment, the workflow we follow, and the standards that code must meet before it
can be merged.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Branch Naming](#branch-naming)
4. [Commit Messages](#commit-messages)
5. [Running Tests](#running-tests)
6. [Code Style](#code-style)
7. [Type Checking](#type-checking)
8. [Pull Request Process](#pull-request-process)
9. [Issue Templates](#issue-templates)

---

## Getting Started

### Fork and clone

```bash
# 1. Fork the repository on GitHub, then:
git clone https://github.com/<your-username>/faultscope.git
cd faultscope

# 2. Add the upstream remote so you can pull future changes
git remote add upstream https://github.com/your-org/faultscope.git
```

### Bootstrap your environment

```bash
# Requires Python 3.12 and uv (pip install uv)
make setup
```

`make setup` creates a virtual environment at `.venv/`, installs all development and lint
dependencies with `uv`, and registers the pre-commit hooks that run on every `git commit`.

### Configure environment variables

```bash
cp .env.example .env
# Edit .env — at minimum you need:
#   FAULTSCOPE_KAFKA_BOOTSTRAP_SERVERS
#   FAULTSCOPE_KAFKA_CONSUMER_GROUP
#   FAULTSCOPE_DB_PASSWORD
```

For unit tests no infrastructure is required. Integration tests spin up ephemeral containers
automatically via testcontainers — you only need Docker.

---

## Development Workflow

1. **Sync with upstream** before starting any work:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create a feature branch** from `main` (see naming conventions below).

3. **Write your code** with tests. We aim for 80 % line coverage on all new code.

4. **Run the fast checks** before committing:
   ```bash
   make lint
   make typecheck
   make test-unit
   ```

5. **Commit** using [Conventional Commits](#commit-messages).

6. **Push** and open a pull request against `main`.

---

## Branch Naming

| Prefix | Use for |
|---|---|
| `feature/` | New functionality |
| `fix/` | Bug fixes |
| `docs/` | Documentation-only changes |
| `refactor/` | Code restructuring without behaviour change |
| `test/` | New or improved tests |
| `chore/` | Dependency updates, build config, CI |
| `perf/` | Performance improvements |

**Examples:**

```
feature/autoencoder-anomaly-detection
fix/kafka-consumer-rebalance-timeout
docs/deployment-ssl-guide
chore/bump-tensorflow-2.16
```

Branch names must be lowercase with hyphens, no spaces, no underscores.

---

## Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.
Every commit message must have the form:

```
<type>(<optional scope>): <short summary>

<optional body>

<optional footer(s)>
```

### Types

| Type | When to use |
|---|---|
| `feat` | A new feature visible to users or callers |
| `fix` | A bug fix |
| `docs` | Documentation only |
| `style` | Formatting, whitespace — no logic change |
| `refactor` | Code change that is neither a fix nor a feature |
| `test` | Adding or updating tests |
| `chore` | Build process, dependency, CI configuration |
| `perf` | Performance improvement |
| `ci` | CI/CD pipeline changes |

### Rules

- The summary line must be **50 characters or fewer** and written in the **imperative mood**
  ("add feature" not "adds feature" or "added feature").
- Do not end the summary with a period.
- The body (when present) wraps at **72 characters** and explains *why*, not *what*.
- Breaking changes must include `BREAKING CHANGE:` in the footer.

### Examples

```
feat(inference): add MC Dropout uncertainty intervals to RUL predictions

Previously the inference API returned only a point estimate. This adds
T=30 forward passes with dropout active and reports the 5th and 95th
percentile as rul_lower_bound and rul_upper_bound.

Closes #42
```

```
fix(alerting): prevent duplicate incidents during Kafka rebalance

The cooldown check was keyed on (machine_id, rule_id) but used wall-clock
time from the consumer host. During a partition rebalance the same event
could be re-processed by a different consumer instance, generating a
duplicate incident. Now the event timestamp from the Kafka message is used.

Closes #87
```

```
chore: bump aiokafka to 0.11.0

BREAKING CHANGE: AIOKafkaConsumer no longer accepts loop= parameter.
Updated all consumer instantiation sites.
```

---

## Running Tests

```bash
# Fast unit tests — pure Python, no containers
make test-unit

# Integration tests — spins up Kafka and TimescaleDB via testcontainers
make test-integration

# Full suite (unit + integration)
make test

# End-to-end tests — requires the full stack to be running
make run-all
make test-e2e

# Coverage report
make coverage
# Open reports/coverage/index.html
```

### Test layout

```
tests/
├── unit/           # @pytest.mark.unit  — no external dependencies
├── integration/    # @pytest.mark.integration — testcontainers
└── e2e/            # @pytest.mark.e2e  — full stack
```

When adding tests:
- Unit tests must not import infrastructure code that requires live services.
- Use `pytest-mock` (`mocker` fixture) to isolate units.
- Integration tests should use the testcontainers fixtures defined in
  `tests/conftest.py` rather than hardcoding connection strings.
- Every new public function or class must have at least one unit test.

---

## Code Style

All Python code must pass `ruff` checks at the project's configuration (79-character line limit,
`pyproject.toml`). The pre-commit hook runs this automatically, but you can run it manually:

```bash
# Check only (no changes)
make lint

# Auto-fix what ruff can fix
make lint-fix
```

Additional style rules not enforced by ruff:

- Prefer `from __future__ import annotations` at the top of every module.
- Use f-strings; never `%`-formatting or `str.format()`.
- Do not use bare `except:` — always name the exception class.
- Mark non-production fallbacks with `# pragma: no cover`.
- Avoid mutable default arguments; use `dataclasses.field(default_factory=...)`.
- Every public module, class, method, and function must have a docstring.
  Docstrings follow the NumPy style.

---

## Type Checking

All code must pass `mypy` at strict settings:

```bash
make typecheck
```

Rules:
- Every function parameter and return type must be annotated.
- Use `from __future__ import annotations` to enable postponed evaluation.
- Third-party libraries that lack type stubs are exempted via `[[tool.mypy.overrides]]`
  in `pyproject.toml` — add new exemptions there, not inline `# type: ignore` comments.
- `# type: ignore` comments require an inline explanation comment.

---

## Pull Request Process

1. Ensure all CI checks pass (lint, typecheck, unit tests, integration tests).
2. Update `CHANGELOG.md` under the `[Unreleased]` section with a brief description.
3. If your change modifies a public API, update the relevant file in `docs/`.
4. Request at least one review from a maintainer. Reviews focus on correctness,
   test coverage, and whether the change is consistent with the architecture.
5. Squash your commits on merge if the history is noisy; preserve them if each
   commit tells a meaningful story.

**Do not:**
- Merge your own PR without a review.
- Force-push to `main` or any shared branch.
- Commit `.env` files, credentials, or large binary files.

---

## Issue Templates

### Bug report

When filing a bug please include:

```
**Environment**
- FaultScope version (git sha or tag):
- Python version:
- OS:
- Docker / Docker Compose version:

**Steps to reproduce**
1. ...
2. ...
3. ...

**Expected behaviour**
...

**Actual behaviour**
...

**Logs**
```
make health
docker compose logs --tail=50 <service>
```

**Additional context**
...
```

### Feature request

```
**Problem statement**
What problem does this feature solve? Who benefits?

**Proposed solution**
How do you envision this working? Code sketches, API design, etc.

**Alternatives considered**
What other approaches did you consider and why were they rejected?

**Acceptance criteria**
- [ ] ...
- [ ] ...
```
