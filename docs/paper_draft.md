# AI-Guided Adversary Simulation: Draft Paper

## Abstract
We present an AI-guided adversary simulation platform that integrates autonomous agents with human-over-the-loop controls to rehearse cyber defense scenarios. The system couples a central orchestrator, specialized red team agents, and real-time telemetry pipelines to explore hostile actions while maintaining analyst oversight. By combining structured planning with adaptive scoring, the platform accelerates threat emulation, surfaces defensive gaps, and supports repeatable experimentation for enterprise security teams.

## Architecture Overview
- **Orchestrator:** Coordinates scenario execution, schedules agent activities, and enforces policy constraints while aggregating state across the environment.
- **Agents:** Modular offensive actors that encapsulate reconnaissance, exploitation, and post-exploitation skills, each driven by large language models augmented with tool access.
- **Telemetry:** Unified logging and measurement layer that streams host, network, and agent-level events into analytics backends for situational awareness and evaluation.

## Methodology
1. **Beam Search Planner:** The orchestrator maintains a beam of candidate action sequences, expanding and pruning plans based on predicted utility and policy compliance to balance exploration with risk.
2. **Scoring Functions:** Composite scores blend mission progress, stealth, resource usage, and defensive signal generation. Domain-specific heuristics and learned predictors guide both beam selection and post-hoc evaluation.
3. **Human Checkpoints:** Analysts may inject feedback or halt actions at predefined decision points, ensuring the autonomy remains bounded by operational directives.

## Experimental Design
- **Policies Compared:** Evaluate default permissive policy, defense-in-depth policy emphasizing stealth penalties, and a rapid-response policy prioritizing speed over covertness.
- **Metrics:** Measure mission success rate, mean time-to-impact, detection likelihood derived from telemetry triggers, and resource consumption across repeated scenarios.
- **Datasets & Environments:** Use synthetic enterprise lab networks seeded with known vulnerabilities, alongside replay of anonymized incident traces for validation.

## Results
- Placeholder for quantitative tables summarizing mission outcomes.
- Placeholder for qualitative analysis highlighting notable attack paths and defensive responses.

## Ethical Safeguards
- Strict access controls, audit trails, and approval workflows gate scenario execution.
- Embedded safety filters restrict disallowed exploit classes and enforce environmental isolation.
- Responsible disclosure procedures align simulations with legal and organizational policies.

## Reproducibility Statement
All experiments rely on containerized lab topologies, versioned agent configurations, and scripted deployment playbooks. Beam search parameters, scoring weights, and policy definitions are logged per run, enabling independent reproduction of results given equivalent infrastructure.

## RL Extension
- **Goal:** Evaluate reinforcement learning policies that adapt orchestrated attack sequences to maximize mission progression while minimizing detection signals across simulated enterprise labs.
- **Safety:** Training runs inherit the platform's isolation guarantees—policies interact only with sandboxed lab services, and scenario gating plus audit logging remain enforced throughout experimentation.
- **Compute Footprint:** CPU-only execution; validated on a 16 vCPU, 32 GB RAM host where the default `total_timesteps=200000` run completes in approximately 2.5 hours.
- **Reproducibility Commands:**
  1. `docker compose -f lab/docker-compose.yml up -d`
  2. `uvicorn orchestrator.app.main:app --reload`
  3. `python analysis/train_loop.py --config analysis/experiment_config.yaml --seed 42`
- **Hardware Notes:** Prioritize consistent CPU frequency scaling (disable aggressive power saving) to ensure stable episode timing; SSD-backed storage is recommended for log ingestion bursts during training checkpoints.
