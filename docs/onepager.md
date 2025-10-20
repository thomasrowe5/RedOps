# RedOps One-Pager

## Project Objective
RedOps is a research-oriented platform for experimenting with defensive automation in a controlled lab. It provides an orchestrated pipeline that simulates adversary behavior, captures the resulting telemetry, and evaluates defensive hypotheses without touching production infrastructure.

## Architecture Overview
```
                         +----------------------+
                         |  Scenario Catalogue  |
                         +----------+-----------+
                                    |
                                    v
+----------------+      poll/queue      +------------------+
| FastAPI        | <------------------> | Simulated Agents |
| Orchestrator   |                      +------------------+
| (orchestrator/) |                           |
+--------+-------+                           | emits events
         |                                     v
         |                          +-----------------------+
         |                          | Telemetry Collectors  |
         |                          | (telemetry/, Filebeat |
         v                          |  + PCAP generators)   |
+--------------------+              +-----------+-----------+
| Run Artifacts &    |                          |
| Event Log Storage  |<-------------------------+
| (data/runs)        |
+--------+-----------+
         |
         v
+----------------------+        +----------------------+        +-----------------------+
| Detection Content    | -----> | Analysis & Scoring   | -----> | Visualization & Lab   |
| (detection/, Sigma)  |        | (analysis/, mapping/) |        | Services (lab/ stack) |
+----------------------+        +----------------------+        +-----------------------+
```

* **Orchestrator** exposes the scheduling API and maintains state for simulated runs.
* **Agents** poll for actions, execute scripted behaviors, and emit structured events.
* **Telemetry** components enrich lab data streams with network captures and log forwarding.
* **Detection** pipelines and **analysis** tooling translate events into MITRE ATT&CK-aligned insights stored in the lab services (Elasticsearch, Kibana, Neo4j).

## Running the MVP Locally
1. Install Docker and Docker Compose v2.
2. From the repository root, boot the lab services:
   ```bash
   cd lab
   docker compose up -d
   ```
3. (Optional) Start the orchestrator API for interactive scenarios:
   ```bash
   cd ../orchestrator
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```
4. Launch a simulated agent pointing at the orchestrator when needed:
   ```bash
   python agents/agent_python/agent.py \
     --orchestrator http://localhost:8000 \
     --agent-id demo-agent \
     --run-id <run returned by the orchestrator>
   ```

## Safety Note
This project is designed strictly for isolated lab environments. Do **not** expose RedOps components, data stores, or intentionally vulnerable services to production networks or untrusted users.

## RL Extension
- **Goal:** Train reinforcement learning policies that refine simulated adversary playbooks to progress missions while suppressing defensive telemetry in the lab.
- **Safety:** Runs stay confined to the orchestrated sandbox with existing approval workflows, scenario gating, and audit logging.
- **Compute Footprint:** CPU-only baseline; a 16 vCPU, 32 GB RAM node completes the default `total_timesteps=200000` configuration in roughly 2.5 hours.
- **Quick Reproduction:**
  1. `docker compose -f lab/docker-compose.yml up -d`
  2. `uvicorn orchestrator.app.main:app --reload`
  3. `python analysis/train_loop.py --config analysis/experiment_config.yaml --seed 42`
- **Hardware Notes:** Keep CPU frequency scaling steady and use SSD-backed storage to avoid I/O stalls during checkpoint logging.

## Recommended Next Steps for Research Publication
### Experiments
- **Automated Detection Efficacy:** Run controlled attack simulations with varying tactics to benchmark how quickly and accurately detections trigger.
- **Response Orchestration Timing:** Measure end-to-end latency from action dispatch to detection alert under different load conditions.
- **Telemetry Coverage Analysis:** Stress-test log and packet collectors using high-volume scenarios to quantify data loss or backlog behavior.

### Evaluation Metrics
- **True/False Positive Rates** per detection rule mapped to MITRE ATT&CK techniques.
- **Mean Time to Detect (MTTD)** and **Mean Time to Respond (MTTR)** for orchestrated runs.
- **Event Throughput & Drop Rate** across telemetry pipelines.
- **Graph Coverage Scores** showing how many scenario steps are enriched with mapping context in Neo4j.
- **Usability Feedback Scores** from analysts running tabletop exercises with the platform.
