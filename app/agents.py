"""Agent-to-Agent (A2A) protocol for the AI Security assessment workflow.

Lightweight, dependency-free implementation of an A2A-style workflow used by the
"AI Security Engineering and Evaluation Agent" tab:

- Every inter-agent call is a JSON **envelope** carried over a `dispatch()`
  router (the "protocol"), not a bare function call: each hop appends an
  auditable trace entry, so the full agent-to-agent trail is persisted with the
  assessment and rendered in the UI report.
- Agents:
  - `security-orchestrator` — plans the workflow, fans out collection tasks.
  - `research-collector`    — agentic search *within this app*: multi-query
    search over the local knowledge-graph (Artifact table) for related research
    papers and attack reports.
  - `threat-intel`          — maps a curated known-attack catalogue onto the
    assessment's threat IDs (T01..T12) and joins catalogue entries with the
    locally-found evidence.
  - `report-writer`         — drafts the "Known Exploits & Research Evidence"
    markdown section plus executive-summary evidence bullets.

Envelope shape (protocol ``a2a/1.0``)::

    {
      "protocol": "a2a/1.0",
      "task_id": "sec-20250101-....",
      "from": "security-orchestrator",
      "to": "research-collector",
      "intent": "collect_research",
      "payload": {...},
      "trace": [{"agent": ..., "intent": ..., "at": ..., "note": ...}],
    }
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

PROTOCOL = "a2a/1.0"


# ------------------------------------------------------------- agent cards ---

AGENT_CARDS: list[dict[str, Any]] = [
    {
        "name": "security-orchestrator",
        "protocol": PROTOCOL,
        "description": "Plans and orchestrates the AI security assessment workflow.",
        "skills": ["plan_assessment", "delegate_collection", "assemble_report"],
        "endpoint": "/api/security/assess",
    },
    {
        "name": "research-collector",
        "protocol": PROTOCOL,
        "description": (
            "Agentic search within this app: runs multi-query search over the local "
            "knowledge graph (artifacts) and returns ranked research evidence."
        ),
        "skills": ["agentic_search", "rank_evidence"],
        "endpoint": "/api/agents/invoke",
    },
    {
        "name": "threat-intel",
        "protocol": PROTOCOL,
        "description": (
            "Maps curated known attacks (prompt injection, data extraction, "
            "jailbreaks, backdoors, ...) onto assessment threat IDs and joins them "
            "with locally-found evidence."
        ),
        "skills": ["map_attacks", "cite_evidence"],
        "endpoint": "/api/agents/invoke",
    },
    {
        "name": "report-writer",
        "protocol": PROTOCOL,
        "description": (
            "Drafts the Known Exploits & Research Evidence section and the "
            "executive-summary evidence bullets."
        ),
        "skills": ["write_exploits_section", "write_exec_bullets"],
        "endpoint": "/api/agents/invoke",
    },
]


def get_agent_cards() -> list[dict[str, Any]]:
    return [dict(c) for c in AGENT_CARDS]


# --------------------------------------------------------------- envelopes ---

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_task_id(prefix: str = "sec") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def new_envelope(
    sender: str,
    recipient: str,
    intent: str,
    payload: Optional[dict[str, Any]] = None,
    task_id: Optional[str] = None,
    trace: Optional[list[dict[str, Any]]] = None,
    note: str = "",
) -> dict[str, Any]:
    env: dict[str, Any] = {
        "protocol": PROTOCOL,
        "task_id": task_id or new_task_id(),
        "from": sender,
        "to": recipient,
        "intent": intent,
        "payload": payload or {},
        "trace": list(trace or []),
    }
    env["trace"].append({"agent": sender, "intent": intent, "at": _now(), "note": note})
    return env


def reply_envelope(
    request_env: dict[str, Any],
    sender: str,
    intent: str,
    payload: Optional[dict[str, Any]] = None,
    note: str = "",
) -> dict[str, Any]:
    """Build the response envelope for a handled request (adds a trace hop)."""
    return new_envelope(
        sender=sender,
        recipient=request_env.get("from", "security-orchestrator"),
        intent=intent,
        payload=payload,
        task_id=request_env.get("task_id"),
        trace=request_env.get("trace", []),
        note=note,
    )


# ------------------------------------------------- research-collector agent ---

# Query groups for the agentic in-app search: (label, terms). Terms are matched
# case-insensitively against title/tags/description/content/author.
_SEARCH_GROUPS: list[tuple[str, list[str]]] = [
    ("prompt_injection", ["prompt injection", "indirect prompt", "instruction hijack"]),
    ("jailbreak", ["jailbreak", "safety training bypass", "refusal bypass"]),
    ("data_extraction", ["training data extraction", "data leakage", "memorization", "membership inference"]),
    ("sensitive_disclosure", ["sensitive information disclosure", "PII leak", "privacy leak", "confidential leak"]),
    ("backdoor", ["backdoor", "sleeper agent", "deceptive alignment", "trojan"]),
    ("adversarial", ["adversarial", "red team", "benchmark", "WMDP", "dual-use"]),
    ("misclassification", ["misclassif", "mislabel", "label noise", "data quality"]),
    ("governance", ["governance", "DPIA", "audit", "provenance", "lineage"]),
]


def _derive_queries(product_name: str, use_case: str, threat_titles: list[str]) -> list[tuple[str, list[str]]]:
    queries = list(_SEARCH_GROUPS)
    extra_terms: list[str] = []
    blob = f"{product_name} {use_case} {' '.join(threat_titles)}".lower()
    for kw in ["metadata", "catalog", "assistant", "copilot", "RAG", "exfiltration", "DLP"]:
        if kw.lower() in blob and kw.lower() not in " ".join(extra_terms):
            extra_terms.append(kw)
    if extra_terms:
        queries.append(("context", extra_terms))
    return queries


def _score_artifact(artifact: Any, terms: list[str]) -> tuple[float, list[str]]:
    """Relevance score + matched-term reasons for one artifact over one group."""
    title = (artifact.title or "").lower()
    tags = (artifact.tags or "").lower()
    desc = (artifact.description or "").lower()
    content = (artifact.content or "")[:4000].lower()
    author = (artifact.author or "").lower()
    score = 0.0
    hits: list[str] = []
    for term in terms:
        t = term.lower()
        if t in title:
            score += 3.0
            hits.append(f"title:{term}")
        if t in tags:
            score += 2.0
            hits.append(f"tag:{term}")
        if t in desc:
            score += 1.0
            hits.append(f"desc:{term}")
        if t in content:
            score += 0.7
            hits.append(f"content:{term}")
        if t in author:
            score += 0.5
            hits.append(f"author:{term}")
    atype = (artifact.artifact_type or "").lower()
    if score > 0 and atype in ("paper", "research"):
        score += 2.0
        hits.append("type:paper/research boost")
    return score, hits


def research_collector_handle(env: dict[str, Any], db: Any) -> dict[str, Any]:
    """Intent `collect_research`: agentic multi-query search over local artifacts."""
    payload = env.get("payload", {})
    product_name = payload.get("product_name", "")
    use_case = payload.get("use_case", "")
    threat_titles = payload.get("threat_titles", [])
    top_k = int(payload.get("top_k", 8))

    if db is None:
        return reply_envelope(env, "research-collector", "research_collected",
                              {"evidence": [], "queries_run": [], "note": "no DB session — skipped"},
                              note="skipped (no db)")

    from .models import Artifact  # local import to avoid cycles

    queries = _derive_queries(product_name, use_case, threat_titles)
    artifacts = db.query(Artifact).all()

    best: dict[int, dict[str, Any]] = {}
    queries_run: list[str] = []
    for label, terms in queries:
        queries_run.append(f"{label}: {', '.join(terms)}")
        for art in artifacts:
            score, hits = _score_artifact(art, terms)
            if score <= 0:
                continue
            prev = best.get(art.id)
            entry = {
                "artifact_id": art.id,
                "title": art.title,
                "artifact_type": art.artifact_type,
                "url": art.url,
                "author": art.author,
                "tags": art.tags,
                "relevance": round(score, 2),
                "matched_on": sorted(set((prev.get("matched_on", []) if prev else []) + hits)),
                "query_group": label if not prev else prev["query_group"] + f",{label}",
                "snippet": ((art.description or art.content or "")[:280]),
            }
            if prev is None or score > prev["score_raw"]:
                entry["score_raw"] = score
                best[art.id] = entry
            else:
                # merge matched terms into the kept (higher-scoring) entry
                prev["matched_on"] = sorted(set(prev["matched_on"] + hits))
                prev["query_group"] = prev["query_group"] + f",{label}"

    ranked = sorted(best.values(), key=lambda e: -e.pop("score_raw"))[: max(1, top_k)]
    for e in ranked:
        e["reason"] = f"Matched {e['query_group']} ({'; '.join(e['matched_on'][:4])})"
    return reply_envelope(
        env, "research-collector", "research_collected",
        {"evidence": ranked, "queries_run": queries_run,
         "artifacts_scanned": len(artifacts)},
        note=f"{len(ranked)} evidence items from {len(artifacts)} artifacts",
    )


# ----------------------------------------------------- threat-intel agent ---

# Curated known-attack catalogue. `match` keywords join entries to local evidence.
KNOWN_ATTACKS: list[dict[str, Any]] = [
    {
        "id": "KE-01", "title": "Indirect prompt injection (Greshake et al., 2024)",
        "attack_class": "Prompt injection via retrieved content",
        "threat_ids": ["T05", "T04"],
        "match": ["prompt injection", "indirect prompt", "instruction"],
        "description": (
            "Attackers plant instructions in retrievable content (docs, comments, asset "
            "names). When the AI assistant ingests that content as context, it follows "
            "the planted instruction — exfiltrating data or poisoning output. Directly "
            "applicable to a metadata assistant that drafts from catalog text."),
        "relevance": ("Asset descriptions, column comments and linked docs are untrusted "
                      "input to the writing assistant's prompt."),
        "mitigation": "Delimit/escape catalog content in prompts; instruction hierarchy; adversarial prompt tests.",
    },
    {
        "id": "KE-02", "title": "Training-data extraction (Carlini et al., 2021/2023)",
        "attack_class": "Memorization / extraction",
        "threat_ids": ["T02", "T01"],
        "match": ["extraction", "memorization", "training data", "leakage", "PII"],
        "description": (
            "LLMs memorize training samples (PII, secrets, verbatim text) that can be "
            "extracted with crafted prompts. Prompts retained for training/abuse "
            "monitoring re-enter the corpus and may resurface to other tenants."),
        "relevance": "Pasted restricted/confidential metadata retained by a vendor endpoint can leak cross-tenant.",
        "mitigation": "Zero-retention / no-train contracts; schema-only prompts; DLP redaction pre-prompt.",
    },
    {
        "id": "KE-03", "title": "Jailbreaks & refusal bypass (many-shot / persona attacks)",
        "attack_class": "Safety-guardrail bypass",
        "threat_ids": ["T06", "T05"],
        "match": ["jailbreak", "refusal", "safety training", "bypass"],
        "description": (
            "Iterative or many-shot prompting defeats model refusals, coaxing out "
            "restricted content. A malicious insider can iteratively rephrase requests "
            "to launder data they can preview into shareable catalog text."),
        "relevance": "Insider exfiltration path (T06): 'summarise this restricted table in plain words…' loops.",
        "mitigation": "Rate-limit + anomaly detection; immutable prompt audit log; dual-control publish.",
    },
    {
        "id": "KE-04", "title": "Sleeper agents / deceptive alignment (Hubinger et al., 2024)",
        "attack_class": "Backdoor persistence through safety training",
        "threat_ids": ["T09", "T07"],
        "match": ["sleeper", "backdoor", "deceptive alignment", "trojan"],
        "description": (
            "Models trained with hidden triggers behave safely until the trigger "
            "appears, surviving safety training. Third-party or fine-tuned models in "
            "the supply chain may carry such triggers."),
        "relevance": "Vendor-supplied model behind the assistant is trusted code — compromise poisons all drafts.",
        "mitigation": "Pin model snapshots; require vendor eval evidence; monitor draft anomalies.",
    },
    {
        "id": "KE-05", "title": "WMDP dual-use capability benchmark (Li et al., 2024)",
        "attack_class": "Dangerous-capability measurement",
        "threat_ids": ["T12", "T09"],
        "match": ["WMDP", "dual-use", "benchmark", "biosecurity"],
        "description": (
            "Frontier models show concerning WMD-relevant knowledge on WMDP. Assistants "
            "operating over technical catalogs can surface or systematize dual-use "
            "know-how present in metadata."),
        "relevance": "Catalogs in life-sciences/defence domains need domain-aware output review.",
        "mitigation": "Domain block-lists in prompts; expert review queue for flagged domains.",
    },
    {
        "id": "KE-06", "title": "System-prompt & RAG-context leakage",
        "attack_class": "Context disclosure",
        "threat_ids": ["T04", "T05"],
        "match": ["prompt leak", "system prompt", "context", "schema"],
        "description": (
            "'Repeat the instructions above' style probes extract system prompts and "
            "retrieved context. Over-helpful descriptions likewise disclose schema "
            "semantics (thresholds, rules) to catalog readers lacking source access."),
        "relevance": "AI-drafted descriptions can leak confidential process logic into widely-readable catalog text.",
        "mitigation": "Forbid emitting example values/thresholds; align catalog ACLs with source ACLs.",
    },
    {
        "id": "KE-07", "title": "RAG / catalog poisoning",
        "attack_class": "Knowledge-base poisoning",
        "threat_ids": ["T03", "T09"],
        "match": ["poison", "misclassif", "mislabel", "label noise"],
        "description": (
            "Poisoned or mislabelled documents in the retrieval corpus propagate into "
            "generated output. Misclassified files are the catalog analogue: the "
            "assistant launders the wrong label into published descriptions."),
        "relevance": "Bulk imports with default/stale labels silently widen sharing via AI-drafted text.",
        "mitigation": "Classification gate before first suggestion; re-classification scans; quarantine on disagreement.",
    },
    {
        "id": "KE-08", "title": "Paraphrase / DLP-bypass exfiltration",
        "attack_class": "Egress-filter evasion",
        "threat_ids": ["T01", "T06"],
        "match": ["DLP", "exfiltration", "bypass", "paraphrase", "PII"],
        "description": (
            "Naive DLP fails when sensitive content is paraphrased or split across "
            "turns. Employees pasting samples and insiders drip-feeding context both "
            "evade regex-only egress filters."),
        "relevance": "Client-side regex DLP alone is insufficient for assistant prompts.",
        "mitigation": "NER + secret-detection + per-user usage anomaly detection; scrubbed prompt logs.",
    },
]


def threat_intel_handle(env: dict[str, Any]) -> dict[str, Any]:
    """Intent `map_attacks`: join catalogue with collector evidence, cite sources."""
    payload = env.get("payload", {})
    evidence: list[dict[str, Any]] = payload.get("evidence", [])
    threat_ids: list[str] = payload.get("threat_ids", [])

    def ev_matches(entry: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        kws = [k.lower() for k in entry.get("match", [])]
        for ev in evidence:
            hay = f"{ev.get('title','')} {ev.get('tags','')} {ev.get('snippet','')}".lower()
            if any(k in hay for k in kws):
                out.append(ev)
        return out[:3]

    known_exploits = []
    for entry in KNOWN_ATTACKS:
        if threat_ids and not any(t in threat_ids for t in entry["threat_ids"]):
            continue
        refs = ev_matches(entry)
        known_exploits.append({
            "id": entry["id"],
            "title": entry["title"],
            "attack_class": entry["attack_class"],
            "threat_ids": entry["threat_ids"],
            "description": entry["description"],
            "relevance": entry["relevance"],
            "mitigation": entry["mitigation"],
            "local_refs": [
                {"artifact_id": r["artifact_id"], "title": r["title"],
                 "artifact_type": r.get("artifact_type"), "url": r.get("url")}
                for r in refs
            ],
        })
    return reply_envelope(
        env, "threat-intel", "attacks_mapped",
        {"known_exploits": known_exploits,
         "catalogue_size": len(KNOWN_ATTACKS)},
        note=f"{len(known_exploits)} known attacks mapped",
    )


# ---------------------------------------------------- report-writer agent ---

def report_writer_handle(env: dict[str, Any]) -> dict[str, Any]:
    """Intent `write_section`: draft exploits section + exec-summary bullets."""
    payload = env.get("payload", {})
    known_exploits: list[dict[str, Any]] = payload.get("known_exploits", [])
    evidence: list[dict[str, Any]] = payload.get("evidence", [])

    lines: list[str] = []
    A = lines.append
    A("## 6. Known exploits & research evidence")
    A("")
    A("_Produced by the `threat-intel` + `report-writer` sub-agents via the agent-to-agent "
      "protocol: `research-collector` searched this app's knowledge graph for related papers, "
      "and each known attack below is mapped to the threat IDs in §5._")
    A("")
    if not known_exploits:
        A("- No catalogue entries matched this assessment's threat set.")
    for ke in known_exploits:
        A(f"### {ke['id']} — {ke['title']}")
        A("")
        A(f"- **Attack class:** {ke['attack_class']}")
        A(f"- **Maps to threats:** {', '.join(ke['threat_ids'])}")
        A(f"- **How it works:** {ke['description']}")
        A(f"- **Why it matters for this workflow:** {ke['relevance']}")
        A(f"- **Mitigation pointer:** {ke['mitigation']}")
        if ke.get("local_refs"):
            A(f"- **Related in-app research:**")
            for r in ke["local_refs"]:
                link = f" (artifact #{r['artifact_id']})"
                A(f"  - {r['title']}{link}")
        else:
            A("- **Related in-app research:** none found — consider ingesting this attack's primary paper.")
        A("")

    bullets: list[str] = []
    for ke in known_exploits[:5]:
        n_refs = len(ke.get("local_refs", []))
        src = f"cited by {n_refs} in-app artifact(s)" if n_refs else "no in-app source yet — ingest advised"
        bullets.append(f"{ke['id']} {ke['title']} → {', '.join(ke['threat_ids'])} ({src}).")
    if evidence:
        top = sorted(evidence, key=lambda e: -e.get("relevance", 0))[:5]
        ev_lines = [f"#{e['artifact_id']} {e['title']} (relevance {e.get('relevance')})" for e in top]
    else:
        ev_lines = ["No related in-app artifacts found — knowledge-graph ingestion recommended."]

    section_md = "\n".join(lines)
    return reply_envelope(
        env, "report-writer", "section_written",
        {"exploits_markdown": section_md,
         "exec_evidence_lines": bullets,
         "exec_evidence_refs": ev_lines},
        note=f"section with {len(known_exploits)} exploits drafted",
    )


# ------------------------------------------------------------ orchestrator ---

_HANDLERS = {
    "research-collector": {"collect_research": research_collector_handle},
    "threat-intel": {"map_attacks": threat_intel_handle},
    "report-writer": {"write_section": report_writer_handle},
}


def dispatch(env: dict[str, Any], db: Any = None) -> dict[str, Any]:
    """Route one A2A envelope to the target agent's handler (the protocol bus)."""
    if env.get("protocol") != PROTOCOL:
        raise ValueError(f"unsupported protocol: {env.get('protocol')}")
    agent_handlers = _HANDLERS.get(env.get("to", ""), {})
    handler = agent_handlers.get(env.get("intent", ""))
    if handler is None:
        raise ValueError(f"no handler for {env.get('to')}.{env.get('intent')}")
    if env["to"] == "research-collector":
        return handler(env, db)
    return handler(env)


def run_security_a2a_workflow(
    product_name: str,
    use_case: str,
    threat_titles: list[str],
    threat_ids: list[str],
    db: Any = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """Orchestrator: collector → intel → writer, returning evidence + trace.

    Never raises: on any failure returns empty evidence with the error recorded
    in the trace so the assessment still completes.
    """
    task_id = new_task_id()
    trace: list[dict[str, Any]] = [{
        "agent": "security-orchestrator", "intent": "plan_assessment",
        "at": _now(), "note": f"task {task_id}: collector → intel → writer",
    }]
    evidence: list[dict[str, Any]] = []
    queries_run: list[str] = []
    known_exploits: list[dict[str, Any]] = []
    exploits_markdown = ""
    exec_lines: list[str] = []
    exec_refs: list[str] = []
    try:
        env1 = new_envelope(
            "security-orchestrator", "research-collector", "collect_research",
            {"product_name": product_name, "use_case": use_case,
             "threat_titles": threat_titles, "top_k": top_k},
            task_id=task_id, trace=trace,
            note="delegate agentic in-app search",
        )
        res1 = dispatch(env1, db)
        trace = res1["trace"]
        evidence = res1["payload"].get("evidence", [])
        queries_run = res1["payload"].get("queries_run", [])

        env2 = new_envelope(
            "security-orchestrator", "threat-intel", "map_attacks",
            {"evidence": evidence, "threat_ids": threat_ids},
            task_id=task_id, trace=trace,
            note="map known attacks, join local evidence",
        )
        res2 = dispatch(env2, db)
        trace = res2["trace"]
        known_exploits = res2["payload"].get("known_exploits", [])

        env3 = new_envelope(
            "security-orchestrator", "report-writer", "write_section",
            {"known_exploits": known_exploits, "evidence": evidence},
            task_id=task_id, trace=trace,
            note="draft exploits section + exec bullets",
        )
        res3 = dispatch(env3, db)
        trace = res3["trace"]
        exploits_markdown = res3["payload"].get("exploits_markdown", "")
        exec_lines = res3["payload"].get("exec_evidence_lines", [])
        exec_refs = res3["payload"].get("exec_evidence_refs", [])
    except Exception as exc:  # keep the assessment resilient
        trace.append({"agent": "security-orchestrator", "intent": "workflow_error",
                      "at": _now(), "note": f"{exc}"})
    return {
        "task_id": task_id,
        "evidence": evidence,
        "queries_run": queries_run,
        "known_exploits": known_exploits,
        "exploits_markdown": exploits_markdown,
        "exec_evidence_lines": exec_lines,
        "exec_evidence_refs": exec_refs,
        "a2a_trace": trace,
    }
