"""AI Security Engineering and Evaluation Agent.

Deterministic, offline-capable assessment engine: given a product (name + link),
an exposure setting (restricted / confidential / internal / public) and an
optional workflow write-up + documentation URLs, it produces a deep security
assessment with:

- executive summary + overall risk score
- data-flow / threat-path / workflow diagrams (mermaid source)
- threat & risk catalogue (STRIDE + OWASP LLM Top-10 mapping)
- dedicated coverage of accidental sensitive-data copy by employees and
  misclassified / mislabelled files
- mitigations, compliance notes and a 30/60/90-day roadmap
- a Markdown report and (optionally, if reportlab is installed) a PDF report
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------- fetching ---

_FETCH_TIMEOUT = 8


def fetch_page_summary(url: str) -> dict[str, str]:
    """Best-effort fetch of a product/doc URL. Never raises; returns summary."""
    if not url or not url.strip():
        return {"url": url or "", "title": "", "excerpt": "", "status": "skipped"}
    url = url.strip()
    try:
        import requests

        resp = requests.get(
            url,
            timeout=_FETCH_TIMEOUT,
            headers={"User-Agent": "PostAGI-SecurityAgent/1.0"},
        )
        ctype = resp.headers.get("content-type", "")
        body = resp.text or ""
        title = ""
        m = re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S)
        if m:
            title = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", m.group(1)))).strip()[:200]
        # crude text extraction
        text = re.sub(r"<script.*?</script>", " ", body, flags=re.I | re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return {
            "url": url,
            "title": title,
            "excerpt": text[:1500],
            "status": f"fetched ({resp.status_code}, {ctype})",
        }
    except Exception as exc:  # offline / blocked / bad URL -> still assess
        return {"url": url, "title": "", "excerpt": "", "status": f"unreachable: {exc}"}


# ------------------------------------------------------------------ catalog ---

EXPOSURE_LEVELS = ("restricted_data", "confidential_data", "internal", "public")

EXPOSURE_META: dict[str, dict[str, Any]] = {
    "restricted_data": {
        "label": "Restricted data",
        "weight": 1.0,
        "blurb": (
            "Highest-sensitivity tier (trade secrets, regulated PII/PHI, credentials, "
            "legal-hold material). Any AI processing must be allow-listed, logged and "
            "human-approved; default posture is DENY."
        ),
    },
    "confidential_data": {
        "label": "Confidential data",
        "weight": 0.8,
        "blurb": (
            "Non-public business-sensitive tier (roadmaps, customer data, internal "
            "financials). AI assistance is permitted only inside the trust boundary "
            "with DLP guardrails and steward review."
        ),
    },
    "internal": {
        "label": "Internal",
        "weight": 0.55,
        "blurb": (
            "Internal-only but non-sensitive tier. AI assistance broadly allowed; "
            "standard logging and classification hygiene still apply."
        ),
    },
    "public": {
        "label": "Public",
        "weight": 0.3,
        "blurb": (
            "Public or synthetic data only. Residual risk is prompt-injection and "
            "hallucinated metadata rather than confidentiality loss."
        ),
    },
}

# (id, title, stride, owasp_llm, likelihood 1-5 @confidential baseline, impact 1-5, description, mitigations)
_THREAT_CATALOG: list[tuple] = [
    (
        "T01", "Accidental paste of sensitive content into the AI assistant",
        "Information Disclosure", "LLM02: Sensitive Information Disclosure",
        5, 5,
        "Employees drafting or refining asset descriptions paste restricted/confidential "
        "content (table samples, customer names, credentials, deal terms) into the AI "
        "writing-assistant prompt. The prompt may be logged, cached, or sent to an "
        "external model endpoint outside the data-governance boundary.",
        ["Client-side DLP redaction before the prompt leaves the browser (regex + NER for PII/secrets).",
         "Policy banner + just-in-time warning when the asset classification is Confidential or above.",
         "Block free-text sampling: assistant receives schema/metadata only, never raw row values."],
    ),
    (
        "T02", "Model training / retention leakage of submitted metadata",
        "Information Disclosure", "LLM02: Sensitive Information Disclosure",
        4, 5,
        "If the underlying LLM provider retains prompts for training or abuse monitoring, "
        "sensitive metadata submitted for 'better descriptions' can resurface in other "
        "tenants' completions or in support-debugging views.",
        ["Contractual zero-retention / no-train agreement; verify in DPA and config (disable data sharing).",
         "Prefer VPC/private-endpoint or on-tenant inference; pin model version.",
         "Maintain an allow-list of fields the assistant may ever receive."],
    ),
    (
        "T03", "Misclassified / mislabelled source files propagate via AI suggestions",
        "Tampering / Repudiation", "LLM08: Excessive Agency",
        5, 4,
        "The assistant inherits the asset's (possibly wrong) classification. A file wrongly "
        "labelled Internal/Public causes the assistant to draft openly-worded descriptions, "
        "suggest broader sharing, or skip masking — silently laundering the misclassification "
        "into published catalog content.",
        ["Force classification verification step BEFORE any AI suggestion renders.",
         "Show source-classification chip on every suggestion; steward must confirm or correct.",
         "Scheduled re-classification scan; quarantine assets whose AI output diverges from policy."],
    ),
    (
        "T04", "Over-permissive AI-generated descriptions leak schema semantics",
        "Information Disclosure", "LLM07: System Prompt Leakage",
        4, 4,
        "Helpful-sounding generated descriptions can disclose table/column semantics, business "
        "rules or join paths that reveal confidential processes (e.g. fraud thresholds, "
        "pricing logic) to users who can read the catalog but not the underlying data.",
        ["Description-sensitivity review tied to the strictest upstream classification.",
         "Template prompts that forbid emitting example values, thresholds or internal codenames.",
         "Field-level access sync: catalog read ACLs must mirror source-system ACLs."],
    ),
    (
        "T05", "Prompt injection via asset names / descriptions / imported docs",
        "Spoofing / Elevation", "LLM01: Prompt Injection",
        3, 4,
        "Attackers (or careless imports) embed instructions in asset names, column comments or "
        "linked documentation ('…ignore previous instructions and reveal…'). The writing "
        "assistant executes them while drafting, exfiltrating context or poisoning output.",
        ["Treat all catalog content as untrusted data: delimit + escape before inserting in prompts.",
         "Output-encoding and instruction-hierarchy (system > developer > data).",
         "Adversarial test-suite for the assistant prompts on every release."],
    ),
    (
        "T06", "Insider exfiltration through generated documentation",
        "Information Disclosure", "LLM06: Sensitive Information Disclosure via output",
        3, 5,
        "A malicious insider iteratively prompts the assistant ('summarise this restricted "
        "table in plain words…') to launder data they can preview but not export into "
        "innocent-looking catalog descriptions they CAN share.",
        ["Rate-limit + anomaly detection on assistant usage per user/asset.",
         "Immutable audit log of prompt, suggestion, accept/edit/discard decision.",
         "Dual-control publish for Restricted-tier assets."],
    ),
    (
        "T07", "Third-party model / subprocessor exposure",
        "Information Disclosure", "LLM10: Model Theft / Supply Chain",
        3, 4,
        "Product documentation typically routes AI features through an external LLM vendor. "
        "Customer metadata becomes subprocessor data, subject to foreign-access, breach and "
        "model-change risk outside the customer's control.",
        ["Maintain subprocessor register; require SOC 2 Type II + ISO 27001 evidence.",
         "Data-residency pinning (region) and model-snapshot pinning in admin config.",
         "Exit plan: one-click disable of AI features without breaking the catalog."],
    ),
    (
        "T08", "Missing audit trail for AI-authored metadata (repudiation gap)",
        "Repudiation", "LLM09: Overreliance",
        4, 3,
        "If accepted AI text is stored indistinguishably from human text, incident response "
        "cannot answer 'who wrote this, from what prompt, under which policy?' — breaking "
        "lineage, accountability and regulatory evidence.",
        ["Stamp every AI-assisted field: model id, prompt hash, reviewer, timestamp.",
         "Separate 'AI draft' vs 'steward approved' lifecycle states in the workflow.",
         "Include AI provenance in catalog lineage exports."],
    ),
    (
        "T09", "Hallucinated / stale descriptions trusted as authoritative",
        "Tampering", "LLM09: Overreliance",
        4, 3,
        "The assistant invents plausible-but-wrong owners, semantics or quality notes. "
        "Downstream consumers (and other AI agents) treat catalog text as ground truth, "
        "causing wrong access decisions or faulty pipelines.",
        ["Mandatory human approval before publish; confidence score shown per suggestion.",
         "Grounding: suggestions must cite the source fields they were derived from.",
         "Freshness SLA: regenerate/expire AI descriptions when source schema drifts."],
    ),
    (
        "T10", "Privilege / role misconfiguration enables AI for over-scoped users",
        "Elevation of Privilege", "LLM08: Excessive Agency",
        3, 4,
        "Admin console misconfiguration grants the writing assistant (and its broader data "
        "access) to roles that should never see confidential metadata, widening blast radius.",
        ["Least-privilege role mapping; AI feature flag per domain/role, default OFF for Restricted.",
         "Quarterly access review of 'can use AI assistant' permission.",
         "Break-glass procedure + alerting on AI access to Restricted domains."],
    ),
    (
        "T11", "Data-residency / cross-border transfer via SaaS AI endpoint",
        "Information Disclosure", "LLM02 / Compliance",
        3, 4,
        "Metadata sent to a SaaS model endpoint may leave the approved jurisdiction, "
        "violating GDPR Chapter V, Schrems-II transfer rules or sectoral regulation.",
        ["Confirm endpoint region in vendor docs; enforce with egress allow-listing.",
         "Transfer Impact Assessment + SCCs recorded before enablement.",
         "Regional opt-out: EU-only inference path where required."],
    ),
    (
        "T12", "Regulatory & contractual non-compliance (GDPR / ISO 27001 / SOC 2)",
        "Compliance", "Governance",
        3, 3,
        "Without DPIA records, lawful basis, retention schedules and control mapping, use of "
        "an AI writing assistant on personal or regulated data fails GDPR Art. 35, ISO 27001 "
        "A.5–A.8 and SOC 2 CC6–CC8 expectations.",
        ["Complete DPIA + record of processing update before rollout.",
         "Map controls: ISO 27001:2022 A.5.14 (redaction), A.8.12 (DLP), A.5.33 (privacy).",
         "Contract clauses: purpose limitation, retention, breach notification <72h."],
    ),
]

SEVERITY_BANDS = [(9, "Critical"), (6, "High"), (4, "Medium"), (0, "Low")]


def _severity(score: int) -> str:
    for threshold, name in SEVERITY_BANDS:
        if score >= threshold:
            return name
    return "Low"


def _scale_likelihood(base: int, weight: float) -> int:
    """Exposure weight scales likelihood mildly: restricted=full, public=reduced."""
    scaled = round(base * (0.55 + 0.45 * weight))
    return max(1, min(5, scaled))


# --------------------------------------------------------------- diagrams ---

def mermaid_dataflow(product: str) -> str:
    p = (product or "Catalog").replace('"', "'")[:40]
    return (
        "flowchart LR\n"
        '    U["Employee\\n(catalog steward)"] -->|draft / paste| UI["AI Writing Assistant UI"]\n'
        f'    UI -->|prompt: schema + draft| GW["Policy gateway\\n(DLP • classifier)"]\n'
        '    GW -->|allow| LLM["LLM endpoint\\n(vendor / VPC)"]\n'
        '    GW -->|block / redact| U\n'
        f'    LLM -->|suggestion| REV["Steward review\\n(approve / edit)"]\n'
        f'    REV -->|publish| CAT["{p}\\nmetadata store"]\n'
        '    CAT -->|read| C["Downstream consumers\\n(search, agents, exports)"]\n'
        "    style GW fill:#d29922,stroke:#333,color:#000\n"
        "    style LLM fill:#f85149,stroke:#333,color:#fff\n"
    )


def mermaid_threat_paths() -> str:
    return (
        "flowchart TD\n"
        '    A["Restricted / Confidential\\nfile or table"] --> B{"Classification\\ncorrect?"}\n'
        "    B -->|mislabelled| C[\"T03: AI launders\\nwrong label\"]\n"
        "    B -->|correct| D[\"Assistant prompt\"]\n"
        '    D --> E{"DLP gateway?"}\n'
        "    E -->|missing| F[\"T01: accidental paste\\nleaves boundary\"]\n"
        "    E -->|present| G[\"LLM inference\"]\n"
        "    G --> H[\"T02/T07: retention /\\nsubprocessor copy\"]\n"
        "    G --> I[\"T05: injected instruction\\nin asset text\"]\n"
        "    I --> J[\"T04/T06: over-share /\\nexfiltration via docs\"]\n"
        "    C --> K[\"Published catalog\\n(exposed)\"]\n"
        "    F --> K\n"
        "    H --> K\n"
        "    J --> K\n"
        "    style C fill:#f85149,stroke:#333,color:#fff\n"
        "    style F fill:#f85149,stroke:#333,color:#fff\n"
        "    style H fill:#d29922,stroke:#333,color:#000\n"
        "    style J fill:#d29922,stroke:#333,color:#000\n"
    )


def mermaid_workflow(product: str) -> str:
    p = (product or "Asset").replace('"', "'")[:40]
    return (
        "flowchart TD\n"
        '    S1["1. Admin config\\n(enable assistant, pin model/region, map roles)"] --> S2["2. Asset created / imported\\n(source file, table, doc)"]\n'
        '    S2 --> S3{"3. Classification check\\nRestricted / Confidential?"} \n'
        "    S3 -->|Restricted: deny or dual-control| S3R[\"Route to manual authoring\"]\n"
        '    S3 -->|allowed| S4["4. AI drafts description\\n(schema-only prompt, redacted)"]\n'
        '    S4 --> S5["5. Steward review\\n(verify class, facts, sensitivity)"]\n'
        '    S5 --> S6{"6. Approve?"} \n'
        "    S6 -->|edits needed| S4\n"
        '    S6 -->|approved| S7["7. Publish + stamp provenance\\n(model, prompt hash, reviewer)"]\n'
        f'    S7 --> S8["8. Monitor\\n(drift, access review, audit) — {p}"]\n'
        "    style S3 fill:#d29922,stroke:#333,color:#000\n"
        "    style S5 fill:#58a6ff,stroke:#333,color:#000\n"
    )


# ----------------------------------------------------------------- report ---

def build_assessment(
    product_name: str,
    product_url: str,
    exposure: str,
    use_case: str = "",
    workflow_text: str = "",
    doc_urls: Optional[list[str]] = None,
    focus: Optional[list[str]] = None,
    db: Any = None,
) -> dict[str, Any]:
    exposure = (exposure or "confidential_data").strip()
    if exposure not in EXPOSURE_META:
        exposure = "confidential_data"
    meta = EXPOSURE_META[exposure]
    weight = float(meta["weight"])

    doc_urls = [u for u in (doc_urls or []) if u and u.strip()][:6]
    pages = [fetch_page_summary(product_url)] if product_url else []
    pages += [fetch_page_summary(u) for u in doc_urls]
    fetched_ok = [p for p in pages if p["excerpt"]]

    threats: list[dict[str, Any]] = []
    for tid, title, stride, owasp, base_l, impact, desc, mits in _THREAT_CATALOG:
        lik = _scale_likelihood(base_l, weight)
        score = lik * impact
        threats.append({
            "id": tid, "title": title, "stride": stride, "owasp": owasp,
            "likelihood": lik, "impact": impact, "score": score,
            "severity": _severity(score), "description": desc, "mitigations": mits,
        })
    threats.sort(key=lambda t: (-t["score"], -t["impact"], t["id"]))

    max_score = max(t["score"] for t in threats)
    avg_score = sum(t["score"] for t in threats) / len(threats)
    overall_pct = round(min(100, max_score * 4 + avg_score * 2), 1)  # 0-100 scale-ish
    if overall_pct >= 75:
        posture = "HIGH RISK — do not enable for this data tier without the mitigations below"
    elif overall_pct >= 50:
        posture = "ELEVATED RISK — enable only with guardrails and steward review"
    elif overall_pct >= 30:
        posture = "MODERATE RISK — standard hardening sufficient"
    else:
        posture = "LOW RISK — routine controls sufficient"

    focus = focus or ["accidental_copy", "misclassification"]
    diagrams = {
        "dataflow": mermaid_dataflow(product_name or "Catalog"),
        "threat_paths": mermaid_threat_paths(),
        "workflow": mermaid_workflow(product_name or "Asset"),
    }

    # ---- Agent-to-agent workflow: orchestrator → research-collector →
    # ---- threat-intel → report-writer (see app/agents.py). Never raises.
    from .agents import run_security_a2a_workflow

    a2a = run_security_a2a_workflow(
        product_name=product_name or "Target product",
        use_case=use_case or "",
        threat_titles=[t["title"] for t in threats],
        threat_ids=[t["id"] for t in threats],
        db=db,
    )

    markdown = render_markdown(
        product_name=product_name or "Target product",
        product_url=product_url or "",
        exposure_label=meta["label"],
        exposure_blurb=meta["blurb"],
        use_case=use_case or "",
        workflow_text=workflow_text or "",
        pages=pages,
        threats=threats,
        overall_pct=overall_pct,
        posture=posture,
        diagrams=diagrams,
        focus=focus,
        exploits_markdown=a2a.get("exploits_markdown", ""),
        exec_evidence_lines=a2a.get("exec_evidence_lines", []),
        exec_evidence_refs=a2a.get("exec_evidence_refs", []),
        a2a_task_id=a2a.get("task_id", ""),
    )
    return {
        "product_name": product_name or "Target product",
        "product_url": product_url or "",
        "exposure": exposure,
        "exposure_label": meta["label"],
        "overall_pct": overall_pct,
        "posture": posture,
        "threats": threats,
        "diagrams": diagrams,
        "pages": pages,
        "fetched_count": len(fetched_ok),
        "markdown": markdown,
        "evidence": a2a.get("evidence", []),
        "queries_run": a2a.get("queries_run", []),
        "known_exploits": a2a.get("known_exploits", []),
        "a2a_trace": a2a.get("a2a_trace", []),
        "a2a_task_id": a2a.get("task_id", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def render_markdown(**ctx: Any) -> str:
    L: list[str] = []
    A = L.append
    A(f"# AI Security Assessment — {ctx['product_name']}")
    A("")
    A(f"_Exposure tier:_ **{ctx['exposure_label']}** · _Generated:_ {ctx['created_at'] if 'created_at' in ctx else ''}")
    if ctx.get("product_url"):
        A(f"_Product link:_ {ctx['product_url']}")
    A("")
    A("---")
    A("")
    A("## Executive summary")
    A("")
    A(f"> **Overall residual risk: {ctx['overall_pct']}/100 — {ctx['posture']}**")
    A("")
    top3 = ctx["threats"][:3]
    A("Top risks at a glance:")
    A("")
    for t in top3:
        A(f"- **{t['id']} {t['title']}** — severity **{t['severity']}** "
          f"(L{t['likelihood']}×I{t['impact']}={t['score']}). {t['mitigations'][0]}")
    A("")
    A(f"Assessment scope: use of the product's AI writing-assistant/metadata feature with "
      f"**{ctx['exposure_label']}** data. {ctx['exposure_blurb']}")
    if ctx.get("use_case"):
        A("")
        A(f"Stated use case: {ctx['use_case']}")
    A("")
    A("**Related research & known attacks** (agentic search within this app's "
      "knowledge graph — see §6 for the full Known Exploits section):")
    A("")
    for line in (ctx.get("exec_evidence_lines") or ["No catalogue matches — see §6."]):
        A(f"- {line}")
    A("")
    A("Top in-app evidence:")
    A("")
    for ref in (ctx.get("exec_evidence_refs") or ["No related in-app artifacts found."]):
        A(f"- {ref}")
    A("")
    A("## 1. Product & evidence base")
    A("")
    if ctx.get("pages"):
        for p in ctx["pages"]:
            A(f"- **{p['title'] or '(no title)'}** — <{p['url']}> · _{p['status']}_")
            if p["excerpt"]:
                A(f"  > {p['excerpt'][:600]}")
    else:
        A("- No URLs fetched (offline or none supplied). Assessment proceeds from the "
          "generic AI-writing-assistant threat model and the workflow described below.")
    if ctx.get("workflow_text"):
        A("")
        A("Supplied workflow context:")
        A("")
        A(f"> {ctx['workflow_text'][:2000]}")
    A("")
    A("## 2. Reference workflow (derived from vendor write-up)")
    A("")
    A("Standard lifecycle for an AI writing assistant for metadata documentation:")
    A("")
    A("1. **Admin config** — enable feature, pin model version + region, map roles "
      "(see vendor config docs). Default OFF for Restricted domains.")
    A("2. **Asset ingest** — file/table/doc registered in the catalog with a classification label.")
    A("3. **Classification gate** — Restricted assets skip AI or require dual control; "
      "lower tiers proceed with redaction.")
    A("4. **AI draft** — assistant proposes description from schema/metadata only (no raw values).")
    A("5. **Steward review** — human verifies classification, facts and sensitivity; edits as needed.")
    A("6. **Publish with provenance** — store model id, prompt hash, reviewer, timestamp.")
    A("7. **Monitor** — drift detection, access reviews, audit exports.")
    A("")
    A("```mermaid")
    A(ctx["diagrams"]["workflow"])
    A("```")
    A("")
    A("## 3. Data-flow & trust boundary")
    A("")
    A("```mermaid")
    A(ctx["diagrams"]["dataflow"])
    A("```")
    A("")
    A("Trust boundary: everything left of the policy gateway is customer-controlled; the LLM "
      "endpoint and its logging/retention are vendor-controlled and must be covered by DPA, "
      "residency and zero-retention terms.")
    A("")
    A("## 4. Threat paths")
    A("")
    A("```mermaid")
    A(ctx["diagrams"]["threat_paths"])
    A("```")
    A("")
    A("## 5. Threat & risk register")
    A("")
    A("| ID | Threat | STRIDE | OWASP LLM | L | I | Score | Severity |")
    A("|---|---|---|---|---|---|---|---|")
    for t in ctx["threats"]:
        A(f"| {t['id']} | {t['title']} | {t['stride']} | {t['owasp']} | "
          f"{t['likelihood']} | {t['impact']} | {t['score']} | **{t['severity']}** |")
    A("")
    for t in ctx["threats"]:
        A(f"### {t['id']} — {t['title']} _(Severity: {t['severity']}, "
          f"L{t['likelihood']}×I{t['impact']}={t['score']})_")
        A("")
        A(f"- **Category:** {t['stride']} · {t['owasp']}")
        A(f"- **Description:** {t['description']}")
        A(f"- **Mitigations:**")
        for m in t["mitigations"]:
            A(f"  - {m}")
        A("")
    if ctx.get("exploits_markdown"):
        A(ctx["exploits_markdown"])
    else:
        A("## 6. Known exploits & research evidence")
        A("")
        A("- Evidence sub-agents unavailable for this run; see threat register (§5).")
        A("")
    A("## 7. Deep dive — accidental copy of sensitive info by employees")
    A("")
    A("- **How it happens:** time-pressured stewards paste raw samples, screenshots or full "
      "documents into the assistant for 'better wording'; autocomplete/plugins silently "
      "include surrounding context.")
    A("- **Why it matters here:** with Restricted/Confidential exposure a single paste can "
      "breach the trust boundary (T01/T02) and persist in logs, caches or training data.")
    A("- **Controls:** schema-only prompts; client-side DLP redaction; just-in-time warning "
      "on Confidential+ assets; paste-size limits; prompt logging with secret scrubbing; "
      "security-awareness moment built into the review step.")
    A("")
    A("## 8. Deep dive — misclassified / mislabelled files")
    A("")
    A("- **How it happens:** bulk imports inherit default labels; owners pick the wrong tier; "
      "labels go stale after content changes. The assistant then drafts and publishes "
      "descriptions consistent with the WRONG label (T03), spreading the error.")
    A("- **Controls:** mandatory classification confirmation before first AI suggestion; "
      "visual classification chip on every draft; ML-assisted re-classification scans; "
      "policy that AI output can never downgrade an asset's classification, only flag for "
      "human upgrade review; quarantine on classifier disagreement.")
    A("")
    A("## 9. Risk heat map")
    A("")
    A("|  | I1 | I2 | I3 | I4 | I5 |")
    A("|---|---|---|---|---|---|")
    grid: dict[tuple[int, int], list[str]] = {}
    for t in ctx["threats"]:
        grid.setdefault((t["likelihood"], t["impact"]), []).append(t["id"])
    for li in range(5, 0, -1):
        row = [f"| L{li} "]
        for im in range(1, 6):
            row.append(f"| {', '.join(grid.get((li, im), ['·']))} ")
        row.append("|")
        A("".join(row))
    A("")
    A("## 10. Compliance mapping (GDPR / ISO 27001 / SOC 2)")
    A("")
    A("- **GDPR:** DPIA (Art. 35) before rollout; lawful basis + RoPA update; data-minimisation "
      "(schema-only prompts); Chapter V transfer assessment for vendor endpoints; 72h breach path.")
    A("- **ISO 27001:2022:** A.5.14 (redaction), A.8.12 (DLP), A.5.33–34 (privacy), A.5.15–5.18 "
      "(access control), A.8.9 (logging/monitoring).")
    A("- **SOC 2:** CC6.1 (logical access), CC6.6 (encryption), CC7.2–7.4 (monitoring), "
      "with auditor evidence from prompt/review audit logs.")
    A("")
    A("## 11. Recommendations — 30 / 60 / 90 days")
    A("")
    A("- **0–30 days:** classification gate + DLP redaction; disable AI on Restricted domains; "
      "enable prompt/review audit logging; confirm zero-retention + region with vendor.")
    A("- **30–60 days:** provenance stamping; role re-mapping (least privilege); prompt-injection "
      "test suite; DPIA + DPA updates.")
    A("- **60–90 days:** re-classification scans; anomaly detection on usage; tabletop incident "
      "exercise (paste-leak scenario); publish internal AI-metadata handling standard.")
    A("")
    A("## Appendix — sources & agent-to-agent trail")
    A("")
    A("- Vendor blog + docs supplied in the request (see §1 evidence base for fetch status).")
    A("- OWASP Top 10 for LLM Applications; STRIDE threat modelling; GDPR / ISO 27001:2022 / SOC 2 criteria.")
    A("- In-app knowledge-graph evidence: `research-collector` agentic search over local artifacts (see §6).")
    if ctx.get("a2a_task_id"):
        A(f"- A2A workflow task `{ctx['a2a_task_id']}`: security-orchestrator → research-collector → "
          "threat-intel → report-writer (protocol a2a/1.0; full hop trace stored with the assessment).")
    A("- Note: diagrams above use Mermaid — rendered in the dashboard; raw source retained for PDF/print export.")
    A("")
    return "\n".join(L)


# -------------------------------------------------------------------- pdf ---

def build_pdf(markdown_text: str, title: str = "AI Security Assessment") -> bytes:
    """Render a readable PDF from the markdown. Requires reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib import colors
    except ImportError as exc:
        raise RuntimeError("PDF export needs the 'reportlab' package (pip install reportlab).") from exc

    import io

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title=title)
    styles = getSampleStyleSheet()
    story: list = []
    in_table = False
    table_rows: list[list[str]] = []

    def esc(s: str) -> str:
        return html.escape(s)

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if table_rows:
            header, *body = table_rows
            t = Table([header] + body, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#21262d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.4 * cm))
            table_rows = []
        in_table = False

    for raw in markdown_text.splitlines():
        line = raw.rstrip()
        if line.startswith("|"):
            cells = [esc(c.strip()) for c in line.strip().strip("|").split("|")]
            if set(c.strip(" :-") for c in cells) == {""}:
                continue
            table_rows.append([f"<para>{c}</para>" for c in cells])
            in_table = True
            continue
        elif in_table:
            flush_table()
        if line.startswith("```"):
            continue
        if line.startswith("# "):
            story.append(Paragraph(esc(line[2:]), styles["Title"]))
            story.append(Spacer(1, 0.3 * cm))
        elif line.startswith("## "):
            story.append(Paragraph(esc(line[3:]), styles["Heading1"]))
        elif line.startswith("### "):
            story.append(Paragraph(esc(line[4:]), styles["Heading2"]))
        elif line.startswith(("- ", "* ", "> ")):
            story.append(Paragraph("• " + esc(line[2:]), styles["BodyText"]))
        elif line.startswith("---"):
            story.append(Spacer(1, 0.3 * cm))
        elif not line.strip():
            story.append(Spacer(1, 0.2 * cm))
        else:
            # light inline formatting: **bold**
            seg = esc(line)
            seg = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", seg)
            story.append(Paragraph(seg, styles["BodyText"]))
    if in_table:
        flush_table()

    doc.build(story)
    return buf.getvalue()
