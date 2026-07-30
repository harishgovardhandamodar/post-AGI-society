from datetime import datetime, timezone
from sqlalchemy.orm import Session
from .models import Artifact, Sentiment, Projection, Relationship

SEED_ARTIFACTS = [
    {
        "title": "Superintelligence: Paths, Dangers, Strategies",
        "artifact_type": "book",
        "author": "Nick Bostrom",
        "date_published": datetime(2014, 9, 3, tzinfo=timezone.utc),
        "description": "Seminal work exploring how superintelligence could be created and what risks it poses. Introduces the orthogonality thesis and instrumental convergence.",
        "tags": "superintelligence,safety,alignment,risks",
        "sentiments": [{"score": -0.3, "label": "negative", "source": "initial"}],
        "projections": [{"projection_type": "cautionary", "confidence": 0.85, "timeframe": "2040-2060", "summary": "AGI poses existential risks if not aligned"}],
    },
    {
        "title": "The Singularity Is Near",
        "artifact_type": "book",
        "author": "Ray Kurzweil",
        "date_published": datetime(2005, 9, 26, tzinfo=timezone.utc),
        "description": "Predicts the technological singularity by 2045, where AI surpasses human intelligence and triggers runaway technological growth.",
        "tags": "singularity,accelerationist,transhumanism",
        "sentiments": [{"score": 0.8, "label": "very_positive", "source": "initial"}],
        "projections": [{"projection_type": "utopian", "confidence": 0.9, "timeframe": "2045", "summary": "Singularity by 2045 leading to radical life extension and abundance"}],
    },
    {
        "title": "GPT-4 Technical Report",
        "artifact_type": "paper",
        "author": "OpenAI",
        "date_published": datetime(2023, 3, 15, tzinfo=timezone.utc),
        "description": "Technical report detailing GPT-4, a large multimodal model exhibiting human-level performance on various professional benchmarks.",
        "url": "https://arxiv.org/abs/2303.08774",
        "tags": "llm,transformer,scaling,benchmarks",
        "sentiments": [{"score": 0.6, "label": "positive", "source": "initial"}],
    },
    {
        "title": "AGI Ruin: A List of Lethalities",
        "artifact_type": "essay",
        "author": "Eliezer Yudkowsky",
        "date_published": datetime(2022, 5, 1, tzinfo=timezone.utc),
        "description": "Argues that alignment of superhuman AGI is extremely difficult and humanity is not on track to solve it. The default outcome is catastrophic.",
        "tags": "alignment,xrisk,safety,doom",
        "sentiments": [{"score": -0.9, "label": "very_negative", "source": "initial"}],
        "projections": [{"projection_type": "dystopian", "confidence": 0.95, "timeframe": "2030-2040", "summary": "Default outcome of AGI development is human extinction"}],
    },
    {
        "title": "Situational Awareness: The Decade Ahead",
        "artifact_type": "essay",
        "author": "Leopold Aschenbrenner",
        "date_published": datetime(2024, 6, 1, tzinfo=timezone.utc),
        "description": "Argues transformative AGI is likely by 2027-2030, exploring geopolitical and economic implications of the sharp left turn in AI capabilities.",
        "tags": "takeoff,transformation,geopolitics,scaling",
        "sentiments": [{"score": 0.2, "label": "neutral", "source": "initial"}],
        "projections": [{"projection_type": "accelerationist", "confidence": 0.8, "timeframe": "2027-2030", "summary": "Transformative AGI within 3-5 years, leading to massive economic and geopolitical shifts"}],
    },
    {
        "title": "What Matters Most: AGI and Human Flourishing",
        "artifact_type": "essay",
        "author": "Sam Altman",
        "date_published": datetime(2025, 1, 15, tzinfo=timezone.utc),
        "description": "Altman's vision for AGI as the most powerful technology ever created, emphasizing broad distribution of benefits and careful stewardship.",
        "tags": "agi,optimism,governance,openai",
        "sentiments": [{"score": 0.7, "label": "positive", "source": "initial"}],
        "projections": [{"projection_type": "utopian", "confidence": 0.7, "timeframe": "2030-2035", "summary": "AGI will amplify human capabilities and create unprecedented prosperity"}],
    },
    {
        "title": "Debate: AGI Timelines — Short vs Long",
        "artifact_type": "interview",
        "author": "Dwarkesh Patel / Various",
        "date_published": datetime(2024, 8, 1, tzinfo=timezone.utc),
        "description": "Series of podcast debates between AI researchers on timelines for AGI: short-timeline believers (2027-2030) vs long-timeline skeptics (2050+).",
        "tags": "timelines,debate,forecasting",
        "sentiments": [{"score": 0.0, "label": "neutral", "source": "initial"}],
        "projections": [{"projection_type": "neutral", "confidence": 0.5, "timeframe": "2027-2050", "summary": "Wide disagreement on timelines reflecting deep uncertainty"}],
    },
    {
        "title": "DeepMind's AlphaFold and AI-Driven Scientific Discovery",
        "artifact_type": "news",
        "author": "DeepMind",
        "date_published": datetime(2024, 5, 8, tzinfo=timezone.utc),
        "description": "AlphaFold 3 predicts structures of proteins and other molecules, accelerating drug discovery. Represents AI's growing role in fundamental science.",
        "url": "https://deepmind.google/alphafold/",
        "tags": "science,drug-discovery,ai-capabilities",
        "sentiments": [{"score": 0.8, "label": "very_positive", "source": "initial"}],
    },
    {
        "title": "Anthropic's Research on AI Interpretability",
        "artifact_type": "research",
        "author": "Anthropic",
        "date_published": datetime(2024, 5, 21, tzinfo=timezone.utc),
        "description": "Golden Gate Claude: Research on understanding features inside neural networks using sparse autoencoders at scale.",
        "url": "https://transformer-circuits.pub/",
        "tags": "interpretability,mechanistic-interpretability,safety",
        "sentiments": [{"score": 0.5, "label": "positive", "source": "initial"}],
    },
    {
        "title": "Elon Musk: 'There's a 10-20% chance of AGI going bad'",
        "artifact_type": "tweet",
        "author": "Elon Musk",
        "date_published": datetime(2024, 4, 1, tzinfo=timezone.utc),
        "description": "Elon Musk estimates 10-20% chance of AGI going bad, advocating for cautious development and regulation.",
        "tags": "risk-assessment,timelines,xai",
        "sentiments": [{"score": -0.3, "label": "negative", "source": "initial"}],
        "projections": [{"projection_type": "cautionary", "confidence": 0.5, "timeframe": "2030-2040", "summary": "Non-trivial chance of catastrophic outcomes from AGI"}],
    },
    {
        "title": "The Alignment Problem",
        "artifact_type": "book",
        "author": "Brian Christian",
        "date_published": datetime(2020, 10, 6, tzinfo=timezone.utc),
        "description": "Accessible overview of the AI alignment problem, covering reinforcement learning, inverse reinforcement learning, and the practical challenges of specifying human values.",
        "tags": "alignment,safety,ethics,overview",
        "sentiments": [{"score": -0.1, "label": "neutral", "source": "initial"}],
    },
    {
        "title": "Open Letter: Pause Giant AI Experiments",
        "artifact_type": "news",
        "author": "Future of Life Institute",
        "date_published": datetime(2023, 3, 22, tzinfo=timezone.utc),
        "description": "Open letter signed by thousands calling for a 6-month pause on training AI systems more powerful than GPT-4 due to profound risks.",
        "url": "https://futureoflife.org/open-letter/pause-giant-ai-experiments/",
        "tags": "governance,safety,pause,regulation",
        "sentiments": [{"score": -0.4, "label": "negative", "source": "initial"}],
    },
    {
        "title": "Nuclear Fusion Breakthrough: AI-Optimized Reactors",
        "artifact_type": "news",
        "author": "Various",
        "date_published": datetime(2025, 2, 10, tzinfo=timezone.utc),
        "description": "AI-guided plasma control achieves sustained net-positive fusion energy. A key example of AI accelerating solutions to grand challenges.",
        "tags": "fusion,energy,ai-science,breakthrough",
        "sentiments": [{"score": 0.9, "label": "very_positive", "source": "initial"}],
        "projections": [{"projection_type": "utopian", "confidence": 0.6, "timeframe": "2030-2035", "summary": "AI-accelerated scientific breakthroughs could solve energy, disease, and climate challenges"}],
    },
]


def seed_database(db: Session):
    existing = db.query(Artifact).count()
    if existing > 0:
        return

    for data in SEED_ARTIFACTS:
        sentiments_data = data.pop("sentiments", [])
        projections_data = data.pop("projections", [])

        artifact = Artifact(**data)
        db.add(artifact)
        db.flush()

        for s_data in sentiments_data:
            sentiment = Sentiment(artifact_id=artifact.id, **s_data)
            db.add(sentiment)

        for p_data in projections_data:
            projection = Projection(artifact_id=artifact.id, **p_data)
            db.add(projection)

    db.commit()

    artifacts_map = {a.title: a.id for a in db.query(Artifact).all()}

    relationships = [
        ("Superintelligence: Paths, Dangers, Strategies", "AGI Ruin: A List of Lethalities", "supports", "Yudkowsky builds on Bostrom's risk analysis"),
        ("Superintelligence: Paths, Dangers, Strategies", "The Alignment Problem", "references", "Christian references Bostrom's work extensively"),
        ("The Singularity Is Near", "What Matters Most: AGI and Human Flourishing", "supports", "Altman continues Kurzweil's accelerationist vision"),
        ("AGI Ruin: A List of Lethalities", "The Alignment Problem", "supports", "Both focus on alignment difficulty"),
        ("Situational Awareness: The Decade Ahead", "Debate: AGI Timelines — Short vs Long", "supports", "Aschenbrenner represents the short-timeline camp"),
        ("GPT-4 Technical Report", "Situational Awareness: The Decade Ahead", "supports", "Scaling evidence used to argue for rapid timelines"),
        ("Open Letter: Pause Giant AI Experiments", "AGI Ruin: A List of Lethalities", "supports", "Pause call motivated by alignment concerns"),
        ("Nuclear Fusion Breakthrough: AI-Optimized Reactors", "What Matters Most: AGI and Human Flourishing", "supports", "Example of AI-driven prosperity Altman describes"),
        ("Anthropic's Research on AI Interpretability", "Superintelligence: Paths, Dangers, Strategies", "responds_to", "Interpretability research responds to alignment challenge"),
        ("Elon Musk: 'There's a 10-20% chance of AGI going bad'", "AGI Ruin: A List of Lethalities", "supports", "Both express caution about AGI risk"),
    ]

    for source_title, target_title, rel_type, desc in relationships:
        src_id = artifacts_map.get(source_title)
        tgt_id = artifacts_map.get(target_title)
        if src_id and tgt_id:
            db.add(Relationship(
                source_id=src_id,
                target_id=tgt_id,
                relationship_type=rel_type,
                description=desc,
            ))

    db.commit()
