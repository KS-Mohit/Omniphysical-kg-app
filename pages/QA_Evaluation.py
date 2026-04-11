"""
QA Evaluation UI
Simple Streamlit interface for running QA evaluations on processed documents.
"""
import os
import sys
import json
import time
import random
import streamlit as st
from pathlib import Path
from datetime import datetime
from neo4j import GraphDatabase
from openai import OpenAI

# ============================================================
# SPINNER VERBS
# ============================================================

SPINNER_VERBS = [
    'Quizzing', 'Testing', 'Evaluating', 'Grading', 'Checking', 'Verifying',
    'Analyzing', 'Examining', 'Assessing', 'Questioning', 'Probing', 'Investigating',
    'Scrutinizing', 'Reviewing', 'Inspecting', 'Pondering', 'Deliberating'
]

def get_spinner_message() -> str:
    return f"{random.choice(SPINNER_VERBS)}..."

# ============================================================
# CONFIGURATION
# ============================================================

def get_secret(key: str) -> str:
    """Try Streamlit secrets first (cloud), fall back to env vars (local)."""
    try:
        return st.secrets[key]
    except:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(key, "")

LLM_MODEL = "gpt-5-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Retrieval settings
ENTITY_TOP_K = 20
ENTITY_THRESHOLD = 0.20
RELATIONSHIP_TOP_K = 30
RELATIONSHIP_THRESHOLD = 0.20
PARAGRAPH_TOP_K = 5
PARAGRAPH_THRESHOLD = 0.25

# Properties to exclude from display
SKIP_KEYS = {
    'rel_id', 'chunk_id', 'isLatest', 'created_at', 'updated_at',
    'superseded_at', 'updates_rel_id', 'extends_rel_id',
    'embedding', 'embedding_updated_at'
}

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="KG QA Evaluation",
    page_icon="Q",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
    }
    
    .main-header h1 {
        font-size: 2rem;
        font-weight: 600;
    }
    
    .score-perfect { color: #22c55e; font-weight: bold; }
    .score-good { color: #84cc16; }
    .score-partial { color: #eab308; }
    .score-wrong { color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# AUTHENTICATION
# ============================================================

def check_password() -> bool:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if st.session_state.authenticated:
        return True
    
    st.markdown('<div class="main-header"><h1>KG QA Evaluation</h1><p>Enter password to continue</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        password = st.text_input("Password", type="password", label_visibility="collapsed")
        
        if st.button("Login", use_container_width=True):
            if password == get_secret("APP_PASSWORD"):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
    
    return False

# ============================================================
# DATABASE CONNECTION
# ============================================================

@st.cache_resource
def get_driver():
    return GraphDatabase.driver(
        get_secret("NEO4J_URI"),
        auth=(get_secret("NEO4J_USERNAME"), get_secret("NEO4J_PASSWORD"))
    )

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=get_secret("OPENAI_API_KEY"))

# ============================================================
# FAMILY MAPPING
# ============================================================

@st.cache_data
def load_family_mapping() -> dict:
    """Load family_mapping.json if available."""
    try:
        # Try common locations
        paths = [
            Path("family_mapping.json"),
            Path("../family_mapping.json"),
            Path(__file__).parent.parent / "family_mapping.json"
        ]
        for path in paths:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
    except:
        pass
    return {}

# ============================================================
# PROCESSED DOCUMENTS
# ============================================================

PROCESSED_DOCS = [
    {"filename": "2014 Italy Wedding of Jon and Amy .docx", "chunks": 7},
    {"filename": "2023 Honeymoon Trip to Ottawa.docx", "chunks": 9},
    {"filename": "Hilley 2018.docx", "chunks": 10},
    {"filename": "Hilley 2024.docx", "chunks": 7},
    {"filename": "Lehigh People.docx", "chunks": 9},
    {"filename": "Mitchell and Daschle People.docx", "chunks": 15},
    {"filename": "People - Windy Hollow Years.docx", "chunks": 15},
    {"filename": "Solebury Friends.docx", "chunks": 9},
    {"filename": "The Hard Problem of Consciousness1.docx", "chunks": 7},
    {"filename": "White House People.docx", "chunks": 15},
]

def get_paragraphs(driver, filename: str, num_chunks: int) -> list:
    """Get paragraphs for a document."""
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Paragraph {source_filename: $filename})
            RETURN p.chunk_id AS chunk_id,
                   p.chunk_index AS chunk_index,
                   p.text AS text,
                   p.source_filename AS source_filename
            ORDER BY p.chunk_index
            LIMIT $limit
        """, filename=filename, limit=num_chunks)
        return [dict(r) for r in result]

# ============================================================
# EMBEDDING & RETRIEVAL
# ============================================================

def get_embedding(client, text: str) -> list:
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding

def format_relationship_props(props: dict) -> str:
    if not props:
        return ""
    formatted = []
    for key, value in props.items():
        if key in SKIP_KEYS or value is None or value == '':
            continue
        formatted.append(f"{key.replace('_', ' ')}: {value}")
    return "; ".join(formatted) if formatted else ""

def retrieve_context(driver, client, question: str) -> str:
    """Retrieve context from KG.DL for a question."""
    embedding = get_embedding(client, question)
    
    with driver.session() as session:
        # Search entities
        entities = session.run("""
            MATCH (e)
            WHERE e.embedding IS NOT NULL AND NOT e:Paragraph AND NOT e:Document
            WITH e, gds.similarity.cosine(e.embedding, $embedding) AS score
            WHERE score >= $threshold
            RETURN e.name AS name, labels(e)[0] AS type, score
            ORDER BY score DESC LIMIT $top_k
        """, embedding=embedding, threshold=ENTITY_THRESHOLD, top_k=ENTITY_TOP_K)
        entities = [dict(r) for r in entities]
        
        # Search relationships
        relationships = session.run("""
            MATCH (a)-[r]->(b)
            WHERE r.embedding IS NOT NULL
            AND NOT a:Paragraph AND NOT a:Document AND NOT b:Paragraph AND NOT b:Document
            WITH a, r, b, gds.similarity.cosine(r.embedding, $embedding) AS score
            WHERE score >= $threshold
            RETURN a.name AS from_name, type(r) AS rel_type, b.name AS to_name,
                   properties(r) AS props, score
            ORDER BY score DESC LIMIT $top_k
        """, embedding=embedding, threshold=RELATIONSHIP_THRESHOLD, top_k=RELATIONSHIP_TOP_K)
        relationships = [dict(r) for r in relationships]
        
        # Search paragraphs
        paragraphs = session.run("""
            MATCH (p:Paragraph)
            WHERE p.embedding IS NOT NULL
            WITH p, gds.similarity.cosine(p.embedding, $embedding) AS score
            WHERE score >= $threshold
            RETURN p.text AS text, p.source_filename AS source, p.chunk_index AS chunk_index, score
            ORDER BY score DESC LIMIT $top_k
        """, embedding=embedding, threshold=PARAGRAPH_THRESHOLD, top_k=PARAGRAPH_TOP_K)
        paragraphs = [dict(r) for r in paragraphs]
    
    # Format context
    sections = []
    
    if entities:
        lines = ["ENTITIES:"]
        for e in entities[:10]:
            lines.append(f"  - {e['name']} ({e['type']})")
        sections.append("\n".join(lines))
    
    if relationships:
        lines = ["RELEVANT RELATIONSHIPS:"]
        for rel in relationships[:20]:
            props_str = format_relationship_props(rel.get('props', {}))
            if props_str:
                lines.append(f"  - {rel['from_name']} -[{rel['rel_type']}]-> {rel['to_name']} ({props_str})")
            else:
                lines.append(f"  - {rel['from_name']} -[{rel['rel_type']}]-> {rel['to_name']}")
        sections.append("\n".join(lines))
    
    if paragraphs:
        lines = ["SOURCE TEXT:"]
        for para in paragraphs:
            lines.append(f"\n[From: {para['source'][:40]}..., chunk {para['chunk_index']}]")
            lines.append(para['text'])
        sections.append("\n".join(lines))
    
    return "\n\n".join(sections) if sections else "No relevant context found."

# ============================================================
# LLM CALLS
# ============================================================

def generate_questions(client, paragraph_text: str, family_mapping: dict) -> list:
    """Generate questions from a paragraph."""
    family_json = json.dumps(family_mapping, indent=2) if family_mapping else "{}"
    
    system_prompt = f"""You are a Quiz Question Generator for testing knowledge graph completeness.

Generate factual questions from the provided text that can be answered using structured knowledge.

RULES:
1. Use FULL CANONICAL NAMES from FAMILY CONTEXT, never just first names
2. Do NOT ask time-specific questions for preferences/likes (BAD: "What ice cream in March?")
3. Only use dates for actual events/milestones
4. Focus on meaningful facts, not trivial details

FAMILY CONTEXT:
{family_json}

Output JSON only:
{{"questions": [{{"question": "string", "answer": "string"}}]}}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate 1 factual question from:\n\n{paragraph_text}"}
        ],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    return result.get("questions", [])


def generate_answer(client, question: str, context: str) -> str:
    """Generate answer from context."""
    system_prompt = """You are a knowledgeable assistant answering questions from a personal knowledge graph.

Guidelines:
- Answer directly and naturally
- ALWAYS provide an answer — never say "not specified" or "not available"
- If exact info isn't in context, make reasonable inferences
- Never ask follow-up questions"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
        ]
    )
    return response.choices[0].message.content


def grade_answer(client, question: str, expected: str, generated: str) -> dict:
    """Grade the generated answer."""
    system_prompt = """Grade the answer on 1-5 scale:
5: PERFECT — All key facts correct
4: MOSTLY CORRECT — Core answer right, minor details differ
3: PARTIALLY CORRECT — Main idea right but some facts wrong
2: RELATED BUT WRONG — Right topic but wrong answer
1: COMPLETELY WRONG

DO NOT penalize misspellings — they come from source documents.

Output JSON: {"score": 1-5, "reasoning": "brief explanation"}"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"QUESTION: {question}\nEXPECTED: {expected}\nGENERATED: {generated}"}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation(driver, client, filename: str, num_chunks: int, progress_callback=None):
    """Run QA evaluation on a document."""
    family_mapping = load_family_mapping()
    paragraphs = get_paragraphs(driver, filename, num_chunks)
    
    results = []
    scores = []
    
    for i, para in enumerate(paragraphs):
        if progress_callback:
            progress_callback(i, len(paragraphs), f"Processing chunk {i+1}/{len(paragraphs)}")
        
        # Skip short paragraphs
        if len(para['text'].strip()) < 50:
            continue
        
        try:
            # Generate question
            questions = generate_questions(client, para['text'], family_mapping)
            if not questions:
                continue
            
            q = questions[0]
            question = q['question']
            expected = q['answer']
            
            # Retrieve context and generate answer
            context = retrieve_context(driver, client, question)
            generated = generate_answer(client, question, context)
            
            # Grade
            grade_result = grade_answer(client, question, expected, generated)
            score = grade_result.get('score', 1)
            reasoning = grade_result.get('reasoning', '')
            
            scores.append(score)
            results.append({
                "chunk_index": para['chunk_index'],
                "question": question,
                "expected": expected,
                "generated": generated,
                "score": score,
                "reasoning": reasoning
            })
            
        except Exception as e:
            results.append({
                "chunk_index": para['chunk_index'],
                "error": str(e)
            })
    
    # Calculate metrics
    avg_score = sum(scores) / len(scores) if scores else 0
    
    return {
        "results": results,
        "metrics": {
            "total": len(scores),
            "average_score": round(avg_score, 2),
            "average_percent": f"{(avg_score / 5 * 100):.1f}%",
            "distribution": {
                5: sum(1 for s in scores if s == 5),
                4: sum(1 for s in scores if s == 4),
                3: sum(1 for s in scores if s == 3),
                2: sum(1 for s in scores if s == 2),
                1: sum(1 for s in scores if s == 1)
            }
        }
    }

# ============================================================
# MAIN APP
# ============================================================

def main():
    if not check_password():
        return
    
    st.markdown('<div class="main-header"><h1>KG QA Evaluation</h1></div>', unsafe_allow_html=True)
    
    driver = get_driver()
    client = get_openai_client()
    
    # Use hardcoded document list
    docs = PROCESSED_DOCS
    
    # Document selection
    doc_options = {f"{d['filename']} ({d['chunks']} chunks)": d for d in docs}
    selected = st.selectbox("Select document to evaluate", options=list(doc_options.keys()))
    
    if selected:
        doc = doc_options[selected]
        
        # Number of chunks
        max_chunks = doc['chunks']
        num_chunks = st.slider("Number of chunks to evaluate", min_value=1, max_value=min(max_chunks, 50), value=min(10, max_chunks))
        
        # Run evaluation
        if st.button("Run Evaluation", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, message):
                progress_bar.progress(current / total)
                status_text.text(message)
            
            with st.spinner(get_spinner_message()):
                results = run_evaluation(driver, client, doc['filename'], num_chunks, update_progress)
            
            progress_bar.progress(1.0)
            status_text.text("Complete!")
            
            # Display results
            st.markdown("---")
            
            # Metrics
            metrics = results['metrics']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Questions", metrics['total'])
            with col2:
                st.metric("Average Score", f"{metrics['average_score']}/5")
            with col3:
                st.metric("Percentage", metrics['average_percent'])
            
            # Distribution
            st.markdown("**Score Distribution**")
            dist = metrics['distribution']
            st.text(f"""  5 (Perfect):        {dist[5]}
  4 (Mostly correct): {dist[4]}
  3 (Partial):        {dist[3]}
  2 (Related/wrong):  {dist[2]}
  1 (Wrong):          {dist[1]}""")
            
            # Detailed results
            st.markdown("---")
            st.markdown("**Detailed Results**")
            
            for r in results['results']:
                if 'error' in r:
                    st.error(f"Chunk {r['chunk_index']}: {r['error']}")
                    continue
                
                score = r['score']
                if score == 5:
                    icon = "✅"
                    color = "score-perfect"
                elif score >= 4:
                    icon = "🟢"
                    color = "score-good"
                elif score >= 3:
                    icon = "🟡"
                    color = "score-partial"
                else:
                    icon = "🔴"
                    color = "score-wrong"
                
                with st.expander(f"{icon} Chunk {r['chunk_index']} — Score: {score}/5"):
                    st.markdown(f"**Question:** {r['question']}")
                    st.markdown(f"**Expected:** {r['expected']}")
                    st.markdown(f"**Generated:** {r['generated']}")
                    st.markdown(f"**Reasoning:** {r['reasoning']}")

if __name__ == "__main__":
    main()