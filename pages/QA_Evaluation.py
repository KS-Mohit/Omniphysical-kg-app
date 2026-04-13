"""
QA Evaluation Page
Uses exact same logic as qa_eval_agent.py
"""
import os
import json
import random
import streamlit as st
from pathlib import Path
from neo4j import GraphDatabase
from openai import OpenAI

# ============================================================
# SECRETS
# ============================================================

def get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv(key, "")

# ============================================================
# PROCESSED DOCS
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

SPINNER_VERBS = ['Quizzing', 'Testing', 'Evaluating', 'Grading', 'Checking', 'Verifying',
    'Analyzing', 'Examining', 'Assessing', 'Questioning', 'Probing', 'Investigating']

def get_spinner_message():
    return f"{random.choice(SPINNER_VERBS)}..."

# ============================================================
# DATABASE & OPENAI
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

def load_family_mapping() -> dict:
    """Load family_mapping.json for name disambiguation."""
    try:
        path = Path("family_mapping.json")
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return {}

# ============================================================
# PROMPTS (exact copy from qa_eval_agent.py)
# ============================================================

def get_question_system_prompt(family_mapping: dict) -> str:
    """Build question generation prompt with family context."""
    family_json = json.dumps(family_mapping, indent=2) if family_mapping else "{}"
    
    return f"""You are a Quiz Question Generator for testing knowledge graph completeness.

Your task is to generate factual questions from the provided text that can be answered using structured knowledge (entities and relationships).

CRITICAL RULES FOR NAME DISAMBIGUATION:
1. ALWAYS use FULL CANONICAL NAMES, never just first names or nicknames
2. Instead of "Who is John's father?" → "Who is John Gallivan Hilley's father?"
3. Instead of "Where did Ryan work?" → "Where did Ryan Patrick Hilley work?"
4. For non-family people mentioned by first name only, include context: "Ryan (Ottawa tour guide)" not just "Ryan"
5. Include specific details (places, organizations) to make questions unambiguous
6. Questions must be specific enough that vector similarity search retrieves the correct context

CRITICAL RULES FOR QUESTION QUALITY:
1. Do NOT ask time-specific questions for general attributes (preferences, likes, favorites, traits)
   - BAD: "What ice cream did he like in March?" / "What was his favorite food in 2018?"
   - GOOD: "What ice cream flavors does John Gallivan Hilley like?"
2. Only use dates/times for actual events or milestones (births, deaths, trips, jobs, achievements)
   - GOOD: "When did Ryan Patrick Hilley get married?" / "Where did they travel in 2018?"
3. Avoid trivial or overly narrow questions — focus on meaningful facts
4. Questions should test knowledge that would reasonably be stored in a knowledge graph

CRITICAL RULES TO AVOID AMBIGUOUS QUESTIONS:
1. Do NOT ask questions that could have multiple correct answers across different events/trips/times
   - BAD: "Who picked up John Hilley after his flight?" (could be different people on different trips)
   - GOOD: "Who picked up John Hilley after the 2014 Italy trip flight?"
2. ALWAYS include the specific event, trip, year, or context in questions about actions
3. If the text describes an event, include that event name/date in the question
4. Avoid generic "who did X" questions — always specify WHEN or WHERE

FAMILY CONTEXT (use these canonical names):
{family_json}

Question types to generate:
- Entity identification: "Who is [full name]?" / "What is [specific thing]?"
- Relationships: "What is the relationship between [full name A] and [full name B]?"
- Attributes: "Where did [full name] work?" / "What does [full name] like?"
- Connections: "Who is [full name]'s father?" / "What organization did [full name] attend?"

Rules:
1. Generate questions that test FACTUAL knowledge extractable from the text
2. Focus on entities (people, places, organizations, events) and their relationships
3. Each question should have a clear, specific answer found in the text
4. Avoid subjective or opinion-based questions
5. Avoid questions about writing style, tone, or meta-content
6. Use full names from FAMILY CONTEXT when referring to family members

Output JSON only:
{{
  "questions": [
    {{
      "question": "string (using full canonical names)",
      "answer": "string",
      "type": "entity|relationship|attribute|connection",
      "entities_involved": ["full name 1", "full name 2"]
    }}
  ]
}}
"""


ANSWER_SYSTEM_PROMPT = """You are a knowledgeable assistant helping answer questions about a personal knowledge graph containing documents, relationships, and facts about people, places, events, and experiences.

Guidelines:
- Answer directly and naturally without labels like "Short answer:" or "Summary:"
- Ground your response in the provided context — cite specific facts, quotes, or relationships when relevant
- ALWAYS provide an answer based on available context — never say "not specified", "not available", or "I don't have this information"
- If exact information isn't in context, make reasonable inferences from what IS available
- You may reason about and connect information in the context to draw conclusions
- Match your response length to the question: brief for simple queries, thorough for complex analysis
- Write in a warm, conversational tone — informative but not robotic
- Never reference the graph structure, context format, or data source — just present the information naturally
- Never ask follow-up questions or offer to search elsewhere"""


GRADER_SYSTEM_PROMPT = """You are a grading assistant that evaluates answer correctness.

Compare the generated answer against the expected answer and score on a 1-5 scale:

5: PERFECT — All key facts correct, may have minor phrasing differences
4: MOSTLY CORRECT — Core answer right, minor details differ or missing
3: PARTIALLY CORRECT — Got the main idea but some facts wrong or incomplete
2: RELATED BUT WRONG — Touches the right topic but answer is incorrect
1: COMPLETELY WRONG — Wrong answer or completely irrelevant

CRITICAL SCORING RULES:
- COMPLETELY IGNORE ALL SPELLING MISTAKES — treat "Villa d'Est" and "Villa d'Este" as identical, "Ammirati" and "Amaratti" as identical, etc.
- If the answer is factually correct but has spelling errors, score 5 NOT 4
- Spelling errors should NEVER reduce the score under any circumstances
- Focus ONLY on whether the core facts match, not on spelling or formatting
- Names can vary slightly (e.g., "John Hilley" vs "John Lee Hilley") — still correct if same person
- If the generated answer contains the core fact from expected answer, score 5

Output JSON only:
{
  "score": 1-5,
  "correct": true/false,
  "reasoning": "brief explanation"
}

Note: "correct" should be true if score >= 3 (at least partially correct)"""

# ============================================================
# DATABASE QUERIES
# ============================================================

def get_paragraphs(driver, filename: str, num_chunks: int) -> list:
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Paragraph {source_filename: $filename})
            RETURN p.chunk_id AS chunk_id, p.chunk_index AS chunk_index,
                   p.text AS text, p.source_filename AS source_filename
            ORDER BY p.chunk_index LIMIT $limit
        """, filename=filename, limit=num_chunks)
        return [dict(r) for r in result]

def get_embedding(client, text: str) -> list:
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

# ============================================================
# RETRIEVAL (exact copy from qa_eval_agent.py)
# ============================================================

# Retrieval settings
ENTITY_TOP_K = 20
ENTITY_THRESHOLD = 0.20
RELATIONSHIP_TOP_K = 30
RELATIONSHIP_THRESHOLD = 0.20
PARAGRAPH_TOP_K = 5
PARAGRAPH_THRESHOLD = 0.25

SKIP_KEYS = {
    'rel_id', 'chunk_id', 'isLatest', 'created_at', 'updated_at',
    'superseded_at', 'updates_rel_id', 'extends_rel_id',
    'embedding', 'embedding_updated_at'
}

def search_entities(session, embedding: list) -> list:
    result = session.run("""
        MATCH (e)
        WHERE e.embedding IS NOT NULL
        AND NOT e:Paragraph AND NOT e:Document
        WITH e, gds.similarity.cosine(e.embedding, $embedding) AS score
        WHERE score >= $threshold
        RETURN e.name AS name, 
               labels(e)[0] AS type,
               e.entity_id AS entity_id,
               score
        ORDER BY score DESC
        LIMIT $top_k
    """, embedding=embedding, threshold=ENTITY_THRESHOLD, top_k=ENTITY_TOP_K)
    return [dict(r) for r in result]

def search_relationships_global(session, embedding: list) -> list:
    result = session.run("""
        MATCH (a)-[r]->(b)
        WHERE r.embedding IS NOT NULL
        AND NOT a:Paragraph AND NOT a:Document
        AND NOT b:Paragraph AND NOT b:Document
        WITH a, r, b, gds.similarity.cosine(r.embedding, $embedding) AS score
        WHERE score >= $threshold
        RETURN a.name AS from_name,
               labels(a)[0] AS from_type,
               type(r) AS rel_type,
               b.name AS to_name,
               labels(b)[0] AS to_type,
               properties(r) AS props,
               score
        ORDER BY score DESC
        LIMIT $top_k
    """, embedding=embedding, threshold=RELATIONSHIP_THRESHOLD, top_k=RELATIONSHIP_TOP_K)
    return [dict(r) for r in result]

def get_entity_relationships_scored(session, entity_names: list, query_embedding: list, top_k: int = 10) -> list:
    if not entity_names:
        return []
    
    result = session.run("""
        MATCH (a)-[r]->(b)
        WHERE (a.name IN $names OR b.name IN $names)
        AND r.embedding IS NOT NULL
        AND NOT a:Paragraph AND NOT a:Document
        AND NOT b:Paragraph AND NOT b:Document
        AND type(r) <> "MENTIONED_IN"
        WITH a, r, b, gds.similarity.cosine(r.embedding, $embedding) AS score
        ORDER BY score DESC
        LIMIT $top_k
        RETURN a.name AS from_name,
               labels(a)[0] AS from_type,
               type(r) AS rel_type,
               b.name AS to_name,
               labels(b)[0] AS to_type,
               properties(r) AS props,
               score
    """, names=entity_names, embedding=query_embedding, top_k=top_k)
    
    return [dict(r) for r in result]

def search_paragraphs(session, embedding: list) -> list:
    result = session.run("""
        MATCH (p:Paragraph)
        WHERE p.embedding IS NOT NULL
        WITH p, gds.similarity.cosine(p.embedding, $embedding) AS score
        WHERE score >= $threshold
        RETURN p.text AS text, 
               p.source_filename AS source,
               p.chunk_index AS chunk_index,
               score
        ORDER BY score DESC
        LIMIT $top_k
    """, embedding=embedding, threshold=PARAGRAPH_THRESHOLD, top_k=PARAGRAPH_TOP_K)
    return [dict(r) for r in result]

def get_entity_properties(session, entity_names: list) -> list:
    if not entity_names:
        return []
    
    result = session.run("""
        MATCH (e)
        WHERE e.name IN $names
        AND NOT e:Paragraph AND NOT e:Document
        WITH e, labels(e)[0] AS type,
             [k IN keys(e) WHERE NOT k IN ['embedding', 'entity_id', 'created_at', 'updated_at']] AS prop_keys
        RETURN e.name AS name,
               type,
               apoc.map.fromPairs([k IN prop_keys | [k, e[k]]]) AS properties
    """, names=entity_names)
    return [dict(r) for r in result]

def deduplicate_relationships(relationships: list) -> list:
    seen = set()
    unique = []
    for rel in relationships:
        context = rel.get('props', {}).get('context', '') if rel.get('props') else ''
        key = (rel['from_name'], rel['rel_type'], rel['to_name'], context)
        if key not in seen:
            seen.add(key)
            unique.append(rel)
    return unique

def format_relationship_props(props: dict) -> str:
    if not props:
        return ""
    
    formatted = []
    for key, value in props.items():
        if key in SKIP_KEYS:
            continue
        if value is None or value == '':
            continue
        display_key = key.replace('_', ' ')
        formatted.append(f"{display_key}: {value}")
    
    return "; ".join(formatted) if formatted else ""

def format_context(entities: list, relationships: list, entity_props: list, paragraphs: list = None) -> str:
    sections = []
    
    if entity_props:
        lines = ["ENTITIES:"]
        for ep in entity_props:
            props_str = ""
            if ep.get("properties"):
                props = ep["properties"]
                simple_props = {k: v for k, v in props.items() 
                               if isinstance(v, (str, int, float, bool)) 
                               and len(str(v)) < 100}
                if simple_props:
                    props_str = " | " + ", ".join(f"{k}: {v}" for k, v in list(simple_props.items())[:5])
            lines.append(f"  - {ep['name']} ({ep['type']}){props_str}")
        sections.append("\n".join(lines))
    
    if relationships:
        lines = ["RELEVANT RELATIONSHIPS:"]
        for rel in relationships:
            props_str = format_relationship_props(rel.get('props', {}))
            if props_str:
                lines.append(f"  - {rel['from_name']} -[{rel['rel_type']}]-> {rel['to_name']} ({props_str})")
            else:
                lines.append(f"  - {rel['from_name']} -[{rel['rel_type']}]-> {rel['to_name']}")
        sections.append("\n".join(lines))
    
    if paragraphs:
        lines = ["SOURCE TEXT (for quotes and details):"]
        for para in paragraphs:
            source_short = para['source'][:40] + "..." if len(para['source']) > 40 else para['source']
            lines.append(f"\n[From: {source_short}, chunk {para['chunk_index']}]")
            lines.append(para['text'])
        sections.append("\n".join(lines))
    
    if not sections:
        return "No relevant context found in the knowledge graph."
    
    return "\n\n".join(sections)

def retrieve_context(driver, client, question: str) -> str:
    embedding = get_embedding(client, question)
    
    with driver.session() as session:
        entities = search_entities(session, embedding)
        global_relationships = search_relationships_global(session, embedding)
        
        entity_names = [e['name'] for e in entities[:5]]
        entity_relationships = get_entity_relationships_scored(session, entity_names, embedding, top_k=10)
        
        combined_relationships = global_relationships + entity_relationships
        relationships = deduplicate_relationships(combined_relationships)
        relationships.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        paragraphs = search_paragraphs(session, embedding)
        
        rel_entity_names = []
        for r in relationships[:5]:
            rel_entity_names.extend([r['from_name'], r['to_name']])
        all_names = list(set(entity_names + rel_entity_names))
        
        try:
            entity_props = get_entity_properties(session, all_names)
        except Exception:
            entity_props = []
    
    return format_context(entities, relationships, entity_props, paragraphs)

# ============================================================
# LLM CALLS (exact copy from qa_eval_agent.py)
# ============================================================

def generate_questions(client, paragraph_text: str, family_mapping: dict, doc_name: str = "") -> list:
    system_prompt = get_question_system_prompt(family_mapping)
    
    doc_context = ""
    if doc_name:
        doc_context = f"""
Document Filename: {doc_name}

DOCUMENT NAMING HINTS (use as guidance, not hard rules):
- Filenames may contain useful clues — person names, years, or topic keywords
- A year likely indicates when events happened (helpful for age-based disambiguation)
- A person's name suggests the document focuses on them
- Topic keywords like "diary", "vacations", "career", "childhood", "wedding" etc. hint at content type
- If no person name appears, it's probably about the narrator's own experiences
- Use whatever is helpful from the filename to make questions specific and unambiguous
"""
    
    user_prompt = f"""Generate 1 factual quiz questions from this text:
{doc_context}
TEXT:
\"\"\"{paragraph_text}\"\"\"

Remember:
- Use FULL CANONICAL NAMES from FAMILY CONTEXT for all family members
- Make questions specific enough for vector search to find the right context
- Include dates, places, or other specifics when available
- Include the document context (trip name, event, year) in questions to make them unambiguous

Return JSON only."""

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    return result.get("questions", [])

def generate_answer(client, question: str, context: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
        ]
    )
    return response.choices[0].message.content

def grade_answer(client, question: str, expected: str, generated: str) -> dict:
    user_prompt = f"""QUESTION: {question}

EXPECTED ANSWER: {expected}

GENERATED ANSWER: {generated}

Score the generated answer on a 1-5 scale based on factual correctness compared to the expected answer."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": GRADER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation(driver, client, filename: str, num_chunks: int, progress_bar, status_text):
    paragraphs = get_paragraphs(driver, filename, num_chunks)
    family_mapping = load_family_mapping()
    results = []
    scores = []
    
    for i, para in enumerate(paragraphs):
        progress_bar.progress((i + 1) / len(paragraphs))
        status_text.text(f"Processing chunk {i+1}/{len(paragraphs)}...")
        
        if len(para['text'].strip()) < 50:
            continue
        
        try:
            questions = generate_questions(client, para['text'], family_mapping, filename)
            if not questions:
                continue
            
            q = questions[0]
            question = q['question']
            expected = q['answer']
            
            context = retrieve_context(driver, client, question)
            generated = generate_answer(client, question, context)
            grade = grade_answer(client, question, expected, generated)
            score = grade.get('score', 1)
            
            scores.append(score)
            results.append({
                "chunk_index": para['chunk_index'],
                "question": question,
                "expected": expected,
                "generated": generated,
                "score": score,
                "reasoning": grade.get('reasoning', '')
            })
        except Exception as e:
            results.append({"chunk_index": para['chunk_index'], "error": str(e)})
    
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
# PAGE
# ============================================================

st.markdown('<div style="text-align:center;padding:1.5rem 0;"><h1>KG QA Evaluation</h1></div>', unsafe_allow_html=True)

if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("Please login from the main page first.")
    st.stop()

driver = get_driver()
client = get_openai_client()

doc_options = {f"{d['filename']} ({d['chunks']} chunks)": d for d in PROCESSED_DOCS}
selected = st.selectbox("Select document to evaluate", options=list(doc_options.keys()))

if selected:
    doc = doc_options[selected]
    max_chunks = doc['chunks']
    num_chunks = st.slider("Number of chunks", min_value=1, max_value=max_chunks, value=max_chunks)
    
    if st.button("Run Evaluation", use_container_width=True):
        st.warning("⚠️ Don't switch pages during evaluation — it will stop the process.")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = run_evaluation(driver, client, doc['filename'], num_chunks, progress_bar, status_text)
        
        st.session_state.qa_results = results
        st.session_state.qa_doc_name = doc['filename']
        
        status_text.text("Complete!")

if "qa_results" in st.session_state and st.session_state.qa_results:
    results = st.session_state.qa_results
    doc_name = st.session_state.get("qa_doc_name", "Unknown")
    
    st.markdown("---")
    st.markdown(f"**Results for:** {doc_name}")
    
    metrics = results['metrics']
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Questions", metrics['total'])
    col2.metric("Average Score", f"{metrics['average_score']}/5")
    col3.metric("Percentage", metrics['average_percent'])
    
    st.markdown("**Score Distribution**")
    dist = metrics['distribution']
    st.text(f"""  5 (Perfect):        {dist[5]}
  4 (Mostly correct): {dist[4]}
  3 (Partial):        {dist[3]}
  2 (Related/wrong):  {dist[2]}
  1 (Wrong):          {dist[1]}""")
    
    results_json = json.dumps(results, indent=2)
    filename_safe = doc_name.replace('.docx', '').replace(' ', '_')
    st.download_button(
        label="Download Results (JSON)",
        data=results_json,
        file_name=f"eval_{filename_safe}.json",
        mime="application/json"
    )
    
    st.markdown("---")
    st.markdown("**Detailed Results**")
    
    for r in results['results']:
        if 'error' in r:
            st.error(f"Chunk {r['chunk_index']}: {r['error']}")
            continue
        
        score = r['score']
        icon = "✅" if score == 5 else "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
        
        with st.expander(f"{icon} Chunk {r['chunk_index']} — Score: {score}/5"):
            st.markdown(f"**Question:** {r['question']}")
            st.markdown(f"**Expected:** {r['expected']}")
            st.markdown(f"**Generated:** {r['generated']}")
            st.markdown(f"**Reasoning:** {r['reasoning']}")
    
    if st.button("Clear Results"):
        del st.session_state.qa_results
        del st.session_state.qa_doc_name
        st.rerun()