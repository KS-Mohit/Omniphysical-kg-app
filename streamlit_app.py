"""
Omniphysical Knowledge Graph - Query Interface
Simple chatbot UI for querying the knowledge graph.
"""
import os
import sys
import time
import random
import streamlit as st
from neo4j import GraphDatabase
from openai import OpenAI

# ============================================================
# SPINNER VERBS
# ============================================================

SPINNER_VERBS = [
    'Accomplishing', 'Actualizing', 'Architecting', 'Baking', 'Beaming', 'Beboppin',
    'Befuddling', 'Billowing', 'Blanching', 'Bloviating', 'Boogieing', 'Boondoggling',
    'Booping', 'Bootstrapping', 'Brewing', 'Bunning', 'Burrowing', 'Calculating',
    'Canoodling', 'Caramelizing', 'Cascading', 'Catapulting', 'Cerebrating', 'Channeling',
    'Channelling', 'Choreographing', 'Churning', 'Clauding', 'Coalescing', 'Cogitating',
    'Combobulating', 'Composing', 'Computing', 'Concocting', 'Considering', 'Contemplating',
    'Cooking', 'Crafting', 'Creating', 'Crunching', 'Crystallizing', 'Cultivating',
    'Deciphering', 'Deliberating', 'Determining', 'Dilly-dallying', 'Discombobulating',
    'Doing', 'Doodling', 'Drizzling', 'Ebbing', 'Effecting', 'Elucidating', 'Embellishing',
    'Enchanting', 'Envisioning', 'Evaporating', 'Fermenting', 'Fiddle-faddling',
    'Finagling', 'Flambéing', 'Flibbertigibbeting', 'Flowing', 'Flummoxing', 'Fluttering',
    'Forging', 'Forming', 'Frolicking', 'Frosting', 'Gallivanting', 'Galloping',
    'Garnishing', 'Generating', 'Gesticulating', 'Germinating', 'Gitifying', 'Grooving',
    'Gusting', 'Harmonizing', 'Hashing', 'Hatching', 'Herding', 'Honking', 'Hullaballooing',
    'Hyperspacing', 'Ideating', 'Imagining', 'Improvising', 'Incubating', 'Inferring',
    'Infusing', 'Ionizing', 'Jitterbugging', 'Julienning', 'Kneading', 'Leavening',
    'Levitating', 'Lollygagging', 'Manifesting', 'Marinating', 'Meandering',
    'Metamorphosing', 'Misting', 'Moonwalking', 'Moseying', 'Mulling', 'Mustering',
    'Musing', 'Nebulizing', 'Nesting', 'Newspapering', 'Noodling', 'Nucleating',
    'Orbiting', 'Orchestrating', 'Osmosing', 'Perambulating', 'Percolating', 'Perusing',
    'Philosophising', 'Photosynthesizing', 'Pollinating', 'Pondering', 'Pontificating',
    'Pouncing', 'Precipitating', 'Prestidigitating', 'Processing', 'Proofing',
    'Propagating', 'Puttering', 'Puzzling', 'Quantumizing', 'Razzle-dazzling',
    'Razzmatazzing', 'Recombobulating', 'Reticulating', 'Roosting', 'Ruminating',
    'Sautéing', 'Scampering', 'Schlepping', 'Scurrying', 'Seasoning', 'Shenaniganing',
    'Shimmying', 'Simmering', 'Skedaddling', 'Sketching', 'Slithering', 'Smooshing',
    'Sock-hopping', 'Spelunking', 'Spinning', 'Sprouting', 'Stewing', 'Sublimating',
    'Swirling', 'Swooping', 'Symbioting', 'Synthesizing', 'Tempering', 'Thinking',
    'Thundering', 'Tinkering', 'Tomfoolering', 'Topsy-turvying', 'Transfiguring',
    'Transmuting', 'Twisting', 'Undulating', 'Unfurling', 'Unravelling', 'Vibing',
    'Waddling', 'Wandering', 'Warping', 'Whatchamacalliting', 'Whirlpooling', 'Whirring',
    'Whisking', 'Wibbling', 'Working', 'Wrangling', 'Zesting', 'Zigzagging'
]

def get_spinner_message() -> str:
    """Get a random spinner verb with ellipsis."""
    return f"{random.choice(SPINNER_VERBS)}..."

# ============================================================
# LOGGING SETUP
# ============================================================

def log(msg: str):
    print(msg, flush=True)

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

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Omniphysical KG",
    page_icon="O",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
    }
    
    .main-header h1 {
        font-size: 2rem;
        font-weight: 600;
    }
    
    /* Chat input styling */
    .stChatInput > div {
        border-radius: 20px;
        border: 1px solid #444;
    }
    
    /* Hide "Press Enter to apply" text */
    .stTextInput small {
        display: none !important;
    }
    .stTextInput [data-testid="InputInstructions"] {
        display: none !important;
    }
    div[data-testid="InputInstructions"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# AUTHENTICATION
# ============================================================

def check_password() -> bool:
    """Simple password authentication."""
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if st.session_state.authenticated:
        return True
    
    st.markdown('<div class="main-header"><h1>Omniphysical KG</h1><p>Enter password to continue</p></div>', unsafe_allow_html=True)
    
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
# KNOWLEDGE GRAPH RETRIEVER
# ============================================================

class KGRetriever:
    """Handles knowledge graph queries."""
    
    def __init__(self):
        self.neo4j_uri = get_secret("NEO4J_URI")
        self.neo4j_auth = (get_secret("NEO4J_USERNAME"), get_secret("NEO4J_PASSWORD"))
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=self.neo4j_auth,
            connection_timeout=60,
            max_connection_lifetime=300
        )
        self.openai = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
        self.embedding_model = "text-embedding-3-small"
        self.llm_model = "gpt-5-mini"
    
    def close(self):
        self.driver.close()
    
    def refresh_connection(self):
        """Refresh Neo4j connection if stale"""
        try:
            self.driver.close()
        except:
            pass
        self.driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=self.neo4j_auth,
            connection_timeout=60,
            max_connection_lifetime=300
        )
        log("[*] Refreshed Neo4j connection")
    
    def get_embedding(self, text: str) -> list:
        response = self.openai.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def get_entity_relationships_scored(self, session, entity_names: list, 
                                         query_embedding: list, top_k: int = 10) -> list:
        """Get top relationships from matched entities, scored against question."""
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
                   r.context AS context,
                   score
        """, names=entity_names, embedding=query_embedding, top_k=top_k)
        
        return [dict(r) for r in result]
    
    def get_entity_properties(self, session, entity_names: list) -> list:
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
    
    def deduplicate_relationships(self, relationships: list) -> list:
        """Deduplicate relationships using (from, type, to, context) as key."""
        seen = set()
        unique = []
        for rel in relationships:
            key = (rel['from_name'], rel['rel_type'], rel['to_name'], rel.get('context', ''))
            if key not in seen:
                seen.add(key)
                unique.append(rel)
        return unique
    
    def format_context(self, entities: list, relationships: list, entity_props: list, paragraphs: list = None) -> str:
        sections = []
        
        # Entity properties section
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
        
        # Relationship matches section (already deduplicated before calling this)
        if relationships:
            lines = ["RELEVANT RELATIONSHIPS:"]
            for rel in relationships:
                context_str = f" ({rel['context']})" if rel.get('context') else ""
                lines.append(f"  - {rel['from_name']} -[{rel['rel_type']}]-> {rel['to_name']}{context_str}")
            sections.append("\n".join(lines))
        
        # Paragraph section
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
    
    def generate_answer(self, question: str, context: str) -> tuple:
        """Generate answer and return (answer, usage_dict)."""
        system_prompt = """You are a knowledgeable assistant helping answer questions about a personal knowledge graph containing documents, relationships, and facts about people, places, events, and experiences.

Guidelines:
- Answer directly and naturally without labels like "Short answer:" or "Summary:"
- Ground your response in the provided context — cite specific facts, quotes, or relationships when relevant
- If the question asks you to distinguish or categorize (e.g., "strongly supported vs speculative"), do so; otherwise, just answer naturally
- You may reason about and connect information in the context to draw conclusions
- Stay grounded in the context; if key information is missing, briefly note it rather than guessing
- Match your response length to the question: brief for simple queries, thorough for complex analysis
- Write in a warm, conversational tone — informative but not robotic
- Never reference the graph structure, context format, or data source — just present the information naturally"""

        response = self.openai.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
            ]
        )
        
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        return response.choices[0].message.content, usage
    
    def verify_connection(self):
        """Verify Neo4j connection is alive, refresh if not"""
        try:
            self.driver.verify_connectivity()
        except Exception as e:
            log(f"[!] Connection dead: {e}")
            self.refresh_connection()
    
    def update_spinner(self, spinner_placeholder):
        """Update spinner with new random message."""
        if spinner_placeholder:
            spinner_placeholder.markdown(f"*{get_spinner_message()}*")
    
    def query(self, question: str, retry: bool = True, spinner_placeholder=None) -> str:
        query_start_time = time.time()
        
        log(f"\n[QUERY] {question}")
        log("-" * 50)
        
        # Verify connection before starting
        self.verify_connection()
        
        # Embed
        self.update_spinner(spinner_placeholder)
        embed_start = time.time()
        query_embedding = self.get_embedding(question)
        embed_time = time.time() - embed_start
        log(f"[1] Embedded question (dim: {len(query_embedding)}) [{embed_time:.2f}s]")
        
        try:
            neo4j_start = time.time()
            with self.driver.session() as session:
                
                self.update_spinner(spinner_placeholder)
                
                # [2] Search entities (with threshold)
                result = session.run("""
                    MATCH (e)
                    WHERE e.embedding IS NOT NULL
                    AND NOT e:Paragraph AND NOT e:Document
                    WITH e, gds.similarity.cosine(e.embedding, $embedding) AS score
                    WHERE score >= $threshold
                    RETURN e.name AS name, labels(e)[0] AS type, score, 
                           e.entity_id AS entity_id
                    ORDER BY score DESC
                    LIMIT $top_k
                """, embedding=query_embedding, threshold=0.25, top_k=10)
                entities = [dict(r) for r in result]
                
                log(f"[2] Found {len(entities)} relevant entities (threshold=0.25):")
                for e in entities[:5]:
                    log(f"    - {e['name']} ({e['type']}) [score: {e['score']:.3f}]")
                
                self.update_spinner(spinner_placeholder)
                
                # [3a] Search relationships globally (with threshold)
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
                           r.context AS context,
                           score
                    ORDER BY score DESC
                    LIMIT $top_k
                """, embedding=query_embedding, threshold=0.25, top_k=20)
                global_relationships = [dict(r) for r in result]
                
                log(f"[3a] Found {len(global_relationships)} global relationships (threshold=0.25):")
                for r in global_relationships[:5]:
                    log(f"    - {r['from_name']} -[{r['rel_type']}]-> {r['to_name']} [score: {r['score']:.3f}]")
                
                self.update_spinner(spinner_placeholder)
                
                # [3b] Get relationships from matched entities, scored against question
                entity_names = [e['name'] for e in entities[:5]]
                entity_relationships = self.get_entity_relationships_scored(
                    session, entity_names, query_embedding, top_k=10
                )
                
                log(f"[3b] Found {len(entity_relationships)} entity-based relationships:")
                for r in entity_relationships[:5]:
                    log(f"    - {r['from_name']} -[{r['rel_type']}]-> {r['to_name']} [score: {r['score']:.3f}]")
                
                # [3c] Combine and deduplicate relationships
                combined_relationships = global_relationships + entity_relationships
                relationships = self.deduplicate_relationships(combined_relationships)
                
                # Sort by score descending
                relationships.sort(key=lambda x: x.get('score', 0), reverse=True)
                
                log(f"[3c] Combined: {len(relationships)} unique relationships")
                
                self.update_spinner(spinner_placeholder)
                
                # [4] Search paragraphs (always, threshold filters automatically)
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
                """, embedding=query_embedding, threshold=0.45, top_k=5)
                paragraphs = [dict(r) for r in result]
                
                log(f"[4] Found {len(paragraphs)} relevant paragraphs (threshold=0.45):")
                for p in paragraphs[:3]:
                    source_short = p['source'][:35] if p['source'] else "?"
                    log(f"    - {source_short}... chunk {p['chunk_index']} [score: {p['score']:.3f}]")
                
                self.update_spinner(spinner_placeholder)
                
                # [5] Get entity properties (top 5 entities + top 5 relationship entities)
                rel_entity_names = []
                for r in relationships[:5]:
                    rel_entity_names.extend([r['from_name'], r['to_name']])
                all_names = list(set(entity_names + rel_entity_names))
                
                try:
                    result = session.run("""
                        MATCH (e)
                        WHERE e.name IN $names
                        AND NOT e:Paragraph AND NOT e:Document
                        WITH e, labels(e)[0] AS type,
                             [k IN keys(e) WHERE NOT k IN ['embedding', 'entity_id', 'created_at', 'updated_at']] AS prop_keys
                        RETURN e.name AS name,
                               type,
                               apoc.map.fromPairs([k IN prop_keys | [k, e[k]]]) AS properties
                    """, names=all_names)
                    entity_props = [dict(r) for r in result]
                except Exception:
                    entity_props = []
                
                neo4j_time = time.time() - neo4j_start
                log(f"[5] Retrieved properties for {len(entity_props)} entities [{neo4j_time:.2f}s total Neo4j]")
            
            self.update_spinner(spinner_placeholder)
            
            # Format context (relationships already deduplicated)
            context = self.format_context(entities, relationships, entity_props, paragraphs)
            log(f"\n[CONTEXT]\n{context}\n")
            log("-" * 50)
            
            self.update_spinner(spinner_placeholder)
            
            # Generate answer
            llm_start = time.time()
            answer, usage = self.generate_answer(question, context)
            llm_time = time.time() - llm_start
            
            total_time = time.time() - query_start_time
            
            log(f"[TOKENS] Input: {usage['input_tokens']} | Output: {usage['output_tokens']} | Total: {usage['total_tokens']}")
            log(f"[LATENCY] Embed: {embed_time:.2f}s | Neo4j: {neo4j_time:.2f}s | LLM: {llm_time:.2f}s | Total: {total_time:.2f}s")
            log(f"[ANSWER] {answer}\n")
            
            return answer
            
        except Exception as e:
            error_msg = str(e).lower()
            if retry and ("routing" in error_msg or "defunct" in error_msg or "no data" in error_msg or "timeout" in error_msg or "timed out" in error_msg):
                log(f"[!] Connection error: {e}")
                log(f"[!] Refreshing connection and retrying...")
                self.refresh_connection()
                return self.query(question, retry=False, spinner_placeholder=spinner_placeholder)
            else:
                raise

# ============================================================
# MAIN APP
# ============================================================

def main():
    if not check_password():
        return
    
    # Header
    st.markdown('<div class="main-header"><h1>Omniphysical KG</h1></div>', unsafe_allow_html=True)
    
    # Initialize retriever
    if "retriever" not in st.session_state:
        try:
            st.session_state.retriever = KGRetriever()
        except Exception as e:
            st.error(f"Failed to connect: {e}")
            return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    retriever = st.session_state.retriever
    
    # Display chat history using native chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if query := st.chat_input("Ask a question..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.write(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display assistant response with dynamic spinner
        with st.chat_message("assistant"):
            spinner_placeholder = st.empty()
            
            try:
                # Show initial spinner
                spinner_placeholder.markdown(f"*{get_spinner_message()}*")
                
                # Run query with spinner updates
                answer = retriever.query(query, spinner_placeholder=spinner_placeholder)
                
                # Replace spinner with answer
                spinner_placeholder.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                spinner_placeholder.error(f"Error: {e}")

if __name__ == "__main__":
    main()