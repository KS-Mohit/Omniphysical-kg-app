"""
Omniphysical Knowledge Graph - Query Interface
Simple chatbot UI for querying the knowledge graph.
"""
import os
import sys
import streamlit as st
from neo4j import GraphDatabase
from openai import OpenAI

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
    
    def search_entities(self, query_embedding: list, top_k: int = 10, threshold: float = 0.5) -> list:
        with self.driver.session() as session:
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
            """, embedding=query_embedding, threshold=threshold, top_k=top_k)
            return [dict(r) for r in result]
    
    def search_relationships(self, query_embedding: list, top_k: int = 20, threshold: float = 0.5) -> list:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a)-[r]->(b)
                WHERE r.embedding IS NOT NULL
                AND (r.isLatest IS NULL OR r.isLatest = true)
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
            """, embedding=query_embedding, threshold=threshold, top_k=top_k)
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
    
    def format_context(self, entities: list, relationships: list, entity_props: list) -> str:
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
        
        # Relationship matches section (deduplicated)
        if relationships:
            lines = ["RELEVANT RELATIONSHIPS:"]
            seen = set()
            for rel in relationships:
                key = (rel['from_name'], rel['rel_type'], rel['to_name'])
                if key in seen:
                    continue
                seen.add(key)
                context_str = f" ({rel['context']})" if rel.get('context') else ""
                lines.append(f"  - {rel['from_name']} -[{rel['rel_type']}]-> {rel['to_name']}{context_str}")
            sections.append("\n".join(lines))
        
        if not sections:
            return "No relevant context found in the knowledge graph."
        
        return "\n\n".join(sections)
    
    def generate_answer(self, question: str, context: str) -> str:
        system_prompt = """You are a helpful assistant answering questions based on a personal knowledge graph.
Use ONLY the provided context to answer. If the context doesn't contain enough information, say so.
Be concise and direct. Reference specific facts from the context."""

        response = self.openai.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}
            ]
        )
        
        return response.choices[0].message.content
    
    def verify_connection(self):
        """Verify Neo4j connection is alive, refresh if not"""
        try:
            self.driver.verify_connectivity()
        except Exception as e:
            log(f"[!] Connection dead: {e}")
            self.refresh_connection()
    
    def query(self, question: str, retry: bool = True) -> str:
        log(f"\n[QUERY] {question}")
        log("-" * 50)
        
        # Verify connection before starting
        self.verify_connection()
        
        # Embed
        query_embedding = self.get_embedding(question)
        log(f"[1] Embedded question (dim: {len(query_embedding)})")
        
        try:
            # Use single session for all operations
            with self.driver.session() as session:
                # Search entities
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
                """, embedding=query_embedding, threshold=0.5, top_k=10)
                entities = [dict(r) for r in result]
                
                log(f"[2] Found {len(entities)} relevant entities:")
                for e in entities[:5]:
                    log(f"    - {e['name']} ({e['type']}) [score: {e['score']:.3f}]")
                
                # Search relationships
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
                """, embedding=query_embedding, threshold=0.5, top_k=20)
                relationships = [dict(r) for r in result]
                
                log(f"[3] Found {len(relationships)} relevant relationships:")
                for r in relationships[:5]:
                    log(f"    - {r['from_name']} -[{r['rel_type']}]-> {r['to_name']} [score: {r['score']:.3f}]")
                
                # Get entity properties (top 5 entities + top 5 relationship entities)
                entity_names = [e['name'] for e in entities[:5]]
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
                
                log(f"[4] Retrieved properties for {len(entity_props)} entities")
            
            # Format context (uses ALL entities and relationships, not just top 5)
            context = self.format_context(entities, relationships, entity_props)
            log(f"\n[CONTEXT]\n{context}\n")
            log("-" * 50)
            
            # Generate answer
            answer = self.generate_answer(question, context)
            log(f"[ANSWER] {answer}\n")
            
            return answer
            
        except Exception as e:
            error_msg = str(e).lower()
            if retry and ("routing" in error_msg or "defunct" in error_msg or "no data" in error_msg or "timeout" in error_msg or "timed out" in error_msg):
                log(f"[!] Connection error: {e}")
                log(f"[!] Refreshing connection and retrying...")
                self.refresh_connection()
                return self.query(question, retry=False)
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
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = retriever.query(query)
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()