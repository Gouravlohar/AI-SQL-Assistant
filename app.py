import streamlit as st
from pathlib import Path
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
from langchain_groq import ChatGroq


st.set_page_config(page_title="AI SQL Assistant", page_icon="🤖", layout="wide")

st.title("🤖 AI SQL Assistant")

MYSQL = "USE_MYSQL"

with st.sidebar:
    st.header("Database Configuration")
    selected_opt = st.radio("Choose an option", ["How to Use", "Connect to MySQL Database"])

    if selected_opt == "Connect to MySQL Database":
        db_uri = MYSQL
        mysql_host = st.text_input("MySQL Host")
        mysql_user = st.text_input("MySQL User")
        mysql_password = st.text_input("MySQL Password", type="password")
        mysql_db = st.text_input("MySQL Database")
        api_key = st.text_input("Groq API Key", type="password")
    else:
        db_uri = None
        api_key = None

    if st.button("Clear Chat History"):
        st.session_state["messages"] = []

    st.markdown("---")
    st.subheader("About")
    st.info("This AI SQL Assistant uses LangChain and Groq to provide a natural language interface to your MySQL database. Ask questions in plain English, and get SQL-powered answers!")

if db_uri == MYSQL and not api_key:
    st.sidebar.error("Please add the Groq API key")
    st.stop()

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-70b-versatile", streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    st.header("How to Use")
    st.markdown("""
    1. Select "Connect to MySQL Database" in the sidebar.
    2. Fill in your MySQL connection details and Groq API key.
    3. Once connected, you can start chatting with your SQL database!
    4. Ask questions in natural language, and the AI will translate them into SQL queries.
    """)
    st.stop()

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

streamlit_callback = StreamlitCallbackHandler(st.container())

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callbacks=[streamlit_callback]
)

st.header("Chat Interface")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        response_container = st.container()
        with response_container:
            response = agent.run(user_query)
            response_container.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})