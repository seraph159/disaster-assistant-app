# Disaster Assistant App
import os
import warnings
import logging

import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Disable warnings and info logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Page configuration
st.set_page_config(
    page_title="Disaster Assistant",
    page_icon="🚨",
    layout="wide"
)

# Sidebar for app navigation
st.sidebar.title("Disaster Assistant")
page = st.sidebar.radio("Navigation", ["Home", "Emergency Chat", "Resources"])

# Setup a session state variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to load disaster-related documents
@st.cache_resource
def get_vectorstore():
    # Create a directory for disaster documents if it doesn't exist
    docs_dir = "./disaster_docs"
    os.makedirs(docs_dir, exist_ok=True)
    
    # Check if documents exist, otherwise use sample data
    if not os.path.exists(f"{docs_dir}/preparedness_guide.pdf"):
        # Create a sample document with disaster information
        st.warning("No disaster documents found. Using built-in knowledge.")
    
    try:
        # Load all PDFs from the disaster_docs directory
        loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        
        # Create chunks for vector database
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        ).from_loaders([loader])
        
        return index.vectorstore
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return None

# Function to get response from LLM
def get_disaster_assistant_response(prompt, disaster_type=None):
    try:
        model = "llama3-8b-8192"
        groq_chat = ChatGroq(
            groq_api_key=os.environ.get("GROQ_API_KEY"), 
            model_name=model
        )
        
        disaster_context = f"The user is asking about a {disaster_type} disaster. " if disaster_type else ""
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", f"{disaster_context}You are a disaster assistance AI that provides accurate, "
                       "helpful, and calming information during emergencies. "
                       "Prioritize safety advice and clear instructions. "
                       "If you're unsure about specific local details, make that clear and suggest "
                       "contacting local authorities.\n\n{context}"),
            ("human", "{input}")
        ])
        
        vectorstore = get_vectorstore()
        if vectorstore is not None:
            retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
            question_answer_chain = create_stuff_documents_chain(groq_chat, system_prompt)
            chain = create_retrieval_chain(retriever, question_answer_chain)
            result = chain.invoke({"input": prompt})
            return result["answer"]
        else:
            chain = system_prompt | groq_chat | StrOutputParser()
            return chain.invoke({"input": prompt})
    except Exception as e:
        return f"I'm sorry, I encountered an error: {str(e)}. Please try again or contact emergency services if this is an urgent situation."

# Home page
if page == "Home":
    st.title("🚨 Disaster Assistance App")
    st.markdown("""
    ## Welcome to the Disaster Assistance App
    
    This application provides information and guidance during disaster situations.
    
    ### Features:
    - **Emergency Chat**: Get answers to your questions about disaster preparedness and response
    - **Resources**: Access important emergency contacts and resources
    
    ### Current Disaster Types Supported:
    - Earthquakes
    - Floods
    - Hurricanes/Typhoons
    - Wildfires
    - Winter Storms
    - Pandemics
    """)
    
    st.info("⚠️ IMPORTANT: In a real emergency, always contact local emergency services first!")

# Emergency Chat page
elif page == "Emergency Chat":
    st.title("🆘 Emergency Assistant Chat")
    
    # Disaster type selector
    disaster_type = st.selectbox(
        "Select Disaster Type (Optional)",
        [None, "Earthquake", "Flood", "Hurricane/Typhoon", "Wildfire", "Winter Storm", "Pandemic"]
    )
    
    # Display all the historical messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    # Chat input
    prompt = st.chat_input('Ask a question about disaster preparedness or response...')
    
    if prompt:
        # Display user message
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        # Get and display assistant response
        with st.spinner("Getting information..."): 
            response = get_disaster_assistant_response(prompt, disaster_type)
            
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

# Resources page
elif page == "Resources":
    st.title("📚 Emergency Resources")
    
    st.markdown("""
    ## Emergency Contacts
    
    - **Emergency Services**: 911 (US) or your local emergency number
    - **FEMA**: 1-800-621-3362
    - **Red Cross**: 1-800-733-2767
    
    ## Disaster Preparedness Resources
    
    - [Ready.gov](https://www.ready.gov/) - Official disaster preparedness website
    - [Red Cross Disaster Preparedness](https://www.redcross.org/get-help/how-to-prepare-for-emergencies.html)
    - [FEMA Mobile App](https://www.fema.gov/about/news-multimedia/mobile-app-text-messages)
    
    ## Disaster Recovery Resources
    
    - [DisasterAssistance.gov](https://www.disasterassistance.gov/)
    - [FEMA Disaster Recovery Centers](https://egateway.fema.gov/ESF6/DRCLocator)
    """)
    
    # Location-based information (placeholder)
    st.subheader("Local Resources")
    st.info("In a full implementation, this section would show resources specific to your location.")