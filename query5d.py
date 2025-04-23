import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

api_key = st.secrets["OPENAI_API_KEY"]

def sanitize_folder_name(name):
    """Convert company name to a format that matches directory naming conventions."""
    return name.strip().replace(" ", "_").replace(".", "").upper()

def load_database(database_dir, folder_name):
    """Loads the FAISS database for a given folder."""
    embeddings = HuggingFaceEmbeddings()
    
    # Try the original folder name first
    folder_path = os.path.join(database_dir, folder_name)
    abs_path = os.path.abspath(folder_path)

    print(f"🔍 Checking folder: {abs_path}")

    # If the exact folder doesn't exist, try to find a matching one
    if not os.path.isdir(abs_path):
        print(f"🔍 Folder not found with exact name, trying alternative formats...")
        
        # Try sanitized version of the folder name
        sanitized_name = sanitize_folder_name(folder_name)
        sanitized_path = os.path.join(database_dir, sanitized_name)
        abs_sanitized_path = os.path.abspath(sanitized_path)
        
        if os.path.isdir(abs_sanitized_path):
            print(f"✅ Found matching folder: {abs_sanitized_path}")
            abs_path = abs_sanitized_path
        else:
            # Try to find a case-insensitive match
            found = False
            for dir_name in os.listdir(database_dir):
                if folder_name.lower() == dir_name.lower() or sanitized_name.lower() == dir_name.lower():
                    folder_path = os.path.join(database_dir, dir_name)
                    abs_path = os.path.abspath(folder_path)
                    print(f"✅ Found case-insensitive match: {abs_path}")
                    found = True
                    break
            
            if not found:
                print(f"🚫 No matching folder found for: {folder_name}")
                return None

    try:
        print(f"📦 Loading FAISS vector store from: {abs_path}")
        # Check if required files exist
        required_files = ['index.faiss', 'index.pkl']
        if not all(os.path.exists(os.path.join(abs_path, f)) for f in required_files):
            print(f"❌ Missing required FAISS files in {abs_path}")
            return None
        
        # Fix the FAISS loading issue - Custom deserializer handling
        try:
            vector_store = FAISS.load_local(
                abs_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
        except AttributeError as e:
            if '__fields_set__' in str(e):
                print(f"⚠️ Detected pydantic compatibility issue, trying alternative loading method...")
                # Attempt to load FAISS with a different approach
                import pickle
                import faiss
                from langchain_core.documents import Document
                
                # Load the index
                index = faiss.read_index(os.path.join(abs_path, "index.faiss"))
                
                # Load the docstore and other data
                with open(os.path.join(abs_path, "index.pkl"), "rb") as f:
                    data = pickle.load(f)
                    
                # Create a new FAISS instance with the loaded data
                docstore = {}
                if hasattr(data, 'docstore') and hasattr(data.docstore, '_dict'):
                    # For newer LangChain versions
                    docstore = data.docstore._dict
                elif hasattr(data, '_dict'):
                    # For older LangChain versions
                    docstore = data._dict
                else:
                    # Try to extract documents from whatever structure we have
                    print("⚠️ Unusual docstore format, attempting recovery...")
                    for key, value in vars(data).items():
                        if 'docstore' in key or 'dict' in key:
                            if hasattr(value, '_dict'):
                                docstore = value._dict
                                break
                
                # Create a new vector store using the loaded components
                vector_store = FAISS(
                    embeddings.embed_query,
                    index,
                    docstore,
                    {}, # Empty dict for index_to_docstore_id mapping (will be populated from docstore keys)
                    allow_dangerous_deserialization=True
                )
            else:
                raise e

        # Increase k to retrieve more context documents
        retriever = vector_store.as_retriever(search_kwargs={"k": 25})

        llm = ChatOpenAI(
            model_name="gpt-4o",  # Using a more capable model for better understanding
            api_key=api_key
        )

        # Improved prompt template specifically for IPO/financial document analysis
        template = """You are a specialized financial document analyzer focused on IPO and company information. 
        Your task is to find relevant information from {company_name}'s documents to answer specific questions.

        You're primarily looking for information about:
        - IPO details (objects of issue, lead managers, counsels, etc.)
        - Financial information (auditors, financial statements)
        - Company structure (directors, promoters, reservations)
        - Regulatory compliance details

        IMPORTANT INSTRUCTIONS:
        1. If the exact information is present in the context, provide it concisely
        2. If similar or related information exists that can answer the question, use your judgment to formulate a response
        3. For questions about "Legal Counsels to the Issue", "Industry Agency", or similar IPO terms, look for any mention of legal firms, consultants, or agencies in the context
        4. For questions about reservations (shareholder/employee), look for any mention of reserved portions or allocations
        5. For yes/no questions about inclusions (like "Whether Proforma Financial Statements included?"), clearly state yes or no if you can determine this

        Context:
        {context}

        Question: {input}

        Answer the question as specifically as possible. If you absolutely cannot find relevant information to answer the question, respond with "Based on the available documents, I cannot find information about [the specific topic]."
        """

        prompt = ChatPromptTemplate.from_template(template.replace("{company_name}", folder_name))
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        print(f"✅ Successfully loaded database for: {folder_name}")
        return retrieval_chain

    except Exception as e:
        print(f"❌ Error loading database for {folder_name}: {e}")
        import traceback
        traceback.print_exc()  # Print detailed traceback
        return None

# Enhanced query function with specialized handling for common IPO questions
def query_specific_database(retrieval_chain, user_query, company_name):
    """Queries a specific database with enhanced handling for IPO questions."""
    if not retrieval_chain:
        return f"Database for {company_name} could not be loaded."
    
    # Normalize query for better matching
    normalized_query = user_query.lower().strip()
    
    # Create enhanced queries for specific question types to improve retrieval
    enhanced_query = user_query
    
    # Enhance queries to improve retrieval
    query_enhancements = {
        "legal counsel": "legal counsel OR legal advisor OR law firm OR legal firm",
        "industry agency": "industry agency OR research agency OR industry report OR research report OR industry analysis",
        "reservation": "reservation OR reserved portion OR allocated shares OR allocation",
        "auditor": "auditor OR statutory auditor OR chartered accountant",
        "promoter": "promoter OR promoter group OR founding member",
        "board of director": "board of director OR director OR management team OR leadership",
        "object of the issue": "object of the issue OR use of proceeds OR utilization of proceeds OR purpose of the offer",
        "lead manager": "lead manager OR book running lead manager OR BRLM OR investment banker"
    }
    
    # Apply enhancements if keywords are found in the query
    for keyword, enhancement in query_enhancements.items():
        if keyword in normalized_query:
            enhanced_query = f"{user_query} ({enhancement})"
            break
    
    try:
        print(f"Querying with enhanced prompt: {enhanced_query}")
        response = retrieval_chain.invoke({"input": enhanced_query})
        answer = response['answer'].strip()
        
        # Post-process answer for specific question types
        if "whether" in normalized_query.lower() and ("yes" not in answer.lower() and "no" not in answer.lower()):
            if "cannot find" in answer.lower() or "don't know" in answer.lower() or "doesn't mention" in answer.lower():
                answer = f"No, based on the available documents, {answer.replace('I cannot find information about', 'there is no clear mention of')}"
        
        return answer
    except Exception as e:
        print(f"Error querying database: {e}")
        return f"An error occurred while querying the database: {str(e)}"

def load_all_databases(database_dir):
    """Loads all available databases in the directory."""
    retrieval_chains = {}
    
    if not os.path.exists(database_dir):
        print(f"Error: Database directory not found: {database_dir}")
        return retrieval_chains

    for folder_name in os.listdir(database_dir):
        folder_path = os.path.join(database_dir, folder_name)
        if os.path.isdir(folder_path):
            print(f"Found database directory: {folder_name}")
            chain = load_database(database_dir, folder_name)
            if chain:
                # Store the chain with the original display name from UI
                for company in ["Continuum Green Energy Limited", "Hexaware Technologies Limited", 
                              "WeWork India Management Limited", "Dr. Agarwal's Health Care Limited"]:
                    if company.lower().replace(" ", "_").replace(".", "") in folder_name.lower().replace(" ", "_").replace(".", ""):
                        retrieval_chains[company] = chain
                        print(f"Mapped {folder_name} to {company}")
                        break
                else:
                    # If no match found in the predefined list, use the directory name as is
                    retrieval_chains[folder_name] = chain
    
    return retrieval_chains

def format_response(response, user_query, company_name):
    """Formats the response from the database."""
    if not response or "cannot find information" in response.lower():
        return f"Based on {company_name}'s available documents, I cannot find specific information about '{user_query}'."
    
    return response