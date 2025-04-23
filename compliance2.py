import os
import json
import logging
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def sanitize_folder_name(name):
    """Convert company name to a format that matches directory naming conventions."""
    return name.strip().replace(" ", "_").replace(".", "").upper()

def load_database(database_dir, folder_name):
    """Loads the FAISS database for a given folder."""
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    
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
        retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        
        api_key = st.secrets["openai_api_key"] 
        llm = ChatOpenAI(
            model_name="gpt-4o",  # Using a more capable model for better understanding
            temperature=0.1,      # Lower temperature for more consistent responses
            api_key=api_key
        )

        print(f"✅ Successfully loaded database for: {folder_name}")
        return {
            "retriever": retriever,
            "llm": llm,
            "vector_store": vector_store
        }

    except Exception as e:
        print(f"❌ Error loading database for {folder_name}: {e}")
        import traceback
        traceback.print_exc()  # Print detailed traceback
        return None

# Get standard headings from JSON structure
def get_standard_headings():
    """Returns the standard headings structure for IPO documents."""
    return [
        {
            "name": "Cover Pages",
            "description": "Section with front and back cover pages, including issue and issuer details, selling shareholders, and other specified information.",
            "synonyms": ["Front Cover", "Back Cover", "Cover Page"]
        },
        {
            "name": "Definitions and Abbreviations",
            "description": "Section defining conventional, issue-related, issuer/industry-related terms, and abbreviations.",
            "synonyms": ["Glossary", "Terms and Definitions", "Abbreviations"]
        },
        {
            "name": "Offer Document Summary",
            "description": "Summary of key information like issuer's business, promoters, issue size, and financial details.",
            "synonyms": ["Summary", "Executive Summary", "Offer Summary"]
        },
        {
            "name": "Risk Factors",
            "description": "Section listing internal and external risk factors in descending order of materiality.",
            "synonyms": ["Risks", "Risk Disclosures", "Potential Risks"]
        },
        {
            "name": "Introduction",
            "description": "Section with brief issue details and consolidated financial information summary.",
            "synonyms": ["Overview", "Preface", "Introductory Section"]
        },
        {
            "name": "General Information",
            "description": "Details of issuer's registered office, board of directors, and key contacts.",
            "synonyms": ["Company Information", "Issuer Details", "Corporate Information"]
        },
        {
            "name": "Capital Structure",
            "description": "Details of authorized, issued, and paid-up capital of the issuer.",
            "synonyms": ["Share Capital", "Equity Structure", "Capital Details"]
        },
        {
            "name": "Particulars of the Issue",
            "description": "Section on objects of the issue, funding requirements, and issue details.",
            "synonyms": ["Issue Details", "Offer Particulars", "Issue Objectives"]
        },
        {
            "name": "Legal and Other Information",
            "description": "Section covering outstanding litigations, material developments, and dues to creditors.",
            "synonyms": ["Legal Disclosures", "Litigation Details", "Other Disclosures"]
        },
        {
            "name": "Information with Respect to Group Companies",
            "description": "Details about group companies, including financial information hosted on their websites.",
            "synonyms": ["Group Companies", "Related Entities", "Affiliate Information"]
        },
        {
            "name": "Other Regulatory and Statutory Disclosures",
            "description": "Disclosures on authority for the issue, compliance, and disclaimer clauses.",
            "synonyms": ["Regulatory Disclosures", "Statutory Disclosures", "Compliance Details"]
        },
        {
            "name": "Offering Information",
            "description": "Details of terms of the issue and issue procedures.",
            "synonyms": ["Offer Terms", "Issue Procedures", "Offering Details"]
        },
        {
            "name": "Description of Equity Shares and Terms of the Articles of Association",
            "description": "Main provisions of the Articles of Association, including members' rights.",
            "synonyms": ["Equity Shares", "Articles of Association", "Share Terms"]
        },
        {
            "name": "Other Information",
            "description": "List of material contracts and documents available for inspection.",
            "synonyms": ["Additional Information", "Material Contracts", "Supplementary Details"]
        }
    ]

# Create a retrieval chain for a specific heading
def create_heading_chain(database_components, heading, description, synonyms, company_name):
    """Creates a retrieval chain for a specific heading."""
    try:
        retriever = database_components["retriever"]
        llm = database_components["llm"]
        
        template = f"""
        You are an expert in financial regulatory compliance analyzing the {company_name} IPO offer document. Your task is to determine if a section titled '{heading}' (or variations like {', '.join(synonyms)}) is present, as described: '{description}'. This is an IPO document with sections like Table of Contents, Risk Factors, etc., typically following a standard structure.

        Context:
        {{context}}

        Consider:
        - Variations in wording (e.g., "Contents" or "Index" for "Table of Contents").
        - Structural cues (e.g., a Table of Contents lists sections with page numbers, appearing early in the document).
        - Surrounding text (e.g., section titles followed by numbered items).

        Answer in the following format:
        Present: Yes/No
        Followed: Yes/No
        Matched Text: The most relevant text snippet
        Explanation: Brief explanation of the decision

        If no relevant information is found, state that the section is not present.
        """
        prompt = ChatPromptTemplate.from_template(template)
        document_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, document_chain)
    except Exception as e:
        logger.error(f"Failed to create chain for '{heading}': {e}")
        return None

# Process a single heading for compliance check
def process_heading(heading_info, database_components, company_name):
    """Process a single heading for compliance check."""
    heading = heading_info["name"]
    description = heading_info["description"]
    synonyms = heading_info["synonyms"]
    
    logger.info(f"Processing heading for {company_name}: {heading}")
    
    # Create query combining heading, synonyms, and description
    query = f"{heading} or {', '.join(synonyms)}: {description}"
    
    chain = create_heading_chain(database_components, heading, description, synonyms, company_name)
    if not chain:
        return {
            "heading": heading,
            "present": False,
            "followed": False,
            "matched_text": "Error creating chain",
            "explanation": "Failed to create retrieval chain"
        }
    
    try:
        response = chain.invoke({"input": query})
        answer = response['answer'].strip()
        
        # Parse response
        lines = answer.split('\n')
        present = False
        followed = False
        matched_text = "No match found"
        explanation = "No explanation provided"
        
        for line in lines:
            if line.startswith("Present:"):
                present = line.split(":")[1].strip().lower() == "yes"
            elif line.startswith("Followed:"):
                followed = line.split(":")[1].strip().lower() == "yes"
            elif line.startswith("Matched Text:"):
                matched_text = line.split(":", 1)[1].strip()[:200] + "..."
            elif line.startswith("Explanation:"):
                explanation = line.split(":", 1)[1].strip()
        
        return {
            "heading": heading,
            "present": present,
            "followed": followed,
            "matched_text": matched_text,
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Error processing '{heading}' for {company_name}: {e}")
        return {
            "heading": heading,
            "present": False,
            "followed": False,
            "matched_text": "Error in processing",
            "explanation": f"Error: {str(e)}"
        }

# Enhanced check_compliance function 
def check_compliance(database_dir, company_name):
    """
    Checks compliance of a company's IPO document against standard headings.
    Returns a compliance report in JSON format.
    """
    # Load database components
    database_components = load_database(database_dir, company_name)
    if not database_components:
        return {"error": f"Database for {company_name} could not be loaded."}
    
    # Get standard headings
    headings = get_standard_headings()
    
    # Process headings concurrently
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_heading, heading, database_components, company_name) 
                  for heading in headings]
        for future in futures:
            results.append(future.result())

    # Generate report
    report = {
        "company": company_name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "headings_checked": len(headings),
        "results": results,
        "summary": {
            "sections_present": sum(1 for r in results if r["present"]),
            "sections_followed": sum(1 for r in results if r["followed"]),
            "compliance_score": round(sum(1 for r in results if r["present"] and r["followed"]) / len(headings) * 100, 2)
        }
    }
    
    return report

# Maintain original query function with enhanced prompt template
def query_specific_database_compliance(retrieval_chain, user_query, company_name):
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

# Function to create a single database retrieval chain
def create_database_chain(database_components, company_name):
    """Creates a retrieval chain for general questions about a company."""
    retriever = database_components["retriever"]
    llm = database_components["llm"]
    
    # Improved prompt template specifically for IPO/financial document analysis
    template = """You are a specialized compliance analyst reviewing financial and regulatory documents submitted by {company_name}. Your task is to verify whether specific compliance-related sections or guidelines are present and followed in the company's documentation. Use the reference structure from the provided guidelines which outlines expected sections, their descriptions, and synonym labels.
    
    INSTRUCTIONS
    Start your response with YES, NO, or PARTIAL.
    If YES, include the exact matched content or description found.
    If NO, briefly explain what was expected and not found.
    If PARTIAL, mention what was found and what remains missing or incomplete.
    Keep the explanation to 1–3 concise sentences.
    Focus on clarity and document traceability.
    Avoid unnecessary details or jargon.
    
    Context:
    {context}
    
    Query:
    {input}
    """
    
    prompt = ChatPromptTemplate.from_template(template.replace("{company_name}", company_name))
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

# Function to load all databases
def load_all_databases_compliance(database_dir):
    """Loads all available databases in the directory."""
    retrieval_chains = {}
    database_components = {}
    
    if not os.path.exists(database_dir):
        print(f"Error: Database directory not found: {database_dir}")
        return retrieval_chains, database_components

    for folder_name in os.listdir(database_dir):
        folder_path = os.path.join(database_dir, folder_name)
        if os.path.isdir(folder_path):
            print(f"Found database directory: {folder_name}")
            components = load_database(database_dir, folder_name)
            if components:
                # Store the chain with the original display name from UI
                for company in ["Continuum Green Energy Limited", "Hexaware Technologies Limited", 
                              "WeWork India Management Limited", "Dr. Agarwal's Health Care Limited"]:
                    if company.lower().replace(" ", "_").replace(".", "") in folder_name.lower().replace(" ", "_").replace(".", ""):
                        chain = create_database_chain(components, company)
                        retrieval_chains[company] = chain
                        database_components[company] = components
                        print(f"Mapped {folder_name} to {company}")
                        break
                else:
                    # If no match found in the predefined list, use the directory name as is
                    chain = create_database_chain(components, folder_name)
                    retrieval_chains[folder_name] = chain
                    database_components[folder_name] = components
    
    return retrieval_chains, database_components

# Function to format the response
def format_response_compliance(response, user_query, company_name):
    """Formats the response from the database."""
    if not response or "cannot find information" in response.lower():
        return f"Based on {company_name}'s available documents, I cannot find specific information about '{user_query}'."
    
    return response

# Function to generate a compliance report and save to file
def generate_compliance_report(database_dir, company_name, output_file=None):
    """
    Generates a comprehensive compliance report for a company and saves it to a JSON file.
    """
    report = check_compliance(database_dir, company_name)
    
    if output_file:
        try:
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to {output_file}")
        except Exception as e:
            print(f"Failed to save report: {e}")
    
    return report

# Main function to test the code
if __name__ == "__main__":
    database_dir = "database"
    company = "WeWork India Management Limited"
    
    # Generate and display a compliance report
    report = generate_compliance_report(database_dir, company, "compliance_report.json")
    
    # Print summary
    print("\nCompliance Check Summary:")
    print(f"Company: {report['company']}")
    print(f"Date: {report['date']}")
    print(f"Sections Present: {report['summary']['sections_present']}/{report['headings_checked']}")
    print(f"Sections Followed: {report['summary']['sections_followed']}/{report['headings_checked']}")
    print(f"Compliance Score: {report['summary']['compliance_score']}%")