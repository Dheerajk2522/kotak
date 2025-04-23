# --- Disable File Watcher (Attempt to mitigate torch error) ---
import os
os.environ["STREAMLIT_SERVER_ENABLE_WATCHER"] = "false"
# --- End Disable File Watcher ---

import streamlit as st
import sys
import time
import logging
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid, GridOptionsBuilder
import pandas as pd
from web import search_perplexity_api, setup_perplexity_api
import os
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Import query functions ---
try:
    from query5d import (
        load_all_databases,
        query_specific_database,
        format_response
    )
    logger.info("Successfully imported query5d.py")
except ImportError as e:
    error_msg = f"Failed to import 'query5d.py'. Error: {str(e)}"
    logger.error(error_msg)
    st.error(error_msg)
    st.stop()

try:
    from observations import (
        load_all_databases_observation,
        query_specific_database_observation,
        format_response_observation
    )
    logger.info("Successfully imported observations.py")
except ImportError as e:
    error_msg = f"Failed to import 'observations.py'. Error: {str(e)}"
    logger.error(error_msg)
    st.error(error_msg)
    st.stop()
    
try:
    from compliance2 import (
        load_all_databases_compliance,
        query_specific_database_compliance,
        format_response_compliance
    )
    logger.info("Successfully imported compliance.py")
except ImportError as e:
    error_msg = f"Failed to import 'compliance.py'. Error: {str(e)}"
    logger.error(error_msg)
    st.error(error_msg)
    st.stop()




# Load environment variables
load_dotenv()

# Initialize Perplexity API (add this after your RAG initialization)
api_key = st.secrets["PERPLEXITY_API_KEY"]
if api_key:
    base_url, headers = setup_perplexity_api(api_key)
else:
    st.warning("Perplexity API key not found. Web search functionality will be limited.")
    base_url, headers = None, None

api_key1 =  st.secrets["OPENAI_API_KEY"]
if not api_key1:
    st.error("OPENAI_API_KEY not found. Please set it in your environment variables or .env file.")
else:
    logger.info("OPENAI_API_KEY found")

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize RAG System ---
# @st.cache_resource(show_spinner="Connecting Document Databases...")
# def initialize_rag():
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     database_directory = os.path.join(base_dir, 'database')
#     logger.info(f"Looking for database directory at: {database_directory}")
    
#     # Log directory structure for debugging
#     if os.path.exists(database_directory):
#         logger.info(f"Database directory contents: {os.listdir(database_directory)}")
#     else:
#         logger.error(f"Database directory not found: {database_directory}")
#         return None, None, None, [], f"Database directory not found: {database_directory}"
    
#     if not os.listdir(database_directory):
#         logger.warning(f"Database directory '{database_directory}' is empty.")
#         return None, None, None, [], f"Database directory '{database_directory}' is empty."

#     try:
#         # Load all three types of databases
#         retrieval_chains = load_all_databases(database_directory)
#         retrieval_chains_observation = load_all_databases_observation(database_directory)
#         retrieval_chains_compliance, compliance_database_components = load_all_databases_compliance(database_directory)
        
#         logger.info(f"Loaded regular database chains: {list(retrieval_chains.keys())}")
#         logger.info(f"Loaded observation database chains: {list(retrieval_chains_observation.keys())}")
#         logger.info(f"Loaded compliance database chains: {list(retrieval_chains_compliance.keys())}")
        
#         if not retrieval_chains and not retrieval_chains_observation and not retrieval_chains_compliance:
#             logger.warning("No document databases loaded.")
#             return None, None, None, [], "No document databases were loaded successfully."

#         # Use the regular chains to determine company names
#         company_names = list(retrieval_chains.keys())
#         logger.info(f"Loaded databases for: {company_names}")
#         return retrieval_chains, retrieval_chains_observation, retrieval_chains_compliance, company_names, None
#     except Exception as e:
#         error_msg = f"Failed to initialize RAG system: {e}"
#         logger.error(error_msg)
#         import traceback
#         logger.error(traceback.format_exc())
#         return None, None, None, [], error_msg

# # Update the unpacking of return values
# retrieval_chains, retrieval_chains_observation, retrieval_chains_compliance, company_names, init_error = initialize_rag()
# rag_options = ["All Companies"] + company_names if company_names else ["No Companies Available"]
# logger.info(f"Available companies: {rag_options}")

@st.cache_resource(show_spinner="Connecting Document Databases...")
def initialize_rag():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    database_directory = os.path.join(base_dir, 'database')
    logger.info(f"Looking for database directory at: {database_directory}")
    
    # Log directory structure for debugging
    if os.path.exists(database_directory):
        logger.info(f"Database directory contents: {os.listdir(database_directory)}")
    else:
        logger.error(f"Database directory not found: {database_directory}")
        return None, None, None, None, [], f"Database directory not found: {database_directory}"
    
    if not os.listdir(database_directory):
        logger.warning(f"Database directory '{database_directory}' is empty.")
        return None, None, None, None, [], f"Database directory '{database_directory}' is empty."

    try:
        # Load all three types of databases
        retrieval_chains = load_all_databases(database_directory)
        retrieval_chains_observation = load_all_databases_observation(database_directory)
        
        # Capture both the retrieval chains and database components
        retrieval_chains_compliance, compliance_database_components = load_all_databases_compliance(database_directory)
        
        logger.info(f"Loaded regular database chains: {list(retrieval_chains.keys())}")
        logger.info(f"Loaded observation database chains: {list(retrieval_chains_observation.keys())}")
        logger.info(f"Loaded compliance database chains: {list(retrieval_chains_compliance.keys())}")
        
        if not retrieval_chains and not retrieval_chains_observation and not retrieval_chains_compliance:
            logger.warning("No document databases loaded.")
            return None, None, None, None, [], "No document databases were loaded successfully."

        # Use the regular chains to determine company names
        company_names = list(retrieval_chains.keys())
        logger.info(f"Loaded databases for: {company_names}")
        return retrieval_chains, retrieval_chains_observation, retrieval_chains_compliance, compliance_database_components, company_names, None
    except Exception as e:
        error_msg = f"Failed to initialize RAG system: {e}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, None, [], error_msg

# Update the unpacking of return values
retrieval_chains, retrieval_chains_observation, retrieval_chains_compliance, compliance_database_components, company_names, init_error = initialize_rag()
rag_options = ["All Companies"] + company_names if company_names else ["No Companies Available"]
logger.info(f"Available companies: {rag_options}")



# --- Session State Initialization ---
default_page = "Company IPO"
if "current_page" not in st.session_state:
    st.session_state.current_page = default_page
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = { 
        "Company IPO": [], 
        "Observations": [],
        "Compliance check": [], 
        "Web search": []  
    }
if "selected_company" not in st.session_state:
    st.session_state.selected_company = rag_options[0] if rag_options else None
if "research_tab" not in st.session_state:
    st.session_state.research_tab = "Reference Data"  # Default to Reference Data tab

# --- Helper Functions ---
def set_page(page_name):
    st.session_state.current_page = page_name

def set_research_tab(tab_name):
    st.session_state.research_tab = tab_name

def get_company_data():
    return {
        "Continuum Green Energy Limited": {
            "1) List of Objects of the Issue": [
                "Repayment or prepayment of certain borrowings availed by the Company and its subsidiaries.",
                "Funding capital expenditure requirements for the development of renewable energy projects.",
                "General corporate purposes, subject to applicable laws."
            ],
            "2) Whether Proforma Financial Statements included?": "Yes, Proforma Financial Statements are included in the Draft Red Herring Prospectus of Continuum Green Energy Limited.",
            "3) Name of Statutory Auditors": "Deloitte Haskins & Sells LLP — Chartered Accountants (Current statutory auditors of the Company)",
            "4) Whether Regulation 6(1) or 6(2) IPO": "The IPO of Continuum Green Energy Limited is being made under Regulation 6(2) of the SEBI ICDR Regulations.",
            "5) Book Running Lead Managers (BRLMs) to the Issue": [
                "Kotak Mahindra Capital Company Limited",
                "Ambit Private Limited",
                "Citigroup Global Markets India Private Limited",
                "JM Financial Limited"
            ],
            "6) Legal Counsels to the Issue": [
                "Indian Legal Counsel to the Company: Trilegal",
                "International Legal Counsel to the Company: Herbert Smith Freehills LLP",
                "Indian Legal Counsel to the BRLMs: Shardul Amarchand Mangaldas & Co.",
                "International Legal Counsel to the BRLMs: Sidley Austin LLP"
            ],
            "7) Name of the Industry Agency": "CRISIL Market Intelligence & Analytics (CRISIL MI&A), a division of CRISIL Limited",
            "8) Names of the Board of Directors": [
                "Shailesh Vishnubhai Haribhakti - Chairperson & Non-Executive Independent Director",
                "Arvind Bansal - Whole-time Director & CEO",
                "Nandiwada Venkatesan Venkataramanan - Whole-time Director & COO",
                "Vikash Saraf - Non-Executive Director",
                "Kumar Tushar - Non-Executive Director",
                "Raja Parthasarathy - Non-Executive Director",
                "Mohit Batra - Non-Executive Independent Director",
                "Purvi Sheth - Non-Executive Independent Director",
                "Girija Krishan Varma - Non-Executive Independent Director"
            ],
            "9) Names of the Promoters": [
                "Arvind Bansal",
                "Vikash Saraf",
                "Continuum Green Energy Holdings Limited (CGEHL)",
                "Continuum Energy Pte. Ltd. (CEPL)",
                "Clean Joules Pte. Ltd. (CJPL)",
                "Starlight Pacific Ventures Pte. Ltd. (SPVPL)"
            ],
            "10) Whether there is Shareholder Reservation": "Yes, for Eligible Shareholders of CGEHL, one of the corporate promoters.",
            "11) Whether there is Employee Reservation": "Yes, for eligible employees, with a discount offered on the equity shares.",
            "12) List of Key Performance Indicators (KPIs)": [
                "Revenue from Operations",
                "EBITDA",
                "EBITDA Margin (%)",
                "Profit for the Year",
                "ROCE (%) – Return on Capital Employed",
                "Net Debt to EBITDA (x)",
                "Debt Equity Ratio",
                "Cash Flow from Operations",
                "Installed Capacity (MW)",
                "PLF (%) – Plant Load Factor",
                "Revenue per MW"
            ]
        },

        "Hexaware Technologies Limited": {
            "1) List of Objects of the Issue": [
                "To carry out the Offer for Sale of up to ₹99,500 million by the Promoter Selling Shareholder, CA Magnum Holdings.",
                "To achieve the benefits of listing the Equity Shares on the Stock Exchanges, which include: o	Enhancing visibility and brand image o	Providing a public market for the Equity Shares in India."
            ],
            "2) Whether Proforma Financial Statements included?": "No, Proforma Financial Statements are not included. Instead, the Draft Red Herring Prospectus of Hexaware Technologies Limited includes Restated Consolidated Financial Information",
            "3) Name of Statutory Auditors": "BSR & Co. LLP — Chartered Accountants",
            "4) Whether Regulation 6(1) or 6(2) IPO": "The IPO is made under Regulation 6(1)of SEBI ICDR Regulations.",
            "5) Book Running Lead Managers to the Issue": [
                "Kotak Mahindra Capital Company Limited",
                "JM Financial Limited",
                "CitiGroup Global Markets India Private Limited",
                "HSBC Securities and Capital Markets (India) Private Limited",
                "IIFL Securities Limited"
            ],
            "6) Legal Counsels to the Issue": [
                "Shardul Amarchand Mangaldas & Co – Indian Legal Counsel",
                "Herbert Smith Freehills LLP – International Legal Counsel",
                "ZeusIP Advocates LLP – Intellectual Property Consultant",
                "MMJB & Associates LLP – Company Secretarial Compliance Advisor"
            ],
            "7) Name of the Industry Agency": "The industry report was prepared by Everest Business Advisory India Private Limited (Everest Group).",
         "8) Names of the Board of Directors": [
            "R Srikrishna – Executive Director and CEO",
            "Joseph McLaren Quinlan - Non-Executive Independent Director and Chairman",
            "Sandra Joy Horbach - Non-Executive Non-Independent Director",
            "Julius Michael Genachowski - Non-Executive Non-Independent Director",
            "Lucia De Fatima Soares - Non-Executive Non-Independent Director",
            "Kapil Modi - Non-Executive Non Independent Director",
            "Shawn Albert Devilla - Non-Executive Non Independent Director",
            "Milind Shripad Sarwate - Non-Executive Independent Director",
            "Vivek Sharma - Non-Executive Independent Director",
            "Sukanya Kripalu - Non-Executive Independent Director",
            "Neeraj Bhargava – Non-Executive Non Independent Director"
            ],
            "9) Names of the Promoters": [
            "HT Global IT Solutions Holdings Limited",
            "CA Magnum Holdings",
            "CA Ancona Investments",
            "CA Houdan Investments",
            "CA Sebright Investments",
            "CA Silkie Investments",
            "Hexaware Global Limited"
            ],
            "10) Whether there is Shareholder Reservation": "No",
            "11) Whether there is Employee Reservation": "Yes, with a discount offered to eligible employees.",
           "12) List of Key Performance Indicators (KPIs)": [
            "Revenue from Operations",
            "Revenue from operations growth",
            "Revenue by Geography",
            "Revenue from Verticals",
            "Revenue by IT and BPS and others",
            "Revenue by onshore, offshore IT Services",
            "Revenue by customer group",
            "Client Pyramid",
            "EBITDA",
            "EBITDA Margin (%)",
            "Adjusted EBITDA",
            "Adjusted EBITDA Margin (in %)"
        ]
        },

        "WeWork India Management Limited": {
            "1) List of Objects of the Issue": [
                "To carry out the Offer for Sale of up to 43,753,952 Equity Shares.",
                "To achieve the benefits of listing the Equity Shares on the Stock Exchanges."
            ],
            "2) Whether Proforma Financial Statements included?": "Yes, Restated Summary Statements have been included.",
            "3) Name of Statutory Auditors": "S.R. Batliboi & Associates LLP, Chartered Accountants",
            "4) Whether Regulation 6(1) or 6(2) IPO": "The IPO is in accordance with Regulation 6(2) of the SEBI ICDR Regulations.",
            "5) Book Running Lead Managers to the Issue": [
                "JM Financial Limited",
                "ICICI Securities Limited",
                "Jefferies India Private Limited",
                "Kotak Mahindra Capital Company Limited",
                "360 ONE WAM Limited"
            ],
            "6) Legal Counsels to the Issue": "Legal counsels to the Company are Shardul Amarchand Mangaldas & Co.",
            "7) Name of the Industry Agency": "CBRE South Asia Private Limited (CBRE) and AGR Knowledge Services Private Limited (AGR) are the named industry agencies responsible for the industry reports used in the DRHP..",
            "8) Names of the Board of Directors": [
                "Jitendra Mohandas Virwani - Chairman and Non-executive Director",
                "Karan Virwani - Managing Director and CEO",
                "Adnan Mostafa Ahmad - Non-executive Nominee Director",
                "Manoj Kumar Kohli - Independent Director",
                "Mahua Acharya - Independent Director",
                "Anupa Rajiv Sahney - Independent Director"
            ],
            "9) Names of the Promoters": [
                "Jitendra Mohandas Virwani",
                "Karan Virwani",
                "Embassy Buildcon LLP"
            ],
            "10) Whether there is Shareholder Reservation": "No, there is no specific reservation for existing shareholders in the Offer.",
            "11) Whether there is Employee Reservation": "Yes, there is an explicit Employee Reservation Portion in the Offer.",
            "12) List of Key Performance Indicators (KPIs)": [
                "Total income",
                "Total income growth (%)",
                "Revenue from Operations",
                "Revenue from Operations growth (%)",
                "EBITDA",
                "EBITDA margin (%)",
                "Total Equity",
                "Total Assets",
                "Net Debt",
                "Adjusted Capital Employed",
                "Return on Adjusted Capital Employed (%)",
                "Cities (number of cities with geographic presence)",
                "Total Centers",
                "Total Leasable Area",
                "Total Desks Capacity in all Centres",
                "Operational Centres",
                "Renewal Rate",
                "Net Average Revenue per Member / Billed Desk (ARPM)",
                "Revenue to Rent Multiple"
            ]
        },

        "Dr. Agarwal's Health Care Limited": {
            "1) List of Objects of the Issue": [
                "Repayment/prepayment, in part or full, of certain borrowings",
                "General corporate purposes and unidentified inorganic acquisition"
            ],
            "2) Whether Proforma Financial Statements included?": "Yes, Unaudited Pro forma Condensed Combined Financial Information is included.",
            "3) Name of Statutory Auditors": "Deloitte Haskins & Sells, Chartered Accountants",
            "4) Whether Regulation 6(1) or 6(2) IPO": "Regulation 6(1) of SEBI ICDR Regulations (100% Book Built Offer)",
            "5) Book Running Lead Managers to the Issue": [
                "Kotak Mahindra Capital Company Limited",
                "Morgan Stanley India Company Private Limited",
                "Jefferies India Private Limited",
                "Motilal Oswal Investment Advisors Limited"
            ],
            "6) Legal Counsels to the Issue": [
                "Cyril Amarchand Mangaldas",
                "KFin Technologies Limited"],
            
            "7) Name of the Industry Agency": [
                "National Accreditation Board for Hospitals and Healthcare Providers (NABH)",
                "National Accreditation Board for Testing and Calibration Laboratories (NABL)",
                "Insurance Regulatory and Development Authority of India (IRDAI)",
                "Indian Council of Medical Research (ICMR)",
                "Indian Nursing Council (INC)",
                "Medical Councils (e.g., Medical Council of India / respective State Medical Councils)",
                "Food Safety and Standards Authority of India (FSSAI)"
            ],
            "8) Names of the Board of Directors": [
                "Venkatraman Balakrishnan",
                "Ankur Nand Thadani",
                "Ved Prakash Kalanoria",
                "Sanjay Dharambir Anand",
                "Nachiket Madhusudhan Mor",
                "Dr. Ranjan Ramdas Pai",
                "Archana Bhaskar"
            ],
          "9) Names of the Promoters": [
                "Dr. Amar Agarwal",
                "Dr. Athiya Agarwal",
                "Dr. Adil Agarwal",
                "Dr. Anosh Agarwal",
                "Dr. Ashvin Agarwal",
                "Dr. Ashar Agarwal",
                "Dr. Amar Agarwal Family Trust",
                "Dr. Adil Agarwal Family Trust",
                "Dr. Anosh Agarwal Family Trust"
            ],
            "10) Whether there is Shareholder Reservation": "Yes, reservation for Eligible AEHL Shareholders.",
            "11) Whether there is Employee Reservation": "Yes, with a discount of ₹27 per Equity Share.",
            "12) List of Key Performance Indicators (KPIs)": [
                "Revenue from Emerging Facilities",
                "Revenue from Mature Facilities",
                "Revenue from operations",
                "Revenue growth",
                "Revenue from operations – India",
                "EBITDA & EBITDA margin & growth",
                "Restated profit for the year & margin",
                "Return on Equity"
            ]
        }
    }


def get_observations_data():

    return {

        "Hexaware Technologies Limited": {

            "In the CCPS table wherein additional details of ratio, etc. are now being included, please include a column for the total amount raised under each Series": "NO",

            "With respect to the response for query 4(d) of the SEBI Clarification Letter, please provide all the data in place of the placeholders for Risk Factor 2.": "NO",

            "With respect to the response for query 4(m) of the SEBI Clarification Letter, please provide the data in place of the placeholders for the table included in Risk Factor 35.": "NO",

            "Confirmation in the DRHP that all applicable lender consents have been obtained.": "YES",

            "In relation to 5 GWh project, what is the amount of loan disbursed by SBI as on December 31, 2023 out of the sanctioned amount of Rs. 1,910 crore.": "NO",

            "With respect to the response for query 5 of the SEBI Clarification Letter, please disclose the split of organic growth object shown under the 3 separate heads along with amounts in the DRHP.": "NO",

            "Risk Factor 1: LM is advised to modify the heading of the Risk Factor to add that the issuer company has incurred losses and negative cash flow from operating activities in all periods since inception.": "NO",

            "Risk Factor 2: LM is advised to divide the Risk Factor in two parts. The issues relating to defects and quality issues to be merged with Risk Factor 3. The issues relating to supply may be dealt with in this Risk Factor. Provide the distribution of raw materials between domestic supply and import. Top 5 import jurisdictions and their share to be provided.": "NO",

            "Risk Factor 4: LM is advised to disclose different phases and their estimated timelines in a tabular format along with material details": "NO",

            "Risk Factor 5: LM is advised to modify the heading of the Risk Factor to add that the reduction in government incentives may result in increase in retail price of the EVs. Further, disclose different PLI / incentive schemes, their period, targets and the application / claim status, in a tabular format.": "NO",

            "The promoter group companies are loss making. Include relevant details such as cash flow from operating activities etc.": "NO",

            "OCT, where the issue proceeds are proposed to be invested, was incorporated in FY 2023 and has continuously incurred losses since inception.": "NO",

            "LM is advised to provide the reasons for 7X growth in revenues in FY 2023 over FY 2022.": "NO",

            "With respect to losses incurred by the issuer Company in past three financial years, provide the year-wise mapping as to how these losses were financed through debt / fund raise.": "NO",

            "Provide the unit economics of any scooter model.": "NO"

        },

        "WeWork India Management Limited": {

            "To obtain confirmation from company that disclosures in the Offer document related to special rights of shareholders shall not form part of any notice to shareholders for approval of special rights post listing. The same shall also be submitted to the Stock Exchanges.": "NO",

            "To disclose the details of the pledged shares held by the promoters/promoter group in the Issuer Company / its subsidiaries.": "YES",

            "To delete all company specific data disclosed in the industry report and restrict to only disclosing industry overview. Company specific material disclosures may be incorporated in \"Our business\" section.": "NO",

            "To provide Special purpose audited financial statements of the company for the past FYs as part of MCMD.": "YES",

            "To ensure all valuation reports relied on by the company for accounting increase in fair value of NCCCPS are disclosed as part of MCMD.": "NO",

            "To substantiate the significant benefits from the acquisitions by sharing the financials/ revenue generated from the past acquisitions. To also confirm/ disclose the valuation report of the such acquisitions as material documents for inspections.": "YES",

            "To ensure estimated advertisement expenses disclosed in the Offer document have been certified by a chartered accountant.": "NO",

            "To disclose whether Board approvals have been obtained for each object of the issue.": "YES",

            "To disclose details of present and past Nominee Directors and Observers of various shareholders": "YES",

            "To remove content of similar nature, if any, from all section(s) of DRHP. \"None among our company, the selling shareholders or any member of the Syndicate shall be liable for any failure in (i) uploading the bids due to faults in any software/hardware system or otherwise; or (ii) the blocking of Bid Amount in the ASBA Account on receipt of instructions from the Sponsor Bank on account of any errors, omissions or non-compliance by various parties involved in, or any other fault, malfunctioning or breakdown in, or otherwise, in the UPI Mechanism\".": "NO",

            "To delete details of all legal advisers other than legal advisors to the Issuer": "NO",

            "Definitions and Abbreviations- for Technical, Company / Industry related Terms or Abbreviations, along with the expanded form, suitable meaning / explanation to be provided in simple language": "YES",

            "Where mode of acquisition of equity shares is disclosed as \"Transfer\". To specify name of transferor.": "YES"

        },

        "Dr. Agarwal's Health Care Limited": {

            "LM is advised to disclose whether any action has been taken / is pending against the promoter / promoter group/ director of the issuer / Group Companies etc. by any regulatory authority in India or overseas.": "YES",

            "LM is advised to submit whether there has been any instance of issuance of equity shares in the past by the issuer Company, the Group Companies or entities forming part of the Promoter Group to more than 49/200 investors in violation of:": "YES",

            "Section 67(3) of Companies Act, 1956; elevant section(s) of Companies Act, 2013, including Section 42 and the rules notified thereunder; the SEBI Regulations; the SEBI (Disclosure and Investor Protection) Guidelines, 2000, as applicable.": "YES",

            "It has been observed that in various instances disclosures have been made in the offer document stating 'we believe...' LM is advised to provide the basis for making such disclosures in the offer document while also explaining compliance with Regulation 24 (1) and Regulation 25 (2) (b) of SEBI (Issue of Capital and Disclosure Requirements) Regulations, 2018 (\"ICDR Regulations\").": "YES",

            "LM is advised to ensure that all the cross references given in the offer document are correct, leading to the exact page, instead of referring to the beginning of the section. LM may provide the exact risk factor number instead of giving cross referencing of the page no., in all the references in the document.": "YES",

            "LM is advised to confirm and explain compliance with disclosures in the front cover page of the offer document as specified vide circular no. SEBI/HO/CFD/SSEP/CIR/P/2022/14 dated February 4,2022.": "YES",

            "LM is advised to rearrange the risk factors based on materiality. LM is also advised to ensure that risk factors are concise and precise.": "YES",

            "LM is advised to add a separate risk factor stating the fact of debit balance in Profit and Loss account/ Accumulated losses in the top 10 risk factor.": "NO",

            "LM is advised to add a separate risk factor associated with development of new products strategies.": "NO",

            "LM is advised to include a separate risk factor related to dependencies on individual and corporate agents.": "NO",

            "LM is advised to include separate risk factor with respect to complaints pending against the company.": "YES",

            "LM is advised to include a separate risk factor with respect to financial rations compared to listed peers as disclosed on Pg. 110.": "YES",

            "LM is advised to include a separate risk factor on financial and operating ratios of the company.": "NO",

            "LM is advised to include a separate risk factor on impact of various government insurance scheme on the business of the Company.": "YES",

        }

    }
 
def get_regulations_data():

    return {
        "CONTINUUM GREEN ENERGY LIMITED": {
        "Cover Pages": "No",
        "Definitions and Abbreviations": "Yes",
        "Offer Document Summary": "Yes",
        "Risk Factors": "Yes",
        "Introduction": "No",
        "General Information": "Yes",
        "Capital Structure": "Yes",
        "Particulars of the Issue": "Yes",
        "Legal and Other Information": "Yes",
        "Information with Respect to Group Companies": "Yes",
        "Other Regulatory and Statutory Disclosures": "Yes",
        "Offering Information": "Yes",
        "Description of Equity Shares and Terms of the Articles of Association": "Yes",
        "Other Information": "Yes"
        },
        "Hexaware Technologies Limited": {
        "Cover Pages": "No",
        "Definitions and Abbreviations": "Yes",
        "Offer Document Summary": "Yes",
        "Risk Factors": "Yes",
        "Introduction": "No",
        "General Information": "Yes",
        "Capital Structure": "Yes",
        "Particulars of the Issue": "No",
        "Legal and Other Information": "Yes",
        "Information with Respect to Group Companies": "Yes",
        "Other Regulatory and Statutory Disclosures": "Yes",
        "Offering Information": "Yes",
        "Description of Equity Shares and Terms of the Articles of Association": "Yes",
        "Other Information": "Yes"
        },
        "WeWork India Management Limited": {
        "Cover Pages": "No",
        "Definitions and Abbreviations": "Yes",
        "Offer Document Summary": "Yes",
        "Risk Factors": "Yes",
        "Introduction": "Yes",
        "General Information": "Yes",
        "Capital Structure": "Yes",
        "Particulars of the Issue": "No",
        "Legal and Other Information": "Yes",
        "Information with Respect to Group Companies": "Yes",
        "Other Regulatory and Statutory Disclosures": "Yes",
        "Offering Information": "Yes",
        "Description of Equity Shares and Terms of the Articles of Association": "Yes",
        "Other Information": "Yes"
        },
        "Dr. Agarwal's Health Care Limited": {
        "Cover Pages": "No",
        "Definitions and Abbreviations": "Yes",
        "Offer Document Summary": "Yes",
        "Risk Factors": "Yes",
        "Introduction": "Yes",
        "General Information": "Yes",
        "Capital Structure": "Yes",
        "Particulars of the Issue": "Yes",
        "Legal and Other Information": "Yes",
        "Information with Respect to Group Companies": "Yes",
        "Other Regulatory and Statutory Disclosures": "Yes",
        "Offering Information": "Yes",
        "Description of Equity Shares and Terms of the Articles of Association": "Yes",
        "Other Information": "Yes"
        }
    }

# --- Sidebar Navigation ---    
with st.sidebar:
    # Logo
    logo_path = os.path.join(os.path.dirname(__file__), 'images.png')
    if os.path.exists(logo_path):
         st.image(logo_path, width=150)
    else:
         st.markdown("### Intelligent Assistant")

    st.markdown("---", unsafe_allow_html=True)

    # Sidebar Option Menu
    selected_option = option_menu(
        menu_title="",
        options=["Company IPO", "Observations", "Compliance check", "Web search"],
        icons=["building", "person-lines-fill","book", "search"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f0f2f6"},
            "icon": {"color": "Red", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#22709E"},
        }
    )
    # Store selected page in session_state
    st.session_state.current_page = selected_option
    st.markdown("""<style>
.st-emotion-cache-t1wise {
    padding-left: 5rem;
    padding-right: 5rem;
    padding-top: 2.5rem;
}

.st-emotion-cache-1voybx5 {
border: 1px solid rgba(49, 51, 63, 0.2);
border-radius: 0.5rem;
padding: calc(-1px + 1rem);
height: 450px;
overflow: auto;
}
</style>""", unsafe_allow_html=True)

    if init_error:
         st.error(f"RAG Init Failed:\n{init_error}")
         # Show additional debug information
         with st.expander("Debug Information"):
             st.write(f"Available database chains: {retrieval_chains}")
             st.write(f"Company names: {company_names}")
             base_dir = os.path.dirname(os.path.abspath(__file__))
             database_directory = os.path.join(base_dir, 'database')
             st.write(f"Database directory: {database_directory}")
             if os.path.exists(database_directory):
                 st.write(f"Database directory contents: {os.listdir(database_directory)}")
             else:
                 st.write("Database directory does not exist")

# --- Main Content Area ---
if st.session_state.current_page == "Company IPO":
    st.header("🏢 Company IPO")
    
    # Company selector at the top
    company_data = get_company_data()
    selected_company = st.selectbox(
        "Select Company:",
        options=list(company_data.keys()),
        index=0,
        key="company_selector"
    )
    
    # Create tabs within the Company Research page
    tab1, tab2 = st.tabs(["📋 Standard Questions", "🔍 Query Bot"])
    
    with tab1:
        st.subheader(f"Reference Data for {selected_company}")
        
        # Display data for selected company in a point-to-point table
        data = company_data[selected_company]
        
        # Create a DataFrame for the table view
        table_data = []
        for key, value in data.items():
            if isinstance(value, list):
                formatted_value = "\n".join([f"• {item}" for item in value])
            else:
                formatted_value = value
            
            table_data.append({
                "Question": key,
                "Answer": formatted_value
            })
        
        df = pd.DataFrame(table_data)
        
        # Configure grid options
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(
            wrapText=True,
            autoHeight=True,
            editable=False,
            resizable=True
        )
        gb.configure_column("Question", width=200, headerName="Question", pinned=True)
        gb.configure_column("Answer", width=600, headerName="Answer", wrapText=True,
                            cellStyle={'whiteSpace': 'pre-line'})
        
        grid_options = gb.build()
        
        # Display the AgGrid table
        AgGrid(
            df,
            gridOptions=grid_options,
            height=600,
            width='100%',
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            theme='streamlit'
        )
    
    # In your Company IPO tab2 section:
    with tab2:
        st.subheader(f"Query {selected_company} Documents")
        
        # # Debug information about available databases
        # with st.expander("Debug Information"):
        #     st.write(f"Available database chains: {list(retrieval_chains.keys()) if retrieval_chains else 'None'}")
        #     st.write(f"Selected company: {selected_company}")
        #     if retrieval_chains and selected_company in retrieval_chains:
        #         st.write(f"Database for '{selected_company}' is available")
        #     else:
        #         st.write(f"Database for '{selected_company}' is NOT available")
                
        #         # Check for close matches
        #         if retrieval_chains:
        #             st.write("Available databases:")
        #             for key in retrieval_chains.keys():
        #                 st.write(f"- {key}")
        
        # Chat display container
        chat_display_container = st.container(height=500)
        
        with chat_display_container:
            if "Company IPO" in st.session_state.chat_histories:
                for role, message in st.session_state.chat_histories["Company IPO"]:
                    with st.chat_message(role):
                        st.markdown(message)
        
        # Chat input
        user_input = st.chat_input(
            f"Ask about '{selected_company}'...",
            key="company_research_chat_input"
        )
        
        if user_input:
            # Append user message
            st.session_state.chat_histories["Company IPO"].append(("user", user_input))
            
            # Generate response
            with st.spinner("Searching documents..."):
                try:
                    if retrieval_chains and selected_company in retrieval_chains:
                        logger.info(f"Querying database for '{selected_company}'")
                        response = query_specific_database(
                            retrieval_chains[selected_company], 
                            user_input,company_name=selected_company
                        )
                        bot_response = format_response(response, user_input, selected_company)
                    else:
                        # Try to find a case-insensitive match
                        matched = False
                        if retrieval_chains:
                            for chain_key in retrieval_chains.keys():
                                if selected_company.lower() in chain_key.lower() or chain_key.lower() in selected_company.lower():
                                    logger.info(f"Found close match: '{chain_key}' for '{selected_company}'")
                                    response = query_specific_database(
                                        retrieval_chains[chain_key], 
                                        user_input,company_name=chain_key
                                    )
                                    bot_response = format_response(response, user_input, selected_company)
                                    matched = True
                                    break
                        
                        if not matched:
                            logger.warning(f"No database found for '{selected_company}'")
                            bot_response = f"Document database for '{selected_company}' is not available or not loaded properly."
                except Exception as e:
                    logger.error(f"Error querying database: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    bot_response = f"An error occurred: {str(e)}"
            
            # Append bot response
            st.session_state.chat_histories["Company IPO"].append(("assistant", bot_response))
            st.rerun()


elif st.session_state.current_page == "Observations":   
    st.header("📖 Observations")
    
    # Company selector at the top
    company_data = get_observations_data()
    selected_company = st.selectbox(
        "Select Company:",
        options=list(company_data.keys()),
        index=0,
        key="company_selector"
    )
    
    # Create tabs within the Company Research page
    tab1, tab2 = st.tabs(["📋 Standard observation", "🔍 Query Bot"])
        
    with tab1:
        st.subheader(f"Reference Data for {selected_company}")
        
        # Display data for selected company in a point-to-point table
        if selected_company in company_data:
            data = company_data[selected_company]
            
            # Create a DataFrame for the table view
            table_data = []
            for observation, response in data.items():
                table_data.append({
                    "Past Observations": observation,
                    "Yes or No": response
                })
            
            df = pd.DataFrame(table_data)
            
            # Configure grid options
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_default_column(
                wrapText=True,
                autoHeight=True,
                editable=False,
                resizable=True
            )
            gb.configure_column("Past Observations", width=600, headerName="Past Observations", pinned=True)
            gb.configure_column("Yes or No", width=100, headerName="Response", wrapText=True)
            
            grid_options = gb.build()
            
            # Display the AgGrid table
            AgGrid(
                df,
                gridOptions=grid_options,
                height=600,
                width='100%',
                fit_columns_on_grid_load=False,
                allow_unsafe_jscode=True,
                theme='streamlit'
            )
        else:
            st.error(f"No data found for {selected_company}")
    
    # In your Company IPO tab2 section:
    with tab2:
        st.subheader(f"Query {selected_company} Documents")
        
        # Chat display container
        chat_display_container = st.container(height=500)
        
        with chat_display_container:
            if "Observations" in st.session_state.chat_histories:
                for role, message in st.session_state.chat_histories["Observations"]:
                    with st.chat_message(role):
                        st.markdown(message)
        
        # Chat input
        user_input = st.chat_input(
            f"Ask about '{selected_company}'...",
            key="company_research_chat_input"
        )
        
        if user_input:
            # Append user message
            st.session_state.chat_histories["Observations"].append(("user", user_input))
            
            # Generate response
            with st.spinner("Searching documents..."):
                try:
                    if retrieval_chains_observation and selected_company in retrieval_chains_observation:
                        logger.info(f"Querying database for '{selected_company}'")
                        response = query_specific_database_observation(
                            retrieval_chains_observation[selected_company], 
                            user_input,company_name=selected_company
                        )
                        bot_response = format_response_observation(response, user_input, selected_company)
                    else:
                        # Try to find a case-insensitive match
                        matched = False
                        if retrieval_chains_observation:
                            for chain_key in retrieval_chains_observation.keys():
                                if selected_company.lower() in chain_key.lower() or chain_key.lower() in selected_company.lower():
                                    logger.info(f"Found close match: '{chain_key}' for '{selected_company}'")
                                    response = query_specific_database_observation(
                                        retrieval_chains_observation[chain_key], 
                                        user_input,company_name=chain_key
                                    )
                                    bot_response = format_response_observation(response, user_input, selected_company)
                                    matched = True
                                    break
                        
                        if not matched:
                            logger.warning(f"No database found for '{selected_company}'")
                            bot_response = f"Document database for '{selected_company}' is not available or not loaded properly."
                except Exception as e:
                    logger.error(f"Error querying database: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    bot_response = f"An error occurred: {str(e)}"
            
            # Append bot response
            st.session_state.chat_histories["Observations"].append(("assistant", bot_response))
            st.rerun()


elif st.session_state.current_page == "Compliance check":
    st.header("🧑‍💼 Compliance check")
    
    # Company selector at the top
    company_data = get_regulations_data()
    selected_company = st.selectbox(
        "Select Company:",
        options=list(company_data.keys()),
        index=0,
        key="company_selector_compliance"
    )
    
    # Create tabs within the Company Research page
    tab1, tab2 = st.tabs(["📋 Compliance check", "🔍 Query Bot"])
        
    with tab1:
        st.subheader(f"Reference Data for {selected_company}")
        
        # Display data for selected company in a point-to-point table
        if selected_company in company_data:
            data = company_data[selected_company]
            
            # Create a DataFrame for the table view
            table_data = []
            for regulations, response in data.items():
                table_data.append({
                    "Regulations": regulations,
                    "Yes or No": response
                })
            
            df = pd.DataFrame(table_data)
            
            # Configure grid options
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_default_column(
                wrapText=True,
                autoHeight=True,
                editable=False,
                resizable=True
            )
            gb.configure_column("Regulations", width=600, headerName="Regulations", pinned=True)
            gb.configure_column("Yes or No", width=100, headerName="Response", wrapText=True)
            
            grid_options = gb.build()
            
            # Display the AgGrid table
            AgGrid(
                df,
                gridOptions=grid_options,
                height=600,
                width='100%',
                fit_columns_on_grid_load=False,
                allow_unsafe_jscode=True,
                theme='streamlit'
            )
        else:
            st.error(f"No data found for {selected_company}")
    
    with tab2:
        st.subheader(f"Query {selected_company} Documents")
        
        # Chat display container
        chat_display_container = st.container(height=500)
        
        with chat_display_container:
            if "Compliance check" in st.session_state.chat_histories:
                for role, message in st.session_state.chat_histories["Compliance check"]:
                    with st.chat_message(role):
                        st.markdown(message)
        
        # Chat input
        user_input = st.chat_input(
            f"Ask about '{selected_company}'...",
            key="compliance_chat_input"
        )
        
        if user_input:
            # Append user message
            st.session_state.chat_histories["Compliance check"].append(("user", user_input))
            
            # Generate response
            with st.spinner("Searching documents..."):
                try:
                    if retrieval_chains_compliance and selected_company in retrieval_chains_compliance:
                        logger.info(f"Querying database for '{selected_company}'")
                        response = query_specific_database_compliance(
                            retrieval_chains_compliance[selected_company], 
                            user_input,company_name=selected_company
                        )
                        bot_response = format_response_compliance(response, user_input, selected_company)
                    else:
                        # Try to find a case-insensitive match
                        matched = False
                        if retrieval_chains_compliance:
                            for chain_key in retrieval_chains_compliance.keys():
                                if selected_company.lower() in chain_key.lower() or chain_key.lower() in selected_company.lower():
                                    logger.info(f"Found close match: '{chain_key}' for '{selected_company}'")
                                    response = query_specific_database_compliance(
                                        retrieval_chains_compliance[chain_key], 
                                        user_input,company_name=chain_key
                                    )
                                    bot_response = format_response_compliance(response, user_input, selected_company)
                                    matched = True
                                    break
                        
                        if not matched:
                            logger.warning(f"No database found for '{selected_company}'")
                            bot_response = f"Document database for '{selected_company}' is not available or not loaded properly."
                except Exception as e:
                    logger.error(f"Error querying database: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    bot_response = f"An error occurred: {str(e)}"
            
            # Append bot response - FIXED THE KEY HERE
            st.session_state.chat_histories["Compliance check"].append(("assistant", bot_response))
            st.rerun()


elif st.session_state.current_page == "Web search":
    st.header("🔍 Web Search")
    
    # Initialize chat history for web search if not exists
    if "Web search" not in st.session_state.chat_histories:
        st.session_state.chat_histories["Web search"] = []
    
    # # Chat display container
    # chat_display_container = st.container(height=500)
    
    # with chat_display_container:
    #     for role, message in st.session_state.chat_histories["Web search"]:
    #         with st.chat_message(role):
    #             st.markdown(message)
    
    # # Chat input
    # user_input = st.chat_input(
    #     "Enter a company name to search for litigations or SEBI actions...",
    #     key="web_search_chat_input"
    # )


    # Chat display container
    chat_display_container = st.container(height=500)

    with chat_display_container:
        # Add a custom wrapper with a class name
        st.markdown(
            "<div class='custom-chat-container'>",
            unsafe_allow_html=True
        )

        for role, message in st.session_state.chat_histories["Web search"]:
            with st.chat_message(role):
                st.markdown(message)

        st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input(
        "Enter a company name to search for litigations or SEBI actions...",
        key="web_search_chat_input"
    )

    # # Custom CSS scoped to the class
    # st.markdown(
    #     """
    #     <style>
    #     .custom-chat-container {
    #         background-color: #f0f4ff;
    #         padding: 1rem;
    #         border-radius: 10px;
    #         height: 100%;
    #         overflow-y: auto;
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )
    
    
    if user_input:
        # Append user message
        st.session_state.chat_histories["Web search"].append(("user", user_input))
        
        # Generate response
        with st.spinner("Searching the web..."):
            if base_url and headers:
                try:
                    result, _, _, _ = search_perplexity_api(
                        base_url, 
                        headers, 
                        user_input
                    )
                    
                    if result:
                        bot_response = result
                    else:
                        bot_response = "Sorry, I couldn't find any information about this company."
                except Exception as e:
                    bot_response = f"An error occurred during the web search: {str(e)}"
            else:
                bot_response = "Web search functionality is currently unavailable (missing API key)."
        
        # Append bot response
        st.session_state.chat_histories["Web search"].append(("assistant", bot_response))
        st.rerun()