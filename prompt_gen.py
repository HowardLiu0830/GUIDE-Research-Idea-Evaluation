import os
import re
import json
from typing import Dict, List
from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
import argparse
from huggingface_hub import hf_hub_download
import tempfile


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate paper reviews using LLM')
    parser.add_argument('--openai_key', required=True, help='OpenAI API key')
    parser.add_argument('--paper_path', required=True, help='Path to paper data JSON')
    parser.add_argument('--output_path', required=True, help='Path for output JSON')
    parser.add_argument('--num_related', required=True, type=int, help='number of related papers to include per section')
    parser.add_argument('--start_idx', required=True, type=int, help='start index for paper processing')
    parser.add_argument('--end_idx', required=True, type=int, help='end index for paper processing')
    
    # Replace multiple flags with comma-separated section lists
    parser.add_argument('--paper_sections', type=str, default='', 
                        help='Comma-separated list of paper sections to include (e.g., "introduction,method,conclusion")')
    parser.add_argument('--search_types', type=str, default='abstract,contribution,method,experiments',
                        help='Comma-separated list of sections to search by (e.g., "abstract,contribution,method,experiments")')
    
    return parser.parse_args()

def download_hf_database(repo_id: str, local_dir: str = None):
    if local_dir is None:
        local_dir = tempfile.mkdtemp(prefix="hf_database_")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        
        for file_path in repo_files:
            hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
        
        print(f"Database downloaded to: {local_dir}")
        return local_dir
    except Exception as e:
        print(f"Error downloading database from {repo_id}: {e}")
        return None
    

def load_data(path: str) -> List[str]:
    paper_list = []
    with open(path, 'r') as f:
        data = json.load(f)
    for paper in data:
        paper_list.append(paper)
    return paper_list


def user_prompt_gen(paper, database_map, args=None):
    # Parse section sets - use lists instead of sets to maintain order
    paper_sections = [section.strip() for section in args.paper_sections.split(',') if section.strip()]
    search_types = [section.strip() for section in args.search_types.split(',') if section.strip()]
    
    # Debug information
    print(f"Paper sections to include: {paper_sections}")
    print(f"Section types to search by: {search_types}")
    
    # Define the canonical order of sections
    section_order = ["introduction", "related_work", "background", "method", "experiments", "results", "discussion", "conclusion"]
    
    # Sort the sections according to the canonical order
    def sort_by_canonical_order(sections):
        return sorted(sections, key=lambda x: section_order.index(x) if x in section_order else len(section_order))
    
    # Sort the sections
    paper_sections = sort_by_canonical_order(paper_sections)
    
    if "summary" in paper:
        print(f"Available paper summary keys: {list(paper['summary'].keys())}")
    else:
        print("Warning: Paper does not have a 'summary' key")
    
    # Build the prompt - start with the target paper details
    prompt = f"""
    Here is the target idea's title, abstract, contribution and method&experiments overview:
    **Paper Title**: {paper["title"]}
    **Abstract**: {paper["abstract"]}
    **Contribution**: {paper["contribution"]}
    """
    
    # Add the paper's own summary sections based on the paper_sections set
    if "summary" in paper and isinstance(paper["summary"], dict) and paper_sections:
        prompt += "\n**Paper Sections**:\n"
        summary = paper["summary"]
        
        # Only include sections specified in paper_sections
        sections_added = False
        for section in paper_sections:
            if section in summary:
                prompt += f"{section.capitalize()}: {summary.get(section, 'N/A')}\n"
                sections_added = True
                print(f"Added section '{section}' from paper summary")
            else:
                print(f"Warning: Section '{section}' not found in paper summary")
        
        if not sections_added:
            print("Warning: No requested sections were found in paper summary")
    
    # Map of section names to database keys and field locations
    section_db_map = {
        "abstract": {"db_key": "abstract_db", "field_path": None},  # Direct field access
        "contribution": {"db_key": "contribution_db", "field_path": None},  # Direct field access
        "method": {"db_key": "method_db", "field_path": "summary"},  # In paper["summary"]["method"]
        "experiments": {"db_key": "experiment_db", "field_path": "summary"}  # In paper["summary"]["experiments"]
    }
    
    # Dictionary to store the related papers for each section
    related_papers_by_section = {}
    
    # Query each database for the most similar papers
    for section in search_types:
        if section in section_db_map and section_db_map[section]["db_key"] in database_map:
            db = database_map[section_db_map[section]["db_key"]]
            if db is None:
                print(f"Warning: Database for {section} is not available, skipping...")
                continue
            
            # Get the query text based on the field path
            field_path = section_db_map[section]["field_path"]
            if field_path is None:
                # Direct field (abstract, contribution)
                query_text = paper.get(section, "")
            else:
                # Nested field (in summary dict)
                summary = paper.get(field_path, {})
                query_text = summary.get(section, "")
            
            if query_text:
                # Find related papers based on section content
                similar_papers = db.similarity_search(
                    query_text, 
                    filter={"year": {"$lte": 2024}}, 
                    k=args.num_related
                )
                related_papers_by_section[section] = similar_papers
                print(f"Found {len(similar_papers)} related papers based on {section}")
            else:
                print(f"Warning: No {section} content found in the paper to use for similarity search")
    
    # Descriptions for each section category
    section_descriptions = {
        "abstract": "Below are the abstracts of key related works, outlining each study's scope and main findings. Use this section to evaluate the new paper's novelty and contributions.",
        "contribution": "Below are the key contributions of selected prior works, highlighting their novel ideas and advancements. Use this section to benchmark the new paper's originality and impact.",
        "method": "Below are the methods of key related works, summarizing their technical approaches and algorithms. Use this section to assess the proposed idea's technical novelty, contribution, and soundness.",
        "experiments": "Below are the experimental setups of key prior works, detailing their protocols and evaluation metrics. Use this section to judge whether the proposed experiments are sufficiently sound to support the paper's claims."
    }
    
    # Add related papers to the prompt, grouped by section type
    for section_type in ["abstract", "contribution", "method", "experiments"]:
        if section_type in related_papers_by_section and related_papers_by_section[section_type]:
            papers = related_papers_by_section[section_type]
            prompt += f"\n\n{section_descriptions[section_type]}\n\n"
            
            for i, doc in enumerate(papers, start=1):
                metadata = doc.metadata
                prompt += f"*Related work {i}*: *Title*: {metadata.get('title', 'N/A')} \n"
                
                # Get content for this section
                if section_type in ["abstract", "contribution"]:
                    # Direct field
                    content = metadata.get(section_type, "N/A")
                else:
                    # Might be in the metadata directly or might need extraction
                    content = metadata.get(section_type, "N/A")
                
                prompt += f" *{section_type.capitalize()}*: {content}\n"
    
    prompt += f"""
    Please analyze the paper according to the structured format provided in the system instructions.

    **Return a valid JSON object following this format:**
    ```json
    {{
        "summary": "Overall, the idea ...(summarize the idea's different parts)",
        "comparison_with_previous_work": "Compared to previous works, ...(compare the idea with previous works)",
        "Novelty": "...(Evaluate the novelty/Originality of the idea)",
        "Significance": "...(Evaluate the contribution/significance of the idea)",
        "Soundness": "...(Evaluate the rigor/soundness of the idea)",
        "strengths": "The strengths of the idea are ...(State the strengths of the idea in detail)",
        "weaknesses": "The weaknesses of the idea are ...(State the weaknesses of the idea in detail)",
        "Evaluation": "In conclusion, ... (Give an overall evaluation based on the above analysis)",
        "Suggestion": "To improve the idea, the authors could ... (Provide constructive suggestions)"
    }}
    ```
    **When mentioning a related work, please use the title of the related work.**
    **STRICTLY return only a valid JSON object, without explanations, extra text, or formatting like Markdown.**
    """

    return HumanMessage(content=prompt)


if __name__ == "__main__":
    args = parse_arguments()
    paper_path = args.paper_path
    paper_list = load_data(paper_path)

    openai_api_key = args.openai_key
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
    
    # Create a map of databases
    repo_mappings = {
        "abstract_db": "ResearchAgent-GUIDE/ICLR_abstract",
        "contribution_db": "ResearchAgent-GUIDE/ICLR_contribution", 
        "method_db": "ResearchAgent-GUIDE/ICLR_method",
        "experiment_db": "ResearchAgent-GUIDE/ICLR_experiment"
    }
    database_map = {}
    for db_name, repo_id in repo_mappings.items():
        print(f"Setting up {db_name}...")
        db_path = download_hf_database(repo_id)
        if db_path:
            try:
                database_map[db_name] = Chroma(
                    persist_directory=db_path,
                    embedding_function=embeddings
                )
                print(f"✓ {db_name} loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load {db_name}: {e}")
                database_map[db_name] = None
        else:
            database_map[db_name] = None

    selected_papers = paper_list
    
    response_schemas = [
        ResponseSchema(name="summary", description="The summary of the paper"),
        ResponseSchema(name="comparison_with_previous_work", description="Comparison with prior work and how it informs the novelty/contribution"),
        ResponseSchema(name="Novelty", description="Evaluation of the paper's novelty and originality"),
        ResponseSchema(name="Significance", description="Evaluation of the paper's contribution and significance"),
        ResponseSchema(name="Soundness", description="Evaluation of the paper's rigor and soundness"),
        ResponseSchema(name="strengths", description="The strengths of the paper"),
        ResponseSchema(name="weaknesses", description="The weaknesses of the paper"),
        ResponseSchema(name="Evaluation", description="The overall evaluation of the paper given the above analysis"),
        ResponseSchema(name="Suggestion", description="Constructive suggestions for how the authors could improve the paper"),

    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    system_message = SystemMessage(content="""You are a professional idea evaluator with expertise in machine learning.
    Your task is to evaluate a given target academic idea step by step, with a focus on novelty, significance and contribution.
    You will be given:  
    1. The idea's title, abstract, claimed contribution and section summaries.
    2. A set of relevant prior works, each with abstract, contribution statements, method descriptions and experimental setups.
	**Review Guidelines**
    Read the idea: It's important to carefully read through the given content, and to look up any related work and citations that will help you comprehensively evaluate it. Be sure to give yourself sufficient time for this step.
    **Evaluation Criteria**
    1. Motivation / Objective: What is the goal of the idea? Is it to better address a known application or problem, draw attention to a new application or problem, or to introduce and/or explain a new theoretical finding? A combination of these? Different objectives will require different considerations as to potential value and impact. Is the approach well motivated, including being well-placed in the literature?
    2. Novelty & Originality: Are the tasks or methods new? Is the work a novel combination of well-known techniques? (This can be valuable!) Is it clear how this work differs from previous contributions?
    3. Significance & Contribution: Are the questions being asked important? Does the submission address a difficult task in a better way than previous work? Would researchers or practitioners likely adopt or build on these ideas?
    4. Soundness: Can the proposed method and experiment setup properly substantiate the claimed contributions? Will the claims be well supported under the proposed experiment setup? Are the methods used appropriate? Is this a complete piece of work or work in progress?
    **Related-Works Usage**
    1. **Abstract&Contribution**: frame the problem, scope, and high‑level "what" and "why." Used for evaluating significance and novelty.
    2. **Method**: describe "how" (algorithms, architectures and theoretical derivations). Ssed for checking whether the proposed method is novel or internally consistent, well‑justified, and mathematically rigorous. 
    3. **Experiment setup**: specify experiment design, datasets, baselines, metrics. Used to evaluate whether this work's experiment is appropriately designed and whether the experiment is comprehensive enough to support the claims. This content may also contain expected results.  
    **Criticality**
    Noting the idea will become a paper submitted to top conferences with acceptance rate of 30%, you should be more critical. Feel free to give negative evaluation if the idea's quality is poor.
    For empirical works, there is no need to contain theoretical analysis. For theoretical works, there is no need to contain experimental. Do not give negative evaluation for the two cases.
    **Output Format**  
    Provide a structured evaluation **strictly in valid JSON format**. Include both an overall evaluation and constructive suggestions for improvement. Do not add explanations, extra text, or Markdown formatting.
    When mentioning a related work, please use the title of the related work.
    """)
    
    for i in range(args.start_idx, args.end_idx): 
        paper = selected_papers[i]
        print(f"Processing Paper {i+1}...")

        user_message = user_message = user_prompt_gen(paper, database_map, args)
        custom_id = f"ICLR2024-{i+1}"
        request = {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": system_message.content + '\n' + user_message.content}],"max_completion_tokens": 3000, "temperature": 0.6}}
        with open(args.output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(request, ensure_ascii=False) + '\n')



        