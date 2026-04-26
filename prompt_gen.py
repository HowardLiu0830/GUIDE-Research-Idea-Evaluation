import os
import re
import json
from typing import Dict, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import argparse
from rouge_score import rouge_scorer



def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate paper advising prompts using LLM')
    parser.add_argument('--openai_key', required=True, help='OpenAI API key')
    parser.add_argument('--paper_path', required=True, help='Path to paper data JSON')
    parser.add_argument('--output_path', required=True, help='Path for output JSON')
    parser.add_argument('--num_related', required=True, type=int, help='number of related papers to include per section')
    parser.add_argument('--start_idx', required=True, type=int, help='start index for paper processing')
    parser.add_argument('--end_idx', required=True, type=int, help='end index for paper processing')
    parser.add_argument('--year_threshold', type=int, default=2023,
                        help='Only include related papers with year <= this value (default: 2023)')
    parser.add_argument('--rouge_threshold', type=float, default=0.5,
                        help='Drop retrieved papers whose abstract has ROUGE-L >= this score against the target abstract (default: 0.5)')

    # Replace multiple flags with comma-separated section lists
    parser.add_argument('--paper_sections', type=str, default='',
                        help='Comma-separated list of paper sections to include (e.g., "introduction,method,conclusion")')
    parser.add_argument('--search_types', type=str, default='abstract,contribution,method,experiments',
                        help='Comma-separated list of sections to search by (e.g., "abstract,contribution,method,experiments")')

    return parser.parse_args()


def filter_similar_abstracts(target_abstract: str, retrieved_docs, rouge_threshold: float):
    """Drop retrieved papers whose abstract is too similar to the target (ROUGE-L >= threshold)."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    filtered_docs = []

    print(f"Running ROUGE-L dedup, threshold: {rouge_threshold}")

    for doc in retrieved_docs:
        doc_abstract = doc.metadata.get('abstract', '')
        if not doc_abstract:
            print(f"Warning: '{doc.metadata.get('title', 'Unknown')[:50]}' has no abstract in metadata; keeping it")
            filtered_docs.append(doc)
            continue

        score = scorer.score(target_abstract, doc_abstract)['rougeL'].fmeasure
        if score < rouge_threshold:
            filtered_docs.append(doc)
        else:
            print(f"Filtered (too similar): {doc.metadata.get('title', 'Unknown')[:50]}... (ROUGE-L: {score:.3f})")

    print(f"ROUGE-L dedup: {len(retrieved_docs)} -> {len(filtered_docs)} papers")
    return filtered_docs



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
    target_abstract = paper.get("abstract", "")

    # Query each database for the most similar papers
    for section in search_types:
        if section in section_db_map and section_db_map[section]["db_key"] in database_map:
            db = database_map[section_db_map[section]["db_key"]]
            print(f"\nProcessing {section} section...")

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
                # Over-retrieve so we have headroom for ROUGE-L dedup
                similar_papers = db.similarity_search(
                    query_text,
                    filter={"year": {"$lte": args.year_threshold}},
                    k=args.num_related * 5
                )
                print(f"Initial retrieval: {len(similar_papers)} papers (year <= {args.year_threshold})")

                # ROUGE-L dedup against the target abstract to drop near-duplicates
                if target_abstract:
                    filtered = filter_similar_abstracts(target_abstract, similar_papers, args.rouge_threshold)
                else:
                    print("Warning: target paper has no abstract, skipping ROUGE-L dedup")
                    filtered = similar_papers

                related_papers_by_section[section] = filtered[:args.num_related]
                print(f"Kept {len(related_papers_by_section[section])} related papers for {section} section")
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
    Please analyze the paper according to the structured format specified in the system instructions.

    Return a SINGLE valid JSON object with the following fields ONLY.
    All fields except "summary" MUST be arrays of strings.
    ```json
    {{
    "summary": "A concise paragraph summarizing the paper.",
    "comparison_with_previous_work": [
        "Title of Related Work: ..."
    ],
    "Novelty": ["..."],
    "Significance": ["..."],
    "Soundness": ["..."],
    "strengths": ["..."],
    "weaknesses": ["..."],
    "Suggestion": ["..."]
    }}
    ```
    Rules:
    - STRICTLY output a valid JSON object and nothing else.
    - Do NOT include Markdown, explanations, or extra text.
    - Do NOT add or remove keys.
    - "comparison_with_previous_work" MUST contain EXACTLY 5 items.
    - All other lists (Novelty, Significance, Soundness, strengths, weaknesses, Suggestion) MUST contain EXACTLY 4 items.
    - Each item should be detailed and comprehensive (2 sentences per item minimum).
    - Provide in-depth analysis with specific evidence, examples, and reasoning for each point.
    - For "comparison_with_previous_work":
        - Each item MUST reference the prior work by its paper title (do NOT include URLs).
        - Each item MUST contain EXACTLY TWO sentences:
            1) one sentence describing what the prior work does,
            2) one sentence explaining the difference or relationship to the target paper.
    - For "Novelty": Provide a BALANCED assessment that discusses BOTH the novel aspects AND the limitations in novelty. For each item, explicitly cover (a) what is genuinely new — specific techniques, formulations, or combinations not seen in prior work, AND (b) what is incremental, derivative, or already-explored. Do not only praise or only criticize; every advisor must surface both sides in detail.
    - For "Significance": Provide a BALANCED assessment that discusses BOTH the potential impact AND the limits of significance. For each item, explicitly cover (a) why the problem/result matters — which communities benefit, what downstream applications become possible, what concrete examples of impact are plausible, AND (b) what caps or narrows the significance — niche scope, marginal gains over existing solutions, unclear adoption path, or limited generalizability. Do not only praise or only criticize; every item must contain both positive and negative analysis with concrete reasoning.
    - For "Soundness": Provide a BALANCED assessment BASED ONLY ON WHAT THE SUMMARY REVEALS. Remember: you do NOT have access to the full method or experiment details — no equations, pseudocode, hyperparameters, tables, or numerical results. Therefore you MUST NOT critique (or praise) specifics you cannot see. For each item, explicitly cover (a) what the summary presents as well-grounded — e.g., the proposed approach is internally coherent with its stated motivation, the claimed experimental scope appears aligned with the claims, the chosen problem setting is appropriate, AND (b) what is unclear or questionable AT THE SUMMARY LEVEL — e.g., the summary leaves key design choices unexplained, the link between claim and stated evidence is weak, the described experimental scope seems too narrow to support the claims, or the motivation does not obviously justify the proposed approach. Do not only praise or only criticize; every item must surface both strengths and concerns in detail, and every concern must be framed as a question about what the summary reveals, not as a factual accusation about hidden details.
    - For "strengths": Elaborate on each strength with specific evidence from the paper and its significance.
    - For "weaknesses": Clearly explain each weakness with specific examples and suggestions for improvement.
    - For "Suggestion": Give actionable, specific recommendations with detailed explanations of why and how to implement them.
    - When mentioning related work, ALWAYS use the paper title (no URLs).

    """

    return HumanMessage(content=prompt)


if __name__ == "__main__":
    args = parse_arguments()
    paper_path = args.paper_path
    paper_list = load_data(paper_path)

    openai_api_key = args.openai_key
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
    
    # Create a map of databases
    database_map = {
        "abstract_db": Chroma(
            persist_directory="database/abstract_db/ICLR2016_2024_summary_year",
            embedding_function=embeddings
        ),
        "contribution_db": Chroma(
            persist_directory="database/contribution_db/ICLR2016_2024_contribution_year",
            embedding_function=embeddings
        ),
        "method_db": Chroma(
            persist_directory="database/method_db/ICLR2016_2024_method_year",
            embedding_function=embeddings
        ),
        "experiment_db": Chroma(
            persist_directory="database/experiment_db/ICLR2016_2024_experiments_year",
            embedding_function=embeddings
        )
    }
    

    selected_papers = paper_list
    
    response_schemas = [
        ResponseSchema(name="summary", description="A concise paragraph summarizing the paper"),
        ResponseSchema(name="comparison_with_previous_work", description="List of exactly 5 comparisons with prior work (title-prefixed, two sentences each)"),
        ResponseSchema(name="Novelty", description="List of exactly 4 balanced novelty assessments"),
        ResponseSchema(name="Significance", description="List of exactly 4 balanced significance assessments"),
        ResponseSchema(name="Soundness", description="List of exactly 4 balanced soundness assessments"),
        ResponseSchema(name="strengths", description="List of exactly 4 strengths of the paper"),
        ResponseSchema(name="weaknesses", description="List of exactly 4 weaknesses of the paper"),
        ResponseSchema(name="Suggestion", description="List of exactly 4 actionable suggestions for improvement"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    system_message = SystemMessage(content="""You are a professional idea advisor with expertise in machine learning.
    Your task is to advise on a given target academic idea step by step, with a focus on novelty, contribution and soundness.
    You will be given:
    1. The idea's title, abstract, claimed contribution and section summaries.
    2. A set of relevant prior works, each with abstract, contribution statements, method descriptions and experimental setups.
	**Advising Guidelines**
    Read the given idea's content: It's important to carefully read through the given content, and to look up any related work and citations that will help you comprehensively advise on it. Be sure to give yourself sufficient time for this step.
    **Evaluation Criteria**
    1. Motivation / Objective: What is the goal of the paper? Is it to better address a known application or problem, draw attention to a new application or problem, or to introduce and/or explain a new theoretical finding? A combination of these? Different objectives will require different considerations as to potential value and impact. Is the approach well motivated, including being well-placed in the literature?
    2. Novelty & Originality: Are the tasks or methods new? Is the work a novel combination of well-known techniques? (This can be valuable!) Is it clear how this work differs from previous contributions?
    3. Significance & Contribution: Are the questions being asked important? Does the submission address a difficult task in a better way than previous work? Would researchers or practitioners likely adopt or build on these ideas?
    4. Soundness: Can the proposed method and experiment setup properly substantiate the claimed contributions? Will the claims be well supported under the proposed experiment setup? Are the methods used appropriate? Is this a complete piece of work or work in progress?
    **Related-Works Usage**
    1. **Abstract&Contribution**: frame the problem, scope, and high‑level "what" and "why." Used for evaluating significance and novelty.
    2. **Method**: describe "how" (algorithms, architectures and theoretical derivations). Ssed for checking whether the proposed method is novel or internally consistent, well‑justified, and mathematically rigorous.
    3. **Experiment setup**: specify experiment design, datasets, baselines, metrics. Used to evaluate whether this work's experiment is appropriately designed and whether the experiment is comprehensive enough to support the claims. This content may also contain expected results.
    **Scope of Available Information (IMPORTANT)**
    You are ONLY given the paper's high-level SUMMARY — i.e., the title, abstract, claimed contribution, and short section summaries. You do NOT have access to the full paper, so you CANNOT see:
    - concrete algorithmic details, pseudocode, equations, proofs, or implementation specifics of the proposed method;
    - concrete experimental details such as exact datasets, hyperparameters, training protocols, baseline configurations, ablation studies, numerical results, tables, or figures.
    Therefore, you MUST NOT fabricate or critique such unseen details.
    **Criticality**
    Noting the idea will become a paper submitted to top conferences with acceptance rate of 30%, you should be more critical. Feel free to give negative assessments if the idea's quality is poor.
    For empirical works, there is no need to contain theoretical analysis. For theoretical works, there is no need to contain experimental. Do not give negative assessments for the two cases.
    **Output Format**
    Provide a structured evaluation **strictly in valid JSON format**. Include both an overall evaluation and constructive suggestions for improvement. Do not add explanations, extra text, or Markdown formatting.
    When mentioning a related work, please use the title of the related work (no URLs).
    """)
    
    for i in range(args.start_idx, args.end_idx): 
        paper = selected_papers[i]
        print(f"Processing Paper {i+1}...")

        user_message = user_prompt_gen(paper, database_map, args)
        custom_id = f"ICLR2024-{i+1}"
        request = {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": system_message.content + '\n' + user_message.content}],"max_completion_tokens": 3000, "temperature": 0.6}}
        with open(args.output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(request, ensure_ascii=False) + '\n')



        