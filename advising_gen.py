import os
import re
import json
from typing import Dict, List
from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import argparse

# Import token counting utilities
from langchain_community.callbacks import get_openai_callback
from langchain.callbacks.base import BaseCallbackHandler

# Custom callback handler for token tracking
class TokenUsageCallback(BaseCallbackHandler):
    """Custom callback handler to track token usage across different models"""
    
    def __init__(self):
        super().__init__()
        self.token_usage = {}
        self.current_model = None
        self.sample_token_counts = []
        # Add counters for threshold tracking
        self.thresholds = [14336, 15360, 16384]
        self.samples_exceeding = {threshold: 0 for threshold in self.thresholds}
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called when LLM starts processing"""
        if "model_name" in kwargs and kwargs["model_name"]:
            self.current_model = kwargs["model_name"]
            
    def on_llm_end(self, response, **kwargs):
        """Called when LLM returns a response"""
        if hasattr(response, "llm_output") and response.llm_output:
            model = self.current_model or "unknown_model"
            token_usage = response.llm_output.get("token_usage", {})
            
            if model not in self.token_usage:
                self.token_usage[model] = {
                    "request_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            
            self.token_usage[model]["request_count"] += 1
            self.token_usage[model]["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
            self.token_usage[model]["completion_tokens"] += token_usage.get("completion_tokens", 0)
            self.token_usage[model]["total_tokens"] += token_usage.get("total_tokens", 0)
            
            # Track token count per sample for threshold analysis
            total_tokens = token_usage.get("total_tokens", 0)
            self.sample_token_counts.append(total_tokens)
            
            # Update threshold counters
            for threshold in self.thresholds:
                if total_tokens > threshold:
                    self.samples_exceeding[threshold] += 1
    
    def print_summary(self):
        """Print a summary of token usage"""
        print("\n===== Token Usage Summary =====")
        total_requests = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        
        for model, usage in self.token_usage.items():
            print(f"\nModel: {model}")
            print(f"Request count: {usage['request_count']}")
            print(f"Prompt tokens: {usage['prompt_tokens']:,}")
            print(f"Completion tokens: {usage['completion_tokens']:,}")
            print(f"Total tokens: {usage['total_tokens']:,}")
            
            total_requests += usage['request_count']
            total_prompt_tokens += usage['prompt_tokens']
            total_completion_tokens += usage['completion_tokens']
            total_tokens += usage['total_tokens']
        
        print("\nOverall Summary:")
        print(f"Total requests: {total_requests}")
        print(f"Total prompt tokens: {total_prompt_tokens:,}")
        print(f"Total completion tokens: {total_completion_tokens:,}")
        print(f"Total tokens across all models: {total_tokens:,}")
        
        # Print threshold statistics
        print("\nToken Threshold Statistics:")
        for threshold in sorted(self.thresholds):
            count = self.samples_exceeding[threshold]
            percentage = (count / total_requests) * 100 if total_requests > 0 else 0
            print(f"Samples exceeding {threshold:,} tokens: {count} ({percentage:.2f}%)")
        
        print("===============================")
    
    def save_to_file(self, filepath):
        """Save token usage data to a JSON file"""
        # Add threshold data to the output
        output_data = {
            "model_usage": self.token_usage,
            "threshold_statistics": {
                f"exceeding_{threshold}": self.samples_exceeding[threshold] 
                for threshold in self.thresholds
            },
            "total_samples": len(self.sample_token_counts)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"Token usage data saved to {filepath}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate paper advising using LLM')
    parser.add_argument('--openai_key', required=True, help='OpenAI API key')
    parser.add_argument('--paper_path', required=True, help='Path to paper data JSON')
    parser.add_argument('--output_path', required=True, help='Path for output JSON')
    parser.add_argument('--output_path_clean', required=True, help='Path for output cleaned JSON')
    parser.add_argument('--prompt_path', required=True, help='Path for prompt JSON')
    parser.add_argument('--start_idx', required=True, type=int, help='Starting index for paper processing')
    parser.add_argument('--end_idx', required=True, type=int, help='Ending index for paper processing')
    # Add token tracking arguments
    parser.add_argument('--track_tokens', action='store_true', help='Enable token usage tracking')
    parser.add_argument('--token_output', default='token_usage.json', help='Path for token usage data output')
    parser.add_argument(
        '--model',
        required=False,
        default='gpt-5.4-mini',
        choices=[
            'gpt-5.4-mini',
            'gpt-5.4-nano',
            'gpt-4o-mini',
            'gpt-4o',
            'gpt-4.1-nano',
        ],
        help='Select an OpenAI GPT model'
    )
    return parser.parse_args()


def load_data(path: str) -> List[str]:
    paper_list = []
    with open(path, 'r') as f:
        data = json.load(f)
    for paper in data:
        paper_list.append(paper)
    return paper_list


def extract_json_content(text: str) -> str:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def clean_json_output(raw_output):
    """
    Clean and normalize JSON output from LLM response.
    Handles LaTeX math expressions and ensures proper escaping.
    """
    # Remove code block markers if present
    cleaned_output = re.sub(r"^```json\s*|```$", "", raw_output, flags=re.MULTILINE).strip()
    
    # Clean LaTeX math expressions before JSON parsing
    cleaned_output = re.sub(r'\$([^$]+)\$', lambda m: clean_latex(m.group(1)), cleaned_output)
    
    # Normalize newlines and whitespace
    cleaned_output = re.sub(r"\s*\n\s*", " ", cleaned_output)
    
    # Ensure proper escaping of backslashes that aren't already part of escape sequences
    cleaned_output = re.sub(r'(?<!\\)\\(?!["\\])', r'\\\\', cleaned_output)
    
    return cleaned_output.strip()


def clean_latex(text):
    """
    Clean LaTeX expressions by removing math delimiters and normalizing notation.
    """
    if not isinstance(text, str):
        return text
        
    # Remove math mode delimiters
    text = re.sub(r'\$', '', text)
    
    # Handle common LaTeX commands more gracefully
    text = re.sub(r'\\mathcal{O}', 'O', text)  # Replace \mathcal{O} with just O
    text = re.sub(r'\\sqrt\{([^}]+)\}', lambda m: f"sqrt({m.group(1)})", text)  # Convert sqrt to functional notation
    text = re.sub(r'\^([0-9/]+)', lambda m: f"^{m.group(1)}", text)  # Keep exponents in a cleaner format
    
    return text


def append_to_json_array(new_item, file_path):
    new_item_str = json.dumps(new_item, ensure_ascii=False)
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("[" + new_item_str + "]")
    else:
        with open(file_path, 'r+', encoding='utf-8') as f:
            content = f.read().strip()
            if content == "[]":
                new_content = "[" + new_item_str + "]"
            else:
                new_content = content[:-1] + "," + new_item_str + "]"
            f.seek(0)
            f.write(new_content)
            f.truncate()


def process_paper(i, paper, prompt, llm, max_retries, track_tokens=False):
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Use OpenAI callback for token tracking if enabled and it's an OpenAI model
            if track_tokens and isinstance(llm, ChatOpenAI) and hasattr(llm, 'model_name'):
                with get_openai_callback() as cb:
                    response = llm.invoke(prompt)
                    print(f"   Tokens: Input={cb.prompt_tokens:,} | Output={cb.completion_tokens:,} | Total={cb.total_tokens:,}")
            else:
                response = llm.invoke(prompt)
            
            raw_output = response.content.strip()
            json_content = extract_json_content(raw_output)
            cleaned_output = clean_json_output(json_content)

            advising_result = json.loads(cleaned_output)
            advising_result["raw_output"] = raw_output
            advising_result["title"] = paper["title"] 
            
            # Apply clean_latex to string values, and to each string in a list
            for key in ["summary", "comparison_with_previous_work", "Novelty", "Significance", "Soundness", "strengths", "weaknesses", "Suggestion"]:
                if key not in advising_result:
                    continue
                v = advising_result[key]
                if isinstance(v, str):
                    advising_result[key] = clean_latex(v)
                elif isinstance(v, list):
                    advising_result[key] = [clean_latex(item) if isinstance(item, str) else item for item in v]
                    
            print(f"✅ Advising for '{paper['title']}' generated successfully.")
            return i, advising_result
        except json.JSONDecodeError:
            retry_count += 1
            print(f"❌ JSON Parse Error for '{paper['title']}' (Attempt {retry_count}/{max_retries})")
            print(f"Raw Output:\n{raw_output}")
        except Exception as e:
            print(f"❌ Error processing '{paper['title']}' on attempt {retry_count + 1}: {e}")
            retry_count += 1
    print(f"🚨 Skipping '{paper['title']}' after {max_retries} failed attempts.")
    return i, None


if __name__ == "__main__":
    args = parse_arguments()
    paper_path = args.paper_path
    paper_list = load_data(paper_path)

    openai_api_key = args.openai_key

    # Initialize token tracker if enabled
    token_callback = None
    if args.track_tokens:
        token_callback = TokenUsageCallback()
        print("Token tracking enabled.")
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    

    selected_papers = paper_list[args.start_idx:args.end_idx]
    
    callbacks = [token_callback] if token_callback else None
    llm = ChatOpenAI(api_key=openai_api_key, model=args.model, temperature=0.6, callbacks=callbacks)

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

    results = []
    max_retries = 5
    prompt_list = []
    prompt_path = args.prompt_path
    with open(prompt_path, 'r') as f:
        for line in f:
            json_line = json.loads(line)
            prompt = json_line["body"]["messages"][0]["content"]
            prompt_list.append(prompt)
    print(f"===="*10)
    print(len(prompt_list))

    max_retries = 10
    results = []

    # Process papers sequentially
    for i, paper in enumerate(selected_papers):
        print(f"Processing paper {i+1}/{len(selected_papers)}: {paper['title'][:50]}...")
        paper_index, advising_result = process_paper(i, paper, prompt_list[i], llm, max_retries, args.track_tokens)
        
        if advising_result is not None:
            results.append(advising_result)
            append_to_json_array(advising_result, args.output_path)
                
    # Write formatted output
    with open(args.output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # Create cleaned version of results
    output_path = args.output_path
    
    with open(output_path, "r") as f:
        results = json.load(f)
    print(f"Total advising generated: {len(results)}")
    def _fmt(value):
        if isinstance(value, list):
            return "\n".join(f"  - {item}" for item in value)
        return str(value) if value is not None else ""

    results_copy = results.copy()
    results_cleaned = []
    for i, entry in enumerate(results_copy):
        title = selected_papers[i]["title"]
        abstract = selected_papers[i]["abstract"]
        strengths_content = entry.get('strengths', entry.get('strong_points', 'No strengths provided'))
        weaknesses_content = entry.get('weaknesses', entry.get('weak_points', 'No weaknesses provided'))
        new_advising = "\n".join([
            f"Novelty:\n{_fmt(entry.get('Novelty', ''))}",
            f"Significance:\n{_fmt(entry.get('Significance', ''))}",
            f"Soundness:\n{_fmt(entry.get('Soundness', ''))}",
            f"Strengths:\n{_fmt(strengths_content)}",
            f"Weaknesses:\n{_fmt(weaknesses_content)}",
        ])
        abs_advising = {
            "title": title,
            "abstract": abstract,
            "advising": new_advising
        }
        results_cleaned.append(abs_advising)

    output_path = args.output_path_clean
    with open(output_path, "w") as f:
        json.dump(results_cleaned, f, indent=4, ensure_ascii=False)
    
    if token_callback:
        # Only print and save token data if there's actual usage to report
        if token_callback.token_usage:
            token_callback.print_summary()
            token_callback.save_to_file(args.token_output)
        else:
            print("\nNo token usage data available to summarize (unsupported model or no API calls made).")