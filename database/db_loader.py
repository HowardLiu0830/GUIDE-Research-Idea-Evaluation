import os
import shutil
from typing import Dict, Optional
from huggingface_hub import hf_hub_download, HfApi
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def download_hf_database(repo_id: str, local_dir: str) -> bool:
    """Download database from Hugging Face Hub to specified directory"""
    try:
        print(f"Downloading {repo_id} to {local_dir}...")
        
        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Check if database already exists
        if os.path.exists(os.path.join(local_dir, "chroma.sqlite3")):
            print(f"  Database already exists in {local_dir}")
            return True
        
        api = HfApi()
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        
        print(f"  Found {len(repo_files)} files in repository")
        
        for file_path in repo_files:
            print(f"    Downloading: {file_path}")
            hf_hub_download(
                repo_id=repo_id,
                filename=file_path,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
        
        print(f"  ✓ Database downloaded successfully to: {local_dir}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error downloading database from {repo_id}: {e}")
        return False


def load_databases(base_dir: str = ".", openai_api_key: str = None) -> Dict[str, Optional[object]]:
    """Load all databases into memory"""
    
    # Initialize embeddings if API key is provided
    embeddings = None
    if openai_api_key:
        try:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large", 
                openai_api_key=openai_api_key
            )
            print("✓ OpenAI embeddings initialized")
        except Exception as e:
            print(f"✗ Failed to initialize OpenAI embeddings: {e}")
            return {}
    
    # Database repository mappings
    repo_mappings = {
        "abstract_db": "ResearchAgent-GUIDE/ICLR_abstract",
        "contribution_db": "ResearchAgent-GUIDE/ICLR_contribution", 
        "method_db": "ResearchAgent-GUIDE/ICLR_method",
        "experiment_db": "ResearchAgent-GUIDE/ICLR_experiment"
    }
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    print(f"Using database directory: {os.path.abspath(base_dir)}")
    
    database_map = {}
    
    # Download each database
    for db_name, repo_id in repo_mappings.items():
        print(f"\n{'='*50}")
        print(f"Processing {db_name}")
        print(f"{'='*50}")
        
        db_path = os.path.join(base_dir, db_name)
        
        # Download database
        if download_hf_database(repo_id, db_path):
            # Load into Chroma if embeddings are available
            if embeddings:
                try:
                    db = Chroma(
                        persist_directory=db_path,
                        embedding_function=embeddings
                    )
                    
                    # Test the database
                    collection = db._collection
                    if collection:
                        count = collection.count()
                        print(f"  ✓ Loaded {db_name} with {count} documents")
                        database_map[db_name] = db
                    else:
                        print(f"  ✗ {db_name} collection is empty")
                        database_map[db_name] = None
                        
                except Exception as e:
                    print(f"  ✗ Failed to load {db_name} into Chroma: {e}")
                    database_map[db_name] = None
            else:
                print(f"  ✓ Downloaded {db_name} (not loaded into memory - no API key)")
                database_map[db_name] = None
        else:
            database_map[db_name] = None
    
    return database_map


def test_databases(database_map: Dict, test_query: str = "deep learning neural networks"):
    """Test similarity search on loaded databases"""
    print(f"\n{'='*50}")
    print("Testing Database Functionality")
    print(f"{'='*50}")
    
    working_dbs = {name: db for name, db in database_map.items() if db is not None}
    
    if not working_dbs:
        print("No databases loaded for testing")
        return
    
    print(f"Testing query: '{test_query}'")
    
    for db_name, db in working_dbs.items():
        print(f"\n--- Testing {db_name} ---")
        try:
            results = db.similarity_search(test_query, k=3)
            print(f"✓ Found {len(results)} similar documents")
            
            for i, doc in enumerate(results, 1):
                metadata = doc.metadata
                title = metadata.get('title', 'No title')[:80]
                print(f"  {i}. {title}...")
                
        except Exception as e:
            print(f"✗ Error in similarity search for {db_name}: {e}")


def get_database_info(base_dir: str = "."):
    """Get information about downloaded databases"""
    print(f"Database Directory: {os.path.abspath(base_dir)}")
    print(f"{'='*50}")
    
    if not os.path.exists(base_dir):
        print("Database directory does not exist")
        return
    
    total_size = 0
    db_names = ["abstract_db", "contribution_db", "method_db", "experiment_db"]
    
    for db_name in db_names:
        db_path = os.path.join(base_dir, db_name)
        if os.path.exists(db_path):
            # Calculate directory size
            size = sum(os.path.getsize(os.path.join(dirpath, filename))
                      for dirpath, dirnames, filenames in os.walk(db_path)
                      for filename in filenames)
            size_mb = size / (1024 * 1024)
            total_size += size_mb
            
            files = os.listdir(db_path)
            print(f"{db_name}:")
            print(f"  Size: {size_mb:.1f} MB")
            print(f"  Files: {len(files)}")
            print(f"  Path: {db_path}")
        else:
            print(f"{db_name}: Not downloaded")
    
    print(f"\nTotal size: {total_size:.1f} MB")


def clean_databases(base_dir: str = "."):
    """Remove all downloaded databases"""
    db_names = ["abstract_db", "contribution_db", "method_db", "experiment_db"]
    
    for db_name in db_names:
        db_path = os.path.join(base_dir, db_name)
        if os.path.exists(db_path):
            try:
                shutil.rmtree(db_path)
                print(f"✓ Removed {db_name} directory")
            except Exception as e:
                print(f"✗ Error removing {db_name}: {e}")
        else:
            print(f"{db_name}: Not found")


def main():
    # Set your OpenAI API key here (optional - needed only for testing)
    OPENAI_API_KEY = "your-openai-api-key-here"  # Replace with your actual API key or set to None
    
    import argparse
    parser = argparse.ArgumentParser(description='Download and manage research databases')
    parser.add_argument('--action', choices=['download', 'test', 'info', 'clean'], 
                        default='download', help='Action to perform')
    parser.add_argument('--dir', default='.', help='Database directory (default: current directory)')
    parser.add_argument('--query', default='deep learning neural networks', 
                        help='Test query for similarity search')
    
    args = parser.parse_args()
    
    if args.action == 'download':
        print("Downloading Research Databases")
        print("=" * 60)
        
        # Download and optionally load databases
        api_key = OPENAI_API_KEY if OPENAI_API_KEY != "your-openai-api-key-here" else None
        database_map = load_databases(args.dir, api_key)
        
        # Print summary
        print(f"\n{'='*60}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*60}")
        
        successful = sum(1 for db in database_map.values() if db is not None)
        total = len(database_map)
        
        print(f"Databases downloaded: {total}")
        if api_key:
            print(f"Successfully loaded: {successful}")
            print(f"Failed to load: {total - successful}")
        
        for db_name, db in database_map.items():
            if api_key:
                status = "✓ LOADED" if db is not None else "✗ FAILED"
            else:
                status = "✓ DOWNLOADED"
            print(f"  {db_name}: {status}")
        
        print(f"\nDatabases stored in: {os.path.abspath(args.dir)}")
        
    elif args.action == 'test':
        print("Testing Database Functionality")
        print("=" * 60)
        
        if OPENAI_API_KEY == "your-openai-api-key-here":
            print("✗ Please set your OpenAI API key to test databases")
            return
        
        database_map = load_databases(args.dir, OPENAI_API_KEY)
        test_databases(database_map, args.query)
        
    elif args.action == 'info':
        get_database_info(args.dir)
        
    elif args.action == 'clean':
        print("Cleaning Database Directory")
        print("=" * 60)
        clean_databases(args.dir)


if __name__ == "__main__":
    main()