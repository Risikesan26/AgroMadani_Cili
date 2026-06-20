import sys
import os
from dotenv import load_dotenv

def main():
    print("Welcome to the Chili Disease Detection RAG CLI.")
    
    # Load environment variables
    load_dotenv()
    
    # Check if OPENAI_API_KEY is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("Error: OPENAI_API_KEY is missing or invalid in the .env file.")
        print("Please edit the '.env' file in this directory and replace 'your_openai_api_key_here' with your actual OpenAI API key.")
        sys.exit(1)
        
    # Import the pipeline only after verifying API key, 
    # to avoid unnecessary initializations if key is missing.
    try:
        from rag_pipeline import answer_question, build_vector_store
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Did you install dependencies from requirements.txt?")
        sys.exit(1)
        
    print("Type 'build' to rebuild the vector database from the Articles folder.")
    print("Type 'quit' or 'exit' to stop.")
    
    while True:
        try:
            user_input = input("\nAsk a question about chili diseases: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
            elif user_input.lower() == 'build':
                build_vector_store()
                continue
            elif not user_input:
                continue
            
            print("Thinking...")
            answer = answer_question(user_input)
            print("\nAnswer:")
            print(answer)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
