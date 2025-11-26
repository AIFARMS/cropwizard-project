import dotenv
import time
import os
import json

dotenv.load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

# Check if Tavily API key is available
if not os.environ.get("TAVILY_API_KEY"):
    print("WARNING: TAVILY_API_KEY environment variable not set.")
    print("Please set your Tavily API key:")
    print("export TAVILY_API_KEY='your-api-key'")

def generate_search_query(query: str) -> str:
    final_query = f"""Generate a search query to find information about {query} 
    
    Useful information:
    Current date: {time.strftime('%Y-%m-%d')}

    Return the search query as a string, nothing else not even quotes.
"""

    llm = ChatOpenAI(
        temperature=0, 
        model="gpt-4o-mini",
        request_timeout=60,
        max_retries=2
    )
    return llm.invoke(final_query).content


def search_tool(query: str) -> str:
    # 1. Search for information
    print(f"Searching for: {query}")
    search_tool = TavilySearchResults(
        max_results=3,
        include_answer=True,
        search_depth="advanced"
    )
    
    try:
        search_results = search_tool.invoke(query)
        print(f"Search results type: {type(search_results)}")
        print(f"Search results structure: {json.dumps(search_results, indent=2)[:200]}...")
    except Exception as e:
        print(f"Error during search: {e}")
        raise
    
    # 2. Format the results
    search_results_text = "\n\n".join([
        f"Source {i+1}:\n{result['content']}\nURL: {result['url']}"
        for i, result in enumerate(search_results)
    ])
    
    # 3. Generate follow-up questions based on initial search results
    followup_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research assistant focused on gathering precise information.
        The user has asked a specific question that needs comprehensive data points for comparison.
        
        Based on the initial search results, FIRST DETERMINE IF follow-up questions are actually needed.
        If the initial results provide complete information to answer the original query, NO follow-up 
        questions may be required.
        
        If follow-up questions ARE needed, identify WHAT SPECIFIC INFORMATION IS MISSING to properly 
        answer the original query. Generate 1-3 targeted follow-up questions that will help fill these gaps.
        
        Your follow-up questions should:
        1. Directly address aspects of the original query that weren't covered in the initial results
        2. Focus on obtaining specific data points needed for comparison (prices, specs, features, etc.)
        3. Ask about adjacent model years or specific versions if information is only available for some models
        4. Avoid going off-topic from the original comparison request
        
        Return ONLY the follow-up questions as a numbered list. If no follow-up questions are needed, 
        return exactly: "NO_FOLLOWUP_NEEDED"."""),
        ("user", f"""Original query: {query}
        
        Initial search results:
        {search_results_text}
        
        First determine if follow-up questions are needed. If they are, generate 1-3 FOCUSED follow-up 
        questions that will help gather the MISSING information needed to properly answer the original query:""")
    ])
    
    llm = ChatOpenAI(
        temperature=0.2, 
        model="gpt-4o-mini",
        request_timeout=60,
        max_retries=2
    )
    
    # Format the prompt into messages before invoking the LLM
    formatted_messages = followup_prompt.format_messages()
    print("Formatted follow-up prompt messages for LLM")
    
    followup_questions_response = llm.invoke(formatted_messages)
    followup_content = followup_questions_response.content.strip()
    
    # Check if follow-up is needed
    if followup_content == "NO_FOLLOWUP_NEEDED":
        print("No follow-up questions needed")
        followup_questions = []
    else:
        followup_questions = followup_content.split('\n')
        print(f"Generated {len(followup_questions)} follow-up questions")
    
    # 4. Search for answers to follow-up questions (only if there are questions)
    followup_results = []
    for question in followup_questions:
        # Clean up the question format (remove numbering if present)
        if '.' in question.split()[0]:
            question = ' '.join(question.split()[1:])
        
        print(f"Processing follow-up question: {question}")
        
        # Generate search query for follow-up question
        followup_search_query = generate_search_query(question)
        print(f"Generated search query: {followup_search_query}")
        
        # Search for follow-up information
        try:
            followup_search_results = search_tool.invoke(followup_search_query)
        except Exception as e:
            print(f"Error during follow-up search: {e}")
            followup_search_results = []
        
        # Format the follow-up results
        followup_result_text = "\n\n".join([
            f"Source {i+1}:\n{result['content']}\nURL: {result['url']}"
            for i, result in enumerate(followup_search_results)
        ])
        
        followup_results.append({
            "question": question,
            "results": followup_result_text
        })
    
    # 5. Compile all information into a final response
    followup_sections = []
    for item in followup_results:
        followup_sections.append(f"Follow-up Question: {item['question']}\n\nResults:\n{item['results']}")
    
    all_followup_content = ""
    if followup_sections:
        all_followup_content = "\n\n" + "\n\n".join(followup_sections)
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful research assistant focused on DIRECTLY answering the user's original question.
        
        Your task is to synthesize search results into a CONCISE, FOCUSED answer that:
        1. Directly addresses the specific comparison requested in the original query
        2. Presents clear data points for comparison (prices, features, specs, etc.)
        3. Organizes information in a structured, easy-to-understand format
        4. Prioritizes factual information most relevant to the comparison requested
        5. Avoids including tangential information that doesn't contribute to answering the original query
        
        If search results don't provide enough information to fully answer the query, clearly state what information is missing.
        Be precise with numbers, dates, and specifications.
        Do not make up information that is not supported by the search results."""),
        ("user", f"""Original query: {query}
        
        Initial search results:
        {search_results_text}
        
        {f"Follow-up information:{all_followup_content}" if followup_sections else "No follow-up information was needed as the initial results were comprehensive."}
        
        Please provide a DIRECT answer to the original query based on these search results, focusing specifically on the comparison requested:""")
    ])
    
    # Format the final prompt properly
    formatted_final_messages = final_prompt.format_messages()
    print("Formatted final prompt messages for LLM")
    
    final_response = llm.invoke(formatted_final_messages)
    return final_response.content

def main() -> None:
    # Example query to show users what kind of queries work well
    example_query = "What is the latest Tesla Model 3 price, and how does it compare to the previous model?"
    
    print("Advanced Search Tool with Follow-up Questions")
    print("---------------------------------------------")
    print(f"Example query: {example_query}")
    print("Enter your query or press Enter to use the example:")
    
    user_input = input("> ").strip()
    query = user_input if user_input else example_query
    
    print(f"\nProcessing query: {query}")
    search_query = generate_search_query(query)
    print(f"Generated search query: {search_query}")
    
    search_result = search_tool(search_query)
    
    print("\n=== Final Result ===")
    print(search_result)
    print("\nSearch complete!")

main()