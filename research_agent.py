from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import Graph, StateGraph
from tavily import TavilyClient
import asyncio
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class ResearchAgent:
    def __init__(self):
        # Initialize Ollama with DeepSeek model
        self.llm = Ollama(model="llama3.2")
        
        # Initialize Tavily client
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        # Define the initial context prompt
        self.initial_context_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research assistant. Your task is to analyze the initial web search results and provide a brief overview of the topic.
            This overview will help in generating more focused sub-queries."""),
            ("human", "Query: {query}\n\nInitial Search Results: {search_results}")
        ])
        
        # Define the query breakdown prompt
        self.query_breakdown_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research assistant. Your task is to break down complex queries into specific, focused sub-queries.
            Based on the initial context provided, break down the given query into 5-10 specific sub-queries that will help gather comprehensive information.
            Each sub-query should be clear, focused, and searchable. Please strictly dont add any other text or comments other than the sub-queries.
            Return the sub-queries as a list. Give your response in form like 
            
             <<sub_query1>>
             <<sub_query2>>
             <<sub_query3>>
             <<sub_query4>>
             and so on.....
            """),
            ("human", "Original Query: {query}\n\nInitial Context: {initial_context}")
        ])
        
        # Define the context combination prompt
        self.context_combination_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research synthesizer. Your task is to combine information from multiple sources into a coherent context. 
             You are given a list of sub-queries and the information gathered from the web for a original query.
             Provide as much information as possible from the information scraped from the web for the original query, which is further be used to generate the response for the user.
             Please strictly dont add any other text or comments other than the context.
            Combine the following information into a comprehensive, well-structured context that answers the original query. Remember these are the web scraped information so you might get some web terminologies like
             table of content, print, etc. Please ignore all these type of words and terminologies and focus only on the context provided. Also, focus the original query and provide the most relevant information.
            """)
            ,("human", "Original Query: {query}\n\nInformation: {information}")
        ])
        
        # Define the final response prompt
        self.final_response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research assistant. Using the provided context and original query, generate a comprehensive, well-structured response.
            Your response should be detailed, accurate, and directly address the original query."""),
            ("human", "Original Query: {query}\n\nContext: {context}")
        ])

    async def get_initial_context(self, query: str) -> str:
        """Get initial context about the topic from the web."""
        # First, do a quick search to get an overview
        initial_search = self.tavily_client.search(query=query, search_depth="advanced", include_answer="basic")
        top_sources = initial_search.get("results", [])[:3]
        formatted_results = "\n\n".join([
            f"Content: {source.get('content', '')}"
            for source in top_sources
        ])
        
        # Use LLM to analyze the initial results and provide context
        chain = self.initial_context_prompt | self.llm
        response = await chain.ainvoke({
            "query": query,
            "search_results": formatted_results
        })
        return response

    async def break_down_query(self, query: str, initial_context: str) -> List[str]:
        """Break down the main query into sub-queries using the initial context."""
        chain = self.query_breakdown_prompt | self.llm
        response = await chain.ainvoke({
            "query": query,
            "initial_context": initial_context
        })
        # Parse the response to extract sub-queries
        sub_queries = [q.strip() for q in response.split('\n') if q.strip()]

        ## remomve  << and >> from the sub_queries 
        sub_queries = [q.strip("<>") for q in sub_queries]
        return sub_queries

    async def scrape_information(self, query: str) -> Dict[str, Any]:
        """Scrape and combine information from the top 3 sources using Tavily."""
        
        print("Scraping information for query: ", query)
        search_result = self.tavily_client.search(query=query, search_depth="advanced")
        
        # Get top 3 sources
        top_sources = search_result.get("results", [])[:3]

        # Combine the content from top 3 sources
        combined_content = "\n\n".join([
            source.get("content", "") for source in top_sources if source.get("content")
        ])

        return {
            "query": query,
            "content": combined_content,
        }


    async def process_all_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process all queries in parallel."""
        
        tasks = [self.scrape_information(query) for query in queries]
        results = await asyncio.gather(*tasks)
        return results

    async def combine_context(self, query: str, information: List[Dict[str, Any]]) -> str:
        """Combine all scraped information into a coherent context."""
        combined_info = "\n\n".join([f"Query: {item['query']}\nContent: {item['content']}" for item in information])
        chain = self.context_combination_prompt | self.llm
        response = await chain.ainvoke({"query": query, "information": combined_info})
        return response

    async def generate_final_response(self, query: str, context: str, initial_context: str) -> str:
        """Generate the final response using the combined context."""
        context = context + "\n\n" + initial_context
        chain = self.final_response_prompt | self.llm
        response = await chain.ainvoke({"query": query, "context": context})
        return response

    async def research(self, query: str) -> str:
        """Main research function that orchestrates the entire process."""
        # Step 0: Get initial context about the topic
        print("Getting initial context...")
        initial_context = await self.get_initial_context(query)
        print(f"Initial context: {initial_context}")
        
        # Step 1: Break down the query using initial context
        print("\nBreaking down query into sub-queries...")
        sub_queries = await self.break_down_query(query, initial_context)
        print(f"Generated sub-queries: {sub_queries}")
        
        # Step 2: Scrape information for all queries in parallel
        print("\nScraping information for sub-queries...")
        scraped_info = await self.process_all_queries(sub_queries)
        
        # Step 3: Combine the context
        print("\nCombining context...")
        combined_context = await self.combine_context(query, scraped_info)
        
        # Step 4: Generate final response
        print("\nGenerating final response...")
        final_response = await self.generate_final_response(query, combined_context)
        
        return final_response

# Example usage
async def main():
    agent = ResearchAgent()
    query = "What are transformers in machine learning?"
    response = await agent.research(query)
    print("\nFinal Response:")
    print(response)

if __name__ == "__main__":
    asyncio.run(main()) 