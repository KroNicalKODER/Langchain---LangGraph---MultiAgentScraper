from typing import Dict, List, TypedDict, Annotated
from langgraph.graph import Graph, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient
import asyncio
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Define the state type
class ResearchState(TypedDict):
    query: str
    initial_context: str
    sub_queries: List[str]
    scraped_info: List[Dict]
    combined_context: str
    final_response: str
    history: List[Dict]

class ResearchGraph:
    def __init__(self):
        # Initialize Ollama with DeepSeek model
        self.llm = OllamaLLM(model="llama3.2")
        
        # Initialize Tavily client
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        # Define prompts
        self.initial_context_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research assistant. Your task is to analyze the initial web search results and provide a brief overview of the topic.
            This overview will help in generating more focused sub-queries."""),
            ("human", "Query: {query}\n\nInitial Search Results: {search_results}")
        ])
        
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
        
        self.context_combination_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research synthesizer. Your task is to combine information from multiple sources into a coherent context. 
             You are given a list of sub-queries and the information gathered from the web for a original query.
             Provide as much information as possible from the information scraped from the web for the original query, which is further be used to generate the response for the user.
             Please strictly dont add any other text or comments other than the context.
             Combine the following information into a comprehensive, well-structured context that answers the original query. Remember these are the web scraped information so you might get some web terminologies like
              table of content, print, etc. Please ignore all these type of words and terminologies and focus only on the context provided. Also, focus the original query and provide the most relevant information.
             """),
            ("human", "Original Query: {query}\n\nInformation: {information}")
        ])
        
        self.final_response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research assistant. Using the provided context and original query, generate a comprehensive, well-structured response.
            Your response should be detailed, accurate, and directly address the original query."""),
            ("human", "Original Query: {query}\n\nContext: {context}")
        ])

    def get_initial_context(self, state: ResearchState) -> ResearchState:
        """Get initial context about the topic from the web."""
        print(f"Getting initial context for query: {state['query']}")
        initial_search = self.tavily_client.search(query=state["query"], search_depth="advanced", include_answer="basic")
        top_sources = initial_search.get("results", [])[:3]
        formatted_results = "\n\n".join([
            f"Content: {source.get('content', '')}"
            for source in top_sources
        ])
        
        chain = self.initial_context_prompt | self.llm
        response = chain.invoke({
            "query": state["query"],
            "search_results": formatted_results
        })
        
        state["initial_context"] = response
        return state

    def break_down_query(self, state: ResearchState) -> ResearchState:
        """Break down the main query into sub-queries using the initial context."""
        print(f"Breaking down query: {state['query']}")
        chain = self.query_breakdown_prompt | self.llm
        response = chain.invoke({
            "query": state["query"],
            "initial_context": state["initial_context"]
        })
        
        # Parse the response to extract sub-queries
        sub_queries = [q.strip() for q in response.split('\n') if q.strip()]
        # Remove << and >> from the sub_queries
        sub_queries = [q.strip("<>") for q in sub_queries]
        
        state["sub_queries"] = sub_queries
        return state

    def scrape_information(self, state: ResearchState) -> ResearchState:
        """Scrape information for all queries in parallel."""
        print(f"Scraping information for queries: {state['sub_queries']}")
        async def process_query(query: str) -> Dict:    
            print(f"Scraping information for query: {query}")
            search_result = self.tavily_client.search(query=query, search_depth="advanced")
            top_sources = search_result.get("results", [])[:3]
            combined_content = "\n\n".join([
                source.get("content", "") for source in top_sources if source.get("content")
            ])
            return {
                "query": query,
                "content": combined_content
            }
        
        async def process_all_queries():
            tasks = [process_query(query) for query in state["sub_queries"]]
            return await asyncio.gather(*tasks)
        
        # Create a new event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(process_all_queries())
        finally:
            loop.close()
        
        state["scraped_info"] = results
        return state

    def combine_context(self, state: ResearchState) -> ResearchState:
        """Combine all scraped information into a coherent context."""
        print(f"Combining context for query: {state['query']}")
        combined_info = "\n\n".join([f"Query: {item['query']}\nContent: {item['content']}" for item in state["scraped_info"]])
        chain = self.context_combination_prompt | self.llm
        response = chain.invoke({"query": state["query"], "information": combined_info})
        
        state["combined_context"] = response
        return state

    def generate_final_response(self, state: ResearchState) -> ResearchState:
        """Generate the final response using the combined context."""
        print(f"Generating final response for query: {state['query']}")
        context = state["combined_context"] + "\n\n" + state["initial_context"]
        chain = self.final_response_prompt | self.llm
        response = chain.invoke({"query": state["query"], "context": context})
        
        state["final_response"] = response
        return state

    def build_graph(self) -> Graph:
        """Build the research workflow graph."""
        workflow = StateGraph(ResearchState)
        print("Building graph") 
        # Add nodes
        workflow.add_node("get_initial_context", self.get_initial_context)
        workflow.add_node("break_down_query", self.break_down_query)
        workflow.add_node("scrape_information", self.scrape_information)
        workflow.add_node("combine_context", self.combine_context)
        workflow.add_node("generate_final_response", self.generate_final_response)
        
        # Add edges
        workflow.add_edge("get_initial_context", "break_down_query")
        workflow.add_edge("break_down_query", "scrape_information")
        workflow.add_edge("scrape_information", "combine_context")
        workflow.add_edge("combine_context", "generate_final_response")
        
        # Set entry point
        workflow.set_entry_point("get_initial_context")
        
        # Set finish point
        workflow.set_finish_point("generate_final_response")
        
        return workflow.compile()

    def run_research(self, query: str, history: List[Dict] = None) -> ResearchState:
        """Run the research workflow."""
        if history is None:
            history = []
        
        # Initialize state
        state = ResearchState(
            query=query,
            initial_context="",
            sub_queries=[],
            scraped_info=[],
            combined_context="",
            final_response="",
            history=history
        )
        
        # Build and run the graph
        graph = self.build_graph()
        final_state = graph.invoke(state)
        
        return final_state 