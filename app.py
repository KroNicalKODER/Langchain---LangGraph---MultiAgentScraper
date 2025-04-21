import streamlit as st
from research_graph import ResearchGraph, ResearchState
from typing import List, Dict

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'show_thinking' not in st.session_state:
    st.session_state.show_thinking = False

if 'current_thinking' not in st.session_state:
    st.session_state.current_thinking = {}

# Initialize the research graph
@st.cache_resource
def get_research_graph():
    return ResearchGraph()

# Function to display thinking area
def display_thinking_area(state: ResearchState):
    with st.expander("🤔 Thinking Process", expanded=st.session_state.show_thinking):
        st.markdown("### Initial Context")
        st.markdown(state.get("initial_context", ""))
        
        st.markdown("### Generated Sub-queries")
        for i, query in enumerate(state.get("sub_queries", []), 1):
            st.markdown(f"{i}. {query}")
        
        st.markdown("### Scraped Information")
        for item in state.get("scraped_info", []):
            st.markdown(f"**🔍 {item['query']}**")
            st.markdown(item['content'][:500] + "...")
        
        st.markdown("### Combined Context")
        st.markdown(state.get("combined_context", ""))

# Function to display chat message
def display_message(role: str, content: str):
    with st.chat_message(role):
        st.markdown(content)

# Main Streamlit app
def main():
    st.title("🤖 Research Assistant")
    
    # Sidebar controls
    with st.sidebar:
        st.title("Controls")
        st.session_state.show_thinking = st.checkbox("Show Thinking Process", value=st.session_state.show_thinking)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        display_message(message["role"], message["content"])
        if message.get("thinking"):
            display_thinking_area(message["thinking"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to research?"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt)
        
        # Show thinking spinner
        with st.spinner("Researching..."):
            try:
                # Get research graph
                research_graph = get_research_graph()
                
                # Run research workflow
                final_state = research_graph.run_research(prompt)
                
                # Add assistant response to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_state["final_response"],
                    "thinking": final_state
                })
                
                # Display assistant response
                display_message("assistant", final_state["final_response"])
                
                # Display thinking area if enabled
                if st.session_state.show_thinking:
                    display_thinking_area(final_state)
                
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                display_message("assistant", error_message)

if __name__ == "__main__":
    main() 