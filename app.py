import streamlit as st
import utils
import config
import os
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔍",
    layout="wide"
)

# Initialize Session State
if "reasoning_content" not in st.session_state:
    st.session_state.reasoning_content = ""

if "final_report" not in st.session_state:
    st.session_state.final_report = ""

if "research_complete" not in st.session_state:
    st.session_state.research_complete = False

if "verification_report" not in st.session_state:
    st.session_state.verification_report = ""

# Session History
if "research_history" not in st.session_state:
    st.session_state.research_history = []  # List of {query, report, timestamp}

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    st.info("**Mode:** Comprehensive Research\n\n*Maximum compute + reasoning + search*")

    st.markdown("---")
    
    # Model Selection
    st.subheader("🤖 Model")
    model_choices = config.get_model_choices()
    selected_model_name = st.selectbox(
        "Select Model",
        options=list(model_choices.keys()),
        index=0,
        help="Choose the LLM model for research"
    )
    selected_model_id = model_choices[selected_model_name]
    
    # Search Settings
    st.subheader("🔍 Search Settings")
    search_depth = st.slider(
        "Results per search",
        min_value=config.SEARCH_CONFIG.min_results,
        max_value=config.SEARCH_CONFIG.max_results,
        value=config.SEARCH_CONFIG.default_results,
        help="Number of search results to fetch per query"
    )
    
    st.markdown("---")

    st.subheader("🔑 API Keys")

    # Helper to display key input with status
    def render_key_input(label, env_var, help_text):
        key_input = st.text_input(label, type="password", help=help_text)
        if not key_input and os.environ.get(env_var):
            st.caption(f"✅ Loaded from environment (`{env_var}`)")
        return key_input

    openrouter_api_key = render_key_input("OpenRouter API Key", "OPENROUTER_API_KEY", "Required for the LLM.")
    tavily_api_key = render_key_input("Tavily API Key", "TAVILY_API_KEY", "Optional. Required for Fact Search.")
    exa_api_key = render_key_input("Exa API Key", "EXA_API_KEY", "Optional. Required for Discovery Search.")

    # Prioritize user input, fallback to environment variables
    def get_api_key(user_input, env_var):
        key = user_input or os.environ.get(env_var)
        return key.strip() if key else None

    api_keys = {
        "openrouter": get_api_key(openrouter_api_key, "OPENROUTER_API_KEY"),
        "tavily": get_api_key(tavily_api_key, "TAVILY_API_KEY"),
        "exa": get_api_key(exa_api_key, "EXA_API_KEY")
    }
    
    # Session History Section
    st.markdown("---")
    st.subheader("📜 Research History")
    
    if st.session_state.research_history:
        for i, session in enumerate(reversed(st.session_state.research_history)):
            with st.expander(f"📄 {session['query'][:40]}...", expanded=False):
                st.caption(f"*{session['timestamp']}*")
                st.markdown(session['report'][:500] + "..." if len(session['report']) > 500 else session['report'])
        
        with st.popover("🗑️ Clear History", use_container_width=True):
            st.markdown("Are you sure you want to delete all research history? This action cannot be undone.")
            if st.button("Yes, delete everything", type="primary", use_container_width=True):
                st.session_state.research_history = []
                st.toast("Research history cleared!", icon="✅")
                st.rerun()
    else:
        st.info("No research history yet. Start a new query to build your library!")

# Main Content Area
st.title("Deep Research Agent 🔍")
st.caption("Comprehensive Research Mode — Maximum compute + reasoning + search")

# Input Section
prompt = st.text_area("Enter your research topic or question:", height=120, placeholder="e.g., Generate a complete research report on LLMs released in November 2025")

col1, col2 = st.columns([0.15, 0.85])
with col1:
    has_openrouter = bool(api_keys["openrouter"])
    help_text = "Click to start research" if has_openrouter else "⚠️ OpenRouter API Key required in sidebar"
    start_button = st.button("🚀 Start Research", type="primary", use_container_width=True, disabled=not has_openrouter, help=help_text)

# Research Logic
if start_button:
    if not prompt or not prompt.strip():
        st.error("Please enter a valid research topic.")
    elif not api_keys["openrouter"]:
        # Double check in case of state weirdness, though button should be disabled
        st.error("Please provide an OpenRouter API Key in the sidebar.")
    else:
        # Reset state
        st.session_state.reasoning_content = ""
        st.session_state.final_report = ""
        st.session_state.research_complete = False

        # Create tabs for output
        tab_report, tab_reasoning = st.tabs(["📊 Report", "🧠 Reasoning"])

        with tab_reasoning:
            reasoning_placeholder = st.empty()

        with tab_report:
            report_placeholder = st.empty()
            report_placeholder.info("⏳ Research in progress... Check the **Reasoning** tab to see the agent's thought process.")

        full_response = ""

        with st.spinner("Researching..."):
            messages = [{"role": "user", "content": prompt.strip()}]

            for chunk in utils.stream_deep_research(messages, api_keys, model_id=selected_model_id, num_results=search_depth):
                if chunk.startswith("Error:"):
                    st.error(chunk)
                    full_response = ""
                    break
                full_response += chunk
                # Update reasoning tab in real-time
                reasoning_placeholder.markdown(full_response + "▌")

            if full_response:
                # Final update to reasoning
                reasoning_placeholder.markdown(full_response)
                st.session_state.reasoning_content = full_response

                # Extract final report (everything after "Final Answer:")
                if "Final Answer:" in full_response:
                    final_report = full_response.split("Final Answer:", 1)[1].strip()
                else:
                    # Fallback: use entire response if no marker found
                    final_report = full_response

                # Run Sequential Thinking verification pass (with error handling)
                try:
                    with st.spinner("🔍 Verifying claims with Sequential Thinking..."):
                        # Extract search results from reasoning for verification context
                        search_context = ""
                        for line in full_response.split("\n"):
                            if "Source:" in line or "Content:" in line or "Title:" in line:
                                search_context += line + "\n"
                        
                        if search_context:
                            verification_report, error = utils.verify_claims_with_sequential_thinking(
                                final_report, 
                                search_context, 
                                api_keys
                            )
                            
                            if verification_report and not error:
                                # Apply verification to enhance the report
                                final_report = utils.apply_verification_to_report(final_report, verification_report)
                                st.session_state.verification_report = verification_report
                            else:
                                st.warning(f"⚠️ Verification step skipped: {error if error else 'No search context available'}")
                        else:
                            st.info("ℹ️ No search context found for verification. Report based on model knowledge.")
                except Exception as e:
                    st.warning(f"⚠️ Verification failed: {str(e)}. Report generated without verification.")

                st.session_state.final_report = final_report
                st.session_state.research_complete = True
                
                # Save to session history
                st.session_state.research_history.append({
                    "query": prompt.strip(),
                    "report": final_report,
                    "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M")
                })
                
                # Show completion message inline instead of using st.rerun()
                # st.rerun() can cause WebSocket disconnection after long streaming operations
                st.success("✅ Research Complete! See results below.")

# Show results and download button if research is complete
if st.session_state.research_complete and st.session_state.final_report:
    st.markdown("---")
    st.success("✅ Research Complete!")

    # Show tabs with stored content
    tab_report, tab_reasoning = st.tabs(["📊 Report", "🧠 Reasoning"])

    with tab_report:
        st.markdown(st.session_state.final_report)

    with tab_reasoning:
        st.markdown(st.session_state.reasoning_content)

    # Download buttons
    col_pdf, col_md = st.columns(2)
    
    with col_pdf:
        pdf_bytes = utils.generate_pdf(st.session_state.final_report)
        st.download_button(
            label="📄 Download as PDF",
            data=pdf_bytes,
            file_name="deep_research_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    
    with col_md:
        st.download_button(
            label="📝 Download as Markdown",
            data=st.session_state.final_report,
            file_name="deep_research_report.md",
            mime="text/markdown",
            use_container_width=True
        )
