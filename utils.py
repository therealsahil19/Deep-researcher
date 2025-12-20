import openai
import logging
from fpdf import FPDF
from fpdf.errors import FPDFUnicodeEncodingException
from fpdf.enums import XPos, YPos
import re
import json
import os

from datetime import datetime
from tavily import TavilyClient
from exa_py import Exa

# Import centralized configuration
import config

# Configure logging
logging.basicConfig(
    filename='deep_research.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# ACTION INPUT PARSING AND VALIDATION
# =============================================================================

def parse_action_input(raw_input: str) -> str:
    """
    Parse and clean action input from LLM output.
    
    Handles various malformed formats the LLM might output:
    - JSON arrays: ["query"] or ['query'] -> extracts first element
    - JSON objects: {"query": "..."} -> extracts the value
    - Quoted strings: "query" or 'query' -> unquotes
    - Plain text: query -> returns as-is
    
    Returns the cleaned query string.
    """
    if not raw_input:
        return ""
    
    cleaned = raw_input.strip()

    # Remove markdown code blocks if present (e.g. ```json ... ```)
    # This handles cases where the model wraps the input in code blocks
    if '```' in cleaned:
        # Match content inside ```...```
        # Use DOTALL to match across newlines
        match = re.search(r'```(?:\w+)?\s*(.*?)\s*```', cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        else:
            # Fallback: simple replace if the regex fails for some reason
            cleaned = cleaned.replace("```", "").strip()
            # Also remove language identifier if it remains (e.g. "json\n...")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

    # Remove accidentally captured "Action Input:" prefix
    if cleaned.lower().startswith("action input:"):
        cleaned = cleaned[13:].strip()
    
    # Try to parse as JSON array: ["query"] or ["query1", "query2"]
    if cleaned.startswith('[') and cleaned.endswith(']'):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and len(parsed) > 0:
                # Take the first element if it's a list
                result = str(parsed[0]).strip()
                logger.info(f"Parsed JSON array, extracted: {result}")
                return result
        except json.JSONDecodeError:
            # Not valid JSON, strip the brackets and try to extract
            inner = cleaned[1:-1].strip()
            # Remove quotes if present
            inner = inner.strip('"\'')
            logger.info(f"Stripped brackets, extracted: {inner}")
            return inner
    
    # Try to parse as JSON object: {"query": "...", ...}
    if cleaned.startswith('{') and cleaned.endswith('}'):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                # Look for common keys
                for key in ['query', 'q', 'search', 'input', 'text']:
                    if key in parsed:
                        result = str(parsed[key]).strip()
                        logger.info(f"Parsed JSON object, extracted '{key}': {result}")
                        return result
                # If no common key, take the first value
                if parsed:
                    result = str(list(parsed.values())[0]).strip()
                    logger.info(f"Parsed JSON object, extracted first value: {result}")
                    return result
        except json.JSONDecodeError:
            # Not valid JSON, extract content between braces
            inner = cleaned[1:-1].strip()
            # Try to find a quoted value
            quote_match = re.search(r'["\']([^"\']+)["\']', inner)
            if quote_match:
                result = quote_match.group(1).strip()
                logger.info(f"Extracted quoted value from braces: {result}")
                return result
    
    # Strip surrounding quotes (single or double)
    if (cleaned.startswith('"') and cleaned.endswith('"')) or \
       (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    
    # Also strip escaped quotes that might remain
    cleaned = cleaned.replace('\\"', '"').replace("\\'", "'")
    
    # Remove any trailing "Observation:" that the model might have hallucinated
    if "Observation:" in cleaned:
        cleaned = cleaned.split("Observation:")[0].strip()
    
    # Final cleanup of residual quotes
    cleaned = cleaned.strip('"\'')
    
    return cleaned


def validate_search_query(query: str) -> tuple[bool, str]:
    """
    Validate a search query before sending to search APIs.
    
    Returns:
        (is_valid, error_message)
    """
    if not query:
        return False, "Empty query"
    
    # Minimum length check
    if len(query) < 3:
        return False, f"Query too short ({len(query)} chars): '{query}'"
    
    # Check if query is just punctuation/brackets
    if query in ['[', ']', '{', '}', '(', ')', '[{', '{}', '[]', '```', '``']:
        return False, f"Query is just punctuation: '{query}'"
    
    # Check if query looks like incomplete JSON
    if re.match(r'^[\[\{\(\"\']$', query):
        return False, f"Query looks like incomplete JSON: '{query}'"
    
    return True, ""



def get_react_system_prompt():
    """
    Generates the ReAct system prompt with dynamic current date for temporal grounding.
    This prevents hallucinations about recent events by forcing the model to search.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    return f"""
You are the Deep Research Agent, an expert investigator powered by Xiaomi MiMo-V2-Flash.
Current Date: {current_date}

Your Goal: Answer the user's query by conducting a multi-step investigation.
You have access to a massive 256k context window, so prioritize thoroughness over brevity.

--- RESEARCH METHODOLOGY ---
1. **Decompose**: Break complex queries into smaller, searchable sub-questions.
2. **Diversity**: Use "exa_search" for broad discovery and "tavily_search" for specific fact-checking.
3. **Synthesis**: Combine conflicting information by noting the discrepancy, rather than ignoring it.
4. **Citations**: Every claim must be supported by the search results provided.

--- FORMAT INSTRUCTIONS (STRICT) ---
You must strictly follow this ReAct loop format. Do not use internal <think> tags.

Thought: [Your reasoning about what information is missing or needs verification]
Action: [The tool to use: 'search_discovery' or 'search_fact']
Action Input: [The exact search query - strictly one line]
Observation: [Wait for the tool output]

... (Repeat Thought/Action/Observation as needed) ...

Final Answer: [Your comprehensive report, fully cited]

--- RULES & ANTI-HALLUCINATION ---
1. **No Silent Failures**: If a search fails, admit it in the "Thought" and try a different query.
2. **Zero-Shot Accuracy**: Do not guess specific dates, prices, or version numbers. Search for them.
3. **Date Awareness**: If the user asks for "latest" news, strictly check the date of the search results.
4. **Action Input constraint**: "Action Input" must be a single string. No JSON or Markdown blocks inside the input.
5. If you have sufficient information to answer the user request, go straight to "Final Answer".
"""


def check_and_update_limit(service_name):
    """
    Checks if the rate limit for the given service has been reached.
    Uses settings from config.RATE_LIMIT_CONFIG.
    Returns (True, "") if allowed, or (False, reason) if blocked.
    """
    # Check if rate limiting is enabled
    if not config.RATE_LIMIT_CONFIG.enabled:
        return True, ""
    
    today = datetime.now().strftime("%Y-%m-%d")
    this_month = datetime.now().strftime("%Y-%m")
    
    usage_file = config.RATE_LIMIT_CONFIG.usage_file
    daily_limit = config.RATE_LIMIT_CONFIG.daily_limit
    monthly_limit = config.RATE_LIMIT_CONFIG.monthly_limit

    initial_service_data = {"day": today, "month": this_month, "daily_count": 0, "monthly_count": 0}

    # Initialize usage data if file doesn't exist or is invalid
    usage_data = {}
    if os.path.exists(usage_file):
        try:
            with open(usage_file, "r") as f:
                usage_data = json.load(f)
        except json.JSONDecodeError:
            pass

    # Ensure services exist in data
    if "tavily" not in usage_data:
        usage_data["tavily"] = initial_service_data.copy()
    if "exa" not in usage_data:
        usage_data["exa"] = initial_service_data.copy()

    service_data = usage_data.get(service_name)
    if not service_data:
        service_data = initial_service_data.copy()
        usage_data[service_name] = service_data

    # Check for Month Reset
    if service_data["month"] != this_month:
        service_data["month"] = this_month
        service_data["monthly_count"] = 0
        service_data["day"] = today
        service_data["daily_count"] = 0

    # Check for Day Reset
    elif service_data["day"] != today:
        service_data["day"] = today
        service_data["daily_count"] = 0

    # Check Limits
    if service_data["monthly_count"] >= monthly_limit:
        return False, f"Monthly limit of {monthly_limit} requests reached for {service_name}. Resets next month."

    if service_data["daily_count"] >= daily_limit:
        return False, f"Daily limit of {daily_limit} requests reached for {service_name}. Resets tomorrow."

    # Increment Usage
    service_data["daily_count"] += 1
    service_data["monthly_count"] += 1

    # Save back to file
    try:
        with open(usage_file, "w") as f:
            json.dump(usage_data, f, indent=4)
    except Exception as e:
        return False, f"Error saving usage data: {str(e)}"

    return True, ""

def extract_date_range_from_query(query):
    """
    Extracts date range from a query string.
    Returns (start_date, end_date) in YYYY-MM-DD format, or (None, None) if no date found.
    """
    import calendar
    
    # Pattern for "Month Year" (e.g., "November 2025" or "Nov 2025")
    month_year_pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{4})'
    match = re.search(month_year_pattern, query, re.IGNORECASE)
    
    if match:
        # Normalize month name
        raw_month = match.group(1).lower()
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        # Handle full names by taking first 3 chars
        month_key = raw_month[:3]
        month_num = month_map.get(month_key)

        if not month_num:
             # Fallback to standard parsing if needed (shouldn't happen with regex above)
             try:
                 month_name = raw_month.capitalize()
                 month_num = list(calendar.month_name).index(month_name)
             except ValueError:
                 return None, None

        year = int(match.group(2))
        
        # Get the last day of the month
        _, last_day = calendar.monthrange(year, month_num)
        
        start_date = f"{year}-{month_num:02d}-01"
        end_date = f"{year}-{month_num:02d}-{last_day:02d}"
        
        return start_date, end_date
    
    # Pattern for just a year (e.g., "2025")
    year_pattern = r'\b(202[4-9]|203\d)\b'  # Match years 2024-2039
    year_match = re.search(year_pattern, query)
    
    if year_match:
        year = int(year_match.group(1))
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        return start_date, end_date
    
    return None, None

def tavily_search(query, api_key, num_results=None):
    """
    Performs a targeted search using Tavily with date filtering.
    
    Args:
        query: Search query string
        api_key: Tavily API key
        num_results: Number of results (defaults to config.SEARCH_CONFIG.default_results)
    """
    allowed, message = check_and_update_limit("tavily")
    if not allowed:
        return f"Error: {message}"
    
    if num_results is None:
        num_results = config.SEARCH_CONFIG.default_results

    try:
        # Extract date range from query
        start_date, end_date = extract_date_range_from_query(query)
        
        client = TavilyClient(api_key=api_key)
        
        # Build search parameters
        search_params = {
            "query": query,
            "search_depth": "advanced",
            "max_results": num_results
        }
        
        # Add date filters if dates were extracted
        if start_date and end_date:
            # Tavily uses days parameter for recency - calculate days from now
            query_start = datetime.strptime(start_date, "%Y-%m-%d")
            days_ago = (datetime.now() - query_start).days
            if days_ago > 0:
                search_params["days"] = min(days_ago + 60, 365)  # Add buffer, max 1 year
            logger.info(f"Tavily search with date filter: {start_date} to {end_date}, days={search_params.get('days', 'none')}")
        
        response = client.search(**search_params)
        
        # Include publication dates in output if available
        context = []
        for res in response.get('results', []):
            entry = f"Source: {res['url']}"
            if res.get('published_date'):
                entry += f"\nPublished: {res['published_date']}"
            entry += f"\nContent: {res['content']}"
            context.append(entry)
        
        if not context:
            return f"No results found for query: {query}"
        
        return "\n\n".join(context)
    except Exception as e:
        logger.error(f"Tavily search error: {str(e)}")
        return f"Error performing Tavily search: {str(e)}"

def exa_search(query, api_key, num_results=None):
    """
    Performs a semantic search using Exa with date filtering.
    
    Args:
        query: Search query string
        api_key: Exa API key
        num_results: Number of results (defaults to config.SEARCH_CONFIG.default_results)
    """
    allowed, message = check_and_update_limit("exa")
    if not allowed:
        return f"Error: {message}"
    
    if num_results is None:
        num_results = config.SEARCH_CONFIG.default_results

    try:
        # Extract date range from query
        start_date, end_date = extract_date_range_from_query(query)
        
        exa = Exa(api_key=api_key)
        
        # Build search parameters
        search_params = {
            "query": query,
            "num_results": num_results,
        }
        
        # Add date filters if dates were extracted
        if start_date:
            search_params["start_published_date"] = start_date
            logger.info(f"Exa search with start_published_date: {start_date}")
        if end_date:
            search_params["end_published_date"] = end_date
            logger.info(f"Exa search with end_published_date: {end_date}")
        
        response = exa.search_and_contents(
            **search_params,
            text=True  # Request text content from results
        )
        
        # Include publication dates in output
        context = []
        for res in response.results:
            entry = f"Title: {res.title}\nSource: {res.url}"
            if hasattr(res, 'published_date') and res.published_date:
                entry += f"\nPublished: {res.published_date}"
            # Safely access text content with null check
            text_content = res.text[:500] if res.text else "(No text content available)"
            entry += f"\nContent: {text_content}..."
            context.append(entry)
        
        if not context:
            return f"No results found for query: {query}"
        
        return "\n\n".join(context)
    except Exception as e:
        logger.error(f"Exa search error: {str(e)}")
        return f"Error performing Exa search: {str(e)}"

def stream_deep_research(messages, api_keys, model_id=None, num_results=None):
    """
    Streams the response from the Deep Research model via OpenRouter.
    Implements a ReAct loop if tools (Tavily or Exa) are provided.
    
    Args:
        messages: List of message dicts for the conversation
        api_keys: Dict with 'openrouter', 'tavily', 'exa' keys
        model_id: Model ID to use (defaults to config.DEFAULT_MODEL_ID)
        num_results: Number of search results to return (defaults to config setting)
    """
    logger.info("Starting deep research stream")
    
    # Use config defaults if not specified
    if model_id is None:
        model_id = config.DEFAULT_MODEL_ID
    if num_results is None:
        num_results = config.SEARCH_CONFIG.default_results

    if not api_keys.get("openrouter"):
        logger.error("OpenRouter API Key not provided")
        yield "Error: OpenRouter API Key not provided."
        return

    # Log key details (masked) for debugging
    key = api_keys["openrouter"]
    masked_key = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
    logger.info(f"Using OpenRouter Key: {masked_key} (Length: {len(key)})")

    client = openai.OpenAI(
        base_url=config.API_CONFIG.openrouter_base_url,
        api_key=api_keys["openrouter"],
        timeout=config.API_CONFIG.request_timeout,  # Prevent timeout on long requests
        default_headers={
            "HTTP-Referer": config.API_CONFIG.app_referer,
            "X-Title": config.API_CONFIG.app_title,
        }
    )

    internal_messages = list(messages)

    # Determine available tools
    available_tools = []
    if api_keys.get("exa"):
        available_tools.append("search_discovery")
    if api_keys.get("tavily"):
        available_tools.append("search_fact")

    # Inject System Prompt if tools are available
    if available_tools:
        system_prompt = get_react_system_prompt()  # Dynamic prompt with current date
        system_msg_exists = False
        for msg in internal_messages:
            if msg['role'] == 'system':
                if "Answer the following question. You have access to the following tools" not in msg['content']:
                        msg['content'] += f"\n\n{system_prompt}"
                system_msg_exists = True
                break

        if not system_msg_exists:
            internal_messages.insert(0, {"role": "system", "content": system_prompt})
    else:
        # If no tools, we don't inject ReAct prompt, just rely on model knowledge
        pass

    max_steps = config.REACT_CONFIG.max_steps
    step_count = 0

    # Configure dynamic parameters for MiMo vs others
    temperature = config.TEMPERATURE if "mimo" in model_id or "flash" in model_id else 0.7
    top_p = config.TOP_P

    extra_body = {}
    if "mimo" in model_id:
        extra_body["include_reasoning"] = False

    while step_count < max_steps:
        step_count += 1
        
        # Yield progress indicator
        yield f"\n\n---\n**🔄 Step {step_count}/{max_steps}**: Reasoning...\n\n"

        try:
            stream = client.chat.completions.create(
                model=model_id,
                messages=internal_messages,
                temperature=temperature,
                top_p=top_p,
                extra_body=extra_body,
                stream=True
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content

            # Check if we should look for actions
            if not available_tools:
                break

            # Remove markdown code blocks that might interfere with parsing
            clean_response = full_response.replace("```", "")
            
            # Parse for Action - case insensitive and handles variations
            # Matches: search_discovery, search-discovery, searchdiscovery, etc.
            action_pattern = r"Action:\s*(search[_\-\s]?discovery|search[_\-\s]?fact)"
            action_match = re.search(action_pattern, clean_response, re.IGNORECASE)
            # More flexible input matching - capture everything after "Action Input:" until end of line
            input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", clean_response, re.IGNORECASE)
            
            # Debug logging
            logger.info(f"Step {step_count}: Action match: {action_match.group(0) if action_match else 'None'}")
            logger.info(f"Step {step_count}: Input match raw: {input_match.group(1) if input_match else 'None'}")

            if action_match and input_match:
                # Normalize action name (remove hyphens/spaces, lowercase)
                raw_action = action_match.group(1).lower()
                action_name = re.sub(r'[\-\s]', '_', raw_action)  # Normalize to underscore format
                
                # Use robust parsing for action input
                raw_tool_input = input_match.group(1).strip()
                tool_input = parse_action_input(raw_tool_input)
                
                logger.info(f"Step {step_count}: Raw input: '{raw_tool_input}' -> Parsed: '{tool_input}'")
                
                # Validate the query before executing search
                is_valid, error_msg = validate_search_query(tool_input)
                if not is_valid:
                    logger.warning(f"Step {step_count}: Invalid query rejected: {error_msg}")
                    # Feed error back to model with clear instruction
                    observation = f"SYSTEM ERROR: {error_msg}. Please provide a plain text search query like 'LLMs released November 2025' without JSON formatting."
                    internal_messages.append({"role": "assistant", "content": full_response})
                    internal_messages.append({"role": "user", "content": f"Observation: {observation}"})
                    yield f"\n\n*⚠️ Invalid query format detected. Asking model to retry...*\n\n"
                    continue  # Skip to next iteration
                
                logger.info(f"Executing {action_name} with query: {tool_input}")

                observation = ""

                if action_name == "search_discovery" and "search_discovery" in available_tools:
                    yield f"\n\n**🔍 Step {step_count}/{max_steps}**: Executing Discovery Search (Exa)...\n*Query: {tool_input}*\n\n"
                    observation = exa_search(tool_input, api_keys["exa"], num_results)
                    logger.info(f"Exa search result length: {len(observation)}")
                    
                    # Detect "no results" error (Exa returns ~61 chars for empty results)
                    if len(observation) < 100 or observation.startswith("Error:") or "No results" in observation:
                         yield f"\n\n*{observation}*\n\n"
                         logger.warning(f"Exa search returned minimal results: {observation}")
                         # Feed error back to model with clear instruction
                         observation = f"SYSTEM ERROR: Search returned no results for '{tool_input}'. Please try a different, broader search query."

                elif action_name == "search_fact" and "search_fact" in available_tools:
                    yield f"\n\n**🔍 Step {step_count}/{max_steps}**: Executing Fact Search (Tavily)...\n*Query: {tool_input}*\n\n"
                    observation = tavily_search(tool_input, api_keys["tavily"], num_results)
                    logger.info(f"Tavily search result length: {len(observation)}")
                    
                    # Detect errors or minimal results
                    if len(observation) < 100 or observation.startswith("Error:") or "No results" in observation:
                         yield f"\n\n*{observation}*\n\n"
                         logger.warning(f"Tavily search returned minimal results: {observation}")
                         # Feed error back to model with clear instruction
                         observation = f"SYSTEM ERROR: Search returned no results for '{tool_input}'. Please try a different, broader search query."

                else:
                    # Model tried to use a tool it doesn't have access to or hallucinates
                    msg = f"Model attempted to use {action_name} but tool is unavailable."
                    logger.warning(msg)
                    yield f"\n\n*⚠️ {msg}*\n\n"

                    # Feed error back to model instead of breaking
                    observation = f"SYSTEM ERROR: Tool '{action_name}' is not available. Available tools: {available_tools}. Please use an available tool."

                # Update history
                internal_messages.append({"role": "assistant", "content": full_response})
                internal_messages.append({"role": "user", "content": f"Observation: {observation}"})

                yield f"\n\n*Observation obtained. Analyzing...*\n\n"

                # Loop continues
            else:
                # No action found, we are done
                logger.info(f"Step {step_count}: No action pattern found, ending loop")
                break

        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}", exc_info=True)
            yield f"Error: {str(e)}"
            break

def verify_claims_with_sequential_thinking(claims_text, search_results, api_keys):
    """
    Uses Sequential Thinking methodology to verify claims in the research.
    This function implements a structured verification process to catch hallucinations.
    
    Returns a verification report with confidence levels for each major claim.
    """
    if not api_keys.get("openrouter"):
        return None, "OpenRouter API key required for verification"
    
    client = openai.OpenAI(
        base_url=config.API_CONFIG.openrouter_base_url,
        api_key=api_keys["openrouter"],
        timeout=config.API_CONFIG.request_timeout,
        default_headers={
            "HTTP-Referer": config.API_CONFIG.app_referer,
            "X-Title": f"{config.API_CONFIG.app_title} - Verification",
        }
    )
    
    current_date = datetime.now().strftime("%B %d, %Y")
    
    verification_prompt = f"""You are a fact-checking assistant using Sequential Thinking methodology.
Today's date is: {current_date}

Your task is to verify claims from a research draft against the search results provided.

## VERIFICATION METHODOLOGY (Sequential Thinking):

### Step 1: Extract Claims
Identify all factual claims in the draft (dates, names, numbers, events, product releases, etc.)

### Step 2: Cross-Reference Each Claim
For each claim, check if it appears in the search results. Mark as:
- ✅ VERIFIED: Claim is directly supported by search results
- ⚠️ PARTIALLY VERIFIED: Some support but not exact match
- ❌ UNVERIFIED: No evidence found in search results
- 🚨 CONTRADICTED: Search results contradict this claim

### Step 3: Flag Hallucination Risks
Identify claims that:
- Reference specific dates/versions with no search support
- Mention products or releases not in search results
- Make precise claims (numbers, rankings) without sources

### Step 4: Generate Verification Report
Summarize what is verified vs uncertain.

## RESEARCH DRAFT TO VERIFY:
{claims_text}

## SEARCH RESULTS (Source of Truth):
{search_results}

## OUTPUT FORMAT:
Provide a structured verification report with:
1. List of verified claims (with source citations)
2. List of unverified/uncertain claims
3. Any potential hallucinations detected
4. Overall confidence score (HIGH/MEDIUM/LOW)
5. Recommendations for the final report

Be strict - if a claim cannot be verified from the search results, it should be flagged.
"""

    # Configure dynamic parameters for verification
    # Use default model (MiMo) or fallback to config default
    model_id = config.DEFAULT_MODEL_ID

    temperature = config.TEMPERATURE if "mimo" in model_id or "flash" in model_id else 0.7
    top_p = config.TOP_P

    extra_body = {}
    if "mimo" in model_id:
        extra_body["include_reasoning"] = False

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": verification_prompt}],
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body,
            stream=False
        )
        
        verification_report = response.choices[0].message.content
        return verification_report, None
        
    except Exception as e:
        logger.error(f"Sequential Thinking verification failed: {str(e)}")
        return None, str(e)

def apply_verification_to_report(original_report, verification_report):
    """
    Applies the verification results to enhance the final report.
    Adds uncertainty markers and source citations based on verification.
    """
    if not verification_report:
        return original_report
    
    # Add verification summary at the end of the report
    enhanced_report = f"""{original_report}

---

## 📋 Research Verification Summary

{verification_report}

---
*This report was verified using Sequential Thinking methodology to identify potential inaccuracies.*
"""
    return enhanced_report

def generate_pdf(text):
    """
    Generates a PDF from the provided text and returns the bytes.
    Uses fpdf2.
    """
    class PDF(FPDF):
        def header(self):
            # Attempt to use DejaVuSans if registered, else fallback
            font_family = 'DejaVu' if 'DejaVu' in self.fonts else 'helvetica'
            self.set_font(font_family, 'B', 12)
            self.cell(0, 10, 'Deep Research Report', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            font_family = 'DejaVu' if 'DejaVu' in self.fonts else 'helvetica'
            self.set_font(font_family, 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')

    pdf = PDF()

    # Register the Unicode font
    try:
        pdf.add_font('DejaVu', '', 'assets/fonts/DejaVuSans.ttf')
        pdf.add_font('DejaVu', 'B', 'assets/fonts/DejaVuSans.ttf')  # Fallback: regular font for bold (no styling)
        pdf.add_font('DejaVu', 'I', 'assets/fonts/DejaVuSans.ttf')  # Fallback: regular font for italic (no styling)
        font_family = 'DejaVu'
    except Exception:
        font_family = 'helvetica'

    pdf.add_page()
    pdf.set_font(font_family, size=12)

    try:
        # Try to use markdown=True if available in this version of fpdf2
        # Use a compatible font if loaded
        pdf.multi_cell(0, 10, text, markdown=True)
    except TypeError:
        # Fallback for older versions or if markdown param is not supported in this specific way
        pdf.multi_cell(0, 10, text)
    except (UnicodeEncodeError, FPDFUnicodeEncodingException):
        # Fallback for fpdf2 if it tries to encode and fails (should be rare with DejaVu)
        # We manually sanitize if the automatic handling fails
        sanitized_text = text.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, sanitized_text)

    # fpdf2 output() returns a bytearray if no name is provided
    return bytes(pdf.output())
