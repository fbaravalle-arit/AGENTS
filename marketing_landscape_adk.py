"""
Marketing Intelligence Agent for Italian B2B Market Analysis
Uses Google ADK to analyze competitor websites and generate strategic recommendations
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search_tool, google_search_agent_tool
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Check for Google AI API key
if not os.environ.get("GOOGLE_API_KEY"):
    print("\n" + "="*70)
    print("ERROR: GOOGLE_API_KEY environment variable not set")
    raise ValueError(
        "Missing GOOGLE_API_KEY. Set the environment variable before running this script."
    )
else:
    print(f"✅ Google AI API key loaded successfully")


retry_config = types.HttpRetryOptions(
    attempts=3,
    exp_base=5,
    initial_delay=5,
    http_status_codes=[429, 500, 503, 504]
)

# Create the model instance for the marketing agent
marketing_model = Gemini(model="gemini-3-pro-preview", retry_options=retry_config)

# Define the specific instruction for marketing analysis
MARKETING_INSTRUCTION = """
You are an expert Marketing Strategist. Your goal is to analyze a website URL and extract comprehensive marketing intelligence.

You have access to the 'google_search_tool.GoogleSearchTool()' tool to fetch and analyze the website content.
Read the entire landing page and different sections thoroughly.

CRITICAL: You MUST output your analysis ONLY as valid JSON following this exact structure:

{
  "company_name": "string - The company name",
  "url": "string - The analyzed URL",
  "analysis_date": "string - Today's date in YYYY-MM-DD format",
  "executive_summary": {
    "overview": "string - Brief company overview",
    "brand_details": "string - Brand positioning and identity",
    "market_positioning": "string - How they position in the market"
  },
  "buyer_personas": [
    {
      "title": "string - Persona title/role",
      "description": "string - Detailed description of this persona"
    }
  ],
  "jobs_to_be_done": {
    "functional": ["array of strings - Functional jobs"],
    "emotional": ["array of strings - Emotional jobs"],
    "social": ["array of strings - Social jobs"]
  },
  "problems_addressed": [
    {
      "problem": "string - Problem title",
      "description": "string - Detailed description"
    }
  ],
  "value_proposition": {
    "unique_selling_points": ["array of strings - Key USPs"],
    "differentiation": "string - How they differentiate",
    "competitive_advantages": ["array of strings - Competitive edges"]
  },
  "problem_uvp_fit": {
    "fit_level": "string - High/Medium/Low",
    "analysis": "string - Explanation of the fit"
  },
  "proof_points": {
    "associations": ["array of strings - Professional associations"],
    "key_clients": ["array of strings - Notable clients"],
    "statistics": ["array of strings - Key stats and metrics"],
    "testimonials": ["array of strings - Customer testimonials"]
  },
  "products": [
    {
      "name": "string - Product name",
      "description": "string - Product description",
      "features": ["array of strings - Key features"],
      "target_use_case": "string - Primary use case"
    }
  ],
  "pricing": {
    "model": "string - Pricing model type",
    "purchase_options": "object - Purchase details",
    "rental_options": "object - Rental/subscription details",
    "special_offers": ["array of strings - Discounts or special offers"]
  },
  "branding": {
    "tone": "string - Brand tone",
    "colors": ["array of strings - Brand colors"],
    "imagery_focus": "string - Focus of visual content"
  },
  "seo": {
    "primary_keywords": ["array of strings - Primary SEO keywords"],
    "secondary_keywords": ["array of strings - Secondary keywords"]
  },
  "ctas": {
    "primary": ["array of strings - Primary CTAs"],
    "secondary": ["array of strings - Secondary CTAs"],
    "contact_methods": ["array of strings - Contact options"]
  }
}

OUTPUT ONLY THE JSON. DO NOT include markdown formatting, explanations, or any text outside the JSON structure.
"""

marketing_agent = Agent(
    name="marketing_intelligence_agent",
    model=marketing_model,
    description="Marketing Intelligence Agent specializing in Italian B2B market analysis",
    instruction=MARKETING_INSTRUCTION,
    tools=[google_search_tool.GoogleSearchTool()]
)

marketing_runner = InMemoryRunner(agent=marketing_agent)

def save_company_analysis(response_text: str, url: str) -> Path:
    """
    Extract JSON from LLM response and save to /competitors folder

    Args:
        response_text: The full response text from the LLM
        url: The analyzed URL

    Returns:
        Path to the saved JSON file
    """
    import re

    # Remove markdown code fences if present
    response_text = re.sub(r'```json\s*', '', response_text)
    response_text = re.sub(r'```\s*$', '', response_text)
    response_text = response_text.strip()

    # Try to extract JSON from the response
    # Look for content between first { and last }
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

    if not json_match:
        raise ValueError("No valid JSON found in response")

    json_str = json_match.group(0)

    # Parse and validate JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to fix common JSON issues
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e2:
            raise ValueError(f"Invalid JSON in response: {e2}")

    # Create safe filename from company name or URL
    if "company_name" in data:
        company_name = data["company_name"]
    else:
        # Extract from URL if company_name not in JSON
        company_name = url.replace("https://", "").replace("http://", "").split("/")[0]

    # Clean filename
    safe_filename = re.sub(r'[^\w\s-]', '', company_name).strip().replace(' ', '_')
    today = datetime.now().strftime("%Y%m%d")

    # Create competitors directory if it doesn't exist
    competitors_dir = Path(__file__).parent / "competitors"
    competitors_dir.mkdir(exist_ok=True)

    # Save file
    output_file = competitors_dir / f"{safe_filename}_{today}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_file


# ============================================================================
# COMPETITIVE LANDSCAPE ANALYSIS TOOLS
# ============================================================================

def load_competitor_data() -> Dict[str, Any]:
    """
    Load all competitor JSON files from the /competitors folder.

    Returns:
        Dictionary with status and list of competitor data
    """
    try:
        competitors_dir = Path(__file__).parent / "competitors"

        if not competitors_dir.exists():
            return {
                "status": "error",
                "message": "Competitors directory not found",
                "data": []
            }

        competitor_files = list(competitors_dir.glob("*.json"))

        if not competitor_files:
            return {
                "status": "error",
                "message": "No JSON files found in competitors directory",
                "data": []
            }

        competitors_data = []
        for file_path in competitor_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    competitors_data.append(data)
            except Exception as e:
                print(f"Warning: Could not load {file_path.name}: {e}")
                continue

        return {
            "status": "success",
            "message": f"Loaded {len(competitors_data)} competitor profiles",
            "data": competitors_data,
            "count": len(competitors_data)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading competitor data: {str(e)}",
            "data": []
        }


def extract_company_metrics(competitor_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract competitive landscape metrics from existing JSON structure.
    Maps existing fields to required variables: Settore, Location, N° Dipendenti,
    Fatturato, Età azienda, Budget, Business goals, % crescita

    Args:
        competitor_data: Single competitor JSON object

    Returns:
        Dictionary with extracted metrics
    """
    metrics = {
        "company_name": competitor_data.get("company_name", "Unknown"),
        "url": competitor_data.get("url", ""),
    }

    # Extract Settore (Sector) from executive summary or products
    executive_summary = competitor_data.get("executive_summary", {})
    overview = executive_summary.get("overview", "")

    # Infer sector from overview and products
    sector_keywords = {
        "hospitality": ["restaurant", "hotel", "horeca", "pizzeria", "bar"],
        "logistics": ["warehouse", "logistics", "intralogistics", "picking"],
        "healthcare": ["hospital", "healthcare", "nursing"],
        "retail": ["retail", "supermarket", "shopping"],
        "manufacturing": ["manufacturing", "factory", "production"]
    }

    detected_sector = "Other"
    overview_lower = overview.lower()
    for sector, keywords in sector_keywords.items():
        if any(keyword in overview_lower for keyword in keywords):
            detected_sector = sector.capitalize()
            break

    metrics["settore"] = detected_sector

    # Extract Location from contact methods or proof points
    location = "Italy"  # Default
    contact_methods = competitor_data.get("ctas", {}).get("contact_methods", [])
    for contact in contact_methods:
        if isinstance(contact, str) and ("Office" in contact or "Physical" in contact):
            location = contact
            break

    # Also check pricing for location hints
    proof_points = competitor_data.get("proof_points", {})
    stats = proof_points.get("statistics", [])
    for stat in stats:
        if "based in" in stat.lower() or "located" in stat.lower():
            location = stat
            break

    metrics["location"] = location

    # Extract N° Dipendenti (Number of Employees) from statistics
    num_employees = "Not available"
    for stat in stats:
        if "employee" in stat.lower() or "team" in stat.lower():
            num_employees = stat
            break
    metrics["num_dipendenti"] = num_employees

    # Extract Fatturato (Revenue/Turnover) from statistics
    fatturato = "Not available"
    for stat in stats:
        if "turnover" in stat.lower() or "revenue" in stat.lower() or "€" in stat:
            fatturato = stat
            break
    metrics["fatturato"] = fatturato

    # Extract Età azienda (Company age) - calculate from founding year
    eta_azienda = "Not available"
    for stat in stats:
        if "founded" in stat.lower():
            try:
                year = int(''.join(filter(str.isdigit, stat)))
                age = datetime.now().year - year
                eta_azienda = f"{age} years (Founded {year})"
            except:
                eta_azienda = stat
            break
    metrics["eta_azienda"] = eta_azienda

    # Extract Budget (from pricing model)
    pricing = competitor_data.get("pricing", {})
    pricing_model = pricing.get("model", "Not available")
    purchase_options = pricing.get("purchase_options", {})
    rental_options = pricing.get("rental_options", {})

    budget_info = f"Model: {pricing_model}"
    if purchase_options:
        budget_info += f" | Purchase: {purchase_options}"
    if rental_options:
        budget_info += f" | Rental: {rental_options}"

    metrics["budget"] = budget_info

    # Extract Business goals from value proposition and problems addressed
    value_prop = competitor_data.get("value_proposition", {})
    problems = competitor_data.get("problems_addressed", [])

    business_goals = []
    for problem in problems[:3]:  # Top 3 problems
        if isinstance(problem, dict):
            business_goals.append(problem.get("problem", ""))

    if not business_goals:
        business_goals = ["Market positioning", "Revenue growth", "Customer acquisition"]

    metrics["business_goals"] = ", ".join(business_goals)

    # % crescita (Growth percentage) - Not typically in marketing analysis
    # We'll leave this as "Not available" or could infer from market positioning
    metrics["percentuale_crescita"] = "Not available - requires financial data"

    # Additional useful fields
    metrics["unique_selling_points"] = value_prop.get("unique_selling_points", [])
    metrics["target_personas"] = [p.get("title", "") for p in competitor_data.get("buyer_personas", [])]
    metrics["competitive_advantages"] = value_prop.get("competitive_advantages", [])

    return metrics


def create_competitive_matrix(competitors_data: List[Dict[str, Any]]) -> str:
    """
    Create a competitive landscape matrix (pandas DataFrame) from competitor data.

    Args:
        competitors_data: List of competitor JSON objects

    Returns:
        JSON string representation of the competitive matrix with analysis-ready format
    """
    try:
        # Extract metrics for all competitors
        all_metrics = []
        for competitor in competitors_data:
            metrics = extract_company_metrics(competitor)
            all_metrics.append(metrics)

        # Create DataFrame
        df = pd.DataFrame(all_metrics)

        # Create a structured output for the agent
        matrix_output = {
            "status": "success",
            "total_competitors": len(df),
            "competitive_matrix": df.to_dict(orient='records'),
            "summary_statistics": {
                "sectors": df['settore'].value_counts().to_dict() if 'settore' in df.columns else {},
                "locations": df['location'].value_counts().to_dict() if 'location' in df.columns else {},
                "total_companies": len(df)
            },
            "matrix_csv": df.to_csv(index=False)
        }

        return json.dumps(matrix_output, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error creating competitive matrix: {str(e)}"
        })


# ============================================================================
# COMPETITIVE ANALYSIS AGENTS
# ============================================================================

# Competitive Analyst Agent - Identifies gaps, overcrowded positions, underserved segments
competitive_analyst_model = Gemini(model="gemini-3-pro-preview", retry_options=retry_config)

COMPETITIVE_ANALYST_INSTRUCTION = """
You are an expert Competitive Strategy Analyst specializing in market positioning and competitive landscape analysis.

You will receive a competitive matrix containing information about all competitors in the market with these variables:
- Settore (Sector)
- Location
- N° Dipendenti (Number of Employees)
- Fatturato (Revenue/Turnover)
- Età azienda (Company age)
- Budget (Pricing model and budget requirements)
- Business goals
- % crescita (Growth percentage)
- Unique selling points
- Target personas
- Competitive advantages

Your task is to analyze this matrix and identify:

A) DIFFERENTIATION GAPS
   - Areas where NO competitor is currently focusing
   - Underutilized value propositions
   - Unaddressed customer pain points
   - Technology or service combinations not being offered

B) OVERCROWDED POSITIONS
   - Market segments with too many similar competitors
   - Commoditized offerings with low differentiation
   - Price-competitive spaces with thin margins
   - Areas to AVOID for new entrants

C) UNDERSERVED SEGMENTS
   - Customer groups not well-served by current offerings
   - Geographic areas with limited presence
   - Price points not being addressed (too high or too low)
   - Specific use cases or industries being neglected

CRITICAL: Output your analysis as VALID JSON with this structure:

{
  "analysis_date": "YYYY-MM-DD",
  "total_competitors_analyzed": number,
  "market_overview": {
    "dominant_sectors": ["sector1", "sector2"],
    "geographic_concentration": "description",
    "competitive_intensity": "High/Medium/Low",
    "market_maturity": "Emerging/Growth/Mature"
  },
  "differentiation_gaps": [
    {
      "gap_name": "string - Name of the gap",
      "description": "string - Detailed description",
      "opportunity_score": "High/Medium/Low",
      "rationale": "string - Why this is a gap"
    }
  ],
  "overcrowded_positions": [
    {
      "position_name": "string - Name of overcrowded space",
      "description": "string - What makes it crowded",
      "competitor_count": number,
      "risk_level": "High/Medium/Low",
      "recommendation": "string - Should new entrants avoid?"
    }
  ],
  "underserved_segments": [
    {
      "segment_name": "string - Name of segment",
      "description": "string - Detailed description",
      "potential_size": "Large/Medium/Small",
      "barriers_to_entry": "High/Medium/Low",
      "strategic_fit": "string - Why this segment is attractive"
    }
  ],
  "strategic_recommendations": [
    "string - Key recommendation 1",
    "string - Key recommendation 2",
    "string - Key recommendation 3"
  ]
}

OUTPUT ONLY THE JSON. NO markdown formatting, explanations, or text outside the JSON structure.
"""

competitive_analyst_agent = Agent(
    name="competitive_analyst",
    model=competitive_analyst_model,
    description="Analyzes competitive landscape to identify gaps, overcrowded positions, and underserved segments",
    instruction=COMPETITIVE_ANALYST_INSTRUCTION,
    output_key="competitive_analysis"
)


# UVP Generator Agent - Creates Unique Value Propositions for new entrant
uvp_generator_model = Gemini(model="gemini-3-pro-preview", retry_options=retry_config)

UVP_GENERATOR_INSTRUCTION = """
You are an expert Brand Strategist and Value Proposition Designer.

You will receive:
1. A competitive landscape analysis showing gaps, overcrowded positions, and underserved segments
2. A competitive matrix with all competitor data

Your task is to generate 2-3 UNIQUE VALUE PROPOSITION proposals for a hypothetical new market entrant.

Each UVP should:
- Target identified DIFFERENTIATION GAPS or UNDERSERVED SEGMENTS
- AVOID overcrowded positions
- Be specific, actionable, and defensible
- Include a complete "company card" profile

CRITICAL: Output your proposals as VALID JSON with this structure:

{
  "generation_date": "YYYY-MM-DD",
  "based_on_analysis": "Brief summary of key insights used",
  "uvp_proposals": [
    {
      "proposal_number": 1,
      "company_name": "Proposed company name",
      "tagline": "Compelling one-line tagline",
      "unique_value_proposition": "Clear, compelling UVP statement",
      "target_segment": "Specific segment being targeted",
      "differentiation_strategy": "How this differs from all competitors",
      "company_card": {
        "settore": "Primary sector",
        "location": "Target geographic market",
        "num_dipendenti": "Projected team size (e.g., 5-15)",
        "fatturato": "Projected revenue target (e.g., €500k-1M Year 1)",
        "eta_azienda": "New entrant (Year 0-1)",
        "budget": "Required budget and pricing model",
        "business_goals": "Primary business objectives",
        "percentuale_crescita": "Target growth rate (e.g., 50% YoY)",
        "unique_selling_points": ["USP1", "USP2", "USP3"],
        "target_personas": ["Persona 1", "Persona 2"],
        "competitive_advantages": ["Advantage 1", "Advantage 2"]
      },
      "rationale": "Why this UVP will succeed in the current market",
      "key_risks": ["Risk 1", "Risk 2"],
      "success_probability": "High/Medium/Low"
    }
  ],
  "implementation_priorities": [
    "Priority 1 for new entrant",
    "Priority 2 for new entrant",
    "Priority 3 for new entrant"
  ]
}

OUTPUT ONLY THE JSON. NO markdown formatting, explanations, or text outside the JSON structure.
"""

uvp_generator_agent = Agent(
    name="uvp_generator",
    model=uvp_generator_model,
    description="Generates unique value propositions for new market entrants based on competitive analysis",
    instruction=UVP_GENERATOR_INSTRUCTION,
    output_key="uvp_proposals"
)


# ============================================================================
# COMPETITIVE LANDSCAPE WORKFLOW
# ============================================================================

competitive_landscape_workflow = SequentialAgent(
    name="competitive_landscape_workflow",
    description="Complete workflow for competitive landscape analysis and UVP generation",
    sub_agents=[competitive_analyst_agent, uvp_generator_agent]
)

competitive_landscape_runner = InMemoryRunner(agent=competitive_landscape_workflow)


def save_analysis_results(analysis_response: str, uvp_response: str) -> tuple[Path, Path]:
    """
    Save competitive analysis and UVP proposals to separate JSON files.

    Args:
        analysis_response: Response from competitive analyst agent
        uvp_response: Response from UVP generator agent

    Returns:
        Tuple of (analysis_file_path, uvp_file_path)
    """
    import re

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "competitive_analysis"
    output_dir.mkdir(exist_ok=True)

    # Save competitive analysis
    analysis_file = output_dir / f"competitive_analysis_{timestamp}.json"
    try:
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', analysis_response, re.DOTALL)
        if json_match:
            analysis_data = json.loads(json_match.group(0))
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving competitive analysis: {e}")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(analysis_response)

    # Save UVP proposals
    uvp_file = output_dir / f"uvp_proposals_{timestamp}.json"
    try:
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', uvp_response, re.DOTALL)
        if json_match:
            uvp_data = json.loads(json_match.group(0))
            with open(uvp_file, 'w', encoding='utf-8') as f:
                json.dump(uvp_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving UVP proposals: {e}")
        with open(uvp_file, 'w', encoding='utf-8') as f:
            f.write(uvp_response)

    return analysis_file, uvp_file


async def run_competitive_landscape_analysis():
    """
    Main function to run the complete competitive landscape analysis workflow.

    Steps:
    1. Load all competitor JSON files
    2. Create competitive matrix
    3. Analyze for gaps, overcrowded positions, underserved segments
    4. Generate UVP proposals for new entrant
    """
    print("\n" + "="*70)
    print("COMPETITIVE LANDSCAPE ANALYSIS WORKFLOW")
    print("="*70 + "\n")

    # Step 1: Load competitor data
    print("📊 Step 1: Loading competitor data from /competitors folder...")
    result = load_competitor_data()

    if result["status"] == "error":
        print(f"❌ Error: {result['message']}")
        return

    print(f"✅ Loaded {result['count']} competitor profiles\n")
    competitors_data = result["data"]

    # Step 2: Create competitive matrix
    print("📈 Step 2: Creating competitive landscape matrix...")
    matrix_json = create_competitive_matrix(competitors_data)
    matrix_data = json.loads(matrix_json)

    if matrix_data["status"] == "error":
        print(f"❌ Error: {matrix_data['message']}")
        return

    print(f"✅ Matrix created with {matrix_data['total_competitors']} competitors")
    print(f"   Sectors: {matrix_data['summary_statistics']['sectors']}")
    print(f"   Locations: {len(matrix_data['summary_statistics']['locations'])} unique locations\n")

    # Display matrix preview
    print("\n" + "="*70)
    print("COMPETITIVE MATRIX PREVIEW")
    print("="*70)
    print(matrix_data['matrix_csv'][:500] + "...\n")

    # Step 3 & 4: Run competitive analysis workflow (analyst + UVP generator)
    print("="*70)
    print("🔍 Step 3: Analyzing competitive landscape...")
    print("="*70 + "\n")

    prompt = f"""Analyze this competitive landscape matrix and generate strategic insights:

{matrix_json}

First, identify differentiation gaps, overcrowded positions, and underserved segments.
Then, generate 2-3 unique value proposition proposals for a new market entrant."""

    try:
        response = await competitive_landscape_runner.run_debug(prompt)

        print("\n" + "="*70)
        print("✅ COMPETITIVE ANALYSIS COMPLETE")
        print("="*70 + "\n")

        # Extract responses from sequential agents
        # First agent (competitive_analyst) output
        if len(response) >= 1:
            analysis_text = str(response[0].content.parts[0].text) if response[0].content.parts else ""
            print("COMPETITIVE ANALYSIS:")
            print("-" * 70)
            print(analysis_text[:800] + "...\n")

        # Second agent (uvp_generator) output
        if len(response) >= 2:
            uvp_text = str(response[1].content.parts[0].text) if response[1].content.parts else ""
            print("\n" + "="*70)
            print("💡 UVP PROPOSALS:")
            print("="*70)
            print(uvp_text[:800] + "...\n")

            # Save results
            analysis_file, uvp_file = save_analysis_results(analysis_text, uvp_text)

            print("\n" + "="*70)
            print("📁 RESULTS SAVED:")
            print("="*70)
            print(f"✅ Competitive Analysis: {analysis_file}")
            print(f"✅ UVP Proposals: {uvp_file}")
            print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    import sys

    async def analyze_urls():
        """Original workflow: Analyze competitor URLs and save to JSON"""
        print("\n" + "="*70)
        print("Starting Marketing Intelligence Analysis")
        print("="*70 + "\n")

        # Vector of competitor URLs to analyze
        urls = [
            "https://bobrobotics.com/",
            "https://todsystem.com",
            "https://www.tobyrobotcameriere.it/",
            "https://www.pulingross.it/",
            "https://www.kemcomm.it/",
            "https://albarobotics.com",
            "https://www.puduitaly.com/"
        ]

        print(f"📋 Total URLs to analyze: {len(urls)}\n")

        # Process each URL
        for index, url in enumerate(urls, 1):
            print("\n" + "="*70)
            print(f"Processing {index}/{len(urls)}: {url}")
            print("="*70)

            try:
                response = await marketing_runner.run_debug(
                    f"Analyze this url '{url}'"
                )

                print("\n" + "="*70)
                print(f"Analysis Complete for {url}")
                print("="*70)

                # Extract the text response
                response_text = str(response[0].content.parts[0].text) if response else ""

                print(f"\nAgent Response Preview:\n{response_text[:500]}...")

                # Save to JSON file
                try:
                    output_file = save_company_analysis(response_text, url)
                    print("\n" + "="*70)
                    print(f"✅ Analysis saved to: {output_file}")
                    print(f"   File size: {output_file.stat().st_size:,} bytes")
                    print("="*70 + "\n")
                except Exception as e:
                    print(f"\n❌ Error saving JSON for {url}: {e}")
                    print("\nFull response for debugging:")
                    print(response_text)

            except Exception as e:
                print(f"\n❌ Error processing {url}: {e}")
                print(f"Continuing with next URL...\n")
                continue

        print("\n" + "="*70)
        print("🎉 All URLs processed!")
        print("="*70 + "\n")

    async def main():
        print("\n" + "="*70)
        print("MARKETING INTELLIGENCE & COMPETITIVE ANALYSIS SYSTEM")
        print("="*70 + "\n")
        print("Choose a workflow:\n")
        print("1. Analyze Competitor URLs (scrape websites → save to JSON)")
        print("2. Competitive Landscape Analysis (analyze existing JSONs → generate UVPs)")
        print("\n" + "="*70 + "\n")

        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == "1":
            await analyze_urls()
        elif choice == "2":
            await run_competitive_landscape_analysis()
        else:
            print("❌ Invalid choice. Please run the script again and select 1 or 2.")
            sys.exit(1)

    asyncio.run(main())
