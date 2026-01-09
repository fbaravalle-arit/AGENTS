"""
Marketing Intelligence Agent for Italian B2B Market Analysis
Uses Google ADK to analyze competitor websites and generate strategic recommendations
"""

import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search_tool,google_search_agent_tool

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
Read the entire landing page and different sections thoroughly. PLEASE FOCUS ON INDUSTRIAL AUTOMATION solutions for manufacturing and logistics.

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


if __name__ == "__main__":
    import asyncio

    async def main():
        print("\n" + "="*70)
        print("Starting Marketing Intelligence Analysis")
        print("="*70 + "\n")

        # Vector of competitor URLs to analyze
        # Vector of competitor URLs to analyze
        urls = [
            "https://www.kuka.com",
            "https://www.loccioni.com",
            "https://www.manz.com",
            "https://www.marchesini.com",
            "https://www.micropsi-industries.com",
            "https://www.midlandautomation.co.uk",
            "https://www.mobility.siemens.com",
            "https://www.movu-robotics.com",
            "https://www.neura-robotics.com",
            "https://www.ocme.com",
            "https://www.olympus-tech.com",
            "https://www.pal-robotics.com",
            "https://www.pilz.com",
            "https://www.primaindustrie.com",
            "https://www.rfrautomation.com",
            "https://www.robco.de",
            "https://www.robot-store.co.uk",
            "https://www.rockwellautomation.com",
            "https://www.rockwellautomation.com/en-gb.html",
            "https://www.sauter-controls.com",
            "https://www.se.com",
            "https://www.sick.com",
            "https://www.siemens.com",
            "https://www.sitibt.com",
            "https://www.sptechnology.co.uk",
            "https://www.staubli.com",
            "https://www.universal-robots.com",
            "https://www.wandelbots.com",
            "https://www.woodplc.com",
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

    asyncio.run(main())
