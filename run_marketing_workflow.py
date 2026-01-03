"""
Multi-turn workflow runner for Marketing Intelligence Agent
Continues the conversation until all phases are complete
"""
import os
import asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify API key is set
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY not found in environment variables. "
        "Please add it to your .env file."
    )

from Marketing_adk import marketing_runner

async def run_complete_workflow():
    """Run the complete marketing intelligence workflow with multi-turn conversation"""

    print("\n" + "="*70)
    print("Marketing Intelligence Workflow - Multi-Turn Execution")
    print("="*70 + "\n")

    session_id = "marketing_analysis_session"

    # Phase 1: Start the analysis
    print("Phase 1: Initiating analysis and reading competitor list...")
    response1 = await marketing_runner.run_debug(
        "Read the competitor list from Comp_site_list.md and confirm you have the URLs.",
        session_id=session_id
    )
    print("✓ Phase 1 complete\n")

    # Phase 2: Search for each company
    print("Phase 2: Searching for company information...")
    response2 = await marketing_runner.run_debug(
        "Now use google_search to search for each company URL you found. Search for: company name, products, target market, Italian presence. Gather information for all 7 companies.",
        session_id=session_id
    )
    print("✓ Phase 2 complete\n")

    # Phase 3: Analyze and structure data
    print("Phase 3: Analyzing companies and structuring data...")
    response3 = await marketing_runner.run_debug(
        "Now analyze all the information you gathered and create structured profiles for each company using the schema in your instructions. Include buyer personas, jobs-to-be-done, UVPs, proof points, and Italian market signals.",
        session_id=session_id
    )
    print("✓ Phase 3 complete\n")

    # Phase 4: Competitive landscape
    print("Phase 4: Creating competitive landscape analysis...")
    response4 = await marketing_runner.run_debug(
        "Create the competitive landscape matrix. Identify differentiation gaps, overcrowded positions, and underserved segments in the Italian B2B service robotics market.",
        session_id=session_id
    )
    print("✓ Phase 4 complete\n")

    # Phase 5: Generate recommendations
    print("Phase 5: Generating positioning recommendations...")
    response5 = await marketing_runner.run_debug(
        "Generate 2-3 positioning proposals for a new market entrant. Include strategic rationale, target profile, differentiation strategy, go-to-market suggestions, and Italian market advantages for each proposal.",
        session_id=session_id
    )
    print("✓ Phase 5 complete\n")

    # Phase 6: Save to file
    print("Phase 6: Saving complete analysis to JSON file...")
    today = datetime.now().strftime("%Y%m%d")
    filename = f"marketing_intelligence_analysis_{today}.json"

    response6 = await marketing_runner.run_debug(
        f"""Now compile everything into the final JSON structure with:
- metadata (analysis_date, urls_analyzed, urls_failed, italian_market_focus)
- company_profiles (all 7 companies)
- competitive_landscape
- recommendations (2-3 proposals)

Save this as '{filename}' using file_write_tool.""",
        session_id=session_id
    )
    print("✓ Phase 6 complete\n")

    print("="*70)
    print("Workflow Complete!")
    print("="*70)

    # Check if output file was created
    json_files = list(Path('.').glob('marketing_intelligence_analysis_*.json'))
    if json_files:
        latest_file = max(json_files, key=os.path.getmtime)
        print(f"\n✅ Analysis saved to: {latest_file}")
        print(f"   File size: {latest_file.stat().st_size:,} bytes")

        # Show a preview
        import json
        with open(latest_file, 'r') as f:
            data = json.load(f)
            print(f"\n   Companies analyzed: {len(data.get('company_profiles', []))}")
            print(f"   Recommendations: {len(data.get('recommendations', []))}")
    else:
        print("\n⚠️  No output file found. Check if file_write_tool was called successfully.")

    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    asyncio.run(run_complete_workflow())
