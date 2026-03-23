"""
Run the agentic pipeline for one or more municipalities.
Usage: PYTHONPATH=. python backend/run_agent.py [municipality1] [municipality2] ...
Default: runs for Mississauga and Ottawa
"""
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from backend.config import get_settings
from backend.agents.orchestrator import run_pipeline_for_municipality

DEFAULT_TARGETS = [
    ("Mississauga", "Ontario"),
    ("Ottawa", "Ontario"),
    ("Hamilton", "Ontario"),
    ("Brampton", "Ontario"),
]


async def main():
    settings = get_settings()

    if len(sys.argv) > 1:
        targets = [(name, "Ontario") for name in sys.argv[1:]]
    else:
        targets = DEFAULT_TARGETS

    results = []
    for municipality, province in targets:
        try:
            result = await run_pipeline_for_municipality(
                municipality=municipality,
                province=province,
                database_url=settings.database_url,
            )
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed to process {municipality}: {e}")
            results.append({"municipality": municipality, "status": "error", "error": str(e)})

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['municipality']}: {r.get('regulations', 0)} regulations ({r['status']})")


if __name__ == "__main__":
    asyncio.run(main())
