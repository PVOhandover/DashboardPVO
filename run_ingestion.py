import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "webScrapers"

def run(cmd, cwd=None, check=False):
    cmd = list(map(str, cmd))
    print("\n+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)

def step(name, cmd, cwd=None):
    print(f"\n=== {name} ===")
    try:
        run(cmd, cwd=cwd, check=True)
        print(f"{name} OK")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{name} FAILED (exit {e.returncode})")
        return False



def main():
    py = sys.executable

    ok = True

    ok &= step(
        "NOS RSS (Economy)",
        [
            py, "webScrapers/scrape_nos_feeds.py",
            "--max_feeds", "10",
            "--max_items_per_feed", "5",
            "--out_json", "scrapedArticles/nos_articles.json",
        ],
        cwd=ROOT,
    )

    ok &= step("Brabants Dagblad RSS", [py, "scrape_bd.py"], cwd=WEB)
    ok &= step("Gelderlander RSS", [py, "scrape_gelderlander.py"], cwd=WEB)
    ok &= step("Politie API update", [py, "webScrapers/scrape_police.py", "--update"], cwd=ROOT)
    ok &= step("L1 RSS", [py, "scrape_l1.py"], cwd=WEB)
    ok &= step("Omroep West RSS", [py, "scrape_omroep_west.py"], cwd=WEB)
    ok &= step("RTV Noord RSS", [py, "scrape_rtv_noord.py"], cwd=WEB)

    ok &= step("Merge JSONs", [py, "merge_jsons.py"], cwd=ROOT)

    print("\n==============================")
    print("Ingestion finished:", "SUCCESS" if ok else "PARTIAL (some sources failed)")
    print("==============================\n")

if __name__ == "__main__":
    main()