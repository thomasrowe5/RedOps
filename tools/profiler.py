import cProfile
import io
import argparse
import pstats
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--module",
        required=True,
        help="module:function to profile, e.g., orchestrator.app.main:some_handler",
    )
    ap.add_argument("--out", default="results/profile.prof")
    args = ap.parse_args()
    mod, fn = args.module.split(":")
    m = __import__(mod, fromlist=[fn])
    f = getattr(m, fn)
    pr = cProfile.Profile()
    pr.enable()
    f()
    pr.disable()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pr.dump_stats(args.out)
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())


if __name__ == "__main__":
    main()
