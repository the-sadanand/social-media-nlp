
import time
import sys
from pathlib import Path

# Ensure project root is on the Python path so we can import from src/
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import preprocess_data
from src.sentiment_model import train_and_evaluate
from src.topic_model import train_topic_model


def main() -> None:
    """Execute the full pipeline with timing information."""

    print("+" + "=" * 58 + "+")
    print("|  Social Media Sentiment & Topic Analysis Pipeline        |")
    print("+" + "=" * 58 + "+")

    overall_start = time.time()

    # Ensure output directory exists
    Path("output").mkdir(exist_ok=True)

    # -- Step 1: Preprocess --
    t0 = time.time()
    preprocess_data(
        input_path="data/Tweets.csv",
        output_path="output/preprocessed_data.csv",
    )
    print(f"  [TIME] Preprocessing took {time.time() - t0:.1f}s\n")

    # -- Step 2: Sentiment Classification --
    t0 = time.time()
    _, _, metrics = train_and_evaluate(
        preprocessed_path="output/preprocessed_data.csv",
        raw_data_path="data/Tweets.csv",
        output_dir="output",
    )
    print(f"  [TIME] Sentiment training took {time.time() - t0:.1f}s\n")

    # -- Step 3: Topic Modeling --
    t0 = time.time()
    train_topic_model(
        preprocessed_path="output/preprocessed_data.csv",
        output_dir="output",
    )
    print(f"  [TIME] Topic modeling took {time.time() - t0:.1f}s\n")

    # -- Summary --
    elapsed = time.time() - overall_start
    output_files = sorted(Path("output").iterdir())

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Sentiment accuracy: {metrics['accuracy']:.2%}")
    print(f"\n  Generated artifacts:")
    for f in output_files:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<30} {size_kb:>8.1f} KB")
    print()


if __name__ == "__main__":
    main()
