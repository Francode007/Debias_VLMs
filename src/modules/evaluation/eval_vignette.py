#!/usr/bin/env python
"""
Vignette Social Bias Evaluation CLI
Usage:
    python -m modules.evaluation.eval_vignette --gen_file <path> --axis nationality
"""

import argparse
import sys
from pathlib import Path

from .registry import get_evaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Vignette Social Bias Metrics (Selection Frequency, Log-Odds, PairComp, Polarity)"
    )
    parser.add_argument(
        "--gen_file",
        type=str,
        required=True,
        help="Path to the LLaVA responses CSV file (e.g. nationality_male_1a_llava_responses.csv)"
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="nationality",
        choices=["nationality", "gender_and_sex", "race_ethnicity_color", "religion", 
                 "socioeconomic", "ability", "age", "physical_traits"],
        help="Social axis being evaluated"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory for metrics"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more detailed output"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    gen_file = Path(args.gen_file)
    if not gen_file.exists():
        print(f"❌ Error: File not found: {gen_file}")
        sys.exit(1)

    print(f"🚀 Starting Vignette Evaluation")
    print(f"   Axis       : {args.axis}")
    print(f"   Input File : {gen_file}")

    try:
        evaluator = get_evaluator("vignette")
        
        result = evaluator.evaluate(
            gen_file=str(gen_file),
            axis=args.axis,
            output_dir=args.output_dir
        )

        # Summary
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE")
        print("="*70)
        
        polarity = result["metrics"]["polarity_score"][:5]  # Top 5
        
        print(f"Total Responses     : {result['total_responses']:,}")
        print(f"Total Identities    : {result['total_identities']}")
        
        print("\nTop 5 Most Positively Biased Identities:")
        for item in polarity[:5]:
            print(f"   {item['identity']:20} → Polarity: {item['polarity_score']:.4f}")

        print(f"\n📁 Full results saved in:")
        print(f"   {Path(gen_file).parent / 'vignette_metrics'}")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


'''
python -m modules.evaluation.eval_vignette \
    --gen_file outputs/socialassumptions/test/nationality/nationality_female_1a_llava_responses.csv \
    --axis nationality
'''