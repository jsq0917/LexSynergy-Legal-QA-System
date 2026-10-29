"""核对金额评测集的金标：**用引擎回算每一条，和手写的期望值逐条比对**。

为什么要这一步：金标是手算的，手算会错；引擎也可能有 bug。两边独立算、比对结果，
不一致的地方就必须人工看一眼——**不允许"以引擎为准"自动改写金标**，
那样等于把引擎的 bug 固化进评测集，指标永远正确、答案永远错。

    python -m scripts.check_calc_gold
    python -m scripts.check_calc_gold --verbose
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

from app.calc import engine as E
from app.calc.models import Facts

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "eval" / "calc_cases.jsonl"

#: kind → 引擎函数。新增公式时这里必须同步，否则会漏检
FUNCS = {
    "severance": E.severance,
    "illegal": E.illegal_termination,
    "notice": E.notice_pay,
    "overtime": E.overtime,
    "double_wage": E.double_wage,
    "annual_leave": E.annual_leave,
    "non_compete": E.non_compete,
    "probation": None,          # 单位是"月"，不放进金额评测（见 docs/06）
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="核对金额金标")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    rows = [json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()]
    bad: list[str] = []
    print(f"{'id':<5}{'kind':<13}{'金标':>12}{'引擎':>12}  判定  说明")
    print("-" * 78)
    for row in rows:
        fn = FUNCS.get(row["kind"])
        if fn is None:
            bad.append(row["id"])
            print(f"{row['id']:<5}{row['kind']:<13}{'—':>12}{'—':>12}  SKIP  该类型不纳入金额评测")
            continue
        result = fn(Facts.from_dict(row["facts"]))
        gold = Decimal(str(row["expected_value"]))
        ok = result.value == gold
        if not ok:
            bad.append(row["id"])
        print(
            f"{row['id']:<5}{row['kind']:<13}{gold:>12}{result.value:>12}  "
            f"{'✓' if ok else '✗ 不一致'}  {row.get('note', '')[:28]}"
        )
        if args.verbose or not ok:
            print(f"      公式：{result.formula}")
            print(f"      代入：{result.substitution}")
            print(f"      假设：{'；'.join(result.assumptions)[:110]}")

    print("-" * 78)
    if bad:
        print(f"❌ {len(bad)} 条不一致：{', '.join(bad)}")
        print("   请人工判断是金标错还是引擎错；**不要直接改金标去迁就引擎**。")
        return 1
    print(f"✅ {len(rows)} 条金标与引擎完全一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
