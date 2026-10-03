"""The questions Jev answers on every decision tick (blueprint section 4.4).

One request carries all of them. Each is typed (Choice / Score / Noul -- never free
text), literal and single-hop, and points at the state field it needs by name in
backticks. No arithmetic or counting is asked of Jev; numbers arrive as named
buckets. ``exit_now`` is speculative fan-out: asked every tick, read only when a
position is open. The action is asked three ways for self-consistency.
"""
from __future__ import annotations

from .typesafe import Choice, Noul, Question, Score

ACTIONS = ("buy_calls", "buy_puts", "sell_premium", "no_trade")
ACTION_OPTIONS = {
    "buy_calls": "Structure is bullish with confluence and implied volatility is not elevated, so buying calls is justified now",
    "buy_puts": "Structure is bearish with confluence and implied volatility is not elevated, so buying puts is justified now",
    "sell_premium": "Structure has a clear direction with confluence and implied volatility is elevated, so selling a "
                    "defined-risk credit spread in that direction is justified now",
    "no_trade": "The evidence is mixed, weak, ranging or contradictory, so no new position",
}
ACTION_KEYS = ("action", "action_alt", "action_contrarian_check")


def decision_questions() -> dict[str, Question]:
    return {
        "structure_bias": Choice(
            instructions="Using `structure.daily_bias`, `structure.hourly_trend` and `structure.last_event`, "
                         "which way is market structure biased?",
            criteria={"bullish": "Hourly structure is breaking upward and the daily bias does not oppose it",
                      "bearish": "Hourly structure is breaking downward and the daily bias does not oppose it",
                      "ranging": "Breaks are mixed, there is no break yet, or the hourly trend fights the daily bias"}),
        "liquidity_event": Noul(
            instructions="Does `liquidity.last_event` describe liquidity that was swept and then reclaimed "
                         "(a sweep-and-reverse), rather than taken with price continuing through it?",
            criteria={"true": "A genuine sweep: liquidity was run and price closed back inside",
                      "false": "Liquidity was taken and price continued, or there was no event"}),
        "ob_confluence": Score(
            instructions="How much confluence supports `order_block`?",
            criteria=["None: there is no active order block",
                      "Weak: the block is far from price or already tested, with little else supporting it",
                      "Moderate: the block is near price with one supporting factor",
                      "Strong: an untested block near price formed on a change of character or after a sweep",
                      "Very strong: untested, near price, after a sweep, overlapping a gap, in the right half of the range"]),
        "fvg_fill_expectation": Choice(
            instructions="Given `fair_value_gap` and `structure.hourly_trend`, is the gap more likely to be filled "
                         "or respected as support or resistance?",
            criteria={"filled": "Price will likely trade through the gap",
                      "respected": "The gap will likely hold as support or resistance",
                      "no_gap": "There is no active fair value gap"}),
        "zone_context": Choice(
            instructions="According to `zone`, where is price in the current dealing range?",
            criteria={"premium": "In the upper half, above equilibrium, or above the range",
                      "discount": "In the lower half, below equilibrium, or below the range",
                      "equilibrium": "Near the middle of the range, or there is no valid range"}),
        "iv_regime": Choice(
            instructions="According to `volatility.iv_rank`, is implied volatility elevated, normal or low?",
            criteria={"elevated": "Implied volatility is high relative to the past year",
                      "normal": "Implied volatility is in its usual range",
                      "low": "Implied volatility is low relative to the past year",
                      "unknown": "There is not enough history to judge"}),
        "setup_quality": Score(
            instructions="How strong is the case for a directional options trade now, if `structure`, `liquidity`, "
                         "`order_block`, `fair_value_gap` and `zone` all point one way?",
            criteria=["No setup: the fields disagree or structure is ranging",
                      "Weak: structure has a direction but little else agrees",
                      "Decent: structure and one of the order block, gap or liquidity event agree",
                      "Strong: structure, a recent sweep and an order block or gap agree, from a sensible zone",
                      "Exceptional: everything agrees and price is at a fresh zone"]),
        "action": Choice(instructions="Given `structure`, `liquidity`, `order_block`, `fair_value_gap`, `zone` and "
                                      "`volatility`, what options trade should be made now?", criteria=ACTION_OPTIONS),
        "action_alt": Choice(
            instructions={"question": "Given `structure`, `order_block`, `liquidity` and `volatility`, pick the trade.",
                          "rule": "Only choose a trade when `structure` and at least one of `order_block` or "
                                  "`liquidity` point the same way."},
            criteria=ACTION_OPTIONS),
        "action_contrarian_check": Choice(
            instructions="A skeptical risk manager reviews `structure`, `liquidity`, `order_block`, `zone` and "
                         "`volatility`. Which trade, if any, would they approve?",
            criteria=ACTION_OPTIONS),
        "exit_now": Noul(
            instructions="Given `position`, has `structure.last_event` turned against it, or has the reason for it "
                         "in `order_block` and `liquidity` gone?",
            criteria={"true": "Structure now opposes the position or its reason is gone",
                      "false": "The reason for the position still holds"}),
    }
