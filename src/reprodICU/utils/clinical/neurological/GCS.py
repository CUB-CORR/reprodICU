"""
GCS: helpers for the Glasgow Coma Scale (GCS).

GCS_NT_CARRY_FORWARD carries the last assessed GCS points over not testable
timeframes. The GCS verbal score is 0 when it is not testable (NT, e.g. the
patient is intubated) and the GCS total is then null. Timeframes flagged as NT
(or sedated) without a GCS score take the last assessed GCS points of the stay,
or 0 points (GCS 15) if there is none.

SOURCES
-------
- Lambden S, Laterre PF, Levy MM, Francois B.
  The SOFA score-development, utility and challenges of accurate assessment in clinical trials.
  Crit Care. 2019 Nov 27;23(1):374.
  doi: 10.1186/s13054-019-2663-7. PMID: 31775846; PMCID: PMC6880479.
  (Proposal 1: carry the last pre-intubation GCS forward, else GCS 15)
"""

import polars as pl

from ...common import _to_lazy

STAY_KEY = "Global ICU Stay ID"


def GCS_NT_CARRY_FORWARD(
    data: pl.LazyFrame,
    points_col: str,
    nt_col: str,
) -> pl.LazyFrame:
    """
    Carry the last assessed GCS points forward over not testable timeframes.

    Arguments
    ---------
        data : pl.LazyFrame
            One row per stay and timeframe with columns Global ICU Stay ID,
            timeframe, points_col and nt_col.
        points_col : str
            Column with the GCS points of the timeframe (null if not scored).
        nt_col : str
            Boolean column, true if the timeframe is not testable (verbal
            score 0) or sedated. It is dropped (it would otherwise be summed
            into the score).

    Returns
    -------
        pl.LazyFrame
            data without nt_col; points_col of not testable timeframes is the
            last assessed value in the stay, or 0 if there is none.
    """
    return (
        _to_lazy(data)
        .with_columns(
            pl.when(nt_col)
            .then(
                pl.col(points_col)
                .forward_fill()
                .over(STAY_KEY, order_by="timeframe")
                .fill_null(0)
            )
            .otherwise(points_col)
        )
        .drop(nt_col)
    )
