import unittest
import numpy as np
import pandas as pd
from src.build_dataset import prepare_events, price_at_or_before


def event_row(i, title="Non Farm Payrolls", date="2024-01-05T13:30:00.000Z", actual=200, forecast=150, previous=100):
    return {"id": i, "title": title, "date": date, "actualRaw": actual,
            "forecastRaw": forecast, "previousRaw": previous, "actual": actual,
            "forecast": forecast, "previous": previous, "referencePeriod": "Dec"}


class PipelineTests(unittest.TestCase):
    def test_gold_surprise_sign(self):
        frame = pd.DataFrame([event_row(i, title=t) for i, t in enumerate(
            ["Non Farm Payrolls"]*4 + ["Unemployment Rate"]*4)])
        out = prepare_events(frame)
        self.assertEqual(out[out.event_type.eq("NFP")].gold_surprise_sign.iloc[0], -1)
        self.assertEqual(out[out.event_type.eq("Unemployment")].gold_surprise_sign.iloc[0], 1)

    def test_standardization_has_no_lookahead(self):
        frame = pd.DataFrame([event_row(i, date=f"2024-0{i+1}-05T13:30:00.000Z", actual=a, forecast=150000)
                              for i, a in enumerate([100000, 120000, 140000, 160000, 1000000])])
        out = prepare_events(frame)
        self.assertTrue(np.isnan(out.surprise_sigma.iloc[:4]).all())
        # Fifth scale is the median of the first four equal absolute surprises.
        self.assertAlmostEqual(out.surprise_sigma.iloc[4], 20000/0.67448975)

    def test_conflicting_duplicate_release_is_dropped(self):
        # Same release listed twice with different values: neither can be trusted.
        rows = [event_row(0, date="2024-01-05T13:30:00.000Z", actual=41000, forecast=50000),
                event_row(1, date="2024-01-05T12:30:00.000Z", actual=-140000, forecast=-15000)]
        out = prepare_events(pd.DataFrame(rows))
        self.assertEqual(len(out[out.event_type.eq("NFP")]), 0)

    def test_price_alignment_uses_pre_release_close(self):
        prices = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-05 13:20", periods=3, freq="5min", tz="UTC"),
            "close": [100, 101, 102],
        })
        price, timestamp = price_at_or_before(prices, pd.Timestamp("2024-01-05 13:30", tz="UTC"))
        self.assertEqual(price, 101)
        self.assertEqual(timestamp, pd.Timestamp("2024-01-05 13:25", tz="UTC"))


if __name__ == "__main__":
    unittest.main()
