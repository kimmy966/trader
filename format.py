import sys
from collections import namedtuple
import itertools

Trade = namedtuple("Trade",
    ["date", "time", "code", "buy_price", "buy_amount", "sell_price",
        "sell_amount"])

trades = []
for line in sys.stdin:
    cols = line.split()
    code = cols[2]
    # print(line.strip())
    trade = Trade(cols[0], cols[1], cols[2], cols[5], cols[6], cols[3], cols[4])
    trades.append(trade)

# trades = sorted(trades, key=lambda t: (t.date, t.time, t.code))
for key, trades in itertools.groupby(trades, lambda t: (t.date, t.time, t.code)):
    trades = [t for t in trades]
    print(trades[0].date, trades[0].time, trades[0].code, trades[0].buy_price,
        trades[0].sell_price, " ".join(t.buy_amount for t in trades),
        " ".join(t.sell_amount for t in trades))
