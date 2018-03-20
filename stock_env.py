import itertools
import numpy
from collections import namedtuple


TradeInfo = namedtuple("TradeInfo", ["price", "duration"])

class TradeState(object):
    def __init__(self, text):
        cols = text.split()
        assert len(cols) == 25, cols
        self.date = cols[0]
        self.time = cols[1]
        self.code = cols[2]
        self.buy_price = int(cols[3])
        self.sell_price = int(cols[4])
        self.buy_amounts = [int(c) for c in cols[5:15]]
        self.sell_amounts = [int(c) for c in cols[15:25]]


TAX_RATE = 0.000
class StockEnv(object):
    HOLD = 0
    BUY = 1
    SELL = 2

    def __init__(self, code, episodes_file="prices.log"):
        trade_states = [ TradeState(line) for line in open(episodes_file, "rt")]
        trade_states = sorted(trade_states, key=lambda s: (s.code, s.date, s.time))
        self.episodes = []
        for key, states in itertools.groupby(trade_states, lambda s: (s.code, s.date)):
            if key[0] == code:
                episode = []
                initial = None
                for s in states:
                    if not initial:
                        initial = s.buy_price
                    s.buy_price /= initial
                    s.sell_price /= initial
                    episode.append(s)
                self.episodes.append(episode)

        self.trade_price = 0.0
        self.trade_count = 0
        self.trade_step = 0

    def get_current_state(self):
        s = self.episodes[self.current_episode][self.current_index]
        return numpy.asarray([s.buy_price - self.trade_price, s.buy_price, s.sell_price] + s.buy_amounts + s.sell_amounts)

    def reset(self):
        self.current_episode = numpy.random.randint(len(self.episodes))
        self.current_index = numpy.random.randint(len(self.episodes[self.current_episode]) - 50)
        self.trade_price = 0.0
        self.trade_count = 0
        self.trade_step = 0
        return self.get_current_state()

    def step(self, action):
        '''

        :param action:
        :return: new_state, reward, done, info
        '''
        self.current_index += 1
        s = self.episodes[self.current_episode][self.current_index]
        reward = 0.0
        done = False
        if self.current_index == len(self.episodes[self.current_episode]) - 1:
            if self.trade_count == 0:
                reward += -s.sell_price * TAX_RATE * 2
            elif self.trade_price > 0:
                reward += s.buy_price * (1 - TAX_RATE) - self.trade_price
            done = True
        elif action == StockEnv.BUY and self.trade_price == 0:
            self.trade_price = s.sell_price
            self.trade_count += 1
        elif action == StockEnv.SELL and self.trade_price > 0:
            reward += s.buy_price * (1 - TAX_RATE) - self.trade_price
            done = True

        if self.trade_price > 0:
            self.trade_step += 1
        return self.get_current_state(), reward, done, TradeInfo(self.trade_price, self.trade_step)

    def random_action(self):
        r = numpy.random.randint(5)
        if r == 0:
            return StockEnv.BUY
        elif r == 1:
            return StockEnv.SELL
        else:
            return StockEnv.HOLD

    def num_state(self):
        return 23

    def num_action(self):
        return 3

