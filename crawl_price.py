from bs4 import BeautifulSoup
import requests
import sys

from datetime import datetime

url = "http://finance.naver.com/item/sise.nhn"
# response = requests.get(url, params={"code": "005930", "asktype": 10}) #삼성전자
'''
035720 카카오
005930 삼성전자
000660 SK하이닉스
000120 CJ대한통운
285130 SK케미칼
008970 동양철관
008800 행남자기
033110 이디
001440 대한전선
'''
codes = sys.argv[1] if len(sys.argv) > 1 else "035720,005930,000660,000120,008970,008800,033110,001440"
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
for code in codes.split(","):
    response = requests.get(url, params={"code": code, "asktype": 10})

    soup = BeautifulSoup(response.text, "html.parser")
    # print(response.text)
    elems = soup.select(
        "#content > div.section.inner_sub > div:nth-of-type(2) tbody > tr")
    # elems = soup.select("div.inner_sub")
    sell_prices = []
    buy_prices = []
    sell_amounts = []
    buy_amounts = []
    for elem in elems:
        orders = [elem.text.strip().replace(",", "") for elem in elem.select("td")]
        if len(orders) == 5:
            # print(current_time, code, orders[1], orders[0], orders[3], orders[4])
            sell_prices.append(int(orders[1]))
            sell_amounts.append(orders[0])
            buy_prices.append(int(orders[3]))
            buy_amounts.append(orders[4])
    print(current_time, code, buy_prices[0], sell_prices[0], " ".join(buy_amounts), " ".join(sell_amounts))

'''
#content > div.section.inner_sub > div:nth-child(2) > table:nth-child(2) > tbody > tr:nth-child(3) > td:nth-child(1) > span > strong
#content > div.section.inner_sub > div:nth-child(2) > table:nth-child(2) > tbody > tr:nth-child(3) > td:nth-child(2) > span > strong

#content > div.section.inner_sub > div:nth-child(2) > table:nth-child(2) > tbody > tr:nth-child(4) > td:nth-child(1) > span

#content > div.section.inner_sub > div:nth-child(2) > table:nth-child(2) > tbody > tr:nth-child(3) > td:nth-child(5) > span > strong
#content > div.section.inner_sub > div:nth-child(2) > table:nth-child(2) > tbody > tr:nth-child(3) > td:nth-child(4) > span > strong

#content > div.section.inner_sub > div:nth-child(2) > table:nth-child(2) > tbody > tr:nth-child(11) > td:nth-child(1) > span
'''
