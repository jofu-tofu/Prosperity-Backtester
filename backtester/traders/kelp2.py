from typing import List

from datamodel import Order, OrderDepth, TradingState


class Trader:
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        result = {}
        for product in state.order_depths:
            if product == "KELP":
                order_depth: OrderDepth = state.order_depths[product]
                orders: List[Order] = []
                position: int = state.position.get(product, 0)
                max_buy = 50 - position
                max_sell = -50 - position
                print("POSITION", position)
                # total_amount = 0
                # acceptable_price = 0
                # if len(order_depth.sell_orders) != 0:
                #   for ask, amount in list(order_depth.sell_orders.items()):
                #     acceptable_price += abs(ask * amount)
                #     total_amount += abs(amount)
                # if len(order_depth.buy_orders) != 0:
                #   for bid, amount in list(order_depth.buy_orders.items()):
                #     acceptable_price += abs(bid * amount)
                #     total_amount += abs(amount)
                # acceptable_price = int(acceptable_price/total_amount)
                # if len(order_depth.sell_orders) != 0:
                #     for ask, amount in list(order_depth.sell_orders.items()):
                #         if int(ask) < acceptable_price:
                #             buy_volume = max(min(-amount, max_buy), 0)
                #             print("BUY", str(-buy_volume) + "x", ask)
                #             orders.append(Order(product, ask, buy_volume))
                #             max_buy = max_buy - buy_volume
                #             position = position + buy_volume
                #         elif int(ask) <= acceptable_price and position < 0:
                #             buy_volume = max(min(-amount, -position), 0)
                #             print("BUY", str(-buy_volume) + "x", ask)
                #             orders.append(Order(product, ask, buy_volume))
                #             max_buy = max_buy - buy_volume
                #             position = position + buy_volume

                # if len(order_depth.buy_orders) != 0:
                #     for bid, amount in list(order_depth.buy_orders.items()):
                #         if int(bid) > acceptable_price:
                #             sell_volume = min(max(-amount, max_sell), 0)
                #             print("SELL", str(sell_volume) + "x", bid)
                #             orders.append(Order(product, bid, sell_volume))
                #             max_sell = max_sell - sell_volume
                #             position = position + sell_volume
                #         elif int(bid) >= acceptable_price and position > 0:
                #             sell_volume = min(max(-amount, -position), 0)
                #             print("SELL", str(sell_volume) + "x", bid)
                #             orders.append(Order(product, bid, sell_volume))
                #             max_sell = max_sell - sell_volume
                #             position = position + sell_volume

                # if max_buy < 5:
                #   sell_threshold = list(order_depth.sell_orders.items())[-1][0]
                #   orders.append(Order(product, sell_threshold - 2, -5))
                #   max_sell = max_sell + 5
                # if max_sell > -5:
                #   buy_threshold = list(order_depth.buy_orders.items())[-1][0]
                #   orders.append(Order(product, buy_threshold + 2, 5))
                #   max_buy = max_buy - 5

                if max_sell != 0:
                    sell_threshold = list(order_depth.sell_orders.items())[-1][0]
                    orders.append(Order(product, sell_threshold - 1, max_sell))
                if max_buy != 0:
                    buy_threshold = list(order_depth.buy_orders.items())[-1][0]
                    orders.append(Order(product, buy_threshold + 1, max_buy))

                result[product] = orders

        traderData = "SAMPLE"

        conversions = 1

        return result, conversions, traderData
