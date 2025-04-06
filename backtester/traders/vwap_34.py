from backtester.datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
import jsonpickle

class Trader:
    MAX_KELP_POSITION = 50
    MAX_RESIN_POSITION = 50
    products = ['KELP', 'RAINFOREST_RESIN']
    
    def run(self, state: TradingState):
        print("Current Positions: " + str(state.position))
        if state.traderData:
            traderData = jsonpickle.decode(state.traderData)
        else:
            traderData = {}
        result = {}
        for product in self.products:
            order_depth: OrderDepth = state.order_depths[product]
            print(product + ' SELL ORDERS: ' + str(order_depth.sell_orders))
            print(product + ' BUY ORDERS: ' + str(order_depth.buy_orders))
            current_pos = state.position[product] if product in state.position else 0
            total_dolvol = 0
            total_vol = 0
            for ask, ask_amount in list(order_depth.sell_orders.items()):
                ask_amount = abs(ask_amount)
                total_dolvol += ask * ask_amount
                total_vol += ask_amount
            for bid, bid_amount in list(order_depth.buy_orders.items()):
                total_dolvol += bid * bid_amount
                total_vol += bid_amount
            current_vwap = total_dolvol / total_vol
            rounded_vwap = round(current_vwap)
            orders = []
            if current_pos != 0:
                orders.append(Order(product, rounded_vwap-int(2*current_pos/abs(current_pos)), -current_pos))
            max_buy = -current_pos + self.MAX_KELP_POSITION
            max_sell = -current_pos - self.MAX_KELP_POSITION
            orders.append(Order(product, rounded_vwap+3, int(max_buy//3)))
            orders.append(Order(product, rounded_vwap+4, int(max_sell//3))) # Buy at VWAP-1 if possible
            result[product] = orders
        conversions = 1
        traderData = jsonpickle.encode(traderData)
        return result, conversions, traderData
    