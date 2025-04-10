from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import numpy as np
import jsonpickle

class Trader:

    product_max_positions = {'KELP': 50, 'RAINFOREST_RESIN': 50, 'SQUID_INK': 50, 'CROISSANTS': 250,
                             "JAMS": 350, "DJEMBES": 60, "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100}
    
    def run(self, state: TradingState):
        print("Current Positions: " + str(state.position))
        if state.traderData:
            traderData = jsonpickle.decode(state.traderData)
        else:
            traderData = {}
        result = {}
        for product in self.product_max_positions.keys():
            order_depth: OrderDepth = state.order_depths[product]
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
                orders.append(Order(product, rounded_vwap-int(3*current_pos/abs(current_pos)), -current_pos))
            max_buy = -current_pos + self.product_max_positions[product]
            max_sell = -current_pos - self.product_max_positions[product]
            orders.append(Order(product, rounded_vwap-3, int(max_buy//2)))
            orders.append(Order(product, rounded_vwap+3, int(max_sell//2))) # Buy at VWAP-1 if possible
            result[product] = orders
            print("Orders: " + str(orders))
        conversions = 1
        traderData = jsonpickle.encode(traderData)
        return result, conversions, traderData
    