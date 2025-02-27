from math import exp 
import scipy as sp


def bond_discount(time, rate, type):
    if type == 'continuous':
        return exp(-rate * time)
    elif type == 'discrete':
        return 1 / (1 + rate) ** time
    else:
        raise ValueError("Type must be 'continuous' or 'discrete'")
    

class Bond:
    
    def __init__(self, principal, maturity, interest_rate, capitalization_type):
        self.principal = principal
        self.maturity = maturity
        self.interest_rate = interest_rate # market interest rate
        self.capitalization_type = capitalization_type
        
    def present_value(self, x, n):
        return x * bond_discount(time=n, rate=self.interest_rate, type=self.capitalization_type)
         
        
class ZeroCouponBond(Bond):
    
    def __init__(self, principal, maturity, interest_rate, capitalization_type):
        super().__init__(principal, maturity, interest_rate, capitalization_type)
    
    def calculate_price(self):
        return self.present_value(self.principal, self.maturity)
    
    
class CouponBond(Bond):
    
    def __init__(self, principal, maturity, interest_rate, coupon_rate, capitalization_type):
        super().__init__(principal, maturity, interest_rate, capitalization_type)
        self.coupon_rate = coupon_rate
        
    def calculate_price(self):
        price = 0
        for i in range(1, self.maturity + 1):
            price += self.present_value(self.principal * self.coupon_rate, i)
        price += self.present_value(self.principal, self.maturity)
        return price
        
        