import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class Capm:
    
    RISK_FREE_RATE = 0.05
    # Considering monthly returns and calculating annul returns
    MONTHS_IN_YEAR = 12
    
    def __init__(self, stocks, start_date, end_date):
        self.stocks = stocks
        self.dates = {'start_date': start_date, 'end_date': end_date}
        
    def download_data(self):
        stock_data = {}
        for stock in self.stocks:
            ticker = yf.Ticker(stock)
            # Download historical data
            hist = ticker.history(start=self.dates['start_date'], end=self.dates['end_date'])
            stock_data[stock] = hist['Close']
        data = pd.DataFrame(stock_data)
        return data
    
    def initialize(self):
        
        stock_data = self.download_data()
        # Monthly instead of daily data because monthly are closely to normal distribution
        stock_data = stock_data.resample('M').last()
        self.data = pd.DataFrame({'s_adjclose': stock_data[self.stocks[0]],
                                  'm_adjclose': stock_data[self.stocks[1]]})
        
        self.data[['s_returns', 'm_returns']] = np.log(self.data[['s_adjclose', 'm_adjclose']] / self.data[['s_adjclose', 'm_adjclose']].shift(1))
        
        self.data = self.data[1:]
        
    def calculate_beta_capm(self):
        covariance_matrix = np.cov(self.data['s_returns'], self.data['m_returns'])
        covariance = covariance_matrix[0, 1]
        variance = covariance_matrix[1, 1]
        self.capm_beta = covariance / variance
    
    def calculate_regression(self):
        self.beta_reg, self.alpha_reg = np.polyfit(self.data['m_returns'], self.data['s_returns'], deg=1)
    
        
    def plot_parameters(self):
        if not hasattr(self, 'capm_beta'):
            self.calculate_beta_capm()
        if not hasattr(self, 'beta_reg'):
            self.calculate_regression()
        
        expected_return = self.RISK_FREE_RATE + self.capm_beta * (self.data['m_returns'].mean() * self.MONTHS_IN_YEAR - self.RISK_FREE_RATE)
        fig, axis = plt.subplots(1, figsize=(10, 6))
        axis.scatter(self.data["m_returns"], self.data['s_returns'], label="Data Points")
        axis.plot(self.data["m_returns"], self.beta_reg * self.data["m_returns"] + self.alpha_reg, color='red', label="CAPM Line")

        plt.title('Capital Asset Pricing Model: Finding Alpha and Beta', fontsize=14)
        plt.xlabel('Market Return $R_m$', fontsize=14)
        plt.ylabel('Stock Return $R_a$', fontsize=14)
        plt.text(0.08, 0.05, r'$E\[R_a\] = R_f + \beta \cdot R_m + \alpha$', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
        