import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimization


class MarkowitzPortfolio:
    NUM_TRADING_DAYS = 252  

    def __init__(self, portfolio, start_date, end_date):
        self.portfolio = portfolio
        self.dates = {'start_date': start_date, 'end_date': end_date}
        self.data = self.download_data()

    def calculate_returns(self, period=1):
        log_returns = np.log(self.data / self.data.shift(period))
        return log_returns[period:]

    def calculate_statistics(self, returns, weights, period=NUM_TRADING_DAYS):    
        weights = np.array(weights)
        mean = returns.mean() * period
        cov = returns.cov() * period
        portfolio_return = np.sum(mean * weights)
        portfolio_volatility = np.sqrt(weights.T @ cov @ weights)
        sharpe_ratio = portfolio_return / portfolio_volatility
        return portfolio_return, portfolio_volatility, sharpe_ratio

    def download_data(self):
        stock_data = {}
        for stock in self.portfolio:
            ticker = yf.Ticker(stock)  
            stock_data[stock] = ticker.history(start=self.dates['start_date'], end=self.dates['end_date'])['Close']
        data = pd.DataFrame(stock_data)
        return data

    def show_data(self):
        self.data.plot(figsize=(10, 6))
        plt.show()
        
    def generate_random_portfolios(self, num_portfolios=10000):
        portfolios = []
        returns = self.calculate_returns()

        for _ in range(num_portfolios):
            weights = np.random.random(len(self.portfolio))
            weights /= np.sum(weights)
            
            portfolio_return, portfolio_volatility, sharpe_ratio = self.calculate_statistics(returns, weights)

            portfolios.append({
                'Return': portfolio_return,
                'Volatility': portfolio_volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Weights': weights
            })

        self.portfolios_df = pd.DataFrame(portfolios)
        return self.portfolios_df
    
    def show_portfolios(self):
        if not hasattr(self, 'portfolios_df'):
            self.generate_random_portfolios(num_portfolios=10000)
        cmap = 'jet'
        self.portfolios_df.plot.scatter(x='Volatility', y='Return', c='Sharpe Ratio', cmap=cmap, figsize=(10, 6), grid=True, edgecolors='black')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.title('Portfolio Distribution')
        plt.show()
        
    def optimize_portfolio(self):
        returns = self.calculate_returns()
        num_assets = len(self.portfolio)

        def negative_sharpe(weights, returns):
            return -self.calculate_statistics(returns, weights)[2]  # Retorna o negativo do Sharpe Ratio

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        initial_weights = np.array([1.0 / num_assets] * num_assets)

        optimized = optimization.minimize(
            negative_sharpe,
            initial_weights,
            args=(returns,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if optimized.success:
            optimal_weights = optimized.x
            opt_return, opt_volatility, opt_sharpe = self.calculate_statistics(returns, optimal_weights)

            self.optimal_portfolio = {
                'Return': opt_return,
                'Volatility': opt_volatility,
                'Sharpe Ratio': opt_sharpe,
                'Weights': optimal_weights.round(3)
            }
            return self.optimal_portfolio
        else:
            raise ValueError("Optimization failed!")
        
    def plot_optimal_portfolio(self):
        if not hasattr(self, 'optimal_portfolio'):
            self.optimize_portfolio()
        if not hasattr(self, 'portfolios_df'):
            self.generate_random_portfolios(num_portfolios=10000)

        cmap = 'jet'
        self.portfolios_df.plot.scatter(x='Volatility', y='Return', c='Sharpe Ratio', cmap=cmap, figsize=(10, 6), grid=True, edgecolors='black')
        plt.scatter(self.optimal_portfolio['Volatility'], self.optimal_portfolio['Return'], color='yellow', edgecolors='black', marker='*', s=200)
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.title('Optimal Portfolio')
        plt.show()
