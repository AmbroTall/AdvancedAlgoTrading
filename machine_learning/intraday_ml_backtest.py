import click
import datetime
from qstrader import settings
from qstrader.compat import queue
from qstrader.price_parser import PriceParser
from qstrader.price_handler.iq_feed_intraday_csv_bar import IQFeedIntradayCsvBarPriceHandler
# from qstrader.strategy import
from qstrader.position_sizer.naive import NaivePositionSizer
from qstrader.risk_manager.example import ExampleRiskManager
from qstrader.portfolio_handler import PortfolioHandler
from qstrader.compliance.example import ExampleCompliance
from qstrader.execution_handler.ib_simulated import IBSimulatedExecutionHandler
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading_session import TradingSession
from intraday_ml_strategy import IntradayMachineLearningPredictionStrategy


def run(config, testing, tickers, filename):
    # Set up variables needed for backtest
    events_queue = queue.Queue()
    csv_dir = "data"

    initial_equity = PriceParser.parse(500000.00)

    # Use DTN IQFeed Intraday Bar Price Handler
    start_date = datetime.datetime(2013, 1, 1)

    price_handler = IQFeedIntradayCsvBarPriceHandler(
        csv_dir, events_queue, tickers, start_date=start_date
    )
    # Use the IntradayMachineLearningPredictionStrategy
    model_pickle_file = 'ml_model_rf.pkl'
    strategy = IntradayMachineLearningPredictionStrategy(
        tickers, events_queue, model_pickle_file, lags=5
    )
    # strategy = Strategies(strategy, DisplayStrategy())
    # Use the Naive Position Sizer
    # (suggested quantities are followed)
    position_sizer = NaivePositionSizer()
    # Use an example Risk Manager
    risk_manager = ExampleRiskManager()
    # Use the default Portfolio Handler
    portfolio_handler = PortfolioHandler(
        initial_equity, events_queue, price_handler,
        position_sizer, risk_manager
    )
    # Use the ExampleCompliance component
    compliance = ExampleCompliance(config)
    # Use a simulated IB Execution Handler
    execution_handler = IBSimulatedExecutionHandler(
        events_queue, price_handler, compliance
    )
    # Use the Tearsheet Statistics
    statistics = TearsheetStatistics(
        config, portfolio_handler,
        title=["Intraday AREX Machine Learning Prediction Strategy"],
        periods=int(252 * 6.5 * 60)  # Minutely periods
    )
    # Set up the backtest
    backtest = TradingSession(
        price_handler, strategy,
        portfolio_handler, execution_handler,
        position_sizer, risk_manager,
        statistics, initial_equity
    )
    results = backtest.simulate_trading(testing=testing)
    statistics.save(filename)
    return results


@click.command()
@click.option(
    '--config',
    default=settings.DEFAULT_CONFIG_FILENAME,
    help='Config filename'
)
@click.option(
    '--testing / --no - testing',
    default=False,
    help='Enable testing mode'
)
@click.option(
    '--tickers',
    default='AGG',
    help='Tickers(usecomma)'
)
@click.option(
    '--filename',
    default='',
    help='Pickle(.pkl) statistics filename'
)
def main(config, testing, tickers, filename):
    tickers = tickers.split(",")
    config = settings.from_file(config, testing)
    run(config, testing, tickers, filename)


if __name__ == "__main__":
    main()
