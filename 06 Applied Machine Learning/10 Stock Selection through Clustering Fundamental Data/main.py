# region imports
from AlgorithmImports import *

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRanker
# endregion


class StockSelectionThroughClusteringFundamentalDataAlgorithm(QCAlgorithm):
    """
    This strategy demonstrates how to use learning to rank algorithms 
    to select a subset of stocks that are expected to outperform in 
    terms of raw returns. Specifically, this strategy uses 100 
    fundamental factors, principal component analysis, and a LGBMRanker 
    model.
    """

    def initialize(self):
        self.set_start_date(2018, 12, 31)
        self.set_end_date(2024, 4, 1)
        self.set_cash(100_000)
        self.settings.daily_precise_end_time = False

        self._liquid_universe_size = self.get_parameter(
            'liquid_universe_size', 100
        )
        self._final_universe_size = self.get_parameter(
            'final_universe_size', 10
        )
        self._lookback_period = timedelta(
            self.get_parameter('lookback_period', 365)
        )
        self._components = self.get_parameter('components', 5)
        # The `_prediction_period` is based in trading days. Don't 
        # change this value unless you change the rebalance frequency.
        self._prediction_period = 22 
        self._factors = [
            "market_cap",
            "earning_reports.basic_eps.value",
            "earning_reports.diluted_continuous_operations.value",
            "earning_reports.diluted_discontinuous_operations.value",
            "earning_reports.diluted_extraordinary.value",
            "earning_reports.diluted_eps.value",
            "earning_reports.dividend_per_share.value",
            "earning_reports.basic_eps_other_gains_losses.value",
            "earning_reports.total_dividend_per_share.value",
            "operation_ratios.revenue_growth.value",
            "operation_ratios.operation_income_growth.value",
            "operation_ratios.net_income_growth.value",
            "operation_ratios.net_income_cont_ops_growth.value",
            "operation_ratios.cfo_growth.value",
            "operation_ratios.fcf_growth.value",
            "operation_ratios.operation_revenue_growth_3_month_avg.value",
            "operation_ratios.stockholders_equity_growth.value",
            "operation_ratios.total_assets_growth.value",
            "operation_ratios.total_liabilities_growth.value",
            "operation_ratios.total_debt_equity_ratio_growth.value",
            "operation_ratios.cash_ratio_growth.value",
            "operation_ratios.ebitda_growth.value",
            "operation_ratios.cash_flow_from_financing_growth.value",
            "operation_ratios.cash_flow_from_investing_growth.value",
            "operation_ratios.cap_ex_growth.value",
            "operation_ratios.current_ratio_growth.value",
            "operation_ratios.gross_margin.value",
            "operation_ratios.operation_margin.value",
            "operation_ratios.pretax_margin.value",
            "operation_ratios.net_margin.value",
            "operation_ratios.tax_rate.value",
            "operation_ratios.ebit_margin.value",
            "operation_ratios.ebitda_margin.value",
            "operation_ratios.sales_per_employee.value",
            "operation_ratios.current_ratio.value",
            "operation_ratios.quick_ratio.value",
            "operation_ratios.long_term_debt_total_capital_ratio.value",
            "operation_ratios.interest_coverage.value",
            "operation_ratios.long_term_debt_equity_ratio.value",
            "operation_ratios.financial_leverage.value",
            "operation_ratios.total_debt_equity_ratio.value",
            "operation_ratios.days_in_sales.value",
            "operation_ratios.days_in_inventory.value",
            "operation_ratios.days_in_payment.value",
            "operation_ratios.inventory_turnover.value",
            "operation_ratios.payment_turnover.value",
            "operation_ratios.assets_turnover.value",
            "operation_ratios.roe.value",
            "valuation_ratios.payout_ratio",
            "valuation_ratios.sustainable_growth_rate",
            "valuation_ratios.cash_return",
            "valuation_ratios.sales_per_share",
            "valuation_ratios.book_value_per_share",
            "valuation_ratios.cfo_per_share",
            "valuation_ratios.fcf_per_share",
            "valuation_ratios.earning_yield",
            "valuation_ratios.pe_ratio",
            "valuation_ratios.sales_yield",
            "valuation_ratios.ps_ratio",
            "valuation_ratios.book_value_yield",
            "valuation_ratios.cf_yield",
            "valuation_ratios.pcf_ratio",
            "valuation_ratios.fcf_yield",
            "valuation_ratios.fcf_ratio",
            "valuation_ratios.trailing_dividend_yield",
            "valuation_ratios.forward_dividend_yield",
            "valuation_ratios.forward_earning_yield",
            "valuation_ratios.forward_pe_ratio",
            "valuation_ratios.peg_ratio",
            "valuation_ratios.peg_payback",
            "valuation_ratios.tangible_book_value_per_share",
            "valuation_ratios.forward_dividend",
            "valuation_ratios.working_capital_per_share",
            "valuation_ratios.ev_to_ebitda",
            "valuation_ratios.buy_back_yield",
            "valuation_ratios.total_yield",
            "valuation_ratios.normalized_pe_ratio",
            "valuation_ratios.price_to_ebitda",
            "valuation_ratios.forward_roe",
            "valuation_ratios.forward_roa",
            "valuation_ratios.two_years_forward_earning_yield",
            "valuation_ratios.two_years_forward_pe_ratio",
            "valuation_ratios.total_asset_per_share",
            "valuation_ratios.expected_dividend_growth_rate",
            "valuation_ratios.ev_to_revenue",
            "valuation_ratios.ev_to_pre_tax_income",
            "valuation_ratios.ev_to_total_assets",
            "valuation_ratios.ev_to_fcf",
            "valuation_ratios.ev_to_ebit",
            "valuation_ratios.ffo_per_share",
            "valuation_ratios.price_to_cash_ratio",
            "valuation_ratios.ev_to_forward_ebitda",
            "valuation_ratios.ev_to_forward_revenue",
            "valuation_ratios.ev_to_forward_ebit",
            "valuation_ratios.cape_ratio",
            "valuation_ratios.first_year_estimated_eps_growth",
            "valuation_ratios.second_year_estimated_eps_growth",
            "valuation_ratios.normalized_peg_ratio",
            "company_profile.total_employee_number",
            "company_profile.enterprise_value"
        ]

        schedule_symbol = Symbol.create("SPY", SecurityType.EQUITY, Market.USA)
        date_rule = self.date_rules.month_start(schedule_symbol)
        self.schedule.on(
            date_rule, 
            self.time_rules.after_market_open(schedule_symbol, 1), 
            self._trade
        )

        self._scaler = StandardScaler()
        self._pca = PCA(n_components=self._components, random_state=0)
        self.universe_settings.schedule.on(date_rule)
        self._universe = self.add_universe(self._select_assets)

    def _select_assets(self, fundamental):
        # Select the most liquid assets in the market.
        selected = sorted(
            [f for f in fundamental if f.has_fundamental_data], 
            key=lambda f: f.dollar_volume
        )[-self._liquid_universe_size:]
        liquid_symbols = [f.symbol for f in selected]

        # Get the factors.
        factors_by_symbol = {
            symbol: pd.DataFrame(columns=self._factors) 
            for symbol in liquid_symbols
        }
        history = self.history[Fundamental](
            liquid_symbols, self._lookback_period + timedelta(2)
        )
        for fundamental_dict in history:
            for symbol, asset_fundamentals in fundamental_dict.items():                
                factor_values = []
                for factor in self._factors:
                    factor_values.append(eval(f"asset_fundamentals.{factor}"))
                t = asset_fundamentals.end_time
                factors_by_symbol[symbol].loc[t] = factor_values
        
        # Determine which factors to use for PCA. We can't have any NaN 
        # values. We need to use the same factors for all assets, but 
        # some assets have missing factor values. In some cases we'll 
        # need to drop a factor, in other cases we'll need to drop a 
        # security.
        all_non_nan_factors = []
        tradable_symbols = []
        min_accepted_non_nan_factors = len(self._factors)
        for symbol, factor_df in factors_by_symbol.items():
            non_nan_factors = set(factor_df.dropna(axis=1).columns)
            if len(non_nan_factors) < 20: 
                # Let's say an asset needs at least 20 factors (otherwise
                # the `intersection` operation will remove almost all 
                # factors).
                continue
            min_accepted_non_nan_factors = min(
                min_accepted_non_nan_factors, len(non_nan_factors)
            )
            tradable_symbols.append(symbol)
            all_non_nan_factors.append(non_nan_factors)
        if not all_non_nan_factors:
            return []
        factors_to_use = all_non_nan_factors[0]
        for x in all_non_nan_factors[1:]:
            factors_to_use = factors_to_use.intersection(x)
        factors_to_use = sorted(list(factors_to_use))
        self.plot("Factors", "Count", len(factors_to_use))
        self.plot("Factors", "Min", min_accepted_non_nan_factors)

        # Get the training labels of the universe constituents.
        history = self.history(
            tradable_symbols, 
            self._lookback_period + timedelta(1), 
            Resolution.DAILY
        )
        label_by_symbol = {}
        for symbol in tradable_symbols[:]:
            # Remove the asset if there is not data for it.
            if symbol not in history.index:
                tradable_symbols.remove(symbol)
                continue 
            open_prices = history.loc[symbol]['open'].shift(-1)
            # `shift(-1)` so that the open price here represents the 
            # fill of a MOO order immediately after this universe 
            # selection.

            # Calculate the future return of holding for 22 full 
            # trading days (1 month).
            label_by_symbol[symbol] = open_prices.pct_change(
                self._prediction_period
            ).shift(-self._prediction_period).dropna() 

        # Build the factor matrix and label vector for training.
        X_train = pd.DataFrame()
        y_train = pd.DataFrame()
        for symbol in tradable_symbols:
            labels = label_by_symbol[symbol]
            factors = factors_by_symbol[symbol][factors_to_use].reindex(
                labels.index).ffill()
            X_train = pd.concat([X_train, factors])
            y_train[symbol] = labels
        X_train = X_train.sort_index()

        # Apply PCA.
        X_train_pca = self._pca.fit_transform(
            self._scaler.fit_transform(X_train)
        )

        # A higher value in y_train after this line means greater 
        # expected return. `- 1` to start ranking at 0 instead of 1.
        y_train = y_train.rank(axis=1, method='first').values.flatten() - 1 
        # There can be NaN values in `y_train` if at least one of the 
        # symbols is missing at least 1 label (insufficient history).
        y_train = y_train[~np.isnan(y_train)] 

        # Train the model. We need to set label_gain to avoid error. See 
        # https://github.com/microsoft/LightGBM/issues/1090 and 
        # https://github.com/microsoft/LightGBM/issues/5297.
        model = LGBMRanker(
            objective="lambdarank", 
            label_gain=list(range(len(tradable_symbols)))
        ) 
        # The `group` is a mapping from rebalance time to the number of
        # assets at that time.
        group = X_train.reset_index().groupby("time")["time"].count()
        model.fit(X_train_pca, y_train, group=group) 

        # Predict the ranking of assets over the upcoming month.
        X = pd.DataFrame()
        for symbol in tradable_symbols:
            X = pd.concat(
                [X, factors_by_symbol[symbol][factors_to_use].iloc[-1:]]
            )
        prediction_by_symbol = {
            tradable_symbols[i]: prediction 
            for i, prediction in enumerate(
                model.predict(self._pca.transform(self._scaler.transform(X)))
            )
        }

        # Select the assets that are predicted to rank the highest.
        sorted_predictions = sorted(
            prediction_by_symbol.items(), key=lambda x: x[1]
        )
        return [x[0] for x in sorted_predictions[-self._final_universe_size:]]

    def _trade(self):
        # Rebalance to form an equal-weighted portfolio.
        weight = 1 / len(self._universe.selected)
        self.set_holdings(
            [
                PortfolioTarget(symbol, weight) 
                for symbol in self._universe.selected
            ],
            True
        )

