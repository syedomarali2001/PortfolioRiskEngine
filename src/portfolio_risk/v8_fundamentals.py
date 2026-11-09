"""V8 audited stock fundamentals and scenario DCF, independent of ticker provider.

No inferred zero fields, currency mixing, or valuation claims from nonpositive FCF.
Financial statements must be dated, and DCF is explicitly a sensitivity exercise.
"""
from __future__ import annotations

from datetime import date
from math import isfinite
from numbers import Real
from typing import Mapping

import pandas as pd


class FundamentalsError(ValueError):
    pass


def _number(data: Mapping, key: str, *, required: bool = False) -> float | None:
    x = data.get(key)
    if x is None:
        if required:
            raise FundamentalsError(f"Missing required numeric field {key}")
        return None
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not isfinite(x):
        raise FundamentalsError(f"{key} must be a finite number")
    return float(x)


def _dcf(fcf: float, cash: float, debt: float, shares: float, growth: float,
         discount: float, terminal: float, *, years: int = 5) -> dict:
    if not (-.99 < growth < 1 and 0 < discount < 1 and -.5 < terminal < discount and shares > 0):
        raise FundamentalsError("DCF growth/discount/terminal/shares assumptions invalid")
    projected = [fcf*(1+growth)**yr for yr in range(1,years+1)]
    pv_fcf = sum(v/(1+discount)**i for i,v in enumerate(projected,start=1))
    terminal_cash_flow = projected[-1]*(1+terminal)
    terminal_value = terminal_cash_flow/(discount-terminal)
    pv_terminal = terminal_value/(1+discount)**years
    enterprise = pv_fcf+pv_terminal
    equity = enterprise+cash-debt
    return {"growth":growth,"discount_rate":discount,"terminal_growth":terminal,
            "projection_years":years,"projected_fcf":projected,
            "discounted_fcfs":pv_fcf,"terminal_value":terminal_value,
            "discounted_terminal":pv_terminal,"enterprise_value":enterprise,
            "equity_value":equity, "intrinsic_value_per_share":equity/shares,
            "terminal_share_of_enterprise_value":pv_terminal/enterprise if enterprise>0 else None}



def simple_dcf_limitation(symbol: str, sector: str | None = None,
                          industry: str | None = None) -> str | None:
    """Do not turn a generic operating FCF perpetuity into an insurer/bank price target.

    The classifier is conservative: it blocks only clearly identified financial
    business models and Berkshire share classes, even if classification is missing.
    """
    normalized = str(symbol).strip().upper().replace(".", "-")
    if normalized in {"BRK-A", "BRK-B"}:
        return ("Berkshire Hathaway combines insurance businesses, investments and operating "
                "subsidiaries. Insurance float and investment holdings are not fully represented "
                "by a generic company free-cash-flow calculation. A sum-of-the-parts approach "
                "and insurance-specific measures are more appropriate")
    sector_text = str(sector or "").strip().lower()
    industry_text = str(industry or "").strip().lower()
    if ("insurance" in industry_text or industry_text.startswith("banks") or
            "investment management" in industry_text or "asset management" in industry_text or
            sector_text in {"financial services", "financials"}):
        return ("Financial institutions have balance sheets and cash flows that need an "
                "industry-specific valuation method; an ordinary operating-company DCF is not sufficient")
    return None


def analyze_fundamentals(record: Mapping, *, as_of: date | None = None,
                         scenarios: Mapping[str, Mapping[str,float]] | None = None) -> dict:
    """Latest reported fiscal statement; quote and statement currencies MUST match."""
    symbol = str(record.get("symbol","")).strip().upper()
    if not symbol or not isinstance(record.get("period_end"),str):
        raise FundamentalsError("Symbol and fiscal period_end (YYYY-MM-DD) are required")
    try:
        end = date.fromisoformat(record["period_end"])
    except ValueError as exc:
        raise FundamentalsError("period_end must be YYYY-MM-DD") from exc
    cutoff = as_of or date.today()
    if end > cutoff:
        raise FundamentalsError("Financial statement dates after analysis cutoff are invalid")
    currency = str(record.get("statement_currency","")).upper()
    quote = str(record.get("quote_currency",currency)).upper()
    if currency not in {"USD","CAD"} or quote not in {"USD","CAD"}:
        raise FundamentalsError("Statement/quotation currency must be USD or CAD")
    if quote != currency:
        raise FundamentalsError("Statement and market quote currencies differ; provide audited FX conversion first")
    revenue = _number(record,"revenue")
    ebitda = _number(record,"ebitda")
    net_income = _number(record,"net_income")
    fcf = _number(record,"free_cash_flow")
    cfo = _number(record,"operating_cash_flow")
    capex = _number(record,"capital_expenditures")
    if fcf is None and cfo is not None and capex is not None:
        if capex > 0:
            raise FundamentalsError("capital_expenditures convention must be NEGATIVE cash outflow when deriving FCF")
        fcf = cfo + capex
    shares = _number(record,"shares_outstanding")
    cash = _number(record,"cash")
    debt = _number(record,"total_debt")
    price = _number(record,"price")
    market_cap = _number(record,"market_cap")
    if shares is not None and shares <= 0:
        raise FundamentalsError("shares_outstanding must be positive")
    if (cash is not None and cash < 0) or (debt is not None and debt < 0):
        raise FundamentalsError("Cash/debt balances must be nonnegative")
    if price is not None and price <= 0:
        raise FundamentalsError("Share price must be positive")
    if market_cap is None and shares is not None and price is not None:
        market_cap = shares*price
    ev = market_cap + debt-cash if None not in (market_cap,debt,cash) else None
    multiples = {"price_to_fcf":market_cap/fcf if market_cap is not None and fcf and fcf>0 else None,
                 "price_to_earnings":market_cap/net_income if market_cap is not None and net_income and net_income>0 else None,
                 "ev_to_ebitda":ev/ebitda if ev is not None and ebitda and ebitda>0 else None,
                 "fcf_margin":fcf/revenue if fcf is not None and revenue and revenue>0 else None,
                 "net_debt":debt-cash if debt is not None and cash is not None else None}
    default = {"conservative":{"growth":0.,"discount":.12,"terminal":.015},
               "base":{"growth":.05,"discount":.10,"terminal":.02},
               "optimistic":{"growth":.10,"discount":.08,"terminal":.025}}
    inputs = scenarios if scenarios is not None else default
    if not inputs:
        raise FundamentalsError("At least one DCF scenario is required")
    valuations = {}
    reason = simple_dcf_limitation(symbol, record.get("sector"), record.get("industry"))
    if reason:
        pass  # Retain reported multiples, but omit misleading standard DCF prices.
    elif fcf is None or fcf <= 0:
        reason = "Positive annual free cash flow not available; DCF is not supported for this company"
    elif None in (shares,cash,debt):
        reason = "Shares, cash or debt missing; per-share DCF is unavailable instead of inventing balances"
    else:
        for label,parameters in inputs.items():
            if not label or set(parameters)!={"growth","discount","terminal"}:
                raise FundamentalsError("Each scenario must supply growth, discount and terminal")
            valuation = _dcf(fcf,cash,debt,shares,**parameters)
            if price is not None:
                valuation["illustrative_upside_vs_price"] = valuation["intrinsic_value_per_share"]/price - 1
            valuations[label] = valuation
    warnings = []
    age = (cutoff-end).days
    if age>450:
        warnings.append(f"Financial period ends {age} days before analysis date; historical/stale fundamentals")
    if market_cap is not None and price is not None and shares is not None and abs(market_cap/(price*shares)-1)>.05:
        warnings.append("Market cap differs from quoted price * shares by >5%; verify share basis and dates")
    if fcf is not None and fcf <= 0:
        warnings.append("Negative/nonpositive free cash flow: simple perpetuity DCF inappropriate")
    return {"symbol":symbol,"as_of":str(cutoff),"fiscal_period_end":str(end),
            "statement_currency":currency,"quote_currency":quote,
            "sector":record.get("sector"),"industry":record.get("industry"),
            "source":record.get("source","user-supplied statements (unverified)"),
            "reported": {k:record.get(k) for k in ("revenue","ebitda","net_income","free_cash_flow",
                          "operating_cash_flow","capital_expenditures","shares_outstanding","cash",
                          "total_debt","price","market_cap")},
            "used_free_cash_flow":fcf,"estimated_market_cap":market_cap,
            "enterprise_value":ev,"multiples":multiples,
            "dcf":{"status":"unavailable" if reason else "scenario_analysis",
                   "reason":reason,"scenarios":valuations},
            "warnings":warnings,
            "limitations":["Model-based hypothetical five-year FCF projections; not a price forecast or buy/sell signal.",
                           "FCF may vary significantly by year or industry; financial firms often require a different valuation model.",
                           "No automatic peer multiple comparisons or growth/terminal forecast verification.",
                           "Market quotes may be dated differently from fiscal statements and share counts."]}


def _latest(frame: pd.DataFrame, aliases: tuple[str,...], cutoff: date) -> tuple[float|None,date|None]:
    if not isinstance(frame,pd.DataFrame) or frame.empty:
        return None,None
    # Yahoo row vocabulary evolves; do not assume today's row names are stable.
    normalized = {''.join(ch for ch in str(idx).lower() if ch.isalnum()):idx for idx in frame.index}
    for name in aliases:
        key=''.join(ch for ch in name.lower() if ch.isalnum())
        if key not in normalized:
            continue
        row = frame.loc[normalized[key]]
        for col in sorted(frame.columns, key=lambda x: pd.Timestamp(x), reverse=True):
            timestamp = pd.Timestamp(col)
            if timestamp.date() > cutoff:
                continue
            val = row[col]
            if isinstance(val,Real) and not isinstance(val,bool) and isfinite(val):
                return float(val),timestamp.date()
    return None,None


def yahoo_fundamentals_snapshot(symbol: str, *, as_of: date | None = None,
                                ticker_factory=None) -> dict:
    """Best-effort personal research adapter. No key; missing fields stay missing.

    Yahoo financial statements and information can be inconsistent, changing,
    delayed or blocked. Use the raw record for offline traceable verification.
    """
    cutoff=as_of or date.today()
    if ticker_factory is None:
        import yfinance as yf
        ticker_factory=yf.Ticker
    ticker=ticker_factory(symbol)
    try:
        info=ticker.get_info()
        cashflow=ticker.get_cashflow(freq="yearly")
        income=ticker.get_income_stmt(freq="yearly")
        balance=ticker.get_balance_sheet(freq="yearly")
    except Exception as exc:
        raise FundamentalsError(f"Yahoo statement access failed for {symbol}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(info,dict):
        raise FundamentalsError("Yahoo company info must be a mapping")
    if str(info.get("quoteType","EQUITY")).upper()!="EQUITY":
        raise FundamentalsError("Fundamentals require an individual equity, not an ETF/crypto/bond")
    stmt_ccy=info.get("financialCurrency")
    quote_ccy=info.get("currency")
    if stmt_ccy not in ("CAD","USD") or quote_ccy not in ("CAD","USD"):
        raise FundamentalsError("Unverified financial statement/price currency; explicit manual inputs required")
    values={}
    dates=[]
    names={
        "free_cash_flow":(cashflow,("FreeCashFlow",)),
        "operating_cash_flow":(cashflow,("OperatingCashFlow",)),
        "capital_expenditures":(cashflow,("CapitalExpenditure", "CapitalExpenditures")),
        "revenue":(income,("TotalRevenue", "OperatingRevenue")),
        "ebitda":(income,("NormalizedEBITDA", "EBITDA")),
        "net_income":(income,("NetIncome", "NetIncomeCommonStockholders")),
        "cash":(balance,("CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments")),
        "total_debt":(balance,("TotalDebt", "CurrentDebtAndCapitalLeaseObligation")),
    }
    for field,(frame,aliases) in names.items():
        val,period = _latest(frame,aliases,cutoff)
        if val is not None:
            values[field]=val
            dates.append(period)
    if not dates:
        raise FundamentalsError("No dated annual statement fields available; unable to perform valuation")
    # Conservative: choose a common fiscal year, not latest values of mixed years.
    period = max(dates)
    for field,(frame,aliases) in names.items():
        val,p = _latest(frame,aliases,period)
        if p == period:
            values[field]=val
        else:
            values.pop(field,None)
    for field,key in (("shares_outstanding","sharesOutstanding"),
                      ("price","currentPrice"),("market_cap","marketCap")):
        val=info.get(key)
        if isinstance(val,Real) and not isinstance(val,bool) and isfinite(val):
            values[field]=float(val)
    if "price" not in values:
        val=info.get("regularMarketPrice")
        if isinstance(val,Real) and not isinstance(val,bool) and isfinite(val):
            values["price"]=float(val)
    values.update({"symbol":symbol.upper(), "period_end":str(period),
                   "statement_currency":stmt_ccy,"quote_currency":quote_ccy,
                   "sector":info.get("sector"), "industry":info.get("industry"),
                   "source":"Yahoo Finance financial statements via yfinance; best effort, personal research"})
    return values


def compare_peer_multiples(subject: Mapping, peers: list[Mapping], *,
                           as_of: date | None = None) -> dict:
    """Transparent peer-relative multiple comparison; not a stock recommendation.

    The investor chooses comparison tickers. At least TWO usable peers per
    metric are required; incompatible/unavailable observations stay excluded.
    """
    if not peers:
        raise FundamentalsError("Supply at least one peer; two valid multiples needed per comparison")
    subj = analyze_fundamentals(subject,as_of=as_of)
    seen={subj['symbol']}
    items=[]
    for peer in peers:
        candidate=analyze_fundamentals(peer,as_of=as_of)
        if candidate['symbol'] in seen:
            raise FundamentalsError("Peer tickers must be unique and cannot contain subject")
        seen.add(candidate['symbol'])
        items.append(candidate)
    comparisons={}
    for metric in ('price_to_earnings','price_to_fcf','ev_to_ebitda'):
        valid=[(p['symbol'],float(p['multiples'][metric])) for p in items
               if p['multiples'][metric] is not None and p['multiples'][metric]>0]
        subject_val=subj['multiples'][metric]
        if len(valid)<2 or subject_val is None or subject_val<=0:
            comparisons[metric]={'status':'unavailable','reason':
                'Need a positive subject multiple and >=2 positive peer multiples',
                'usable_peer_count':len(valid)}
            continue
        from statistics import median
        typical=float(median(v for _,v in valid))
        comparisons[metric]={'status':'evaluated','subject_value':float(subject_val),
                             'peer_median':typical,'subject_vs_peer_median_fraction':float(subject_val)/typical-1,
                             'usable_peers':dict(valid),
                             'interpretation':'Higher multiple = greater valuation relative to chosen peers; not proof of overvaluation.'}
    return {'subject':subj['symbol'],'peer_count':len(items),
            'fundamental_periods':{p['symbol']:p['fiscal_period_end'] for p in [subj,*items]},
            'multiples':comparisons,
            'limitations':['Peer group is user-selected; sector/business and accounting comparability not automatically established.',
                           'Peer fiscal years and market price timestamps may differ.',
                           'Negative/zero earnings or FCF cannot produce economically meaningful positive multiples.',
                           'Differences in growth, leverage, accounting, share classes and currency can explain valuation gaps.']}
