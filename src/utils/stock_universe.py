"""
Centralized Stock Universe for Indian SPCM Platform.
Contains 50 NSE equities across 12 distinct industries.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

@dataclass
class StockInfo:
    symbol: str
    ticker: str
    name: str
    sector: str
    exchange: str = "NSE"
    in_training_universe: bool = False
    market_cap_category: str = "Large Cap"
    description: str = ""

SECTOR_CATALOG: List[Dict[str, str]] = [
    {"name": "Banking and Financial Services", "shortCode": "BANKING", "icon": "Landmark"},
    {"name": "Information Technology", "shortCode": "IT", "icon": "Code"},
    {"name": "Pharmaceuticals and Healthcare", "shortCode": "PHARMA", "icon": "HeartPulse"},
    {"name": "Automobile and Electric Vehicles", "shortCode": "AUTO", "icon": "Car"},
    {"name": "Energy and Oil & Gas", "shortCode": "ENERGY", "icon": "Flame"},
    {"name": "FMCG and Consumer Goods", "shortCode": "FMCG", "icon": "ShoppingBag"},
    {"name": "Metals and Mining", "shortCode": "METALS", "icon": "Layers"},
    {"name": "Telecommunications", "shortCode": "TELECOM", "icon": "Radio"},
    {"name": "Infrastructure and Construction", "shortCode": "INFRA", "icon": "Building2"},
    {"name": "Retail and Consumer Services", "shortCode": "RETAIL", "icon": "Store"},
    {"name": "Power and Renewable Energy", "shortCode": "POWER", "icon": "Zap"},
    {"name": "Chemicals", "shortCode": "CHEMICALS", "icon": "FlaskConical"},
]

FULL_STOCK_CATALOG: List[Dict[str, Any]] = [
    # 1. Banking and Financial Services
    {"symbol": "HDFCBANK", "ticker": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "sector": "Banking and Financial Services", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "ICICIBANK", "ticker": "ICICIBANK.NS", "name": "ICICI Bank Ltd", "sector": "Banking and Financial Services", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "KOTAKBANK", "ticker": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank Ltd", "sector": "Banking and Financial Services", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "SBIN", "ticker": "SBIN.NS", "name": "State Bank of India", "sector": "Banking and Financial Services", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "AXISBANK", "ticker": "AXISBANK.NS", "name": "Axis Bank Ltd", "sector": "Banking and Financial Services", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "BANKBARODA", "ticker": "BANKBARODA.NS", "name": "Bank of Baroda", "sector": "Banking and Financial Services", "exchange": "NSE", "in_training_universe": True},

    # 2. Information Technology
    {"symbol": "TCS", "ticker": "TCS.NS", "name": "Tata Consultancy Services Ltd", "sector": "Information Technology", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "INFY", "ticker": "INFY.NS", "name": "Infosys Ltd", "sector": "Information Technology", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "WIPRO", "ticker": "WIPRO.NS", "name": "Wipro Ltd", "sector": "Information Technology", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "HCLTECH", "ticker": "HCLTECH.NS", "name": "HCL Technologies Ltd", "sector": "Information Technology", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "TECHM", "ticker": "TECHM.NS", "name": "Tech Mahindra Ltd", "sector": "Information Technology", "exchange": "NSE", "in_training_universe": True},

    # 3. Pharmaceuticals and Healthcare
    {"symbol": "SUNPHARMA", "ticker": "SUNPHARMA.NS", "name": "Sun Pharmaceutical Industries Ltd", "sector": "Pharmaceuticals and Healthcare", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "DRREDDY", "ticker": "DRREDDY.NS", "name": "Dr. Reddy's Laboratories Ltd", "sector": "Pharmaceuticals and Healthcare", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "CIPLA", "ticker": "CIPLA.NS", "name": "Cipla Ltd", "sector": "Pharmaceuticals and Healthcare", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "DIVISLAB", "ticker": "DIVISLAB.NS", "name": "Divi's Laboratories Ltd", "sector": "Pharmaceuticals and Healthcare", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "APOLLOHOSP", "ticker": "APOLLOHOSP.NS", "name": "Apollo Hospitals Enterprise Ltd", "sector": "Pharmaceuticals and Healthcare", "exchange": "NSE", "in_training_universe": True},

    # 4. Automobile and Electric Vehicles
    {"symbol": "MARUTI", "ticker": "MARUTI.NS", "name": "Maruti Suzuki India Ltd", "sector": "Automobile and Electric Vehicles", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "TATAMOTORS", "ticker": "TATAMOTORS.NS", "name": "Tata Motors Ltd", "sector": "Automobile and Electric Vehicles", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "M&M", "ticker": "M&M.NS", "name": "Mahindra & Mahindra Ltd", "sector": "Automobile and Electric Vehicles", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "BAJAJ-AUTO", "ticker": "BAJAJ-AUTO.NS", "name": "Bajaj Auto Ltd", "sector": "Automobile and Electric Vehicles", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "EICHERMOT", "ticker": "EICHERMOT.NS", "name": "Eicher Motors Ltd", "sector": "Automobile and Electric Vehicles", "exchange": "NSE", "in_training_universe": True},

    # 5. Energy and Oil & Gas
    {"symbol": "RELIANCE", "ticker": "RELIANCE.NS", "name": "Reliance Industries Ltd", "sector": "Energy and Oil & Gas", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "ONGC", "ticker": "ONGC.NS", "name": "Oil and Natural Gas Corporation Ltd", "sector": "Energy and Oil & Gas", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "BPCL", "ticker": "BPCL.NS", "name": "Bharat Petroleum Corporation Ltd", "sector": "Energy and Oil & Gas", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "IOC", "ticker": "IOC.NS", "name": "Indian Oil Corporation Ltd", "sector": "Energy and Oil & Gas", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "POWERGRID", "ticker": "POWERGRID.NS", "name": "Power Grid Corporation of India Ltd", "sector": "Energy and Oil & Gas", "exchange": "NSE", "in_training_universe": True},

    # 6. FMCG and Consumer Goods
    {"symbol": "HINDUNILVR", "ticker": "HINDUNILVR.NS", "name": "Hindustan Unilever Ltd", "sector": "FMCG and Consumer Goods", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "ITC", "ticker": "ITC.NS", "name": "ITC Ltd", "sector": "FMCG and Consumer Goods", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "NESTLEIND", "ticker": "NESTLEIND.NS", "name": "Nestle India Ltd", "sector": "FMCG and Consumer Goods", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "BRITANNIA", "ticker": "BRITANNIA.NS", "name": "Britannia Industries Ltd", "sector": "FMCG and Consumer Goods", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "TATACONSUM", "ticker": "TATACONSUM.NS", "name": "Tata Consumer Products Ltd", "sector": "FMCG and Consumer Goods", "exchange": "NSE", "in_training_universe": True},

    # 7. Metals and Mining
    {"symbol": "TATASTEEL", "ticker": "TATASTEEL.NS", "name": "Tata Steel Ltd", "sector": "Metals and Mining", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "JSWSTEEL", "ticker": "JSWSTEEL.NS", "name": "JSW Steel Ltd", "sector": "Metals and Mining", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "HINDALCO", "ticker": "HINDALCO.NS", "name": "Hindalco Industries Ltd", "sector": "Metals and Mining", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "VEDL", "ticker": "VEDL.NS", "name": "Vedanta Ltd", "sector": "Metals and Mining", "exchange": "NSE", "in_training_universe": True},

    # 8. Telecommunications
    {"symbol": "BHARTIARTL", "ticker": "BHARTIARTL.NS", "name": "Bharti Airtel Ltd", "sector": "Telecommunications", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "IDEA", "ticker": "IDEA.NS", "name": "Vodafone Idea Ltd", "sector": "Telecommunications", "exchange": "NSE", "in_training_universe": True},

    # 9. Infrastructure and Construction
    {"symbol": "LT", "ticker": "LT.NS", "name": "Larsen & Toubro Ltd", "sector": "Infrastructure and Construction", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "ADANIPORTS", "ticker": "ADANIPORTS.NS", "name": "Adani Ports and Special Economic Zone Ltd", "sector": "Infrastructure and Construction", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "ULTRACEMCO", "ticker": "ULTRACEMCO.NS", "name": "UltraTech Cement Ltd", "sector": "Infrastructure and Construction", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "AMBUJACEM", "ticker": "AMBUJACEM.NS", "name": "Ambuja Cements Ltd", "sector": "Infrastructure and Construction", "exchange": "NSE", "in_training_universe": True},

    # 10. Retail and Consumer Services
    {"symbol": "TRENT", "ticker": "TRENT.NS", "name": "Trent Ltd", "sector": "Retail and Consumer Services", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "DMART", "ticker": "DMART.NS", "name": "Avenue Supermarts Ltd (DMart)", "sector": "Retail and Consumer Services", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "TITAN", "ticker": "TITAN.NS", "name": "Titan Company Ltd", "sector": "Retail and Consumer Services", "exchange": "NSE", "in_training_universe": True},

    # 11. Power and Renewable Energy
    {"symbol": "NTPC", "ticker": "NTPC.NS", "name": "NTPC Ltd", "sector": "Power and Renewable Energy", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "TATAPOWER", "ticker": "TATAPOWER.NS", "name": "Tata Power Company Ltd", "sector": "Power and Renewable Energy", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "ADANIGREEN", "ticker": "ADANIGREEN.NS", "name": "Adani Green Energy Ltd", "sector": "Power and Renewable Energy", "exchange": "NSE", "in_training_universe": True},

    # 12. Chemicals
    {"symbol": "PIDILITIND", "ticker": "PIDILITIND.NS", "name": "Pidilite Industries Ltd", "sector": "Chemicals", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "SRF", "ticker": "SRF.NS", "name": "SRF Ltd", "sector": "Chemicals", "exchange": "NSE", "in_training_universe": True},
    {"symbol": "DEEPAKNTR", "ticker": "DEEPAKNTR.NS", "name": "Deepak Nitrite Ltd", "sector": "Chemicals", "exchange": "NSE", "in_training_universe": True},
]

# Canonical 10 benchmark stocks from original Phase 1-5 pipeline
ORIGINAL_BENCHMARK_10 = [
    "HDFCBANK",
    "ICICIBANK",
    "KOTAKBANK",
    "SBIN",
    "TCS",
    "INFY",
    "RELIANCE",
    "HINDUNILVR",
    "ITC",
    "BHARTIARTL",
]

# Full 50-stock universe
ALL_STOCKS_50 = [s["symbol"] for s in FULL_STOCK_CATALOG]

