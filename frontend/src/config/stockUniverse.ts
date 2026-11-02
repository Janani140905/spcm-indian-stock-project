/**
 * Centralized Indian Stock Universe Configuration (NSE) for Frontend.
 * 12 sectors and 50 representative high-liquidity stocks.
 */

export interface StockDefinition {
  symbol: string;
  ticker: string;
  name: string;
  sector: SectorName;
  exchange: "NSE";
  inTrainingUniverse: boolean;
  marketCapCategory: "Large Cap" | "Mid Cap";
  description: string;
}

export type SectorName =
  | "Banking and Financial Services"
  | "Information Technology"
  | "Pharmaceuticals and Healthcare"
  | "Automobile and Electric Vehicles"
  | "Energy and Oil & Gas"
  | "FMCG and Consumer Goods"
  | "Metals and Mining"
  | "Telecommunications"
  | "Infrastructure and Construction"
  | "Retail and Consumer Services"
  | "Power and Renewable Energy"
  | "Chemicals";

export interface SectorDefinition {
  name: SectorName;
  displayName: string;
  shortCode: string;
  description: string;
  iconName: string;
  stockSymbols: string[];
}

export const SECTORS: SectorDefinition[] = [
  {
    name: "Banking and Financial Services",
    displayName: "Banking & Financial Services",
    shortCode: "BANKING",
    description: "Public and private sector commercial banks and lenders",
    iconName: "Landmark",
    stockSymbols: ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK", "BANKBARODA"],
  },
  {
    name: "Information Technology",
    displayName: "Information Technology",
    shortCode: "IT",
    description: "Enterprise software, cloud, and digital transformation providers",
    iconName: "Code",
    stockSymbols: ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
  },
  {
    name: "Pharmaceuticals and Healthcare",
    displayName: "Pharma & Healthcare",
    shortCode: "PHARMA",
    description: "Formulations, active pharmaceutical ingredients, and hospitals",
    iconName: "HeartPulse",
    stockSymbols: ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP"],
  },
  {
    name: "Automobile and Electric Vehicles",
    displayName: "Automobile & EV",
    shortCode: "AUTO",
    description: "Passenger cars, commercial trucks, two-wheelers, and electric vehicles",
    iconName: "Car",
    stockSymbols: ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT"],
  },
  {
    name: "Energy and Oil & Gas",
    displayName: "Energy & Oil / Gas",
    shortCode: "ENERGY",
    description: "Oil exploration, refining, fuel marketing, and energy conglomerates",
    iconName: "Flame",
    stockSymbols: ["RELIANCE", "ONGC", "BPCL", "IOC", "POWERGRID"],
  },
  {
    name: "FMCG and Consumer Goods",
    displayName: "FMCG & Consumer Goods",
    shortCode: "FMCG",
    description: "Packaged foods, household goods, beverages, and personal care",
    iconName: "ShoppingBag",
    stockSymbols: ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM"],
  },
  {
    name: "Metals and Mining",
    displayName: "Metals & Mining",
    shortCode: "METALS",
    description: "Primary steel, aluminum, copper production, and resource extraction",
    iconName: "Layers",
    stockSymbols: ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL"],
  },
  {
    name: "Telecommunications",
    displayName: "Telecommunications",
    shortCode: "TELECOM",
    description: "Wireless network providers, broadband, and telecom towers",
    iconName: "Radio",
    stockSymbols: ["BHARTIARTL", "IDEA"],
  },
  {
    name: "Infrastructure and Construction",
    displayName: "Infrastructure & Construction",
    shortCode: "INFRA",
    description: "Engineering procurement construction (EPC), ports, and cement producers",
    iconName: "Building2",
    stockSymbols: ["LT", "ADANIPORTS", "ULTRACEMCO", "AMBUJACEM"],
  },
  {
    name: "Retail and Consumer Services",
    displayName: "Retail & Consumer Services",
    shortCode: "RETAIL",
    description: "Fashion apparel retail, supermarket hypermarkets, and jewelry lifestyle",
    iconName: "Store",
    stockSymbols: ["TRENT", "DMART", "TITAN"],
  },
  {
    name: "Power and Renewable Energy",
    displayName: "Power & Renewable Energy",
    shortCode: "POWER",
    description: "Thermal, hydro generation, and clean green solar/wind platforms",
    iconName: "Zap",
    stockSymbols: ["NTPC", "TATAPOWER", "ADANIGREEN"],
  },
  {
    name: "Chemicals",
    displayName: "Chemicals & Materials",
    shortCode: "CHEMICALS",
    description: "Specialty chemicals, adhesives, and industrial intermediate compounds",
    iconName: "FlaskConical",
    stockSymbols: ["PIDILITIND", "SRF", "DEEPAKNTR"],
  },
];

export const STOCK_UNIVERSE: StockDefinition[] = [
  // Banking
  { symbol: "HDFCBANK", ticker: "HDFCBANK.NS", name: "HDFC Bank Ltd", sector: "Banking and Financial Services", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "India's largest private commercial bank by assets." },
  { symbol: "ICICIBANK", ticker: "ICICIBANK.NS", name: "ICICI Bank Ltd", sector: "Banking and Financial Services", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Leading private commercial banking powerhouse." },
  { symbol: "KOTAKBANK", ticker: "KOTAKBANK.NS", name: "Kotak Mahindra Bank Ltd", sector: "Banking and Financial Services", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Private bank with high-end wealth and retail management." },
  { symbol: "SBIN", ticker: "SBIN.NS", name: "State Bank of India", sector: "Banking and Financial Services", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Largest public sector bank in India." },
  { symbol: "AXISBANK", ticker: "AXISBANK.NS", name: "Axis Bank Ltd", sector: "Banking and Financial Services", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Prominent private corporate and retail bank." },
  { symbol: "BANKBARODA", ticker: "BANKBARODA.NS", name: "Bank of Baroda", sector: "Banking and Financial Services", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Leading public commercial bank with global branches." },

  // IT
  { symbol: "TCS", ticker: "TCS.NS", name: "Tata Consultancy Services Ltd", sector: "Information Technology", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Global IT consulting flagship of Tata Group." },
  { symbol: "INFY", ticker: "INFY.NS", name: "Infosys Ltd", sector: "Information Technology", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Global enterprise digital services and IT innovator." },
  { symbol: "WIPRO", ticker: "WIPRO.NS", name: "Wipro Ltd", sector: "Information Technology", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Enterprise IT, cloud, and digital consulting leader." },
  { symbol: "HCLTECH", ticker: "HCLTECH.NS", name: "HCL Technologies Ltd", sector: "Information Technology", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Global technology services and engineering R&D firm." },
  { symbol: "TECHM", ticker: "TECHM.NS", name: "Tech Mahindra Ltd", sector: "Information Technology", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Digital transformation leader with deep telecom domain." },

  // Pharma
  { symbol: "SUNPHARMA", ticker: "SUNPHARMA.NS", name: "Sun Pharmaceutical Industries Ltd", sector: "Pharmaceuticals and Healthcare", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "India's largest pharmaceutical formulations manufacturer." },
  { symbol: "DRREDDY", ticker: "DRREDDY.NS", name: "Dr. Reddy's Laboratories Ltd", sector: "Pharmaceuticals and Healthcare", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Global generic and biosimilar pharmaceutical leader." },
  { symbol: "CIPLA", ticker: "CIPLA.NS", name: "Cipla Ltd", sector: "Pharmaceuticals and Healthcare", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Global pharmaceutical leader in respiratory medicine." },
  { symbol: "DIVISLAB", ticker: "DIVISLAB.NS", name: "Divi's Laboratories Ltd", sector: "Pharmaceuticals and Healthcare", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "World-leading active pharmaceutical ingredient (API) producer." },
  { symbol: "APOLLOHOSP", ticker: "APOLLOHOSP.NS", name: "Apollo Hospitals Enterprise Ltd", sector: "Pharmaceuticals and Healthcare", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Premier healthcare provider operating hospitals and digital pharmacy." },

  // Auto
  { symbol: "MARUTI", ticker: "MARUTI.NS", name: "Maruti Suzuki India Ltd", sector: "Automobile and Electric Vehicles", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Dominant market leader in passenger cars in India." },
  { symbol: "TATAMOTORS", ticker: "TATAMOTORS.NS", name: "Tata Motors Ltd", sector: "Automobile and Electric Vehicles", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Pioneer in passenger EVs and commercial trucks." },
  { symbol: "M&M", ticker: "M&M.NS", name: "Mahindra & Mahindra Ltd", sector: "Automobile and Electric Vehicles", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Market leader in SUVs, farm tractors, and electric mobility." },
  { symbol: "BAJAJ-AUTO", ticker: "BAJAJ-AUTO.NS", name: "Bajaj Auto Ltd", sector: "Automobile and Electric Vehicles", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Leading two-wheeler and three-wheeler vehicle exporter." },
  { symbol: "EICHERMOT", ticker: "EICHERMOT.NS", name: "Eicher Motors Ltd", sector: "Automobile and Electric Vehicles", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Manufacturer of Royal Enfield premium leisure motorcycles." },

  // Energy
  { symbol: "RELIANCE", ticker: "RELIANCE.NS", name: "Reliance Industries Ltd", sector: "Energy and Oil & Gas", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Conglomerate across refining, petrochemicals, retail, and Jio telecom." },
  { symbol: "ONGC", ticker: "ONGC.NS", name: "Oil and Natural Gas Corporation Ltd", sector: "Energy and Oil & Gas", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "National crude oil and natural gas exploration company." },
  { symbol: "BPCL", ticker: "BPCL.NS", name: "Bharat Petroleum Corporation Ltd", sector: "Energy and Oil & Gas", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Major downstream oil refiner and fuel distribution retailer." },
  { symbol: "IOC", ticker: "IOC.NS", name: "Indian Oil Corporation Ltd", sector: "Energy and Oil & Gas", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "India's largest national commercial oil refiner and marketer." },
  { symbol: "POWERGRID", ticker: "POWERGRID.NS", name: "Power Grid Corporation of India Ltd", sector: "Energy and Oil & Gas", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Interstate electricity transmission backbone utility." },

  // FMCG
  { symbol: "HINDUNILVR", ticker: "HINDUNILVR.NS", name: "Hindustan Unilever Ltd", sector: "FMCG and Consumer Goods", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "India's largest consumer packaged goods enterprise." },
  { symbol: "ITC", ticker: "ITC.NS", name: "ITC Ltd", sector: "FMCG and Consumer Goods", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Diversified FMCG, agri-business, paperboards, and hotels." },
  { symbol: "NESTLEIND", ticker: "NESTLEIND.NS", name: "Nestle India Ltd", sector: "FMCG and Consumer Goods", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Premium packaged food, nutrition, and beverage brand." },
  { symbol: "BRITANNIA", ticker: "BRITANNIA.NS", name: "Britannia Industries Ltd", sector: "FMCG and Consumer Goods", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Premier bakery, biscuit, and dairy food producer." },
  { symbol: "TATACONSUM", ticker: "TATACONSUM.NS", name: "Tata Consumer Products Ltd", sector: "FMCG and Consumer Goods", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Food and beverage leader with Tata Tea, Salt, and Starbucks JV." },

  // Metals
  { symbol: "TATASTEEL", ticker: "TATASTEEL.NS", name: "Tata Steel Ltd", sector: "Metals and Mining", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Global integrated steel producer with mining operations." },
  { symbol: "JSWSTEEL", ticker: "JSWSTEEL.NS", name: "JSW Steel Ltd", sector: "Metals and Mining", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Flagship integrated steel maker of JSW Group." },
  { symbol: "HINDALCO", ticker: "HINDALCO.NS", name: "Hindalco Industries Ltd", sector: "Metals and Mining", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "World's largest aluminum rolling company via Novelis." },
  { symbol: "VEDL", ticker: "VEDL.NS", name: "Vedanta Ltd", sector: "Metals and Mining", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Diversified natural resources giant across zinc, lead, and aluminum." },

  // Telecom
  { symbol: "BHARTIARTL", ticker: "BHARTIARTL.NS", name: "Bharti Airtel Ltd", sector: "Telecommunications", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Leading telecommunications and digital connectivity operator." },
  { symbol: "IDEA", ticker: "IDEA.NS", name: "Vodafone Idea Ltd", sector: "Telecommunications", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Mid Cap", description: "Major pan-India mobile operator providing cellular services." },

  // Infrastructure
  { symbol: "LT", ticker: "LT.NS", name: "Larsen & Toubro Ltd", sector: "Infrastructure and Construction", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "India's premier engineering, procurement, and construction conglomerate." },
  { symbol: "ADANIPORTS", ticker: "ADANIPORTS.NS", name: "Adani Ports and Special Economic Zone Ltd", sector: "Infrastructure and Construction", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Largest commercial port operator and logistics chain." },
  { symbol: "ULTRACEMCO", ticker: "ULTRACEMCO.NS", name: "UltraTech Cement Ltd", sector: "Infrastructure and Construction", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Largest cement manufacturer in India and top 3 globally." },
  { symbol: "AMBUJACEM", ticker: "AMBUJACEM.NS", name: "Ambuja Cements Ltd", sector: "Infrastructure and Construction", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Leading sustainable cement producer in India." },

  // Retail
  { symbol: "TRENT", ticker: "TRENT.NS", name: "Trent Ltd", sector: "Retail and Consumer Services", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "High-growth fashion retail holding Westside and Zudio." },
  { symbol: "DMART", ticker: "DMART.NS", name: "Avenue Supermarts Ltd (DMart)", sector: "Retail and Consumer Services", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Supermarket hypermarket chain offering value retail essentials." },
  { symbol: "TITAN", ticker: "TITAN.NS", name: "Titan Company Ltd", sector: "Retail and Consumer Services", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Luxury lifestyle leader holding Tanishq jewelry and watches." },

  // Power
  { symbol: "NTPC", ticker: "NTPC.NS", name: "NTPC Ltd", sector: "Power and Renewable Energy", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Largest power generation utility expanding in green solar and wind." },
  { symbol: "TATAPOWER", ticker: "TATAPOWER.NS", name: "Tata Power Company Ltd", sector: "Power and Renewable Energy", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Integrated utility leading EV charging infrastructure and rooftop solar." },
  { symbol: "ADANIGREEN", ticker: "ADANIGREEN.NS", name: "Adani Green Energy Ltd", sector: "Power and Renewable Energy", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "One of the largest global renewable platforms operating mega solar parks." },

  // Chemicals
  { symbol: "PIDILITIND", ticker: "PIDILITIND.NS", name: "Pidilite Industries Ltd", sector: "Chemicals", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Consumer specialty chemicals and adhesives leader holding Fevicol." },
  { symbol: "SRF", ticker: "SRF.NS", name: "SRF Ltd", sector: "Chemicals", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Large Cap", description: "Multi-business chemical manufacturer of fluoro-specialties." },
  { symbol: "DEEPAKNTR", ticker: "DEEPAKNTR.NS", name: "Deepak Nitrite Ltd", sector: "Chemicals", exchange: "NSE", inTrainingUniverse: true, marketCapCategory: "Mid Cap", description: "Premier manufacturer of fine, basic, and performance chemical intermediates." },
];

export const INDIAN_BENCHMARK_STOCKS = [
  "RELIANCE",
  "TCS",
  "HDFCBANK",
  "INFY",
  "ICICIBANK",
  "HINDUNILVR",
  "ITC",
  "SBIN",
  "BHARTIARTL",
  "KOTAKBANK",
] as const;

export const ALL_STOCK_SYMBOLS = STOCK_UNIVERSE.map((s) => s.symbol);

export const STOCK_BY_SYMBOL: Record<string, StockDefinition> = STOCK_UNIVERSE.reduce(
  (acc, s) => {
    acc[s.symbol.toUpperCase()] = s;
    return acc;
  },
  {} as Record<string, StockDefinition>
);
