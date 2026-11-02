/**
 * Centralized Indian Stock Universe Configuration (NSE).
 * Covers 12 distinct sectors and 50 representative, high-liquidity companies.
 *
 * NOTE ON ML MODEL COMPATIBILITY:
 * The original SPCM model bundle was trained exclusively on the canonical 10-stock
 * benchmark universe (`inTrainingUniverse: true`). For stocks where `inTrainingUniverse: true`,
 * SPCM model prediction is explicitly marked as "UNAVAILABLE / TRAINING_REQUIRED"
 * to avoid out-of-distribution hallucinations, while personalized recommendation
 * is computed via multi-factor quantitative and profile-suitability scoring.
 */

export interface StockDefinition {
  symbol: string;
  ticker: string; // e.g. "TCS.NS"
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
    description: "Public and private sector banks, lending institutions, and asset managers",
    iconName: "Landmark",
    stockSymbols: ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK", "BANKBARODA"],
  },
  {
    name: "Information Technology",
    displayName: "Information Technology",
    shortCode: "IT",
    description: "Global IT services, consulting, enterprise software, and cloud providers",
    iconName: "Code",
    stockSymbols: ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
  },
  {
    name: "Pharmaceuticals and Healthcare",
    displayName: "Pharma & Healthcare",
    shortCode: "PHARMA",
    description: "Formulations, active pharmaceutical ingredients (APIs), diagnostics, and hospitals",
    iconName: "HeartPulse",
    stockSymbols: ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP"],
  },
  {
    name: "Automobile and Electric Vehicles",
    displayName: "Automobile & EV",
    shortCode: "AUTO",
    description: "Passenger cars, commercial vehicles, 2-wheelers, and EV mobility manufacturers",
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
    description: "Everyday consumer packaged goods, foods, beverages, and personal care",
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
    description: "Wireless telecom operators, optical fiber, and network infrastructure",
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
    description: "Apparel fashion retail, hypermarket supermarket chains, and jewelry lifestyle",
    iconName: "Store",
    stockSymbols: ["TRENT", "DMART", "TITAN"],
  },
  {
    name: "Power and Renewable Energy",
    displayName: "Power & Renewable Energy",
    shortCode: "POWER",
    description: "Thermal, hydro, transmission, and green solar/wind renewable power",
    iconName: "Zap",
    stockSymbols: ["NTPC", "TATAPOWER", "ADANIGREEN"],
  },
  {
    name: "Chemicals",
    displayName: "Chemicals & Materials",
    shortCode: "CHEMICALS",
    description: "Specialty chemicals, adhesives, fluoro-chemicals, and industrial nitrates",
    iconName: "FlaskConical",
    stockSymbols: ["PIDILITIND", "SRF", "DEEPAKNTR"],
  },
];

export const STOCK_UNIVERSE: StockDefinition[] = [
  // 1. Banking and Financial Services
  {
    symbol: "HDFCBANK",
    ticker: "HDFCBANK.NS",
    name: "HDFC Bank Ltd",
    sector: "Banking and Financial Services",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "India's largest private sector bank by assets and market capitalization.",
  },
  {
    symbol: "ICICIBANK",
    ticker: "ICICIBANK.NS",
    name: "ICICI Bank Ltd",
    sector: "Banking and Financial Services",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Leading private commercial banking and diversified retail lending powerhouse.",
  },
  {
    symbol: "KOTAKBANK",
    ticker: "KOTAKBANK.NS",
    name: "Kotak Mahindra Bank Ltd",
    sector: "Banking and Financial Services",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Prominent private commercial bank with extensive wealth and asset management.",
  },
  {
    symbol: "SBIN",
    ticker: "SBIN.NS",
    name: "State Bank of India",
    sector: "Banking and Financial Services",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "India's largest public sector commercial bank with vast national retail reach.",
  },
  {
    symbol: "AXISBANK",
    ticker: "AXISBANK.NS",
    name: "Axis Bank Ltd",
    sector: "Banking and Financial Services",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Major private sector bank offering corporate and retail banking services.",
  },
  {
    symbol: "BANKBARODA",
    ticker: "BANKBARODA.NS",
    name: "Bank of Baroda",
    sector: "Banking and Financial Services",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Leading public sector commercial bank with significant international footprint.",
  },

  // 2. Information Technology
  {
    symbol: "TCS",
    ticker: "TCS.NS",
    name: "Tata Consultancy Services Ltd",
    sector: "Information Technology",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Global IT services, consulting and business solutions leader of Tata Group.",
  },
  {
    symbol: "INFY",
    ticker: "INFY.NS",
    name: "Infosys Ltd",
    sector: "Information Technology",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Multinational IT consulting and next-generation enterprise digital services firm.",
  },
  {
    symbol: "WIPRO",
    ticker: "WIPRO.NS",
    name: "Wipro Ltd",
    sector: "Information Technology",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Global enterprise IT, cloud consulting and cognitive business solutions provider.",
  },
  {
    symbol: "HCLTECH",
    ticker: "HCLTECH.NS",
    name: "HCL Technologies Ltd",
    sector: "Information Technology",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Global technology services and engineering R&D solutions pioneer.",
  },
  {
    symbol: "TECHM",
    ticker: "TECHM.NS",
    name: "Tech Mahindra Ltd",
    sector: "Information Technology",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Specialized enterprise IT provider with deep telecom and digital transformation expertise.",
  },

  // 3. Pharmaceuticals and Healthcare
  {
    symbol: "SUNPHARMA",
    ticker: "SUNPHARMA.NS",
    name: "Sun Pharmaceutical Industries Ltd",
    sector: "Pharmaceuticals and Healthcare",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "India's largest pharmaceutical firm specializing in generic and branded formulations.",
  },
  {
    symbol: "DRREDDY",
    ticker: "DRREDDY.NS",
    name: "Dr. Reddy's Laboratories Ltd",
    sector: "Pharmaceuticals and Healthcare",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Integrated global pharmaceutical producer with deep generic pipeline in US and Europe.",
  },
  {
    symbol: "CIPLA",
    ticker: "CIPLA.NS",
    name: "Cipla Ltd",
    sector: "Pharmaceuticals and Healthcare",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Global healthcare company with world-renowned respiratory and anti-retroviral therapies.",
  },
  {
    symbol: "DIVISLAB",
    ticker: "DIVISLAB.NS",
    name: "Divi's Laboratories Ltd",
    sector: "Pharmaceuticals and Healthcare",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Leading global active pharmaceutical ingredient (API) and custom synthesis manufacturer.",
  },
  {
    symbol: "APOLLOHOSP",
    ticker: "APOLLOHOSP.NS",
    name: "Apollo Hospitals Enterprise Ltd",
    sector: "Pharmaceuticals and Healthcare",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Asia's premier integrated healthcare provider operating multi-specialty hospitals and pharmacies.",
  },

  // 4. Automobile and Electric Vehicles
  {
    symbol: "MARUTI",
    ticker: "MARUTI.NS",
    name: "Maruti Suzuki India Ltd",
    sector: "Automobile and Electric Vehicles",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "India's market share leader in passenger vehicles and hybrid automobile technology.",
  },
  {
    symbol: "TATAMOTORS",
    ticker: "TATAMOTORS.NS",
    name: "Tata Motors Ltd",
    sector: "Automobile and Electric Vehicles",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Pioneer in Indian electric vehicles (EV), commercial fleet trucks and Jaguar Land Rover.",
  },
  {
    symbol: "M&M",
    ticker: "M&M.NS",
    name: "Mahindra & Mahindra Ltd",
    sector: "Automobile and Electric Vehicles",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Dominant SUV and farm tractor manufacturer expanding aggressively into electric mobility.",
  },
  {
    symbol: "BAJAJ-AUTO",
    ticker: "BAJAJ-AUTO.NS",
    name: "Bajaj Auto Ltd",
    sector: "Automobile and Electric Vehicles",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "World's leading two-wheeler and three-wheeler vehicle exporter with Chetak EV line.",
  },
  {
    symbol: "EICHERMOT",
    ticker: "EICHERMOT.NS",
    name: "Eicher Motors Ltd",
    sector: "Automobile and Electric Vehicles",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Iconic manufacturer of Royal Enfield leisure motorcycles and Volvo Eicher commercial trucks.",
  },

  // 5. Energy and Oil & Gas
  {
    symbol: "RELIANCE",
    ticker: "RELIANCE.NS",
    name: "Reliance Industries Ltd",
    sector: "Energy and Oil & Gas",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "India's highest valued conglomerate with refining, petrochemicals, retail, and Jio telecom.",
  },
  {
    symbol: "ONGC",
    ticker: "ONGC.NS",
    name: "Oil and Natural Gas Corporation Ltd",
    sector: "Energy and Oil & Gas",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "National crude oil and natural gas exploration and domestic extraction enterprise.",
  },
  {
    symbol: "BPCL",
    ticker: "BPCL.NS",
    name: "Bharat Petroleum Corporation Ltd",
    sector: "Energy and Oil & Gas",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Major state-run downstream oil refining, fuel distribution, and retail pump network.",
  },
  {
    symbol: "IOC",
    ticker: "IOC.NS",
    name: "Indian Oil Corporation Ltd",
    sector: "Energy and Oil & Gas",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "India's largest commercial oil refiner, cross-country pipeline operator, and fuel marketer.",
  },
  {
    symbol: "POWERGRID",
    ticker: "POWERGRID.NS",
    name: "Power Grid Corporation of India Ltd",
    sector: "Energy and Oil & Gas",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Central transmission utility carrying ~85% of India's interstate high-voltage electricity.",
  },

  // 6. FMCG and Consumer Goods
  {
    symbol: "HINDUNILVR",
    ticker: "HINDUNILVR.NS",
    name: "Hindustan Unilever Ltd",
    sector: "FMCG and Consumer Goods",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "India's largest fast-moving consumer goods enterprise in home, beauty and personal care.",
  },
  {
    symbol: "ITC",
    ticker: "ITC.NS",
    name: "ITC Ltd",
    sector: "FMCG and Consumer Goods",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Diversified FMCG, paperboards, agricultural exports, packaging and luxury hospitality leader.",
  },
  {
    symbol: "NESTLEIND",
    ticker: "NESTLEIND.NS",
    name: "Nestle India Ltd",
    sector: "FMCG and Consumer Goods",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Premium food and nutrition brand renowned for Maggi, Nescafe, KitKat, and dairy products.",
  },
  {
    symbol: "BRITANNIA",
    ticker: "BRITANNIA.NS",
    name: "Britannia Industries Ltd",
    sector: "FMCG and Consumer Goods",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Premier bakery, biscuit, cake, bread and dairy food brand with widespread retail distribution.",
  },
  {
    symbol: "TATACONSUM",
    ticker: "TATACONSUM.NS",
    name: "Tata Consumer Products Ltd",
    sector: "FMCG and Consumer Goods",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Integrated food and beverage company holding Tata Tea, Tata Salt, and Starbucks JV in India.",
  },

  // 7. Metals and Mining
  {
    symbol: "TATASTEEL",
    ticker: "TATASTEEL.NS",
    name: "Tata Steel Ltd",
    sector: "Metals and Mining",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Top-tier integrated steel producer with extensive mining reserves and European operations.",
  },
  {
    symbol: "JSWSTEEL",
    ticker: "JSWSTEEL.NS",
    name: "JSW Steel Ltd",
    sector: "Metals and Mining",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Flagship company of JSW Group, one of India's largest private primary steel makers.",
  },
  {
    symbol: "HINDALCO",
    ticker: "HINDALCO.NS",
    name: "Hindalco Industries Ltd",
    sector: "Metals and Mining",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Aditya Birla Group metals flagship, world's largest aluminum rolling company via Novelis.",
  },
  {
    symbol: "VEDL",
    ticker: "VEDL.NS",
    name: "Vedanta Ltd",
    sector: "Metals and Mining",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Diversified natural resources giant producing zinc, lead, silver, oil, and aluminum.",
  },

  // 8. Telecommunications
  {
    symbol: "BHARTIARTL",
    ticker: "BHARTIARTL.NS",
    name: "Bharti Airtel Ltd",
    sector: "Telecommunications",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Leading telecom service provider with mobile, broadband, enterprise and Africa operations.",
  },
  {
    symbol: "IDEA",
    ticker: "IDEA.NS",
    name: "Vodafone Idea Ltd",
    sector: "Telecommunications",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Mid Cap",
    description: "Major pan-India mobile operator providing 2G, 3G, 4G, and enterprise connectivity.",
  },

  // 9. Infrastructure and Construction
  {
    symbol: "LT",
    ticker: "LT.NS",
    name: "Larsen & Toubro Ltd",
    sector: "Infrastructure and Construction",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "India's premier engineering, procurement, construction, manufacturing and heavy tech conglomerate.",
  },
  {
    symbol: "ADANIPORTS",
    ticker: "ADANIPORTS.NS",
    name: "Adani Ports and Special Economic Zone Ltd",
    sector: "Infrastructure and Construction",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Largest commercial port operator and integrated transport logistics chain in India.",
  },
  {
    symbol: "ULTRACEMCO",
    ticker: "ULTRACEMCO.NS",
    name: "UltraTech Cement Ltd",
    sector: "Infrastructure and Construction",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Largest manufacturer of grey cement, ready mix concrete and white cement in India.",
  },
  {
    symbol: "AMBUJACEM",
    ticker: "AMBUJACEM.NS",
    name: "Ambuja Cements Ltd",
    sector: "Infrastructure and Construction",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Prominent domestic cement producer renowned for sustainable construction materials.",
  },

  // 10. Retail and Consumer Services
  {
    symbol: "TRENT",
    ticker: "TRENT.NS",
    name: "Trent Ltd",
    sector: "Retail and Consumer Services",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "High-growth fashion retail arm of Tata Group operating Westside, Zudio, and Star Bazaar.",
  },
  {
    symbol: "DMART",
    ticker: "DMART.NS",
    name: "Avenue Supermarts Ltd (DMart)",
    sector: "Retail and Consumer Services",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Supermarket hypermarket chain offering value retail groceries and home essentials.",
  },
  {
    symbol: "TITAN",
    ticker: "TITAN.NS",
    name: "Titan Company Ltd",
    sector: "Retail and Consumer Services",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Tata Group luxury and lifestyle leader across Tanishq jewellery, Fastrack, and watches.",
  },

  // 11. Power and Renewable Energy
  {
    symbol: "NTPC",
    ticker: "NTPC.NS",
    name: "NTPC Ltd",
    sector: "Power and Renewable Energy",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "India's largest power generation utility transitioning into solar, wind, and green hydrogen.",
  },
  {
    symbol: "TATAPOWER",
    ticker: "TATAPOWER.NS",
    name: "Tata Power Company Ltd",
    sector: "Power and Renewable Energy",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Integrated power utility leading EV charging infrastructure and rooftop solar in India.",
  },
  {
    symbol: "ADANIGREEN",
    ticker: "ADANIGREEN.NS",
    name: "Adani Green Energy Ltd",
    sector: "Power and Renewable Energy",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "One of the largest global renewable energy platforms operating massive solar-wind parks.",
  },

  // 12. Chemicals
  {
    symbol: "PIDILITIND",
    ticker: "PIDILITIND.NS",
    name: "Pidilite Industries Ltd",
    sector: "Chemicals",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Consumer specialty chemicals and adhesives market leader holding Fevicol and M-Seal.",
  },
  {
    symbol: "SRF",
    ticker: "SRF.NS",
    name: "SRF Ltd",
    sector: "Chemicals",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Large Cap",
    description: "Multi-business chemicals conglomerate manufacturing fluoro-specialties and industrial fabrics.",
  },
  {
    symbol: "DEEPAKNTR",
    ticker: "DEEPAKNTR.NS",
    name: "Deepak Nitrite Ltd",
    sector: "Chemicals",
    exchange: "NSE",
    inTrainingUniverse: true,
    marketCapCategory: "Mid Cap",
    description: "Premier Indian manufacturer of basic, fine, and performance specialty chemical intermediates.",
  },
];

// Helper maps for quick lookup
export const STOCK_BY_SYMBOL: Record<string, StockDefinition> = STOCK_UNIVERSE.reduce(
  (acc, s) => {
    acc[s.symbol.toUpperCase()] = s;
    return acc;
  },
  {} as Record<string, StockDefinition>
);

export const CANONICAL_BENCHMARK_SYMBOLS = [
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

