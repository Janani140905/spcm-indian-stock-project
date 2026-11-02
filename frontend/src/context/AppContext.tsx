import React, { createContext, useContext, useState, useEffect } from "react";
import { SystemHealth, InvestorProfile } from "../types/index.js";
import { fetchHealth, fetchProfile, saveProfile } from "../services/api.js";
import { loadStoredProfile, saveStoredProfile, DEFAULT_INVESTOR_PROFILE } from "../services/personalization.js";

export interface ToastMessage {
  id: string;
  title: string;
  message: string;
  type: "success" | "error" | "info" | "warning";
}

interface AppContextType {
  demoMode: boolean;
  setDemoMode: (val: boolean) => void;
  activeStock: string | null;
  setActiveStock: (symbol: string | null) => void;
  searchOpen: boolean;
  setSearchOpen: (val: boolean) => void;
  health: SystemHealth | null;
  refreshHealth: () => Promise<void>;
  toasts: ToastMessage[];
  addToast: (title: string, message: string, type?: "success" | "error" | "info" | "warning") => void;
  removeToast: (id: string) => void;
  investorProfile: InvestorProfile | null;
  setInvestorProfile: (profile: InvestorProfile | null) => void;
  profileModalOpen: boolean;
  setProfileModalOpen: (val: boolean) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [demoMode, setDemoMode] = useState<boolean>(false);
  const [activeStock, setActiveStock] = useState<string | null>("RELIANCE");
  const [searchOpen, setSearchOpen] = useState<boolean>(false);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const [investorProfile, setInvestorProfileState] = useState<InvestorProfile | null>(() => {
    return loadStoredProfile() || DEFAULT_INVESTOR_PROFILE;
  });
  const [profileModalOpen, setProfileModalOpen] = useState<boolean>(false);

  const setInvestorProfile = (profile: InvestorProfile | null) => {
    setInvestorProfileState(profile);
    if (profile) {
      saveStoredProfile(profile);
      saveProfile(profile).catch((err) => {
        console.warn("Could not persist investor profile to backend:", err);
      });
    }
  };

  const addToast = (
    title: string,
    message: string,
    type: "success" | "error" | "info" | "warning" = "info"
  ) => {
    const id = "toast_" + Math.random().toString(36).substring(2, 9);
    setToasts((prev) => [...prev, { id, title, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4500);
  };

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  const refreshHealth = async () => {
    try {
      const data = await fetchHealth();
      setHealth(data);
    } catch (err: any) {
      console.warn("Could not reach Express backend:", err);
    }
  };

  useEffect(() => {
    refreshHealth();
    fetchProfile()
      .then((p) => {
        if (p && p.riskTolerance) {
          setInvestorProfileState(p);
          saveStoredProfile(p);
        }
      })
      .catch((err) => {
        // Backend store may be empty or offline; keep local stored profile
      });
    const interval = setInterval(refreshHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <AppContext.Provider
      value={{
        demoMode,
        setDemoMode,
        activeStock,
        setActiveStock,
        searchOpen,
        setSearchOpen,
        health,
        refreshHealth,
        toasts,
        addToast,
        removeToast,
        investorProfile,
        setInvestorProfile,
        profileModalOpen,
        setProfileModalOpen,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("useApp must be used within an AppProvider");
  }
  return context;
};
