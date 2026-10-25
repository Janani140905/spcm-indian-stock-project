import React, { createContext, useContext, useState, useEffect } from "react";
import { SystemHealth } from "../types/index.js";
import { fetchHealth } from "../services/api.js";

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
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const AppProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [demoMode, setDemoMode] = useState<boolean>(false);
  const [activeStock, setActiveStock] = useState<string | null>("RELIANCE");
  const [searchOpen, setSearchOpen] = useState<boolean>(false);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

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
