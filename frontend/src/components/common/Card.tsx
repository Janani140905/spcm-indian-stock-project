import React from "react";

interface CardProps {
  children: React.ReactNode;
  className?: string;
  glow?: boolean;
  title?: string;
  subtitle?: string;
  action?: React.ReactNode;
}

export const Card: React.FC<CardProps> = ({
  children,
  className = "",
  glow = false,
  title,
  subtitle,
  action,
}) => {
  return (
    <div
      className={`rounded-xl border ${
        glow
          ? "bg-gradient-to-br from-[#141f36] via-[#10172a] to-[#121c32] border-[#2a3c5e] shadow-xl shadow-black/40"
          : "bg-[#11192d] border-[#1e2d4a]"
      } p-5 ${className}`}
    >
      {(title || action) && (
        <div className="flex items-start justify-between mb-4 border-b border-[#1b2844] pb-3">
          <div>
            {title && (
              <h3 className="text-base font-semibold text-slate-100 tracking-tight flex items-center gap-2">
                {title}
              </h3>
            )}
            {subtitle && <p className="text-xs text-slate-400 mt-0.5">{subtitle}</p>}
          </div>
          {action && <div>{action}</div>}
        </div>
      )}
      {children}
    </div>
  );
};
