import React from "react";

interface BadgeProps {
  children: React.ReactNode;
  variant?: "bullish" | "bearish" | "neutral" | "demo" | "recommended" | "info" | "purple";
  size?: "sm" | "md";
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = "info",
  size = "md",
  className = "",
}) => {
  const sizeClasses = size === "sm" ? "px-2 py-0.5 text-[10px]" : "px-2.5 py-1 text-xs";

  const variantClasses = {
    bullish: "bg-emerald-950/80 text-emerald-400 border border-emerald-500/40",
    bearish: "bg-rose-950/80 text-rose-400 border border-rose-500/40",
    neutral: "bg-amber-950/70 text-amber-300 border border-amber-500/40",
    demo: "bg-amber-900/60 text-amber-200 border border-amber-600/70 font-mono tracking-wider font-bold",
    recommended: "bg-teal-950/90 text-teal-300 border border-teal-400/50 font-semibold shadow-sm shadow-teal-500/20",
    info: "bg-slate-800/80 text-slate-300 border border-slate-700",
    purple: "bg-violet-950/80 text-violet-300 border border-violet-500/40",
  }[variant];

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-medium ${sizeClasses} ${variantClasses} ${className}`}
    >
      {children}
    </span>
  );
};
