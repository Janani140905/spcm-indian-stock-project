/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        dark: {
          bg: "#0b1020",
          card: "#11192d",
          cardHover: "#16223b",
          border: "#1f2d48",
          muted: "#94a3b8",
        },
        terminal: {
          accent: "#8b5cf6", // Violet accent
          accentLight: "#a78bfa",
          teal: "#14b8a6",   // Teal highlight
          tealLight: "#2dd4bf",
          bullish: "#10b981", // Green/Emerald
          bearish: "#ef4444", // Red/Rose
          neutral: "#f59e0b", // Amber
        },
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "Courier New", "monospace"],
      },
    },
  },
  plugins: [],
}
