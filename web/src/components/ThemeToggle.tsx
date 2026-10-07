export type Theme = "light" | "dark";

interface Props {
  theme: Theme;
  onChange: (theme: Theme) => void;
}

// A sun in dark mode, a moon in light mode: the icon is the mode a click switches to.
export function ThemeToggle({ theme, onChange }: Props) {
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button
      className="theme-toggle icon-button"
      title={`Switch to ${next} mode`}
      aria-label={`Switch to ${next} mode`}
      onClick={() => onChange(next)}
    >
      {theme === "dark" ? (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
          <circle cx="12" cy="12" r="4.2" />
          <path d="M12 2v2.6M12 19.4V22M2 12h2.6M19.4 12H22M4.9 4.9l1.9 1.9M17.2 17.2l1.9 1.9M19.1 4.9l-1.9 1.9M6.8 17.2l-1.9 1.9" strokeLinecap="round" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
          <path d="M20 14.2A8.2 8.2 0 1 1 9.8 4a6.6 6.6 0 0 0 10.2 10.2Z" strokeLinejoin="round" />
        </svg>
      )}
    </button>
  );
}
