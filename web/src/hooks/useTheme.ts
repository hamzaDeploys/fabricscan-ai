import { useCallback, useState } from "react";

/** Light/dark theme. The initial class is set by an inline script in index.html to avoid a flash. */
export function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  const toggle = useCallback(() => {
    setDark((prev) => {
      const next = !prev;
      document.documentElement.classList.toggle("dark", next);
      try {
        localStorage.setItem("fabricscan-theme", next ? "dark" : "light");
      } catch {
        /* storage unavailable (private mode): the choice simply is not remembered */
      }
      return next;
    });
  }, []);
  return { dark, toggle };
}
