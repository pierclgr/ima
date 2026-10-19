// page colors stay separate from the canvas sent to the model
{
  const root = document.documentElement;
  const preference = window.matchMedia('(prefers-color-scheme: dark)');
  let savedTheme = null;
  try {
    savedTheme = localStorage.getItem('ima-snake-theme');
  } catch {
    // browser settings can block storage
  }
  if (!['light', 'dark'].includes(savedTheme)) savedTheme = null;

  function applyTheme(theme) {
    root.dataset.theme = theme;
    document.querySelectorAll('[data-theme-choice]').forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.themeChoice === theme));
    });
  }

  applyTheme(savedTheme ?? (preference.matches ? 'dark' : 'light'));
  preference.addEventListener('change', (event) => {
    if (savedTheme === null) applyTheme(event.matches ? 'dark' : 'light');
  });

  document.addEventListener('DOMContentLoaded', () => {
    applyTheme(root.dataset.theme);
    document.querySelectorAll('[data-theme-choice]').forEach((button) => {
      button.addEventListener('click', () => {
        savedTheme = button.dataset.themeChoice;
        applyTheme(savedTheme);
        try {
          localStorage.setItem('ima-snake-theme', savedTheme);
        } catch {
          // keep the choice for this page when storage is blocked
        }
      });
    });
  }, { once: true });
}
