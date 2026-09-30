// Chart palette.
//
// Mirrors the CSS custom properties in styles/foundation.css. ECharts renders to canvas and
// cannot resolve var(), so the values are duplicated here and must be kept in step.
//
// Deliberately free of React and ECharts imports: the derived analytics reference this palette,
// and those run under `bun test` with no DOM.

export const CHART = {
  text: '#e9eef0',
  muted: '#859197',
  dim: '#5e686d',
  line: '#252c31',
  lineSoft: '#1a2024',
  panel: '#0d1012',
  accent: '#61dbac',
  blue: '#619eff',
  violet: '#b17aff',
  warning: '#e9bd5b',
  danger: '#f35b64',
  /** System One, wherever the dashboard tells the two systems apart. */
  systemOne: '#f08bd0',
} as const
