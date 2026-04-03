import Link from "next/link";
import type { CSSProperties } from "react";

import styles from "./theme-preview.module.css";

type ThemeTemplate = {
  id: string;
  name: string;
  tagline: string;
  description: string;
  fonts: string;
  swatches: string[];
  vars: {
    bg: string;
    bgAlt: string;
    surface: string;
    surfaceStrong: string;
    border: string;
    text: string;
    muted: string;
    accent: string;
    accentSoft: string;
    success: string;
  };
};

const templates: ThemeTemplate[] = [
  {
    id: "obsidian-command",
    name: "Obsidian Command",
    tagline: "Enterprise control center with measured glow",
    description:
      "Deep navy surfaces, high-legibility text, and bright signal accents for dashboards.",
    fonts: "Sora + JetBrains Mono",
    swatches: ["#0b1118", "#122132", "#3ba5ff", "#22c55e"],
    vars: {
      bg: "#0b1118",
      bgAlt: "#101b29",
      surface: "#122132",
      surfaceStrong: "#172a3f",
      border: "rgba(98, 131, 171, 0.38)",
      text: "#e6edf6",
      muted: "#94a8bf",
      accent: "#3ba5ff",
      accentSoft: "rgba(59, 165, 255, 0.22)",
      success: "#22c55e",
    },
  },
  {
    id: "graphite-executive",
    name: "Graphite Executive",
    tagline: "Corporate minimal with confident depth",
    description:
      "A polished graphite stack built for enterprise product teams and executive reporting.",
    fonts: "Manrope + IBM Plex Mono",
    swatches: ["#111317", "#1a1f27", "#5ea0ff", "#7dd3fc"],
    vars: {
      bg: "#111317",
      bgAlt: "#191d24",
      surface: "#1a1f27",
      surfaceStrong: "#232a34",
      border: "rgba(143, 161, 186, 0.34)",
      text: "#edf2f8",
      muted: "#9caec6",
      accent: "#5ea0ff",
      accentSoft: "rgba(94, 160, 255, 0.2)",
      success: "#22c55e",
    },
  },
  {
    id: "carbon-data-lab",
    name: "Carbon Data Lab",
    tagline: "Data-first dark mode with energetic signal colors",
    description:
      "Optimized for metric-heavy products where contrast and graph readability matter.",
    fonts: "Space Grotesk + Fira Code",
    swatches: ["#0e141b", "#16212d", "#00c2ff", "#a3e635"],
    vars: {
      bg: "#0e141b",
      bgAlt: "#13202d",
      surface: "#16212d",
      surfaceStrong: "#203144",
      border: "rgba(87, 116, 147, 0.4)",
      text: "#e3edf8",
      muted: "#94a8be",
      accent: "#00c2ff",
      accentSoft: "rgba(0, 194, 255, 0.2)",
      success: "#a3e635",
    },
  },
  {
    id: "midnight-teal-pro",
    name: "Midnight Teal Pro",
    tagline: "Refined teal atmosphere for modern product UIs",
    description:
      "Dark cyan undertones with calm depth for premium SaaS and platform experiences.",
    fonts: "Plus Jakarta Sans + JetBrains Mono",
    swatches: ["#0b1516", "#122226", "#14b8a6", "#67e8f9"],
    vars: {
      bg: "#0b1516",
      bgAlt: "#122125",
      surface: "#122226",
      surfaceStrong: "#1a3035",
      border: "rgba(101, 152, 154, 0.34)",
      text: "#e6f5f6",
      muted: "#9dbabb",
      accent: "#14b8a6",
      accentSoft: "rgba(20, 184, 166, 0.22)",
      success: "#67e8f9",
    },
  },
  {
    id: "noir-slate-minimal",
    name: "Noir Slate Minimal",
    tagline: "Quiet luxury with minimal light effects",
    description:
      "For teams wanting understated sophistication and excellent typography focus.",
    fonts: "Urbanist + IBM Plex Sans",
    swatches: ["#0f1013", "#171a20", "#94a3b8", "#e2e8f0"],
    vars: {
      bg: "#0f1013",
      bgAlt: "#171a20",
      surface: "#171a20",
      surfaceStrong: "#222833",
      border: "rgba(148, 163, 184, 0.3)",
      text: "#ebf0f7",
      muted: "#a4b1c3",
      accent: "#94a3b8",
      accentSoft: "rgba(148, 163, 184, 0.18)",
      success: "#e2e8f0",
    },
  },
  {
    id: "steel-blue-systems",
    name: "Steel Blue Systems",
    tagline: "Technical operations style with blueprint energy",
    description:
      "High-control interface language for platform ops, automation, and workflow products.",
    fonts: "Exo 2 + Source Code Pro",
    swatches: ["#0a1220", "#132036", "#60a5fa", "#f59e0b"],
    vars: {
      bg: "#0a1220",
      bgAlt: "#111f35",
      surface: "#132036",
      surfaceStrong: "#1b2c47",
      border: "rgba(104, 140, 191, 0.36)",
      text: "#e8eef8",
      muted: "#9aaecb",
      accent: "#60a5fa",
      accentSoft: "rgba(96, 165, 250, 0.22)",
      success: "#f59e0b",
    },
  },
];

function toStyle(template: ThemeTemplate): CSSProperties {
  return {
    ["--tp-bg" as string]: template.vars.bg,
    ["--tp-bg-alt" as string]: template.vars.bgAlt,
    ["--tp-surface" as string]: template.vars.surface,
    ["--tp-surface-strong" as string]: template.vars.surfaceStrong,
    ["--tp-border" as string]: template.vars.border,
    ["--tp-text" as string]: template.vars.text,
    ["--tp-muted" as string]: template.vars.muted,
    ["--tp-accent" as string]: template.vars.accent,
    ["--tp-accent-soft" as string]: template.vars.accentSoft,
    ["--tp-success" as string]: template.vars.success,
  };
}

export default function ThemePreviewPage() {
  return (
    <main className={styles.shell}>
      <header className={styles.hero}>
        <p className={styles.kicker}>Theme Gallery</p>
        <h1 className={styles.title}>Dark Professional Templates</h1>
        <p className={styles.subtitle}>
          Pick one direction and I will apply it site-wide across home, login, and dashboard.
        </p>
        <div className={styles.actions}>
          <Link href="/" className={styles.actionGhost}>
            Back Home
          </Link>
          <Link href="/dashboard" className={styles.actionPrimary}>
            Open Dashboard
          </Link>
        </div>
      </header>

      <section className={styles.grid}>
        {templates.map((template) => (
          <article key={template.id} className={styles.themeCard} style={toStyle(template)}>
            <div className={styles.ambientGlow} aria-hidden="true" />

            <div className={styles.mockShell}>
              <div className={styles.mockTopRow}>
                <span className={styles.badge}>{template.name}</span>
                <span className={styles.pulse} aria-hidden="true" />
              </div>

              <p className={styles.tagline}>{template.tagline}</p>

              <div className={styles.metricGrid}>
                <div className={styles.metricCell}>
                  <span>Queue</span>
                  <strong>1.2s</strong>
                </div>
                <div className={styles.metricCell}>
                  <span>Workers</span>
                  <strong>6</strong>
                </div>
                <div className={styles.metricCell}>
                  <span>Uptime</span>
                  <strong>99.9%</strong>
                </div>
              </div>

              <div className={styles.graph} aria-hidden="true">
                <span style={{ height: "43%" }} />
                <span style={{ height: "72%" }} />
                <span style={{ height: "58%" }} />
                <span style={{ height: "81%" }} />
                <span style={{ height: "65%" }} />
              </div>

              <div className={styles.buttonRow}>
                <button type="button" className={styles.primaryBtn}>
                  Primary
                </button>
                <button type="button" className={styles.ghostBtn}>
                  Secondary
                </button>
              </div>

              <div className={styles.swatchRow}>
                {template.swatches.map((swatch) => (
                  <span key={swatch} style={{ background: swatch }} title={swatch} />
                ))}
              </div>
            </div>

            <div className={styles.meta}>
              <p>{template.description}</p>
              <p className={styles.fontLine}>Suggested fonts: {template.fonts}</p>
            </div>
          </article>
        ))}
      </section>
    </main>
  );
}
