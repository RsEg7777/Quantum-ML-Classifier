import Link from "next/link";

export default function Home() {
  const stats = [
    { label: "Avg Queue Latency", value: "1.2s" },
    { label: "API Uptime", value: "99.95%" },
    { label: "Active Connectors", value: "6" },
    { label: "Typed Endpoints", value: "24" },
  ];

  const features = [
    {
      title: "Hybrid Job Orchestration",
      description:
        "Create training, inference, and experiment runs with dataset-aware configuration and queue dispatch.",
    },
    {
      title: "Live Queue Visibility",
      description:
        "Track running, completed, and failed jobs in real time with adaptive polling and clear status messaging.",
    },
    {
      title: "Secure Access Control",
      description:
        "Use credentials or Google login with signed cookies and protected API routes.",
    },
    {
      title: "Worker-Ready Integrations",
      description:
        "Connect external workers and storage backends for reliable production workflows.",
    },
  ];

  const quickLinks = [
    { href: "/login", label: "Login" },
    { href: "/dashboard", label: "Dashboard" },
    { href: "/api/health", label: "API Health" },
  ];

  return (
    <div className="page-shell min-h-screen pb-10">
      <span className="ambient-orb orb-a" aria-hidden="true" />
      <span className="ambient-orb orb-b" aria-hidden="true" />

      <header className="relative z-10 mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-4 px-6 py-6 md:px-10">
        <div className="flex items-center gap-3">
          <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-blue-200/70 bg-white/80 text-sm font-bold text-blue-700 shadow-[0_8px_22px_rgba(30,64,175,0.15)]">
            QML
          </span>
          <div>
            <p className="text-xs uppercase tracking-[0.16em] text-slate-500">Quantum ML Platform</p>
            <p className="text-sm font-semibold text-slate-900">Control Center</p>
          </div>
        </div>

        <nav className="flex items-center gap-2 rounded-full border border-slate-200 bg-white/75 px-2 py-2 backdrop-blur">
          {quickLinks.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="rounded-full px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-100 hover:text-slate-900"
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </header>

      <main className="relative z-10 mx-auto flex w-full max-w-6xl flex-col gap-6 px-6 pb-8 md:px-10">
        <section className="card-glass fade-up p-8 md:p-10">
          <div className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr]">
            <div>
              <p className="kicker-badge">PRODUCTION FRONTEND</p>
              <h1 className="mt-5 max-w-3xl text-balance text-4xl font-semibold leading-tight text-slate-950 md:text-6xl">
                Quantum-classical ML operations, structured for real teams.
              </h1>

              <p className="mt-6 max-w-2xl text-lg leading-relaxed text-slate-700 md:text-xl">
                Centralize job submission, queue monitoring, and result inspection in one
                reliable interface built for day-to-day experimentation.
              </p>

              <div className="mt-8 flex flex-wrap gap-3">
                <Link className="btn-primary" href="/dashboard">
                  Open Dashboard
                </Link>
                <Link className="btn-ghost" href="/login">
                  Sign In
                </Link>
              </div>
            </div>

            <aside className="surface-muted p-5 md:p-6">
              <p className="text-xs uppercase tracking-[0.14em] text-slate-500">Platform Snapshot</p>
              <h2 className="mt-3 text-xl font-semibold text-slate-900">Operational Readiness</h2>
              <p className="mt-2 text-sm leading-relaxed text-slate-700">
                Built for deployment workflows with typed routes, secure authentication,
                and worker-compatible orchestration.
              </p>

              <div className="mt-5 space-y-2">
                <div className="info-chip text-sm text-slate-700">Signed session cookies + Google OAuth</div>
                <div className="info-chip text-sm text-slate-700">CSV upload pipeline with backend validation</div>
                <div className="info-chip text-sm text-slate-700">Queue and direct worker dispatch support</div>
              </div>
            </aside>
          </div>

          <div className="mt-8 glow-separator" />

          <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {stats.map((stat) => (
              <article key={stat.label} className="metric-card">
                <p className="text-xs uppercase tracking-[0.14em] text-slate-500">{stat.label}</p>
                <p className="mt-2 text-2xl font-semibold text-slate-900">{stat.value}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="fade-up delay-1 grid gap-4 md:grid-cols-2">
          {features.map((feature) => (
            <article key={feature.title} className="card-glass p-6 md:p-7">
              <h3 className="text-xl font-semibold text-slate-900">{feature.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-700">{feature.description}</p>
            </article>
          ))}
        </section>
      </main>

      <footer className="relative z-10 mx-auto w-full max-w-6xl px-6 pb-8 text-xs text-slate-600 md:px-10">
        Quantum ML Control Center • Professional frontend for hybrid experimentation workflows
      </footer>
    </div>
  );
}
