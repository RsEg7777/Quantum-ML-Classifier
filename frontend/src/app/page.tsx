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
        "Track running, completed, and failed jobs in real time with adaptive polling and rich status messaging.",
    },
    {
      title: "Secure Session Auth",
      description:
        "Use credentials or Google sign-in with signed cookies and protected API routes for operational safety.",
    },
    {
      title: "Worker-Ready Integration",
      description:
        "Connect external workers and storage backends for production-class reliability and scale.",
    },
  ];

  const workflow = [
    {
      step: "01",
      title: "Authenticate",
      description:
        "Sign in with credentials or Google to access dashboard controls and protected APIs.",
    },
    {
      step: "02",
      title: "Submit a Job",
      description:
        "Choose job type, select a dataset, and define config JSON for classical or hybrid runs.",
    },
    {
      step: "03",
      title: "Monitor and Iterate",
      description:
        "Watch queue progression, inspect outputs, and tune parameters for the next experiment cycle.",
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
          <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-blue-300/60 bg-white/70 text-sm font-bold text-blue-700 shadow-[0_8px_24px_rgba(30,64,175,0.22)]">
            QML
          </span>
          <div>
            <p className="text-xs uppercase tracking-[0.16em] text-slate-500">Quantum ML Platform</p>
            <p className="text-sm font-semibold text-slate-900">Control Center</p>
          </div>
        </div>

        <nav className="flex items-center gap-2 rounded-full border border-white/55 bg-white/45 px-2 py-2 backdrop-blur">
          {quickLinks.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="rounded-full px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-white/70 hover:text-slate-900"
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
              <p className="kicker-badge">PRODUCTION HOMEPAGE</p>
              <h1 className="mt-5 max-w-3xl text-balance text-4xl font-semibold leading-tight text-slate-950 md:text-6xl">
                Quantum-classical ML operations, organized for real teams.
              </h1>

              <p className="mt-6 max-w-2xl text-lg leading-relaxed text-slate-700 md:text-xl">
                This portal centralizes job submission, queue monitoring, and result
                inspection so experimentation is fast, visible, and repeatable.
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
                Built for deployment workflows with typed routes, secure auth, and
                worker-compatible orchestration.
              </p>

              <div className="mt-5 space-y-2">
                <div className="info-chip text-sm text-slate-700">Signed session cookies + Google OAuth</div>
                <div className="info-chip text-sm text-slate-700">CSV upload pipeline with backend validation</div>
                <div className="info-chip text-sm text-slate-700">Queue mode and direct worker dispatch support</div>
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

        <section className="card-glass fade-up delay-2 p-6 md:p-8">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-2xl font-semibold text-slate-900">How It Works</h2>
            <Link className="btn-ghost" href="/api/health">
              Check System Health
            </Link>
          </div>

          <div className="mt-5 grid gap-3 md:grid-cols-3">
            {workflow.map((item) => (
              <article key={item.step} className="surface-muted p-5">
                <p className="text-xs uppercase tracking-[0.14em] text-blue-700">Step {item.step}</p>
                <h3 className="mt-2 text-lg font-semibold text-slate-900">{item.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-700">{item.description}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="card-glass fade-up delay-3 p-6 text-center md:p-8">
          <p className="text-xs uppercase tracking-[0.15em] text-slate-500">Ready To Run</p>
          <h2 className="mt-2 text-3xl font-semibold text-slate-900">
            Start your next quantum-classical experiment cycle.
          </h2>
          <p className="mx-auto mt-3 max-w-2xl text-sm leading-relaxed text-slate-700">
            Access the dashboard, submit a workload, and monitor every stage from
            queue intake to result payload.
          </p>
          <div className="mt-6 flex flex-wrap justify-center gap-3">
            <Link className="btn-primary" href="/dashboard">
              Launch Dashboard
            </Link>
            <Link className="btn-ghost" href="/login">
              Go to Login
            </Link>
          </div>
        </section>
      </main>

      <footer className="relative z-10 mx-auto w-full max-w-6xl px-6 pb-8 text-xs text-slate-600 md:px-10">
        Quantum ML Control Center • Frontend for hybrid experimentation workflows
      </footer>
    </div>
  );
}
