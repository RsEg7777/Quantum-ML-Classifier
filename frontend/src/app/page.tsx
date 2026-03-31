import Link from "next/link";

export default function Home() {
  return (
    <div className="page-shell flex min-h-screen items-center justify-center px-6 py-16">
      <main className="card-glass mx-auto max-w-4xl p-10 md:p-14">
        <div className="mb-8 inline-flex rounded-full border border-white/40 bg-white/40 px-4 py-2 text-xs font-semibold tracking-[0.18em] text-slate-900">
          QUANTUM ML CONTROL CENTER
        </div>

        <h1 className="max-w-3xl text-balance text-4xl font-semibold leading-tight text-slate-950 md:text-6xl">
          Production frontend for quantum-classical experimentation.
        </h1>

        <p className="mt-6 max-w-2xl text-lg leading-relaxed text-slate-700 md:text-xl">
          Submit training and experiment jobs, monitor queue progression, and
          inspect metrics in one operational dashboard built for Vercel.
        </p>

        <div className="mt-10 flex flex-wrap gap-4">
          <Link className="btn-primary" href="/dashboard">
            Open Dashboard
          </Link>
          <Link className="btn-ghost" href="/api/health">
            Check API Health
          </Link>
        </div>

        <div className="mt-12 grid gap-4 text-sm text-slate-700 md:grid-cols-3">
          <div className="info-chip">Typed API contracts</div>
          <div className="info-chip">Queue-ready job model</div>
          <div className="info-chip">External worker integration path</div>
        </div>
      </main>
    </div>
  );
}
