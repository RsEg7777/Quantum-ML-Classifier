import Link from "next/link";

export default function Home() {
  const navItems = [
    { label: "Home", href: "/", active: true },
    { label: "Pricing", href: "/dashboard" },
    { label: "Video", href: "/dashboard" },
    { label: "Audio", href: "/dashboard" },
    { label: "URL Check", href: "/dashboard" },
    { label: "Image", href: "/dashboard" },
  ];

  const particles = [
    { top: "14%", left: "11%", size: 7, delay: "0s", duration: "8s" },
    { top: "18%", left: "33%", size: 5, delay: "0.4s", duration: "9s" },
    { top: "20%", left: "50%", size: 6, delay: "1s", duration: "10s" },
    { top: "22%", left: "66%", size: 5, delay: "0.8s", duration: "8.2s" },
    { top: "27%", left: "83%", size: 7, delay: "1.3s", duration: "9.5s" },
    { top: "39%", left: "8%", size: 6, delay: "1.5s", duration: "10.5s" },
    { top: "44%", left: "27%", size: 5, delay: "2s", duration: "9.8s" },
    { top: "46%", left: "45%", size: 8, delay: "1.8s", duration: "11s" },
    { top: "41%", left: "64%", size: 6, delay: "0.9s", duration: "8.8s" },
    { top: "47%", left: "89%", size: 7, delay: "1.6s", duration: "9.2s" },
    { top: "63%", left: "16%", size: 6, delay: "2.5s", duration: "10.6s" },
    { top: "67%", left: "34%", size: 5, delay: "1.4s", duration: "9.4s" },
    { top: "72%", left: "49%", size: 7, delay: "0.3s", duration: "8.5s" },
    { top: "68%", left: "68%", size: 6, delay: "2.1s", duration: "10.3s" },
    { top: "76%", left: "84%", size: 8, delay: "1.1s", duration: "9.7s" },
    { top: "86%", left: "26%", size: 7, delay: "2.6s", duration: "10.2s" },
    { top: "88%", left: "62%", size: 6, delay: "0.7s", duration: "8.7s" },
    { top: "90%", left: "93%", size: 5, delay: "1.9s", duration: "9.1s" },
  ];

  return (
    <div className="home-shell min-h-screen">
      <div className="home-particle-layer" aria-hidden="true">
        {particles.map((particle, index) => (
          <span
            key={`particle-${index}`}
            className="home-particle"
            style={{
              top: particle.top,
              left: particle.left,
              width: `${particle.size}px`,
              height: `${particle.size}px`,
              animationDelay: particle.delay,
              animationDuration: particle.duration,
            }}
          />
        ))}
      </div>

      <header className="home-topbar-wrap">
        <div className="home-topbar">
          <div className="home-brand">
            QuantumLens <span>AI</span>
          </div>

          <nav className="home-nav-links">
            {navItems.map((item) => (
              <Link
                key={item.label}
                href={item.href}
                className={`home-nav-link ${item.active ? "active" : ""}`}
              >
                {item.label}
              </Link>
            ))}
          </nav>

          <div className="home-topbar-actions">
            <Link className="home-icon-button" href="/login" aria-label="Open account">
              <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
                <path d="M12 12a4.5 4.5 0 1 0-4.5-4.5A4.5 4.5 0 0 0 12 12zm0 2.25c-3.75 0-6.75 1.9-6.75 4.25a.75.75 0 0 0 .75.75h12a.75.75 0 0 0 .75-.75c0-2.35-3-4.25-6.75-4.25z" />
              </svg>
            </Link>
            <span className="home-icon-button" aria-hidden="true">
              ✶
            </span>
          </div>
        </div>
      </header>

      <main className="home-hero">
        <div className="home-lens-glow" aria-hidden="true" />

        <h1 className="home-title">
          Truth Through
          <br />
          Technology
        </h1>

        <p className="home-subtitle">
          Advanced AI-powered verification for text, images, videos, and audio.
          Detect misinformation fast with production-ready confidence.
        </p>

        <div className="home-actions">
          <Link className="home-cta-primary" href="/dashboard">
            Start Free Trial
          </Link>
          <Link className="home-cta-secondary" href="/login">
            Explore Features
          </Link>
        </div>
      </main>
    </div>
  );
}
