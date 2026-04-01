import { useEffect, useState } from 'react'
import './App.css'

function App() {
  const [scrollY, setScrollY] = useState(0)

  useEffect(() => {
    const onScroll = () => {
      setScrollY(window.scrollY)
    }

    window.addEventListener('scroll', onScroll, { passive: true })
    onScroll()

    return () => {
      window.removeEventListener('scroll', onScroll)
    }
  }, [])

  useEffect(() => {
    const existing = document.querySelector('script[src="https://tenor.com/embed.js"]')
    if (existing) {
      return
    }

    const script = document.createElement('script')
    script.src = 'https://tenor.com/embed.js'
    script.async = true
    document.body.appendChild(script)
  }, [])

  const parallaxY = Math.min(scrollY * 0.16, 140)
  const heroOpacity = Math.max(1 - scrollY / 900, 0.58)

  return (
    <main className="landing-shell">
      <header className="top-nav">
        <p className="brand">Alyssa Lustina</p>
        <nav aria-label="Primary navigation">
          <a href="#work">Work</a>
          <span aria-hidden="true">|</span>
          <a href="#services">Services</a>
          <span aria-hidden="true">|</span>
          <a href="#contact">Contact</a>
        </nav>
      </header>

      <section className="hero-copy" aria-label="Landing introduction">
        <div className="hero-text">
          <div
            className="hero-parallax-group"
            style={{ transform: `translateY(${parallaxY}px)`, opacity: heroOpacity }}
          >
            <p className="hero-headline">
              I design
              <span className="shift-word-stack" aria-hidden="true">
                <span className="word word-1">interior spaces.</span>
                <span className="word word-2">elevated homes.</span>
                <span className="word word-3">intentional spaces.</span>
              </span>
            </p>
            <p className="hero-description">
              Just a pretty girl doing pretty girl things.
            </p>
          </div>
        </div>

        
      </section>

      <section id="work" className="content-section">
        <h2>Selected Work</h2>
        <p>Residential and hospitality concepts crafted with calm luxury.</p>
      </section>

      <section id="services" className="content-section">
        <h2>Services</h2>
        <p>Space planning, material curation, styling, and 3D visualization.</p>
      </section>

      <section id="contact" className="content-section">
        <h2>Contact</h2>
        <p>Let us shape spaces that are elegant, intentional, and personal.</p>
      </section>
    </main>
  )
}

export default App
