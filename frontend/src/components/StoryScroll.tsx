'use client';
import React, { useEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(ScrollTrigger);

function cx(...parts: Array<string | undefined | false | null>): string {
  return parts.filter(Boolean).join(' ');
}

export interface FlowSectionProps {
  className?: string;
  style?: React.CSSProperties;
  children: React.ReactNode;
  'aria-label'?: string;
}

export const FlowSection: React.FC<FlowSectionProps> = ({ className, style = {}, children, 'aria-label': ariaLabel }) => (
  <section
    data-flow-section
    aria-label={ariaLabel}
    className={cx('relative min-h-screen w-full overflow-hidden', className)}
  >
    <div
      data-flow-inner
      className={cx('flow-art-container relative flex min-h-screen w-full flex-col justify-between gap-6 px-[4vw] pt-[clamp(2rem,8vw,4vw)] pb-[4vw]', 'will-change-transform')}
      style={{ transformOrigin: 'bottom left', ...style }}
    >
      {children}
    </div>
  </section>
);

export interface FlowArtProps {
  children: React.ReactNode;
  className?: string;
  'aria-label'?: string;
}

const FlowArt: React.FC<FlowArtProps> = ({ children, className, 'aria-label': ariaLabel = 'Story scroll' }) => {
  const containerRef = useRef<HTMLElement>(null);
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReducedMotion(mq.matches);
    update(); mq.addEventListener('change', update);
    return () => mq.removeEventListener('change', update);
  }, []);

  useGSAP(() => {
    if (!containerRef.current || reducedMotion) return;
    const sections = Array.from(containerRef.current.querySelectorAll<HTMLElement>('[data-flow-section]'));
    if (!sections.length) return;
    const triggers: ScrollTrigger[] = [];
    sections.forEach((section, i) => {
      gsap.set(section, { zIndex: i + 1 });
      const inner = section.querySelector<HTMLElement>('.flow-art-container');
      if (!inner) return;
      if (i > 0) {
        gsap.set(inner, { rotation: 30, transformOrigin: 'bottom left' });
        const tw = gsap.to(inner, { rotation: 0, ease: 'none', scrollTrigger: { trigger: section, start: 'top bottom', end: 'top 25%', scrub: true } });
        if (tw.scrollTrigger) triggers.push(tw.scrollTrigger);
      }
      if (i < sections.length - 1) {
        triggers.push(ScrollTrigger.create({ trigger: section, start: 'bottom bottom', end: 'bottom top', pin: true, pinSpacing: false }));
      }
    });
    ScrollTrigger.refresh();
    return () => { triggers.forEach(t => t.kill()); };
  }, { scope: containerRef, dependencies: [React.Children.count(children), reducedMotion] });

  return (
    <main ref={containerRef as React.RefObject<HTMLElement>} aria-label={ariaLabel} className={cx('w-full overflow-x-hidden', className)}>
      {children}
    </main>
  );
};

export default FlowArt;
