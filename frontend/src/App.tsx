import { useRef, useState, useCallback, useEffect } from 'react';
import Beams from './components/Beams';
import FlowArt, { FlowSection } from './components/StoryScroll';

const API = 'https://angshuman12-anvil.hf.space';

async function runAudit(files: File[], budget: number): Promise<string> {
  const form = new FormData();
  files.forEach(f => form.append('files', f));
  form.append('model', 'resnet18');
  form.append('budget', String(budget));
  const res = await fetch(`${API}/audit/upload`, { method: 'POST', body: form });
  const ct = res.headers.get('content-type') || '';
  if (!ct.includes('application/json')) {
    throw new Error('Server is waking up — wait 30 seconds and try again');
  }
  const data = await res.json();
  return data.job_id;
}

async function pollJob(jobId: string, onLog: (msg: string) => void): Promise<Record<string, unknown>> {
  while (true) {
    await new Promise(r => setTimeout(r, 4000));
    const res = await fetch(`${API}/audit/job/${jobId}`);
    const ct = res.headers.get('content-type') || '';
    if (!ct.includes('application/json')) {
      onLog('Server still processing...');
      continue;
    }
    const data = await res.json();
    if (data.status === 'complete') return data;
    if (data.status === 'error') throw new Error(data.error || 'Audit failed');
    onLog(`Status: ${data.status}...`);
  }
}

function Nav() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4"
      style={{ background: 'rgba(5,14,26,0.85)', backdropFilter: 'blur(12px)', borderBottom: '1px solid var(--border)' }}>
      <span className="text-xl font-black tracking-widest" style={{ color: 'var(--teal)', fontFamily: "'JetBrains Mono', monospace" }}>ANVIL</span>
      <div className="flex items-center gap-6 text-sm font-medium" style={{ color: 'var(--text-muted)' }}>
        <a href="#pipeline" className="hover:text-white transition-colors">Pipeline</a>
        <a href="#demo" className="hover:text-white transition-colors">Demo</a>
        <a href="#stack" className="hover:text-white transition-colors">Stack</a>
        <a href="https://github.com/Ganglet/Anvil" target="_blank" rel="noopener noreferrer"
          className="hover:text-white transition-colors" style={{ color: 'var(--teal)' }}>GitHub ↗</a>
      </div>
    </nav>
  );
}

function StatsBanner() {
  const stats = [
    { value: '8', label: 'Pipeline Phases' },
    { value: '4', label: 'Attack Strategies' },
    { value: '4', label: 'Patch Strategies' },
    { value: '10', label: 'Papers in RAG' },
    { value: '0', label: 'Human Decisions' },
  ];
  return (
    <div className="w-full py-6 flex items-center justify-center flex-wrap gap-0"
      style={{ background: 'var(--surface)', borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)' }}>
      {stats.map((s, i) => (
        <div key={i} className="flex flex-col items-center px-10 py-2"
          style={{ borderRight: i < stats.length - 1 ? '1px solid var(--border)' : 'none' }}>
          <span className="text-3xl font-black" style={{ color: i === 4 ? 'var(--copper-light)' : 'var(--teal)' }}>{s.value}</span>
          <span className="text-xs mt-1 tracking-wide uppercase" style={{ color: 'var(--text-muted)' }}>{s.label}</span>
        </div>
      ))}
    </div>
  );
}

function PipelineSection() {
  const phases = [
    {
      num: '01',
      title: 'Model Interface',
      subtitle: 'Unified Neural Network Wrapper',
      body: 'BaseModel ABC wraps any PyTorch net with predict(), get_gradients(), get_activations(). Standardizes inputs and outputs across architectures for uniform pipeline processing.',
      bg: 'var(--teal-dark)', text: '#fff',
    },
    {
      num: '02',
      title: 'Attack Surface Profiler',
      subtitle: 'Captum Integrated Gradients + Saliency',
      body: 'Runs Captum Integrated Gradients and Saliency analysis to generate a per-class vulnerability score and attack priority list. Identifies the most exposed decision boundaries before any adversarial attack is launched.',
      bg: 'var(--surface2)', text: 'var(--teal-light)',
    },
    {
      num: '03',
      title: 'Attack Engine',
      subtitle: 'Multi-Strategy Adversarial Attacks',
      body: 'FGSM · PGD · Adversarial Patch · Semantic attacks — all built from scratch in PyTorch autograd. No external attack libraries. Budget-controlled perturbation with configurable epsilon and iteration count.',
      bg: '#1a0800', text: 'var(--copper-light)',
    },
    {
      num: '04',
      title: 'Failure Mode Clustering',
      subtitle: 'UMAP + HDBSCAN Manifold Analysis',
      body: 'Penultimate-layer activations from all adversarial failures are extracted, then projected via UMAP (non-linear manifold) and clustered with HDBSCAN (density-based). Falls back to PCA automatically if sample count < 20.',
      bg: 'var(--surface)', text: 'var(--text)',
    },
    {
      num: '05',
      title: 'LLM Explanation Agent',
      subtitle: 'LangGraph + FAISS + Gemini 2.5 Flash',
      body: 'A LangGraph stateful agent runs RAG over a FAISS index of 10 adversarial ML papers embedded with nomic-embed-text. Gemini 2.5 Flash generates grounded, citation-backed explanations for each failure cluster.',
      bg: '#002b22', text: 'var(--teal-light)',
    },
    {
      num: '06',
      title: 'Autonomous Patching',
      subtitle: '4 Strategies · Automated Safety Gate',
      body: '4 autonomous patch strategies: adversarial training, stylized augmentation, counterfactual generation, targeted augmentation. Safety gate rejects any patch where accuracy drop exceeds 3% or robustness score falls below 0.70.',
      bg: 'var(--copper)', text: '#fff',
    },
    {
      num: '07',
      title: 'Audit Report',
      subtitle: 'Professional PDF via ReportLab',
      body: 'ReportLab generates a multi-page PDF: executive cover page, radar charts for attack surface, per-cluster failure analysis cards with LLM explanations, full methodology appendix, and raw metric tables.',
      bg: 'var(--surface2)', text: 'var(--text)',
    },
    {
      num: '08',
      title: 'REST API',
      subtitle: 'FastAPI + Docker on HuggingFace Spaces',
      body: 'FastAPI with async job management. POST /audit/upload → job_id. GET /audit/job/{id} polls status. GET /report/{filename} downloads PDF. Deployed via Docker on HuggingFace Spaces — fully serverless.',
      bg: 'var(--surface)', text: 'var(--text)',
    },
  ];

  return (
    <section id="pipeline" className="w-full">
      <div className="max-w-5xl mx-auto px-6 py-20 text-center">
        <p className="text-xs tracking-[0.3em] uppercase mb-3" style={{ color: 'var(--teal)' }}>The Pipeline</p>
        <h2 className="text-4xl md:text-5xl font-black mb-6 text-white">8 Phases. Zero Human Decisions.</h2>
        <p className="text-lg max-w-2xl mx-auto mb-10" style={{ color: 'var(--text-muted)' }}>
          From raw model weights to signed PDF audit report — fully autonomous adversarial evaluation.
        </p>
        <img
          src="/Anvil/architecture.png"
          alt="ANVIL Architecture"
          className="mx-auto max-w-xs opacity-80 mb-8"
        />
      </div>
      <FlowArt>
        {phases.map((p, i) => (
          <FlowSection
            key={i}
            aria-label={`Phase ${p.num}: ${p.title}`}
            style={{ background: p.bg }}
          >
            <div className="flex flex-col justify-center h-full min-h-screen py-20 px-8 md:px-20 max-w-4xl mx-auto w-full">
              <div className="flex items-start gap-6 mb-6">
                <span className="text-6xl font-black opacity-20 leading-none select-none"
                  style={{ fontFamily: "'JetBrains Mono', monospace", color: p.text }}>{p.num}</span>
                <div>
                  <h3 className="text-3xl md:text-4xl font-black mb-2" style={{ color: p.text }}>{p.title}</h3>
                  <p className="text-sm tracking-widest uppercase" style={{ color: p.text, opacity: 0.6 }}>{p.subtitle}</p>
                </div>
              </div>
              <p className="text-lg leading-relaxed max-w-2xl" style={{ color: p.text, opacity: 0.85 }}>{p.body}</p>
            </div>
          </FlowSection>
        ))}
      </FlowArt>
    </section>
  );
}

function DemoSection() {
  const [files, setFiles] = useState<File[]>([]);
  const [budget, setBudget] = useState(25);
  const [status, setStatus] = useState<'idle' | 'running' | 'complete' | 'error'>('idle');
  const [logs, setLogs] = useState<string[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const dropRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTime = useRef<number>(0);

  const timeEstimate = budget <= 20 ? '~2-3 min' : budget <= 25 ? '~3-4 min' : '~4-5 min';

  const addLog = useCallback((msg: string) => {
    setLogs(prev => [...prev, msg]);
  }, []);

  const startTimer = () => {
    startTime.current = Date.now();
    timerRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTime.current) / 1000));
    }, 1000);
  };

  const stopTimer = () => {
    if (timerRef.current) clearInterval(timerRef.current);
  };

  useEffect(() => () => stopTimer(), []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const dropped = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
    setFiles(prev => [...prev, ...dropped]);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const selected = Array.from(e.target.files).filter(f => f.type.startsWith('image/'));
    setFiles(prev => [...prev, ...selected]);
  };

  const removeFile = (i: number) => setFiles(prev => prev.filter((_, idx) => idx !== i));

  const handleRun = async () => {
    if (files.length === 0) return;
    setStatus('running');
    setLogs([]);
    setDownloadUrl(null);
    setErrorMsg(null);
    setElapsed(0);
    startTimer();
    addLog('Uploading images...');
    try {
      const jobId = await runAudit(files, budget);
      addLog(`Job created: ${jobId}`);
      addLog('Running all 8 pipeline phases...');
      const result = await pollJob(jobId, addLog);
      stopTimer();
      addLog('Audit complete!');
      const reportFile = result.report_filename as string;
      setDownloadUrl(`${API}/report/${reportFile}`);
      setStatus('complete');
    } catch (err) {
      stopTimer();
      setErrorMsg(err instanceof Error ? err.message : 'Unknown error');
      setStatus('error');
    }
  };

  const handleReset = () => {
    setStatus('idle');
    setFiles([]);
    setLogs([]);
    setElapsed(0);
    setDownloadUrl(null);
    setErrorMsg(null);
  };

  const formatTime = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;

  return (
    <section id="demo" className="w-full py-24 px-6" style={{ background: 'var(--surface)' }}>
      <div className="max-w-2xl mx-auto">
        <p className="text-xs tracking-[0.3em] uppercase mb-3 text-center" style={{ color: 'var(--teal)' }}>Live Demo</p>
        <h2 className="text-4xl font-black text-center text-white mb-3">Run the full pipeline on your images</h2>
        <p className="text-center mb-10" style={{ color: 'var(--text-muted)' }}>
          Upload images, hit Run — all 8 phases execute live on HuggingFace Spaces and a real PDF audit report comes back.
        </p>

        <div
          ref={dropRef}
          onDrop={handleDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => document.getElementById('file-input')?.click()}
          className="w-full rounded-xl border-2 border-dashed flex flex-col items-center justify-center py-12 px-6 cursor-pointer transition-colors mb-6"
          style={{ borderColor: 'var(--border)', background: 'var(--bg)' }}
        >
          <input id="file-input" type="file" multiple accept="image/*" className="hidden" onChange={handleFileInput} />
          <div className="text-4xl mb-3">&#128444;&#65039;</div>
          <p className="font-semibold text-white">Click or drag images here</p>
          <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>PNG, JPG, WebP supported</p>
        </div>

        {files.length > 0 && (
          <div className="mb-6 flex flex-wrap gap-2">
            {files.map((f, i) => (
              <div key={i} className="flex items-center gap-2 px-3 py-1 rounded-full text-sm"
                style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--text-muted)' }}>
                <span className="max-w-[120px] truncate">{f.name}</span>
                <button onClick={() => removeFile(i)} className="hover:text-white transition-colors">x</button>
              </div>
            ))}
          </div>
        )}

        <div className="mb-8">
          <div className="flex justify-between items-center mb-2">
            <label className="text-sm font-medium text-white">Sample Budget</label>
            <span className="text-sm" style={{ color: 'var(--teal)' }}>{budget} samples · {timeEstimate}</span>
          </div>
          <input
            type="range" min={15} max={30} value={budget}
            onChange={e => setBudget(Number(e.target.value))}
            className="w-full"
            style={{ accentColor: 'var(--teal)' }}
          />
          <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
            <span>15</span><span>30</span>
          </div>
        </div>

        <p className="text-xs mb-4 text-center" style={{ color: 'var(--text-muted)' }}>
          Upload 5-10 images for best results
        </p>

        {status === 'idle' && (
          <button
            onClick={handleRun}
            disabled={files.length === 0}
            className="w-full py-4 rounded-xl font-bold text-lg transition-all"
            style={{
              background: files.length === 0 ? '#1a2a3a' : 'var(--teal)',
              color: files.length === 0 ? 'var(--text-muted)' : '#050e1a',
              cursor: files.length === 0 ? 'not-allowed' : 'pointer',
            }}
          >
            Run Audit
          </button>
        )}

        {(status === 'running' || status === 'complete' || status === 'error') && (
          <div className="mt-6 rounded-xl p-6" style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
            <div className="flex items-center gap-3 mb-4">
              {status === 'running' && (
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style={{ background: 'var(--teal)' }}></span>
                  <span className="relative inline-flex rounded-full h-3 w-3" style={{ background: 'var(--teal)' }}></span>
                </span>
              )}
              {status === 'complete' && <span style={{ color: 'var(--teal)' }}>✓</span>}
              {status === 'error' && <span style={{ color: 'var(--copper-light)' }}>✗</span>}
              <span className="font-semibold text-white">
                {status === 'running' ? 'Running...' : status === 'complete' ? 'Audit Complete' : 'Error'}
              </span>
              {(status === 'running' || status === 'complete') && (
                <span className="ml-auto text-sm" style={{ color: 'var(--text-muted)' }}>{formatTime(elapsed)}</span>
              )}
            </div>

            {errorMsg && (
              <p className="text-sm mb-4 p-3 rounded-lg" style={{ background: 'rgba(194,65,12,0.1)', color: 'var(--copper-light)', border: '1px solid rgba(194,65,12,0.3)' }}>
                {errorMsg}
              </p>
            )}

            {logs.length > 0 && (
              <div className="mb-4 max-h-40 overflow-y-auto rounded-lg p-3 text-sm space-y-1"
                style={{ background: 'var(--surface2)', fontFamily: "'JetBrains Mono', monospace" }}>
                {logs.map((l, i) => (
                  <div key={i} style={{ color: 'var(--text-muted)' }}>
                    <span style={{ color: 'var(--teal)', marginRight: '8px' }}>›</span>{l}
                  </div>
                ))}
              </div>
            )}

            {downloadUrl && (
              <a
                href={downloadUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="block w-full text-center py-3 rounded-xl font-bold mb-3 transition-all"
                style={{ background: 'var(--teal)', color: '#050e1a' }}
              >
                Download PDF Audit Report
              </a>
            )}

            {(status === 'complete' || status === 'error') && (
              <button
                onClick={handleReset}
                className="w-full py-3 rounded-xl font-medium transition-colors"
                style={{ background: 'var(--surface2)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}
              >
                Try Again
              </button>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

function StackSection() {
  const stack = [
    { name: 'PyTorch', desc: 'Core deep learning framework for all model operations' },
    { name: 'PyTorch Autograd', desc: 'Attacks built from scratch using gradient computation' },
    { name: 'Captum', desc: 'Integrated Gradients and Saliency for attack surface profiling' },
    { name: 'UMAP + HDBSCAN', desc: 'Non-linear manifold projection and density clustering' },
    { name: 'LangGraph + LangChain', desc: 'Stateful agent orchestration for explanation pipeline' },
    { name: 'Gemini 2.5 Flash', desc: 'LLM generating grounded per-cluster vulnerability explanations' },
    { name: 'FAISS + nomic-embed-text', desc: 'Vector search over 10 adversarial ML research papers' },
    { name: 'ReportLab', desc: 'Programmatic PDF generation for the full audit report' },
    { name: 'FastAPI + uvicorn', desc: 'Async REST API with job management and polling' },
    { name: 'Docker + HuggingFace Spaces', desc: 'Containerized deployment with public serverless access' },
  ];

  return (
    <section id="stack" className="w-full py-24 px-6" style={{ background: 'var(--bg)' }}>
      <div className="max-w-5xl mx-auto">
        <p className="text-xs tracking-[0.3em] uppercase mb-3 text-center" style={{ color: 'var(--teal)' }}>Technology Stack</p>
        <h2 className="text-4xl font-black text-center text-white mb-4">Built on production ML tooling</h2>
        <p className="text-center mb-12" style={{ color: 'var(--text-muted)' }}>Every component chosen for correctness, not convenience.</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {stack.map((s, i) => (
            <div key={i} className="rounded-xl p-5 transition-all hover:translate-y-[-2px]"
              style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
              <h3 className="font-bold mb-1" style={{ color: i % 3 === 0 ? 'var(--teal)' : i % 3 === 1 ? 'var(--copper-light)' : 'var(--teal-light)' }}>{s.name}</h3>
              <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{s.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function QuickStartSection() {
  return (
    <section id="quickstart" className="w-full py-24 px-6" style={{ background: 'var(--surface)' }}>
      <div className="max-w-2xl mx-auto">
        <p className="text-xs tracking-[0.3em] uppercase mb-3 text-center" style={{ color: 'var(--teal)' }}>Quick Start</p>
        <h2 className="text-4xl font-black text-center text-white mb-10">Run it yourself</h2>
        <div className="rounded-xl overflow-hidden mb-6" style={{ border: '1px solid var(--border)' }}>
          <div className="flex items-center gap-2 px-4 py-2" style={{ background: 'var(--surface2)', borderBottom: '1px solid var(--border)' }}>
            <span className="w-3 h-3 rounded-full bg-red-500"></span>
            <span className="w-3 h-3 rounded-full bg-yellow-500"></span>
            <span className="w-3 h-3 rounded-full bg-green-500"></span>
            <span className="ml-2 text-xs" style={{ color: 'var(--text-muted)', fontFamily: "'JetBrains Mono', monospace" }}>terminal</span>
          </div>
          <pre className="p-6 text-sm overflow-x-auto" style={{ background: '#020a14', fontFamily: "'JetBrains Mono', monospace", color: 'var(--teal-light)' }}>
            <code>{`git clone https://github.com/Ganglet/Anvil && cd Anvil/Anvil_Project\npip install -r requirements.txt\nuvicorn api:app --host 0.0.0.0 --port 8000\n# Then open ganglet.github.io/Anvil for the demo UI`}</code>
          </pre>
        </div>
        <div className="rounded-xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
          <div className="flex items-center gap-2 px-4 py-2" style={{ background: 'var(--surface2)', borderBottom: '1px solid var(--border)' }}>
            <span className="ml-2 text-xs" style={{ color: 'var(--text-muted)', fontFamily: "'JetBrains Mono', monospace" }}>CLI example</span>
          </div>
          <pre className="p-6 text-sm" style={{ background: '#020a14', fontFamily: "'JetBrains Mono', monospace", color: 'var(--copper-light)' }}>
            <code>{`python run.py --model resnet18 --budget 50`}</code>
          </pre>
        </div>
      </div>
    </section>
  );
}

function Footer() {
  return (
    <footer className="w-full py-10 px-6 text-center" style={{ background: 'var(--bg)', borderTop: '1px solid var(--border)' }}>
      <div className="flex items-center justify-center gap-3 flex-wrap">
        <span className="font-black tracking-widest" style={{ color: 'var(--teal)', fontFamily: "'JetBrains Mono', monospace" }}>ANVIL</span>
        <span style={{ color: 'var(--text-muted)' }}>·</span>
        <span style={{ color: 'var(--text-muted)' }}>Built for the Red Queen project</span>
        <span style={{ color: 'var(--text-muted)' }}>·</span>
        <a href="https://github.com/Ganglet/Anvil" target="_blank" rel="noopener noreferrer"
          className="transition-colors hover:text-white" style={{ color: 'var(--teal)' }}>GitHub ↗</a>
      </div>
    </footer>
  );
}

export default function App() {
  return (
    <div style={{ background: 'var(--bg)', minHeight: '100vh' }}>
      <Nav />

      {/* Hero */}
      <section className="relative w-full flex flex-col items-center justify-center text-center px-6"
        style={{ minHeight: '100vh', background: 'var(--bg)' }}>
        <div className="absolute inset-0 z-0">
          <Beams
            lightColor="#14b8a6"
            beamNumber={14}
            beamWidth={1.8}
            beamHeight={18}
            speed={1.8}
            noiseIntensity={1.5}
            scale={0.18}
            rotation={-10}
          />
        </div>
        <div className="relative z-10 max-w-4xl mx-auto">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full mb-8 text-xs tracking-widest uppercase font-semibold"
            style={{ background: 'rgba(20,184,166,0.12)', border: '1px solid rgba(20,184,166,0.3)', color: 'var(--teal)' }}>
            <span className="w-2 h-2 rounded-full animate-pulse" style={{ background: 'var(--teal)' }}></span>
            Autonomous ML Red-Teaming
          </div>
          <h1 className="font-black leading-none mb-4" style={{ fontSize: 'clamp(5rem,18vw,12rem)', letterSpacing: '-0.04em', color: 'white' }}>
            AN<span style={{ color: 'var(--teal)' }}>VIL</span>
          </h1>
          <p className="text-base tracking-[0.15em] uppercase mb-6 font-medium" style={{ color: 'var(--text-muted)' }}>
            Adversarial Neural Vulnerability Inspection and Learning
          </p>
          <p className="text-lg max-w-2xl mx-auto mb-10 leading-relaxed" style={{ color: 'rgba(241,245,249,0.75)' }}>
            Attack any neural network. Cluster its failures. Explain each vulnerability with RAG-grounded LLM reasoning. Patch autonomously. Deliver a professional audit report. Zero human decisions.
          </p>
          <div className="flex flex-wrap gap-4 justify-center">
            <a href="#demo"
              className="px-8 py-4 rounded-xl font-bold text-base transition-all hover:scale-105"
              style={{ background: 'var(--teal)', color: '#050e1a' }}>
              Live Demo
            </a>
            <a href="https://github.com/Ganglet/Anvil" target="_blank" rel="noopener noreferrer"
              className="px-8 py-4 rounded-xl font-bold text-base transition-all hover:scale-105"
              style={{ background: 'rgba(255,255,255,0.06)', color: 'white', border: '1px solid rgba(255,255,255,0.15)' }}>
              GitHub ↗
            </a>
          </div>
        </div>
      </section>

      <StatsBanner />
      <PipelineSection />
      <DemoSection />
      <StackSection />
      <QuickStartSection />
      <Footer />
    </div>
  );
}
