// Task Pipeline Player — Vanilla JS Web Component (ES Module)
// Usage: <script type="module" src="task-pipeline-player.js"></script>
// Then place <task-pipeline-player></task-pipeline-player> anywhere.
// Optional attributes:
//   farmers-src, farmerrobot-src, openai-src  (asset paths)
//   autoplay ("true" | "false"), speed (milliseconds)
//   width (px) — optional fixed width; component is responsive by default

class TaskPipelinePlayer extends HTMLElement {
  static get observedAttributes() {
    return ["farmers-src", "farmerrobot-src", "openai-src", "autoplay", "speed", "width"]; }

  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._step = 0;
    this._isPlaying = true;
    this._timer = null;
    this._speed = 1600;
    this._scale = 1;

    // Data (positions on a 1200x760 canvas)
    this.BG_W = 1200; this.BG_H = 760;
    this.NODES = [
      { id:'raw', title:'Raw Data', subtitle:'(JSON)', x:20, y:290, w:190, h:160, color:'#38bdf8',
        tooltip:'User–expert conversations, images, image descriptions, and metadata.',
        bullets:['Dialogue','Attachments (Images)','Image Descriptions','Meta Data'], icon:() => this.assets.farmers },
      { id:'clean', title:'Data Cleaning', subtitle:'PII / URLs', x:250, y:300, w:170, h:120, dashed:true,
        tooltip:'Strip PII and scrub URLs before any processing.' },
      { id:'lvlm', title:'LVLM', subtitle:'(image parsing)', x:260, y:450, w:140, h:70,
        tooltip:'Vision-language model used to summarize or describe attached images.', logo:() => this.assets.openai },
      { id:'truncate', title:'Truncate Dialogue', subtitle:'Partial Observability', x:450, y:250, w:220, h:210, dashed:true,
        tooltip:'Iterate over user turns, select a context window, retain context up to the user turn, and extract the next user utterance (revealed fact).',
        bullets:['Iterate through user turns','Select context window','Retain context up to user turn','Extract next user utterance'] },
      { id:'prompt', title:'Prompt Design', x:430, y:500, w:260, h:170, color:'#f59e0b',
        tooltip:'Compose the MMMT prompt from dialogue context, revealed facts, user goal, optional image description, and mark the expert turn.',
        bullets:['Dialogue Context','Revealed Facts','User Goal','Image Description','Expert Turn'] },
      { id:'llm', title:'LLM', x:720, y:530, w:100, h:80,
        tooltip:'Language model converts the prompt into a structured task entry.', logo:() => this.assets.openai },
      { id:'task', title:'MMMT Task', subtitle:'(JSON)', x:860, y:250, w:260, h:310, color:'#fb7185',
        tooltip:'Final structured sample for the benchmark / dataset.',
        bullets:['Source ID','Dialogue Context','Attachments (Images)','Goal','Goal State {...}','Decision','Utterance'], icon:() => this.assets.farmerrobot },
    ];
    this.EDGES = [
      { from:'raw', to:'clean' },{ from:'clean', to:'truncate' },{ from:'raw', to:'lvlm' },
      { from:'lvlm', to:'prompt' },{ from:'truncate', to:'prompt' },{ from:'prompt', to:'llm' },{ from:'llm', to:'task' },
    ];
    this.STEPS = [
      { id:'s1', highlight:['raw'], text:'Start with raw JSON: dialogues, images, descriptions, metadata.' },
      { id:'s2', highlight:['clean'], text:'Clean sensitive info (PII) and strip URLs.' },
      { id:'s3', highlight:['truncate'], text:'Truncate dialogue for partial observability; extract next user fact.' },
      { id:'s4', highlight:['lvlm'], text:'Use LVLM to summarize or describe attached images.' },
      { id:'s5', highlight:['prompt'], text:'Assemble the prompt: context + revealed facts + user goal + (image description).' },
      { id:'s6', highlight:['llm'], text:'LLM converts prompt into a structured MMMT sample.' },
      { id:'s7', highlight:['task'], text:'Output: MMMT Task JSON with fields like Goal, Goal State, Decision, and Utterance.' },
    ];

    this.assets = {
      farmers: this.getAttribute("farmers-src") || "/assets/farmers.png",
      farmerrobot: this.getAttribute("farmerrobot-src") || "/assets/farmer-robot.png",
      openai: this.getAttribute("openai-src") || "/assets/openai.png",
    };
  }

  connectedCallback() {
    this._speed = Number(this.getAttribute('speed')) || this._speed;
    const autoplayAttr = this.getAttribute('autoplay');
    this._isPlaying = autoplayAttr === null ? true : (autoplayAttr === 'true');

    this.shadowRoot.innerHTML = `
      <style>
        :host { display:block; font: 15px/1.45 system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial; color:#1f2937; }
        .card{ background:#fff; border:1px solid #e5e7eb; border-radius:14px; box-shadow:0 10px 18px rgba(0,0,0,.06); }
        .header{ padding:16px 18px 8px; border-bottom:1px solid #f1f5f9; }
        .title{ margin:0 0 6px; font-weight:700; font-size:20px; }
        .controls{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; justify-content:space-between; }
        .left,.right{ display:flex; gap:8px; align-items:center; }
        button{ border:1px solid #d1d5db; background:#fff; border-radius:10px; height:36px; padding:0 12px; cursor:pointer; display:inline-flex; align-items:center; gap:6px; }
        button.icon{ width:36px; justify-content:center; padding:0; }
        button.primary{ background:#f59e0b; color:#fff; border-color:#f59e0b; }
        button:hover{ filter:brightness(.98); }
        input[type=range]{ width:160px; }
        .legend{ color:#6b7280; font-size:13px; display:inline-flex; align-items:center; gap:8px; }
        .dot{ width:8px; height:8px; border-radius:50%; background:#f59e0b; }
        .narration{ margin:8px 0 0; color:#374151; font-size:14px; }
        .canvas{ position:relative; width:100%; overflow:hidden; border-radius:0 0 14px 14px; background:
          radial-gradient(1200px 600px at 20% 35%, rgba(59,130,246,0.08), transparent),
          radial-gradient(1000px 500px at 80% 25%, rgba(244,63,94,0.06), transparent),
          radial-gradient(900px 500px at 50% 75%, rgba(245,158,11,0.06), transparent);
        }
        .node{ position:absolute; background:rgba(255,255,255,.92); backdrop-filter: blur(3px); border:2px solid #d1d5db; border-radius:16px; padding:10px 12px; box-shadow:0 4px 12px rgba(0,0,0,.06); transition: box-shadow .18s, transform .18s, border-color .18s; pointer-events:auto; }
        .node.dashed{ border-style:dashed; }
        .node .head{ display:flex; align-items:start; gap:8px; }
        .node .title{ font-size:15px; margin:0; line-height:1.2; }
        .node .subtitle{ font-size:12px; color:#6b7280; margin-top:-2px; }
        .node ul{ margin:8px 0 0; padding-left:18px; color:#374151; font-size:12px; }
        .node img.icon{ width:40px; height:40px; object-fit:contain; user-select:none; pointer-events:none; }
        .node img.logo{ width:28px; height:28px; margin-left:auto; opacity:.8; object-fit:contain; user-select:none; pointer-events:none; }
        .node.active{ box-shadow:0 0 0 4px rgba(245,158,11,.35); border-color:#f59e0b; }
        .tooltip{ position:absolute; padding:8px 10px; background:#111827; color:#fff; font-size:12px; border-radius:8px; pointer-events:none; transform:translate(-50%,-100%); white-space:nowrap; opacity:0; transition:opacity .12s; }
        .tooltip.show{ opacity:1; }
        svg{ position:absolute; left:0; top:0; pointer-events:none; }
        .wrap{ padding:0 16px 16px; }
      </style>
      <div class="card">
        <div class="header">
          <div class="title">Task Generation Pipeline</div>
          <div class="controls">
            <div class="left">
              <button class="icon" id="back" title="Step Back [←]">⏮</button>
              <button class="icon primary" id="play" title="Play/Pause [Space]">⏸</button>
              <button class="icon" id="fwd" title="Step Forward [→]">⏭</button>
              <button id="reset" title="Reset">↺ Reset</button>
            </div>
            <div class="right">
              <div class="legend"><span class="dot"></span><span id="legend">Step 1 / ${this.STEPS.length}</span></div>
              <label style="color:#6b7280;font-size:13px">Speed</label>
              <input type="range" id="speed" min="600" max="3000" step="200" />
            </div>
          </div>
          <p class="narration" id="narration"></p>
        </div>
        <div class="wrap">
          <div id="canvas" class="canvas"></div>
        </div>
      </div>
    `;

    this.$ = {
      canvas: this.shadowRoot.getElementById('canvas'),
      narration: this.shadowRoot.getElementById('narration'),
      legend: this.shadowRoot.getElementById('legend'),
      back: this.shadowRoot.getElementById('back'),
      play: this.shadowRoot.getElementById('play'),
      fwd: this.shadowRoot.getElementById('fwd'),
      reset: this.shadowRoot.getElementById('reset'),
      speed: this.shadowRoot.getElementById('speed'),
    };

    // initial values
    this.$.speed.value = String(this._speed);

    // events
    this._onResize = this._setScale.bind(this);
    window.addEventListener('resize', this._onResize);

    this.$.play.addEventListener('click', () => { this._isPlaying = !this._isPlaying; this.$.play.textContent = this._isPlaying ? '⏸' : '▶'; this._schedule(); });
    this.$.back.addEventListener('click', () => { this._step = (this._step - 1 + this.STEPS.length) % this.STEPS.length; this._render(); });
    this.$.fwd.addEventListener('click', () => { this._step = (this._step + 1) % this.STEPS.length; this._render(); });
    this.$.reset.addEventListener('click', () => { this._step = 0; this._render(); });
    this.$.speed.addEventListener('input', (e) => { this._speed = Number(e.target.value); this._schedule(); });

    this.shadowRoot.addEventListener('keydown', (e) => {
      if (e.code === 'Space') { e.preventDefault(); this.$.play.click(); }
      else if (e.key === 'ArrowRight') { this.$.fwd.click(); }
      else if (e.key === 'ArrowLeft') { this.$.back.click(); }
    });

    // initial layout
    this._setScale();
    this._render();
    this._schedule();
  }

  disconnectedCallback() {
    clearInterval(this._timer);
    window.removeEventListener('resize', this._onResize);
  }

  attributeChangedCallback(name, oldVal, newVal) {
    if (!this.isConnected) return;
    if (name === 'speed') { this._speed = Number(newVal) || this._speed; this.$.speed.value = String(this._speed); this._schedule(); }
    if (name === 'autoplay') { this._isPlaying = newVal === 'true'; this.$.play.textContent = this._isPlaying ? '⏸' : '▶'; this._schedule(); }
    if (name === 'farmers-src' || name === 'farmerrobot-src' || name === 'openai-src') {
      this.assets = {
        farmers: this.getAttribute('farmers-src') || this.assets.farmers,
        farmerrobot: this.getAttribute('farmerrobot-src') || this.assets.farmerrobot,
        openai: this.getAttribute('openai-src') || this.assets.openai,
      };
      this._render();
    }
    if (name === 'width') {
      this.style.maxWidth = newVal ? `${parseInt(newVal,10)}px` : '';
      this._setScale();
    }
  }

  /* ===== Rendering ===== */
  _setScale() {
    const w = this.$.canvas.clientWidth || this.$.canvas.offsetWidth || this.getBoundingClientRect().width || this.BG_W;
    this._scale = Math.min(w / this.BG_W, 1);
    this.$.canvas.style.height = `${this.BG_H * this._scale}px`;
    this._render();
  }

  _schedule() {
    clearInterval(this._timer);
    if (!this._isPlaying) return;
    this._timer = setInterval(() => { this._step = (this._step + 1) % this.STEPS.length; this._render(); }, this._speed);
  }

  _render() {
    const c = this.$.canvas;
    c.innerHTML = '';

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', this.BG_W * this._scale);
    svg.setAttribute('height', this.BG_H * this._scale);
    svg.style.position = 'absolute'; svg.style.left = '0'; svg.style.top = '0'; svg.style.pointerEvents = 'none';

    const active = new Set(this.STEPS[this._step].highlight);

    // arrows first
    this.EDGES.forEach(e => {
      const from = this.NODES.find(n => n.id === e.from); const to = this.NODES.find(n => n.id === e.to);
      const x1 = (from.x + from.w) * this._scale; const y1 = (from.y + from.h/2) * this._scale;
      const x2 = (to.x) * this._scale; const y2 = (to.y + to.h/2) * this._scale; const midX = (x1 + x2)/2;
      const path = document.createElementNS(svg.namespaceURI,'path');
      path.setAttribute('d', `M ${x1},${y1} C ${midX},${y1} ${midX},${y2} ${x2},${y2}`);
      path.setAttribute('fill','none');
      const isActive = active.has(e.from) || active.has(e.to);
      path.setAttribute('stroke', isActive ? '#f59e0b' : '#9ca3af');
      path.setAttribute('stroke-width', isActive ? '4' : '2');
      svg.appendChild(path);
      const arrow = document.createElementNS(svg.namespaceURI,'polygon');
      arrow.setAttribute('points', `${x2},${y2} ${x2-10},${y2-6} ${x2-10},${y2+6}`);
      arrow.setAttribute('fill', isActive ? '#f59e0b' : '#9ca3af');
      svg.appendChild(arrow);
    });
    c.appendChild(svg);

    // nodes
    this.NODES.forEach(n => {
      const el = document.createElement('div');
      el.className = 'node' + (n.dashed ? ' dashed' : '');
      if (n.color) el.style.borderColor = n.color;
      if (active.has(n.id)) el.classList.add('active');
      el.style.left = (n.x * this._scale) + 'px';
      el.style.top = (n.y * this._scale) + 'px';
      el.style.width = (n.w * this._scale) + 'px';
      el.style.height = (n.h * this._scale) + 'px';

      const head = document.createElement('div'); head.className = 'head';
      if (n.icon) { const img = document.createElement('img'); img.src = n.icon(); img.alt=''; img.className='icon'; head.appendChild(img); }
      const textWrap = document.createElement('div');
      const h = document.createElement('h4'); h.className='title'; h.textContent=n.title; textWrap.appendChild(h);
      if (n.subtitle) { const sub=document.createElement('div'); sub.className='subtitle'; sub.textContent=n.subtitle; textWrap.appendChild(sub); }
      head.appendChild(textWrap);
      if (n.logo) { const lg=document.createElement('img'); lg.src=n.logo(); lg.alt=''; lg.className='logo'; head.appendChild(lg); }
      el.appendChild(head);
      if (n.bullets && n.bullets.length) {
        const ul=document.createElement('ul'); n.bullets.forEach(b=>{ const li=document.createElement('li'); li.textContent=b; ul.appendChild(li); }); el.appendChild(ul);
      }

      const tip = document.createElement('div'); tip.className='tooltip'; tip.textContent = n.tooltip || ''; c.appendChild(tip);
      el.addEventListener('mousemove', (ev)=>{ tip.classList.add('show'); const rect=c.getBoundingClientRect(); tip.style.left=(ev.clientX - rect.left)+'px'; tip.style.top=(ev.clientY - rect.top - 10)+'px'; });
      el.addEventListener('mouseleave', ()=> tip.classList.remove('show'));

      c.appendChild(el);
    });

    // text
    this.$.narration.textContent = this.STEPS[this._step].text;
    this.$.legend.textContent = `Step ${this._step + 1} / ${this.STEPS.length}`;
  }
}

customElements.define('task-pipeline-player', TaskPipelinePlayer);
