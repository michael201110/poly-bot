export const HUD_MODES = ["OFF", "CONTROLS", "NEURAL_NET", "RL_DEBUG", "REWARD", "OBSERVATIONS", "TRAINING", "TRAINING_GRAPH", "GHOST_RACE", "WR_CHASE", "COMPARISON", "RACING_LINE_ANALYSIS", "DETAILED_CONTROLS", "CHAMPION", "MINIMAL_RACE"];

// Presentation only: the policy continues to receive its complete observation.
export function importantInputs(features) {
  const priority = (f) => {
    if (/velocity\.(forward|right)$|route\.(heading_error|lateral_offset)|actual_steering/.test(f.key)) return 100;
    if (f.group === "lookahead" && !/mask$|\.up$/.test(f.key)) return 90;
    if (/ghost\.(target_speed|relative_position\.(forward|right))|wheel_\d\.(contact|skid)|vehicle\.(pitch|roll)|angular_velocity.up/.test(f.key)) return 80;
    if (/acceleration|previous_|controller/.test(f.key)) return 60;
    return 10;
  };
  const chosen = new Set(features.map((f, i) => ({f, i})).sort((a,b) => priority(b.f)-priority(a.f) || a.i-b.i)
    .slice(0, Math.ceil(features.length * .6)).map(({f}) => f.key));
  return features.filter(f => chosen.has(f.key));
}

export function installPolyBotHudOverlay(host = globalThis) {
  const supportedFrameSchema = "polybot.ai-overlay-frame.v1";
  const supportedFeatureSchema = "polybot.observation-features.v1";
  const supportedObservationSchemas = new Set([
    "polybot.observation.v2",
    "polybot.observation.v2.pwm-state",
    "polybot.observation.v2.training-state",
  ]);
  const rows = new Map();
  let root = null;
  let toggle = null;
  let sections = null;
  let lastEvent = "";
  let latestFrame = null;
  let lastRenderedMode = null;
  let hudVisible = false;
  let modeSelector = null;
  let modeOverride = null;
  let extra = null;
  let chart = [];
  let chartRun = null;
  let sectionEntry = null;
  let lastSection = null;
  let previousSection = null;
  let displayedMode = null;
  let modeFadeTimer = null;
  let layoutMode = null;
  let layouts = {};
  let panelScale = 1;
  try { layouts = JSON.parse(host.localStorage?.getItem("polybot.hud.layouts.v1") || "{}"); }
  catch { layouts = {}; }
  if (!layouts || typeof layouts !== "object" || Array.isArray(layouts)) layouts = {};
  const svgNode = (tag, parent, attributes = {}) => {
    const node = host.document.createElementNS ? host.document.createElementNS("http://www.w3.org/2000/svg", tag) : host.document.createElement(tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
    parent.appendChild(node);
    return node;
  };

  const element = (tag, className, parent) => {
    const node = host.document.createElement(tag);
    if (className) node.className = className;
    if (parent) parent.appendChild(node);
    return node;
  };
  const text = (node, value) => {
    node.textContent = value == null ? "" : String(value);
  };
  const number = (value, digits = 2) =>
    typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";

  function ensureRoot() {
    if (root) return true;
    const body = host.document.body;
    if (!body) return false;
    const style = element("style", "", host.document.head || body);
    style.textContent = `
      .polybot-ai-hud-toggle{position:fixed;z-index:2147483647;right:12px;top:12px;
        pointer-events:auto;color:#edf4fa;background:rgba(15,28,44,.28);
        border:1px solid rgba(235,248,255,.38);border-radius:999px;padding:8px 13px;
        backdrop-filter:blur(12px) saturate(150%);-webkit-backdrop-filter:blur(12px) saturate(150%);
        box-shadow:inset 0 1px 0 rgba(255,255,255,.28),0 5px 20px rgba(0,0,0,.12);
        font:12px/1.2 ui-monospace,Consolas,monospace;cursor:pointer}
      .polybot-ai-hud-toggle:hover{background:rgba(125,175,215,.28)}
      .polybot-ai-hud{position:fixed;z-index:2147483647;left:12px;top:12px;
        width:min(454px,calc(100vw - 24px));max-height:calc(100vh - 24px);box-sizing:border-box;
        overflow:auto;display:flex;flex-direction:column;transform-origin:top left;container-type:inline-size;
        opacity:1;transition:opacity 140ms ease;
        color:#f1f7fc;background:linear-gradient(135deg,rgba(230,247,255,.14),rgba(40,70,105,.08)),rgba(9,19,32,.26);
        border:1px solid rgba(235,248,255,.3);border-radius:20px;padding:14px 16px;font:12px/1.35 ui-monospace,Consolas,monospace;
        backdrop-filter:blur(12px) saturate(150%);-webkit-backdrop-filter:blur(12px) saturate(150%);
        text-shadow:0 1px 3px rgba(0,0,0,.8);
        box-shadow:inset 0 1px 0 rgba(255,255,255,.34),inset 0 -1px 0 rgba(255,255,255,.07),0 12px 38px rgba(0,0,0,.18);pointer-events:auto;overscroll-behavior:contain}
      .polybot-ai-hud>*{flex-shrink:0;min-width:0}
      .polybot-ai-hud,.polybot-ai-hud *{scrollbar-width:thin;scrollbar-color:rgba(223,244,255,.48) transparent}
      @supports selector(::-webkit-scrollbar){.polybot-ai-hud,.polybot-ai-hud *{scrollbar-width:auto;scrollbar-color:auto}}
      .polybot-ai-hud::-webkit-scrollbar,.polybot-ai-hud *::-webkit-scrollbar{width:7px;height:7px;background:transparent}
      .polybot-ai-hud::-webkit-scrollbar-track,.polybot-ai-hud *::-webkit-scrollbar-track{background:transparent;margin:12px 0}
      .polybot-ai-hud::-webkit-scrollbar-thumb,.polybot-ai-hud *::-webkit-scrollbar-thumb{border-radius:999px;background:linear-gradient(180deg,rgba(240,252,255,.55),rgba(170,217,241,.28));border:1px solid rgba(255,255,255,.25)}
      .polybot-ai-hud::-webkit-scrollbar-button,.polybot-ai-hud *::-webkit-scrollbar-button{display:none;width:0;height:0}
      .polybot-ai-hud::-webkit-scrollbar-corner,.polybot-ai-hud *::-webkit-scrollbar-corner{background:transparent}
      .polybot-ai-hud-grip,.polybot-ai-hud-resize{border:0!important;background:transparent!important;color:rgba(235,248,255,.65)!important;padding:0!important;min-height:16px;touch-action:none;user-select:none}
      .polybot-ai-hud-grip{order:-2;cursor:grab;position:sticky;top:0;width:100%;z-index:2;line-height:12px}
      .polybot-ai-hud-grip:active{cursor:grabbing}
      .polybot-ai-hud-grip::after{content:"";display:block;width:32px;height:3px;border-radius:999px;background:rgba(235,248,255,.35);margin:auto}
      .polybot-ai-hud-resize{order:1000;position:sticky;bottom:0;align-self:flex-end;cursor:nwse-resize;width:22px;height:18px;font-size:18px!important;line-height:18px}
      .polybot-ai-hud-grip:focus-visible,.polybot-ai-hud-resize:focus-visible{outline:1px solid rgba(235,248,255,.75)!important;border-radius:6px}
      .polybot-ai-hud[hidden],.polybot-ai-hud [hidden]{display:none!important}
      .polybot-ai-hud,.polybot-ai-hud *{font-family:ui-monospace,Consolas,monospace!important;font-style:normal!important}
      .polybot-ai-hud-full{width:min(814px,calc(100vw - 24px))}
      .polybot-ai-hud-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:0 12px}
      .polybot-ai-hud-title{font-size:14px;font-weight:700;color:#fff}
      .polybot-ai-hud-meta{color:#e0ebf5;margin-top:2px}
      .polybot-ai-hud-section{border-top:1px solid rgba(156,190,216,.24);margin-top:7px;padding-top:5px}
      .polybot-ai-hud-heading{font-weight:700;color:#8ed7ff;margin-bottom:2px}
      .polybot-ai-hud-row{display:grid;grid-template-columns:minmax(105px,1fr) auto minmax(48px,.6fr);
        align-items:center;gap:6px;min-height:17px}
      .polybot-ai-hud-label{color:#c9d6e1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
      .polybot-ai-hud-value{text-align:right;color:#fff;white-space:nowrap}
      .polybot-ai-hud-bar{height:4px;background:rgba(211,232,250,.18);border-radius:4px;overflow:hidden}
      .polybot-ai-hud-fill{height:100%;width:0;background:#49c98c}
      .polybot-ai-hud-events{color:#ffd166;font-weight:700;margin-top:4px}
      .polybot-ai-hud-error{color:#ff8c84}
      .polybot-ai-hud-scroll{max-height:56vh;overflow:auto}
      .polybot-ai-hud-mode{position:fixed;z-index:2147483647;right:12px;top:53px;background:rgba(15,28,44,.28);color:#edf4fa;border:1px solid rgba(235,248,255,.38);border-radius:12px;padding:8px;font:12px ui-monospace,monospace;backdrop-filter:blur(12px) saturate(150%);-webkit-backdrop-filter:blur(12px) saturate(150%);box-shadow:inset 0 1px 0 rgba(255,255,255,.28),0 5px 20px rgba(0,0,0,.12)}
      .polybot-ai-hud-mode option{background:#16273a;color:#edf4fa}
      .polybot-ai-hud-hero{font-size:34px;font-weight:700;font-variant-numeric:tabular-nums;color:#fff}
      .polybot-ai-hud-large .polybot-ai-hud-value{font-size:24px}
      .polybot-ai-hud-large .polybot-ai-hud-row{min-height:40px}
      .polybot-ai-hud-large .polybot-ai-hud-bar{height:12px}
      .polybot-ai-hud-diagram{width:100%;height:160px}
      .polybot-ai-hud-network{height:260px;overflow:visible}
      .polybot-ai-hud-network text{fill:#f1f7fc;font-size:11px;font-weight:500}
      .polybot-ai-hud-network .polybot-ai-hud-node-value{fill:#bce8ff;font-size:10px}
      .polybot-ai-hud-network .polybot-ai-hud-node-heading{fill:#bce8ff;font-size:10px;letter-spacing:1.4px}
      .polybot-ai-hud-neuron{transition:fill .18s, r .18s}
      .polybot-ai-hud-small{width:290px}
      .polybot-ai-hud-clean .polybot-ai-hud-section{border:0;margin-top:3px}
      .polybot-ai-hud-clean .polybot-ai-hud-heading{display:none}
      @container(max-width:330px){.polybot-ai-hud-row{grid-template-columns:minmax(60px,1fr) auto;gap:4px}.polybot-ai-hud-bar{display:none}.polybot-ai-hud-network{height:auto;aspect-ratio:480/280}.polybot-ai-hud-hero{font-size:28px}}
      @media(prefers-reduced-motion:reduce){.polybot-ai-hud{transition:none}}
    `;
    toggle = element("button", "polybot-ai-hud-toggle", body);
    toggle.type = "button";
    toggle.setAttribute("aria-pressed", "false");
    toggle.setAttribute("aria-label", "Show AI HUD");
    toggle.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      hudVisible = !hudVisible;
      syncVisibility();
      if (hudVisible) render(latestFrame);
    });
    modeSelector = element("select", "polybot-ai-hud-mode", body);
    modeSelector.setAttribute("aria-label", "AI HUD mode");
    for (const mode of HUD_MODES) {
      const option = element("option", "", modeSelector);
      option.value = mode; text(option, mode.replaceAll("_", " "));
    }
    modeSelector.value = "RL_DEBUG";
    modeSelector.addEventListener("change", () => {
      modeOverride = modeSelector.value;
      hudVisible = modeOverride !== "OFF";
      render(latestFrame);
      syncVisibility();
    });
    root = element("div", "polybot-ai-hud", body);
    root.hidden = true;
    root.setAttribute("aria-hidden", "true");
    installWindowControls();
    const title = element("div", "polybot-ai-hud-title", root);
    const meta = element("div", "polybot-ai-hud-meta", root);
    sections = {
      title,
      meta,
      status: makeSection(root, "Episode"),
      inputs: makeSection(root, "Policy inputs"),
      controls: makeSection(root, "Policy output / applied controls"),
      reward: makeSection(root, "Reward"),
      event: element("div", "polybot-ai-hud-events", root),
    };
    sections.title.textContent = "POLYBOT · AI HUD";
    sections.meta.textContent = "Waiting for Python telemetry";
    for (const section of [sections.status, sections.inputs, sections.controls, sections.reward]) {
      section.section.hidden = true;
    }
    sections.event.hidden = true;
    extra = makeSection(root, "");
    extra.section.style.order = "-1";
    extra.body.style.overflow = "hidden";
    extra.body.style.maxHeight = "none";
    extra.section.hidden = true;
    renderImmediate(null);
    return true;
  }

  function applyLayout() {
    if (!root) return;
    const saved = layouts[layoutMode] || {};
    const viewportWidth = host.innerWidth || 1280, viewportHeight = host.innerHeight || 720;
    const availableWidth = Math.max(120, (viewportWidth - 24) / panelScale);
    const availableHeight = Math.max(100, (viewportHeight - 24) / panelScale);
    root.style.maxWidth = `${availableWidth}px`;
    root.style.maxHeight = `${availableHeight}px`;
    root.style.width = Number.isFinite(saved.width) ? `${Math.min(availableWidth, Math.max(220,saved.width))}px` : "";
    root.style.height = Number.isFinite(saved.height) ? `${Math.min(availableHeight, Math.max(120,saved.height))}px` : "";
    const width = root.getBoundingClientRect?.().width || 454;
    const height = root.getBoundingClientRect?.().height || 300;
    root.style.left = `${Math.max(12,Math.min(Number.isFinite(saved.x)?saved.x:12,viewportWidth-width-12))}px`;
    root.style.top = `${Math.max(12,Math.min(Number.isFinite(saved.y)?saved.y:12,viewportHeight-height-12))}px`;
    if (sections) {
      const heightLimit = Number.isFinite(saved.height) ? Math.max(80,Math.min(saved.height*.45,availableHeight*.45)) : availableHeight*.34;
      sections.inputs.body.style.maxHeight = `${heightLimit}px`;
    }
  }

  function installWindowControls() {
    const grip = element("button", "polybot-ai-hud-grip", root);
    const resize = element("button", "polybot-ai-hud-resize", root);
    grip.type = resize.type = "button";
    grip.setAttribute("aria-label", "Move HUD window");
    resize.setAttribute("aria-label", "Resize HUD window");
    grip.title = "Drag to move · double-click to reset · arrow keys to move";
    resize.title = "Drag to resize · arrow keys to resize";
    text(resize, "\u2921");
    const save = () => {
      try { host.localStorage?.setItem("polybot.hud.layouts.v1",JSON.stringify(layouts)); } catch {}
    };
    for (const [control,resizing] of [[grip,false],[resize,true]]) {
      control.addEventListener("pointerdown", event => {
        if (event.button !== 0) return;
        event.preventDefault(); event.stopPropagation();
        const bounds = root.getBoundingClientRect();
        const startX = event.clientX, startY = event.clientY;
        const initial = {x:bounds.left,y:bounds.top,width:bounds.width/panelScale,height:bounds.height/panelScale};
        control.setPointerCapture(event.pointerId);
        const move = e => {
          e.preventDefault(); e.stopPropagation();
          layouts[layoutMode] = resizing
            ? {...initial,width:initial.width+(e.clientX-startX)/panelScale,height:initial.height+(e.clientY-startY)/panelScale}
            : {...(layouts[layoutMode] || {}),x:initial.x+e.clientX-startX,y:initial.y+e.clientY-startY};
          applyLayout();
        };
        const end = e => {
          e.stopPropagation();
          control.removeEventListener("pointermove",move);
          control.removeEventListener("pointerup",end);
          control.removeEventListener("pointercancel",end);
          save();
        };
        control.addEventListener("pointermove",move);
        control.addEventListener("pointerup",end);
        control.addEventListener("pointercancel",end);
      });
      control.addEventListener("keydown", event => {
        event.stopPropagation();
        if (!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown"].includes(event.key)) return;
        event.preventDefault();
        const bounds = root.getBoundingClientRect(), step = event.shiftKey ? 40 : 10;
        const saved = {...(layouts[layoutMode] || {})};
        const horizontal = ["ArrowLeft","ArrowRight"].includes(event.key);
        const amount = ["ArrowLeft","ArrowUp"].includes(event.key) ? -step : step;
        const key = resizing ? (horizontal?"width":"height") : (horizontal?"x":"y");
        saved[key] = (saved[key] ?? (resizing ? (horizontal?bounds.width:bounds.height)/panelScale : (horizontal?bounds.left:bounds.top))) + amount;
        layouts[layoutMode] = saved; applyLayout(); save();
      });
    }
    grip.addEventListener("dblclick", event => {
      event.preventDefault(); event.stopPropagation();
      delete layouts[layoutMode]; applyLayout(); save();
    });
    for (const type of ["pointerdown","pointerup","click","dblclick","wheel","keydown","keyup"]) {
      root.addEventListener(type,event => event.stopPropagation());
    }
    host.addEventListener?.("resize", applyLayout);
  }
  function syncVisibility() {
    if (modeFadeTimer !== null) return;
    const hidden = !hudVisible || (modeOverride || latestFrame?.settings?.display_mode) === "OFF";
    const buttonText = hudVisible ? "Hide AI HUD" : "Show AI HUD";
    if (root.hidden !== hidden) root.hidden = hidden;
    if (root.getAttribute("aria-hidden") !== String(hidden)) {
      root.setAttribute("aria-hidden", String(hidden));
    }
    if (toggle.textContent !== buttonText) toggle.textContent = buttonText;
    if (toggle.getAttribute("aria-label") !== buttonText) {
      toggle.setAttribute("aria-label", buttonText);
    }
    if (toggle.getAttribute("aria-pressed") !== String(hudVisible)) {
      toggle.setAttribute("aria-pressed", String(hudVisible));
    }
  }
  function makeSection(parent, title) {
    const section = element("section", "polybot-ai-hud-section", parent);
    const heading = element("div", "polybot-ai-hud-heading", section);
    text(heading, title);
    const note = element("div", "polybot-ai-hud-error", section);
    note.hidden = true;
    const body = element("div", "polybot-ai-hud-scroll", section);
    return { section, heading, note, body };
  }

  function setSection(section, visible, heading, message) {
    section.section.hidden = !visible;
    if (heading) text(section.heading, heading);
    section.note.hidden = message === undefined;
    section.body.hidden = message !== undefined;
    if (message !== undefined) text(section.note, message);
  }

  function rowFor(key, parent) {
    let row = rows.get(key);
    if (!row) {
      const container = element("div", "polybot-ai-hud-row", parent);
      row = {
        container,
        label: element("span", "polybot-ai-hud-label", container),
        value: element("span", "polybot-ai-hud-value", container),
        bar: element("div", "polybot-ai-hud-bar", container),
        fill: element("div", "polybot-ai-hud-fill", null),
      };
      row.bar.appendChild(row.fill);
      rows.set(key, row);
    } else if (row.container.parentNode !== parent) {
      parent.appendChild(row.container);
    }
    row.container.hidden = false;
    return row;
  }

  function updateRow(key, label, value, rawValue, unit, minimum, maximum, showLabels) {
    const row = rowFor(key, sections.inputs.body);
    row.label.hidden = !showLabels;
    text(row.label, label);
    text(row.value, `${number(value, 3)}${Number.isFinite(rawValue) ? ` (${number(rawValue, 2)} ${unit || ""})` : ""}`);
    const span = maximum - minimum;
    const ratio = span > 0 ? Math.max(0, Math.min(1, (value - minimum) / span)) : 0.5;
    row.fill.style.width = `${ratio * 100}%`;
    row.fill.style.background = value >= 0 ? "#49c98c" : "#eeb647";
  }

  function renderControls(controls, showLabels) {
    const body = sections.controls.body;
    const definitions = [
      ...((controls.model_output || []).map((item) => [`model:${item.label}`, `Model · ${item.label}`, item.value])),
      ...((controls.transformed_output || []).map((item) => [`transformed:${item.label}`, `Transformed · ${item.label}`, item.value])),
      ...(Object.entries(controls.adapter_demand || {}).map(([name, value]) => [
        `adapter:${name}`,
        `Adapter · ${name}`,
        value,
      ])),
      ...(["steer", "throttle", "brake"].map((name) => [
        `applied:${name}`,
        `Applied · ${name}`,
        controls.applied && controls.applied[name],
      ])),
    ];
    const active = new Set();
    for (const [key, label, value] of definitions) {
      if (typeof value !== "number" || !Number.isFinite(value)) continue;
      active.add(key);
      const row = rowFor(key, body);
      row.label.hidden = !showLabels;
      text(row.label, label);
      text(row.value, number(value, 3));
      row.bar.hidden = true;
    }
    hideUnused(body, active);
  }

  function renderReward(reward, showLabels) {
    const body = sections.reward.body;
    const entries = [
      ["episode", "Accumulated reward", reward.episode_learner],
      ["scale", "Reward scale", reward.scale],
      ...Object.entries(reward.episode_learner_groups || {}).map(([key, value]) => [`group:${key}`, key, value]),
      ...Object.entries(reward.episode_learner_terms || {})
        .filter(([, value]) => Math.abs(value) > 1e-12)
        .map(([key, value]) => [`term:${key}`, key, value]),
      ["raw", "Unscaled accumulated reward", reward.episode_raw],
    ];
    const active = new Set();
    for (const [key, label, value] of entries) {
      if (typeof value !== "number" || !Number.isFinite(value)) continue;
      active.add(key);
      const row = rowFor(`reward:${key}`, body);
      row.label.hidden = !showLabels;
      text(row.label, label);
      text(row.value, number(value, 3));
      row.bar.hidden = true;
    }
    hideUnused(body, active, "reward:");
  }

  function hideUnused(parent, active, prefix = "") {
    for (const [key, row] of rows) {
      if (key.startsWith(prefix) && row.container.parentNode === parent) {
        row.container.hidden = !active.has(key.slice(prefix.length));
      }
    }
  }

  function render(frame) {
    const mode = modeOverride || frame?.settings?.display_mode || null;
    if (modeFadeTimer !== null) return;
    const reducedMotion = host.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (displayedMode !== null && mode !== displayedMode && !root.hidden &&
        !reducedMotion && typeof host.setTimeout === "function") {
      root.style.opacity = "0";
      modeFadeTimer = host.setTimeout(() => {
        modeFadeTimer = null;
        displayedMode = modeOverride || latestFrame?.settings?.display_mode || null;
        renderImmediate(latestFrame || frame);
        root.style.opacity = "1";
      }, 140);
      return;
    }
    const wasHidden = root.hidden;
    displayedMode = mode;
    renderImmediate(frame);
    if (wasHidden && !root.hidden && !reducedMotion && root.animate) {
      root.animate([{opacity:0}, {opacity:1}], {duration:140, easing:"ease"});
    }
  }

  function renderImmediate(frame) {
    if (!frame) {
      const mode = modeOverride || "RL_DEBUG";
      root.className = "polybot-ai-hud";
      sections.title.hidden = sections.meta.hidden = false;
      sections.event.hidden = true;
      text(sections.title,mode.replaceAll("_"," "));
      text(sections.meta,"No model or replay loaded"); text(sections.event, "");
      for (const section of [sections.status,sections.inputs,sections.controls,sections.reward,extra]) {
        section.section.hidden = true;
      }
      for (const row of rows.values()) row.container.hidden = true;
      while (extra.body.firstChild) extra.body.removeChild(extra.body.firstChild);
      const placeholders = (section,heading,names) => {
        setSection(section,true,heading);
        for (const name of names) {
          const row = rowFor(`idle:${heading}:${name}`,section.body);
          row.container.hidden = false; row.label.hidden = false; row.bar.hidden = true;
          text(row.label,name); text(row.value,"\u2014");
        }
      };
      if (["RL_DEBUG","OBSERVATIONS","NEURAL_NET"].includes(mode)) {
        placeholders(sections.inputs,"Policy inputs",["Forward speed","Side speed","Heading error","Line offset","Ahead curvature"]);
      }
      if (!["OFF","TRAINING","TRAINING_GRAPH"].includes(mode)) {
        placeholders(sections.controls,"Controls",["Steering","Throttle","Brake"]);
      }
      if (["RL_DEBUG","REWARD"].includes(mode)) {
        placeholders(sections.reward,"Reward",["Accumulated reward"]);
      }
      if (["TRAINING","TRAINING_GRAPH"].includes(mode)) {
        placeholders(sections.status,"Training",["Model","Stage","Decisions",...(mode === "TRAINING_GRAPH"?["Lap history"]:[])]);
      } else if (!["OFF","RL_DEBUG","REWARD","OBSERVATIONS","CONTROLS","DETAILED_CONTROLS"].includes(mode)) {
        placeholders(sections.status,mode === "NEURAL_NET"?"Neural network":"Race",
          mode === "NEURAL_NET"?["Inputs","Policy","Outputs"]:
          mode === "CHAMPION"?["Timer","Best verified"]:
          mode === "WR_CHASE"?["Timer","WR target","Delta"]:["Timer","Delta"]);
      }
      layoutMode = mode;
      applyLayout();
      syncVisibility();
      return;
    }
    const settings = frame.settings || {};
    root.className = settings.preset === "full"
      ? "polybot-ai-hud polybot-ai-hud-full"
      : "polybot-ai-hud";
    root.style.transform = `scale(${number(settings.scale || 1, 2)})`;
    sections.title.hidden = settings.show_episode_status === false;
    const mode = frame.mode === "replay_swarm" ? "Replay swarm" : String(frame.mode || "AI");
    text(sections.title, `POLYBOT · ${mode.toUpperCase()}`);
    const values = [
      frame.track,
      frame.algorithm && String(frame.algorithm).toUpperCase(),
      frame.model_slot,
      frame.training_step != null ? `policy decisions ${Number(frame.training_step).toLocaleString()}` : null,
      frame.policy_checkpoint_step != null ? `checkpoint ${Number(frame.policy_checkpoint_step).toLocaleString()}` : null,
      frame.episode_total != null ? `episode ${frame.episode}/${frame.episode_total}` : frame.episode != null ? `episode ${frame.episode}` : null,
      frame.simulator_tick != null ? `sim tick ${Number(frame.simulator_tick).toLocaleString()}` : null,
      frame.elapsed_simulation_s != null ? `${number(frame.elapsed_simulation_s, 1)} s` : null,
      frame.frame_skip != null ? `skip ${frame.frame_skip}` : null,
    ].filter(Boolean);
    text(sections.meta, values.join(" · "));
    sections.meta.hidden = settings.show_episode_status === false;
    setSection(sections.status, settings.show_episode_status !== false, "Episode",
      `${String(frame.status || "running").toUpperCase()} · decision ${frame.decision_count ?? 0}` +
      (frame.progress_m != null ? ` · ${number(frame.progress_m, 1)} m` : "") +
      (frame.lap_time_s != null ? ` · lap ${number(frame.lap_time_s, 2)} s` : ""));
    const schemaSupported =
      frame.feature_schema === supportedFeatureSchema &&
      supportedObservationSchemas.has(frame.observation_schema);
    const showInputs = settings.show_observations !== false &&
      (Array.isArray(frame.features) || Boolean(frame.observation_schema_error));
    const features = showInputs && schemaSupported ? frame.features : [];
    if (showInputs && frame.observation_schema_error) {
      setSection(sections.inputs, true, "Policy inputs", frame.observation_schema_error);
      sections.inputs.body.className = "polybot-ai-hud-scroll polybot-ai-hud-error";
    } else if (showInputs && !schemaSupported) {
      setSection(sections.inputs, true, "Policy inputs",
        `Unsupported observation schema: ${frame.observation_schema || frame.feature_schema || "missing"}`);
      sections.inputs.body.className = "polybot-ai-hud-scroll polybot-ai-hud-error";
    } else {
      sections.inputs.body.className = settings.preset === "full"
        ? "polybot-ai-hud-scroll polybot-ai-hud-grid"
        : "polybot-ai-hud-scroll";
      const compactKeys = new Set([
        "velocity.forward", "acceleration.forward", "angular_velocity.up",
        "route.lateral_offset", "route.heading_error", "vehicle.pitch", "vehicle.roll",
        "controls.actual_steering", "wheel_1.contact", "wheel_2.contact",
        "wheel_3.contact", "wheel_4.contact",
      ]);
      const sourceFeatures = settings.display_mode || modeOverride ? importantInputs(features) : features;
      const visible = sourceFeatures.filter((feature) => {
        if (settings.display_mode || modeOverride) return true;
        if (feature.group === "lookahead") {
          const match = /^lookahead\.(\d+)\./.exec(feature.key);
          return Boolean(match && Number(match[1]) < (settings.lookahead_points ?? 0));
        }
        return Boolean(settings.display_mode || modeOverride) || settings.preset === "full" || compactKeys.has(feature.key);
      });
      setSection(sections.inputs, settings.show_observations !== false, "Policy inputs");
      const active = new Set();
      for (const feature of visible) {
        active.add(feature.key);
        updateRow(
          feature.key,
          feature.label,
          feature.value,
          feature.raw_value,
          feature.unit,
          feature.minimum,
          feature.maximum,
          settings.show_labels !== false,
        );
      }
      hideUnused(sections.inputs.body, active);
    }
    const showControls = settings.show_controls !== false && frame.controls;
    setSection(sections.controls, Boolean(showControls), "Policy output / applied controls");
    if (showControls) renderControls(frame.controls, settings.show_labels !== false);
    const showReward = settings.show_reward_breakdown !== false && frame.reward;
    setSection(sections.reward, Boolean(showReward), "Reward");
    if (showReward) renderReward(frame.reward, settings.show_labels !== false);
    const events = settings.show_event_popups === false ? [] : (frame.events || []);
    const eventText = events.length ? events.join(" · ").replaceAll("_", " ").toUpperCase() : "";
    if (eventText && eventText !== lastEvent) {
      lastEvent = eventText;
      text(sections.event, eventText);
    } else if (!eventText) {
      lastEvent = "";
      text(sections.event, "");
    }
    sections.event.hidden = !eventText;
    renderMode(frame, settings, features);
    panelScale = Math.max(.5,Math.min(1.5,Number(settings.scale) || 1));
    layoutMode = modeOverride || settings.display_mode || "RL_DEBUG";
    applyLayout();
    syncVisibility();
  }

  function renderMode(frame, settings, features) {
    const mode = modeOverride || settings.display_mode;
    if (!mode) { extra.section.hidden = true; return; } // Legacy frame compatibility.
    modeSelector.value = mode;
    const inputs = ["RL_DEBUG", "OBSERVATIONS", "NEURAL_NET"].includes(mode);
    const rewards = ["RL_DEBUG", "REWARD"].includes(mode);
    const training = ["TRAINING", "TRAINING_GRAPH"].includes(mode);
    const clean = ["CONTROLS", "DETAILED_CONTROLS", "CHAMPION", "MINIMAL_RACE", "WR_CHASE"].includes(mode);
    sections.inputs.body.style.maxHeight = mode === "RL_DEBUG" ? "25vh" : "34vh";
    sections.inputs.body.className = settings.preset === "full"
      ? "polybot-ai-hud-scroll polybot-ai-hud-grid" : "polybot-ai-hud-scroll";
    sections.inputs.section.hidden = !inputs || settings.show_observations === false;
    if (inputs && (!Array.isArray(frame.features) || !frame.features.length) && !frame.observation_schema_error) {
      setSection(sections.inputs, settings.show_observations !== false, "Policy inputs",
        "Inputs unavailable");
    }
    sections.reward.section.hidden = !rewards || settings.show_reward_breakdown === false;
    sections.controls.section.hidden = training || mode === "OFF" || settings.show_controls === false || !frame.controls;
    if (!frame.controls && !training && mode !== "OFF" && settings.show_controls !== false) {
      setSection(sections.controls, true, "Controls", "Controls unavailable");
    }
    sections.status.section.hidden = mode !== "RL_DEBUG" || settings.show_episode_status === false;
    sections.title.hidden = clean || mode === "OFF";
    sections.meta.hidden = !training && mode !== "RL_DEBUG";
    sections.event.hidden = mode !== "RL_DEBUG" || !frame.events?.length || settings.show_event_popups === false;
    text(sections.title, mode.replaceAll("_", " "));
    root.className += clean ? " polybot-ai-hud-clean polybot-ai-hud-small" : "";
    if (["CONTROLS", "DETAILED_CONTROLS"].includes(mode)) root.className += " polybot-ai-hud-large";
    // Compact modes show the actual applied controls, with model output as a fallback.
    if (frame.controls && mode !== "RL_DEBUG") {
      const applied = frame.controls.applied;
      const values = applied ? Object.entries(applied).filter(([k]) => ["steer","throttle","brake"].includes(k))
        : (frame.controls.model_output || []).map(v => [v.label.toLowerCase(), v.value]);
      for (const row of rows.values()) if (row.container.parentNode === sections.controls.body) row.container.hidden = true;
      for (const [name,value] of values) {
        const row = rowFor(`display:${name}`, sections.controls.body);
        text(row.label, name.toUpperCase()); row.label.hidden = false;
        text(row.value, number(value, mode === "DETAILED_CONTROLS" ? 4 : 2));
        row.bar.hidden = false;
        row.fill.style.width = `${Math.max(0,Math.min(1,Math.abs(value)))*100}%`;
        row.fill.style.background = name === "brake" ? "#ff7676" : "#49c98c";
      }
    }
    setSection(extra, !["OFF", "CONTROLS", "DETAILED_CONTROLS", "RL_DEBUG"].includes(mode), "");
    // Only rebuild the small mode-specific visual; existing input rows are reused.
    while (extra.body.children.length) {
      const child = extra.body.children[0];
      if (extra.body.removeChild) extra.body.removeChild(child);
      else extra.body.children.shift();
    }
    const label = (value, hero=false) => text(element("div", hero ? "polybot-ai-hud-hero" : "polybot-ai-hud-meta", extra.body), value);
    const seconds = value => Number.isFinite(value) ? `${number(value,3)} s` : "\u2014";
    const reference = frame.reference || {};
    const delta = reference.delta_s;
    const showDelta = (value, caption="Delta") => label(`${caption} ${Number.isFinite(value) && value > 0 ? "+" : ""}${seconds(value)}`);
    const time = frame.lap_time_s ?? frame.elapsed_simulation_s;
    // Keep real episode returns, keyed by episode; seeks/reset never create fake improvements.
    const run = `${frame.run_id || ""}:${frame.mode}`;
    if (run !== chartRun) { chart = []; chartRun = run; }
    if (frame.status === "finished" && Number.isFinite(frame.lap_time_s)) {
      const key = `${frame.episode}:${frame.lap_time_s}`;
      if (!chart.some(p => p.key === key)) chart.push({key, value:frame.lap_time_s});
      if (chart.length > 120) chart.shift();
    }
    if (training) {
      label(`${frame.algorithm || "AI"} \u00b7 ${frame.model_slot || "current model"}`);
      label(`${frame.training?.stage || frame.mode || "Training"} \u00b7 ${Number(frame.training_step ?? frame.policy_checkpoint_step ?? 0).toLocaleString()} decisions`, mode === "TRAINING");
      if (frame.training?.budget) label(`Run budget ${Number(frame.training.budget).toLocaleString()} decisions`);
      label(`Episode ${frame.episode ?? "\u2014"} \u00b7 ${frame.status || "\u2014"}`);
      if (mode === "TRAINING_GRAPH") {
        label("Completed lap times \u00b7 this viewing session");
        const graph = svgNode("svg", extra.body, {viewBox:"0 0 360 130", class:"polybot-ai-hud-diagram", role:"img", "aria-label":"Completed lap times graph"});
        if (chart.length < 2) label("Waiting for two completed laps");
        else {
          const low = Math.min(...chart.map(p=>p.value)), high = Math.max(...chart.map(p=>p.value));
          svgNode("polyline",graph,{points:chart.map((p,i)=>`${10+i*340/(chart.length-1)},${110-(p.value-low)/Math.max(.01,high-low)*90}`).join(" "),fill:"none",stroke:"#49c98c","stroke-width":3});
          label(`${seconds(low)} best \u00b7 ${seconds(high)} slowest`);
        }
      }
    } else if (mode === "REWARD") {
      label(number(frame.reward?.episode_learner,3), true); label("Accumulated reward");
    } else if (["NEURAL_NET", "OBSERVATIONS"].includes(mode)) {
      if (mode === "NEURAL_NET") {
        const net = svgNode("svg",extra.body,{viewBox:"0 0 480 280",class:"polybot-ai-hud-diagram polybot-ai-hud-network",role:"img","aria-label":"Labelled live policy input and output schematic"});
        text(svgNode("title",net),"Animated policy schematic; hidden activations are illustrative.");
        const preferred = ["velocity.forward", "velocity.right", "route.heading_error", "route.lateral_offset", "controls.actual_steering", "lookahead.0.curvature", "ghost.target_speed", "wheel_1.contact"];
        const curated = importantInputs(features);
        const chosen = [...curated].sort((a,b) => {
          const rank = f => preferred.includes(f.key) ? preferred.indexOf(f.key) : preferred.length;
          return rank(a)-rank(b);
        }).slice(0,8);
        const output = frame.controls?.model_output || [];
        const columns = [chosen.map(f=>f.value), Array.from({length:6},(_,i)=>Math.sin((frame.decision_count || 0)*.3+i)*.5), output.map(v=>v.value)];
        const y = (c,i) => columns[c].length === 1 ? 150 : 44+i*210/Math.max(1,columns[c].length-1);
        for (let c=0;c<2;c++) columns[c].forEach((v,i)=>columns[c+1].forEach((w,j)=>svgNode("line",net,{x1:170+c*80,y1:y(c,i),x2:250+c*80,y2:y(c+1,j),stroke:"#b6d6ef","stroke-opacity":.15+Math.abs(w)*.28})));
        columns.forEach((col,c)=>col.forEach((v,i)=>svgNode("circle",net,{cx:170+c*80,cy:y(c,i),r:4+Math.min(1,Math.abs(v))*4,fill:v>=0?"#49c98c":"#ffba67",class:"polybot-ai-hud-neuron"})));
        const names = {"velocity.forward":"Forward speed", "velocity.right":"Side speed", "route.heading_error":"Heading error", "route.lateral_offset":"Line offset", "controls.actual_steering":"Steering angle", "lookahead.0.curvature":"Ahead curvature", "ghost.target_speed":"Target speed", "wheel_1.contact":"Front wheel contact"};
        const nodeLabel = (name,value,x,cy,anchor,description) => {
          const title = svgNode("text",net,{x,y:cy-2,"text-anchor":anchor}); text(title,name);
          text(svgNode("title",title),description || name);
          text(svgNode("text",net,{x,y:cy+12,"text-anchor":anchor,class:"polybot-ai-hud-node-value"}),number(value,3));
        };
        for (const [name,x,anchor] of [["INPUTS",154,"end"],["POLICY",250,"middle"],["OUTPUTS",346,"start"]]) {
          text(svgNode("text",net,{x,y:16,"text-anchor":anchor,class:"polybot-ai-hud-node-heading"}),name);
        }
        chosen.forEach((f,i)=>nodeLabel(names[f.key] || f.label,f.value,154,y(0,i),"end",`${f.label}: normalized policy input`));
        output.forEach((o,i)=>nodeLabel(o.label,o.value,346,y(2,i),"start",`${o.label}: raw policy output`));
      } else {
        label("Lookahead");
        const plot=svgNode("svg",extra.body,{viewBox:"0 0 360 160",class:"polybot-ai-hud-diagram",role:"img","aria-label":"Upcoming track lookahead"});
        const lookup = new Map(features.map(f=>[f.key,f.raw_value]));
        const points=[];
        for(let i=0;i<(settings.lookahead_points ?? 3);i++) {
          if(lookup.get(`lookahead.${i}.mask`) === 0) continue;
          const forward=lookup.get(`lookahead.${i}.forward`), right=lookup.get(`lookahead.${i}.right`);
          if(Number.isFinite(forward)&&Number.isFinite(right)) points.push([right,forward]);
        }
        const extent=Math.max(10,...points.flat().map(Math.abs));
        svgNode("polyline",plot,{points:[[180,145],...points.map(([r,f])=>[180+r/extent*140,145-f/extent*130])].map(p=>p.join(",")).join(" "),fill:"none",stroke:"#8ed7ff","stroke-width":3});
        svgNode("circle",plot,{cx:180,cy:145,r:5,fill:"#49c98c"});
        if(!points.length) label("Lookahead unavailable");
      }
    } else {
      label(seconds(time), true);
      if (mode === "WR_CHASE") {
        const target=settings.wr_target_s > 0 ? settings.wr_target_s : null;
        label(`WR target ${seconds(target)}`);
        // A target lap alone cannot supply a truthful live same-position delta.
        const wrDelta = frame.wr_delta_s ?? (target && Math.abs(target-reference.target_time_s) < .0005 ? reference.delta_s : null);
        showDelta(wrDelta, "Live WR delta");
        if(!Number.isFinite(wrDelta)) label("WR split trace unavailable");
        else if(reference.delta_method) label(`Delta estimate: ${reference.delta_method}`);
      } else {
        showDelta(delta);
        if (["GHOST_RACE","COMPARISON","RACING_LINE_ANALYSIS"].includes(mode)) {
          label(`Reference ${reference.name || "\u2014"} \u00b7 target ${seconds(reference.target_time_s)}`);
          label(`Checkpoint ${frame.checkpoint_index ?? "\u2014"} \u00b7 reference split ${seconds(reference.elapsed_s)}`);
          if (reference.delta_method) label(`Delta estimate: ${reference.delta_method}`);
        }
        if(mode === "COMPARISON") {
          label(`AI ${number(frame.speed_mps,1)} m/s \u00b7 reference ${number(frame.reference_speed_mps,1)} m/s`);
        }
        if(mode === "CHAMPION") label(`Best verified ${seconds(frame.best_time_s)}`);
        if(mode === "RACING_LINE_ANALYSIS") {
          const key=`${run}:${frame.episode}:${frame.checkpoint_index}`;
          if(frame.reset || (sectionEntry && time<sectionEntry.time)) { sectionEntry=null; previousSection=null; lastSection=null; }
          if(key !== lastSection) {
            if(sectionEntry) previousSection={entry:sectionEntry.speed,exit:frame.speed_mps,delta:Number.isFinite(delta)&&Number.isFinite(sectionEntry.delta)?delta-sectionEntry.delta:null};
            sectionEntry={time, speed:frame.speed_mps, delta}; lastSection=key;
          }
          label(`Checkpoint section \u00b7 entry ${number(sectionEntry?.speed,1)} m/s`);
          label(`Previous section: entry ${number(previousSection?.entry,1)} \u00b7 exit ${number(previousSection?.exit,1)} m/s`);
          showDelta(previousSection?.delta,"Previous section delta");
          label(`Line offset ${number(frame.lateral_offset_m,2)} m \u00b7 heading ${number(frame.heading_error_rad,2)} rad`);
        }
      }
    }
  }

  function update(frame) {
    if (!ensureRoot()) return;
    if (
      frame && typeof frame === "object" &&
      frame.schema === supportedFrameSchema
    ) {
      const mode = modeOverride || frame.settings?.display_mode;
      // Replays can present the same policy-boundary frame across many render ticks.
      if (frame === latestFrame && mode === lastRenderedMode) return;
      latestFrame = frame;
      render(latestFrame);
      lastRenderedMode = mode;
    }
  }

  update.clear = (source) => {
    const replay = ["replay","replay_swarm"].includes(latestFrame?.mode);
    if ((source === "replay" && !replay) || (source === "live" && (!latestFrame || replay))) return;
    if (modeFadeTimer !== null) { host.clearTimeout(modeFadeTimer); modeFadeTimer = null; }
    latestFrame = null; lastRenderedMode = null; displayedMode = null;
    chart = []; chartRun = null; sectionEntry = previousSection = lastSection = null;
    lastEvent = "";
    if (ensureRoot()) { root.style.opacity = "1"; renderImmediate(null); }
  };

  if (host.document.readyState === "loading") {
    host.document.addEventListener("DOMContentLoaded", () => {
      if (ensureRoot()) syncVisibility();
    }, { once: true });
  } else if (ensureRoot()) {
    syncVisibility();
  }
  return update;
}
