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
  let sections = null;
  let hideTimer = null;
  let lastEvent = "";

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
    if (root) return;
    const style = element("style", "", host.document.head || host.document.body);
    style.textContent = `
      .polybot-ai-hud{position:fixed;z-index:2147483647;left:12px;top:12px;
        width:min(420px,45vw);max-height:88vh;overflow:hidden;transform-origin:top left;
        color:#edf4fa;background:rgba(10,18,27,.91);border:1px solid rgba(156,190,216,.42);
        border-radius:8px;padding:10px 12px;font:12px/1.35 ui-monospace,Consolas,monospace;
        box-shadow:0 5px 22px rgba(0,0,0,.3);pointer-events:none}
      .polybot-ai-hud[hidden]{display:none}
      .polybot-ai-hud-full{width:min(780px,78vw)}
      .polybot-ai-hud-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0 8px}
      .polybot-ai-hud-title{font-size:14px;font-weight:700;color:#fff}
      .polybot-ai-hud-meta{color:#b8c9d8;margin-top:2px}
      .polybot-ai-hud-section{border-top:1px solid rgba(156,190,216,.24);margin-top:7px;padding-top:5px}
      .polybot-ai-hud-heading{font-weight:700;color:#8ed7ff;margin-bottom:2px}
      .polybot-ai-hud-row{display:grid;grid-template-columns:minmax(105px,1fr) auto minmax(48px,.6fr);
        align-items:center;gap:6px;min-height:17px}
      .polybot-ai-hud-label{color:#c9d6e1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
      .polybot-ai-hud-value{text-align:right;color:#fff;white-space:nowrap}
      .polybot-ai-hud-bar{height:4px;background:#344351;border-radius:4px;overflow:hidden}
      .polybot-ai-hud-fill{height:100%;width:0;background:#49c98c}
      .polybot-ai-hud-events{color:#ffd166;font-weight:700;margin-top:4px}
      .polybot-ai-hud-error{color:#ff8c84}
      .polybot-ai-hud-scroll{max-height:56vh;overflow:hidden}
    `;
    root = element("div", "polybot-ai-hud", host.document.body);
    root.setAttribute("aria-hidden", "true");
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
      ["step", "Step reward", reward.learner_step],
      ["episode", "Episode return", reward.episode_learner],
      ["scale", "Reward scale", reward.scale],
      ...Object.entries(reward.learner_groups || {}).map(([key, value]) => [`group:${key}`, key, value]),
      ...Object.entries(reward.learner_terms || {})
        .filter(([, value]) => Math.abs(value) > 1e-12)
        .map(([key, value]) => [`term:${key}`, key, value]),
      ["raw", "Raw reward", reward.raw_step],
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

  function update(frame) {
    if (!frame || frame.enabled !== true) {
      if (root) root.hidden = true;
      if (hideTimer !== null && host.clearTimeout) host.clearTimeout(hideTimer);
      hideTimer = null;
      return;
    }
    ensureRoot();
    root.hidden = false;
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
      const visible = features.filter((feature) => {
        if (feature.group === "lookahead") {
          const match = /^lookahead\.(\d+)\./.exec(feature.key);
          return Boolean(match && Number(match[1]) < (settings.lookahead_points ?? 0));
        }
        return settings.preset === "full" || compactKeys.has(feature.key);
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
    if (hideTimer !== null && host.clearTimeout) host.clearTimeout(hideTimer);
    if (host.setTimeout) {
      hideTimer = host.setTimeout(() => {
        if (root) root.hidden = true;
        hideTimer = null;
      }, 1500);
    }
  }

  return update;
}
