/* LexSynergy 前端：只做三件事——把答案流出来、把证据列清楚、把协作过程画成时间线。
 * 没有框架、没有构建工具：一个文件，浏览器直接跑。
 * 事件与 Pipeline 内部事件一一对应（meta → plan → agent×N → sources → verify → delta → done），
 * 所以页面上的时间线就是后端实际发生的事，不是另行编的动画。 */

const $ = (id) => document.getElementById(id);

/* 流式答案缓冲区：append=true 追加分片，append=false 用终稿覆盖。
 * 终稿覆盖是必须的 —— 草稿还没经过引文对位与五道闸，
 * 校验后的完整文本才允许留在页面上。 */
let answerText = "";

const CHIPS = [
  "试用期最长可以约定多久？",
  "公司裁员，我 2021 年 3 月入职，月薪 12000，能拿多少经济补偿？",
  "竞业限制最长能约定几年？",
  "我们公司的员工手册规定试用期 6 个月，这条合规吗？",
  "公司没提前 30 天通知就让我走，除了补偿还有别的钱吗？",
];

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ESC[c]);

/* 把《法》第X条 高亮成引用样式（先转义再替换，避免注入） */
function renderAnswer(text) {
  const safe = esc(text);
  return safe.replace(/(《[^》]{2,40}》\s*第[一二三四五六七八九十百千零〇两0-9]{1,8}条)/g,
    '<span class="cite">$1</span>');
}

function initChips() {
  const box = $("chips");
  CHIPS.forEach((q) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = q.length > 18 ? q.slice(0, 18) + "…" : q;
    b.title = q;
    b.onclick = () => { $("query").value = q; };
    box.appendChild(b);
  });
}

function badge(text, cls) {
  const span = document.createElement("span");
  span.className = "tag " + (cls || "muted");
  span.textContent = text;
  return span;
}

function resetUI() {
  $("badges").innerHTML = "";
  $("badges").appendChild(badge("进行中…", "muted"));
  $("timeline").innerHTML = "";
  $("evidence").innerHTML = '<li class="empty">尚未检索</li>';
  $("verify").innerHTML = '<span class="muted">尚未校验</span>';
  $("answer").textContent = "";
  $("answer").classList.remove("rejected");
  answerText = "";
  $("timeline-count").textContent = "";
  $("ev-count").textContent = "";
}

/* ---------------- 时间线 ---------------- */
function addTimeline({ name, meta, cls, key }) {
  const li = document.createElement("li");
  if (cls) li.className = cls;
  li.dataset.key = key || "";
  li.innerHTML = `<div class="name">${esc(name)}</div><div class="meta">${esc(meta || "")}</div>`;
  $("timeline").appendChild(li);
  $("timeline-count").textContent = `(${$("timeline").children.length})`;
  return li;
}

function updateTimeline(key, meta, cls) {
  const li = $("timeline").querySelector(`li[data-key="${key}"]`);
  if (!li) return addTimeline({ name: key, meta, cls, key });
  if (cls) li.className = cls;
  li.querySelector(".meta").textContent = meta || "";
  return li;
}

const AGENT_LABEL = {
  statute: "法条检索专家",
  calc: "金额测算专家",
  compliance: "合规审查专家",
  verifier: "引用校验专家",
};

/* ---------------- 证据 ---------------- */
function renderEvidence(sources) {
  const box = $("evidence");
  box.innerHTML = "";
  $("ev-count").textContent = sources.length ? `(${sources.length})` : "";
  if (!sources.length) {
    box.innerHTML = '<li class="empty">本次未检索到证据</li>';
    return;
  }
  sources.forEach((s) => {
    const li = document.createElement("li");
    const score = s.score == null ? "" : ` · 相关度 ${Number(s.score).toFixed(2)}`;
    li.innerHTML =
      `<span class="eid">${esc(s.eid || "")}</span>` +
      `<strong>《${esc(s.law || "")}》${esc(s.article_no || "")}</strong>` +
      `<span class="muted">${esc(score)}</span>` +
      `<span class="quote">${esc((s.quote || "").slice(0, 90))}${(s.quote || "").length > 90 ? "…" : ""}</span>`;
    box.appendChild(li);
  });
}

/* ---------------- 校验 ---------------- */
function renderVerify(v) {
  const box = $("verify");
  if (!v) { box.innerHTML = '<span class="muted">未校验</span>'; return; }
  const passed = v.passed === true;
  const head = passed
    ? '<span class="tag ok">五道闸通过</span>'
    : '<span class="tag bad">未通过</span>';
  const skipped = (v.skipped_gates || []).length
    ? `<div class="muted" style="margin-top:6px">跳过：<span class="gate">${esc((v.skipped_gates || []).join(", "))}</span></div>`
    : "";
  const fails = (v.fails || []).length
    ? `<ul>${v.fails.map((f) => `<li><span class="gate">${esc(f.gate)}</span> ${esc(f.detail || "")}${f.citation ? " · " + esc(f.citation) : ""}</li>`).join("")}</ul>`
    : "";
  const repaired = (v.repaired || []).length
    ? `<div class="muted" style="margin-top:6px">引文对位：${v.repaired.map((r) => `${esc(r.from.split("#")[1] || "")}→${esc(r.to.split("#")[1] || "")}`).join("；")}</div>`
    : "";
  const flags = [];
  if (v.no_rewrite) flags.push("补证据后原稿即通过，未重写");
  if (v.timeout) flags.push("超时降级为模板");
  const flagHtml = flags.length ? `<div class="muted" style="margin-top:6px">${esc(flags.join("；"))}</div>` : "";
  box.innerHTML = head + skipped + fails + repaired + flagHtml;
}

/* ---------------- SSE ---------------- */
function parseFrame(raw) {
  let event = "message";
  let data = "";
  raw.split("\n").forEach((line) => {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  });
  if (!data) return null;
  try { return { event, data: JSON.parse(data) }; } catch { return null; }
}

async function ask() {
  const query = $("query").value.trim();
  if (!query) return;

  let facts = null;
  const rawFacts = $("facts").value.trim();
  if (rawFacts) {
    try { facts = JSON.parse(rawFacts); }
    catch { alert("结构化事实不是合法 JSON"); return; }
  }

  $("ask").disabled = true;
  $("status").textContent = "已发起请求…";
  resetUI();

  const startedAt = performance.now();
  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, facts, session_id: "web" }),
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const frame = parseFrame(buffer.slice(0, idx));
        buffer = buffer.slice(idx + 2);
        if (frame) handle(frame.event, frame.data);
      }
    }
  } catch (err) {
    addTimeline({ name: "请求失败", meta: String(err), cls: "bad", key: "error" });
  } finally {
    $("ask").disabled = false;
    $("status").textContent = `完成 · 前端计时 ${Math.round(performance.now() - startedAt)}ms`;
  }
}

function handle(event, d) {
  if (event === "meta") {
    $("status").textContent = `意图 ${d.intent || "-"} · 角色 ${d.role || "-"} · 复杂度 ${d.complexity || "-"}`;
    $("badges").innerHTML = "";
    const isFast = d.mode === "fast";
    $("badges").appendChild(badge(isFast ? "快速通道" : "全编排", isFast ? "fast" : "full"));
    if (d.route) $("badges").appendChild(badge("路由 " + d.route));
    const why = (d.reasons || []).join("；");
    addTimeline({
      name: isFast ? "定档：快速通道" : "定档：全编排",
      meta: why || "—",
      cls: isFast ? "ok" : "",
      key: "meta",
    });
    return;
  }

  if (event === "plan") {
    const steps = d.steps || [];
    addTimeline({
      name: "计划（DAG）",
      meta: `${steps.length} 步：${steps.map((s) => s.id + ":" + s.agent).join(" → ")}`,
      key: "plan",
    });
    steps.forEach((s) => addTimeline({
      name: `${s.id} · ${AGENT_LABEL[s.agent] || s.agent}`,
      meta: "等待执行",
      key: "step:" + s.id,
    }));
    return;
  }

  if (event === "agent") {
    const name = `${d.step_id} · ${AGENT_LABEL[d.agent] || d.agent}`;
    const meta = `${d.status} · ${d.elapsed_ms}ms · 证据 ${d.evidence} 条 · LLM ${d.llm_calls} 次`;
    const cls = d.status === "ok" ? "ok" : d.status === "input_required" ? "warn" : "bad";
    updateTimeline("step:" + d.step_id, meta, cls);
    return;
  }

  if (event === "sources") {
    renderEvidence(d.sources || []);
    return;
  }

  if (event === "verify") {
    renderVerify(d);
    const n = (d.fails || []).length;
    addTimeline({
      name: d.passed ? "校验通过" : "校验未通过",
      meta: d.passed ? `检查 ${d.checked || 0} 处引用` : `失败 ${n} 项：${(d.fails || []).map((f) => f.gate).join(", ")}`,
      cls: d.passed ? "ok" : "bad",
      key: "verify",
    });
    return;
  }

  if (event === "delta") {
    if (d.append) answerText += d.text || "";
    else answerText = d.text || "";
    $("answer").innerHTML = renderAnswer(answerText);
    /* passed=false：终稿未过闸，答案标红（告示已在文本顶部） */
    $("answer").classList.toggle("rejected", d.passed === false);
    if (d.append) $("status").textContent = "正在生成…";
    return;
  }

  if (event === "done") {
    const t = d.timings || {};
    const parts = ["triage", "dag", "synth", "verify"].filter((k) => t[k + "_ms"] != null)
      .map((k) => `${k} ${t[k + "_ms"]}ms`).join(" · ");
    $("badges").appendChild(badge(`LLM ${d.llm_calls} 次`, "muted"));
    $("badges").appendChild(badge(`${d.latency_ms}ms`, "muted"));
    addTimeline({
      name: "完成",
      meta: `${d.latency_ms}ms（${parts}）· 回路 ${t.loops || 0} 次`,
      cls: "ok",
      key: "done",
    });
    return;
  }

  if (event === "error") {
    addTimeline({ name: "错误", meta: d.message || "未知错误", cls: "bad", key: "error" });
  }
}

$("ask").onclick = ask;
$("clear").onclick = () => {
  $("query").value = "";
  $("facts").value = "";
  $("answer").textContent = "";
  $("answer").classList.remove("rejected");
  answerText = "";
  $("status").textContent = "尚未提问";
  $("badges").innerHTML = '<span class="tag muted">等待提问</span>';
  $("timeline").innerHTML = '<li class="empty">提问后显示每一步</li>';
  $("evidence").innerHTML = '<li class="empty">尚未检索</li>';
  $("verify").innerHTML = '<span class="muted">尚未校验</span>';
};
$("query").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") ask();
});

initChips();
