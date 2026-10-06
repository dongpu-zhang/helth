"""
筋膜枪使用教练 · FastAPI + Edge TTS + OCI Grok 4.7
本地开发: python app.py
云端部署: 设置环境变量后 Render/Railway 自动启动
"""
import os
import json
import tempfile

# ── OCI 证书加载（支持本地文件 + 云端环境变量双模式）───────────────────────────

def _load_oci_cfg():
    """
    云端（Render）：从环境变量读取，动态写出 ~/.oci/config + 私钥文件。
    本地开发：直接读已有的 ~/.oci/config。
    """
    import base64

    oci_dir  = os.path.expanduser("~/.oci")
    cfg_path = os.path.join(oci_dir, "config")
    key_path = os.path.join(oci_dir, "oci_api_key.pem")

    if os.environ.get("OCI_TENANCY"):
        # ── 云端模式：写私钥文件 ──
        key_content = os.environ.get("OCI_PRIVATE_KEY", "").strip()
        if not key_content:
            raise RuntimeError("OCI_PRIVATE_KEY env var is empty")

        # 支持 base64 单行或原始 PEM
        if not key_content.startswith("-----"):
            key_content = base64.b64decode(key_content).decode("utf-8")

        # 统一换行符
        key_content = key_content.replace("\r\n", "\n").replace("\r", "\n")

        os.makedirs(oci_dir, exist_ok=True)
        with open(key_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(key_content)
        os.chmod(key_path, 0o600)

        # ── 写 ~/.oci/config ──
        user        = os.environ["OCI_USER"]
        fingerprint = os.environ["OCI_FINGERPRINT"]
        tenancy     = os.environ["OCI_TENANCY"]
        region      = os.environ.get("OCI_REGION", "us-chicago-1")

        with open(cfg_path, "w", encoding="utf-8") as f:
            f.write(f"[DEFAULT]\n"
                    f"user={user}\n"
                    f"fingerprint={fingerprint}\n"
                    f"tenancy={tenancy}\n"
                    f"region={region}\n"
                    f"key_file={key_path}\n")

    if not os.path.exists(cfg_path):
        raise RuntimeError(
            "OCI credentials not found.\n"
            "Cloud: set OCI_USER / OCI_FINGERPRINT / OCI_TENANCY / OCI_PRIVATE_KEY / OCI_REGION\n"
            "Local: create ~/.oci/config"
        )

    import oci as _oci
    return _oci.config.from_file(cfg_path, "DEFAULT")


_cfg = _load_oci_cfg()

os.environ.update({
    "OCI_USER":                   _cfg["user"],
    "OCI_FINGERPRINT":            _cfg["fingerprint"],
    "OCI_TENANCY":                _cfg["tenancy"],
    "OCI_COMPARTMENT_ID":         _cfg["tenancy"],
    "OCI_KEY_FILE":               _cfg.get("key_file", ""),
    "OCI_REGION":                 _cfg.get("region", "us-chicago-1"),
    "OPENAI_API_KEY":             "sk-oci-local",
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
})
os.environ["LITELLM_LOG"]         = "ERROR"
os.environ["LITELLM_DROP_PARAMS"] = "True"

try:
    import langchain_litellm
except ImportError:
    pass

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel
from typing import List
import uvicorn

OCI_MODEL = "oci/xai.grok-4.7"
PORT      = int(os.environ.get("PORT", 8782))

SYSTEM_PROMPT = """# 角色
你是一位专业、谨慎、实用的「筋膜枪（按摩枪）使用教练」。你的任务是根据用户的身体状况、运动情况和感受，给出安全、个性化的筋膜枪使用建议。

你不是医生，也不提供医疗诊断或治疗方案。你的建议仅用于日常肌肉放松与恢复参考。

# 核心原则
1. 安全第一：任何存在风险的情况，优先提醒停止使用或就医。
2. 个性化：根据用户描述调整部位、档位、时间和手法，避免套模板。
3. 结构清晰：每次建议都按固定格式输出，方便用户执行。
4. 主动追问：信息不足时先问关键问题，再给方案。
5. 诚实边界：不确定或不在能力范围内时，明确说明。

# 严格禁止
- 不诊断疾病、不声称能治疗伤病。
- 不建议在以下情况使用筋膜枪：
  - 急性扭伤、挫伤、骨折、开放性伤口
  - 静脉曲张、血栓、严重水肿
  - 皮肤感染、炎症急性期
  - 怀孕（尤其腹部、腰部）
  - 骨质疏松严重区域、肿瘤部位
  - 脊柱、喉咙、面部、头部、骨突明显处（如肘尖、膝盖骨正面）
- 不鼓励用户强忍尖锐疼痛继续打。
- 不给出极端高强度或超长时间建议。

# 默认安全参数
- 新手或敏感部位：低档（1-2档）
- 常规肌肉放松：中低至中档（2-3档）
- 单个部位建议时间：30-90 秒
- 同一肌群总时间：一般不超过 2-3 分钟
- 移动方式：沿肌肉走向缓慢移动，不要长时间定点猛打
- 出现尖锐痛、麻木、头晕等立即停止

# 标准输出格式
当信息足够时，请严格按以下结构回复：

【使用建议】
- 目标：一句话说明目的
- 推荐部位：具体肌肉/区域
- 建议档位：低 / 中低 / 中（并说明理由）
- 时间：每个部位或每侧多久
- 手法：如何移动、角度、是否需要支撑
- 注意事项：必须避开的位置、疼痛判断标准
- 后续建议：是否配合拉伸、休息或观察

【可选补充】
- 今日可做的简单拉伸或活动
- 什么情况下应停止并考虑就医

如果信息不足，先输出：
【需要补充的信息】
列出 1-3 个最关键的问题，等用户回答后再给完整方案。

# 常见场景处理
1. 运动后肌肉酸痛（DOMS）：低到中低强度，不要在极度酸痛或肿胀时大力打
2. 长期久坐（颈肩、下背、髋部）：优先浅层肌肉放松，提醒姿势问题
3. 明确受伤或急性疼痛：建议暂停使用，建议就医
4. 用户只说"帮我放松一下"：追问主要紧张部位
5. 用户要求打脊柱、骨头、关节缝：明确拒绝，引导到安全区域

# 语气
专业但好懂，像靠谱的健身/康复教练，简洁可执行，用中文回复。"""


def _make_llm():
    from langchain_litellm import ChatLiteLLM
    return ChatLiteLLM(
        model=OCI_MODEL,
        model_kwargs={"compartment_id": _cfg["tenancy"], "timeout": 120},
        temperature=0.4,
        max_tokens=2000,
    )


app = FastAPI(title="筋膜枪使用教练")


class ChatRequest(BaseModel):
    messages: List[dict]


@app.post("/chat")
def chat_stream(req: ChatRequest):
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    def generate():
        msgs = [SystemMessage(content=SYSTEM_PROMPT)]
        for m in req.messages:
            role, content = m.get("role", ""), m.get("content", "")
            if role == "user":
                msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                msgs.append(AIMessage(content=content))
        import time
        last_err = None
        for attempt in range(3):
            try:
                llm     = _make_llm()
                got_any = False
                try:
                    for chunk in llm.stream(msgs):
                        text = getattr(chunk, "content", "") or ""
                        if text:
                            got_any = True
                            yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
                except Exception as se:
                    if "stream_options" in str(se):
                        got_any = False  # fall through to invoke
                    else:
                        raise
                if not got_any:
                    resp = llm.invoke(msgs)
                    text = getattr(resp, "content", str(resp))
                    yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(2 ** attempt)
        if last_err is not None:
            err = f"服务暂时不可用，请稍后重试。({str(last_err)[:120]})"
            yield f"data: {json.dumps({'text': err}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/tts")
async def tts_endpoint(request: Request):
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return Response(status_code=400)
    try:
        import edge_tts
    except ImportError:
        return Response(status_code=503, content=b"pip install edge-tts")

    voice = body.get("voice", "zh-CN-XiaoxiaoNeural")
    try:
        communicate = edge_tts.Communicate(text, voice)
        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return Response(content=b"".join(chunks), media_type="audio/mpeg")
    except Exception as e:
        return Response(status_code=500, content=str(e).encode())


@app.get("/health")
def health():
    tts_ok = True
    try:
        import edge_tts  # noqa
    except ImportError:
        tts_ok = False
    return {"status": "ok", "model": OCI_MODEL, "tts": tts_ok}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=_HTML)


_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>筋膜枪使用教练</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;--surface:#161b22;--surface2:#1c2128;--border:#30363d;
  --text:#e6edf3;--muted:#8b949e;--green:#3fb950;--blue:#58a6ff;
  --red:#f85149;--gold:#d29922;
  --coach-bg:#0f2018;--coach-border:#3fb95055;
  --user-bg:#0d1f38;--user-border:#58a6ff44;
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;overflow:hidden}
.app{display:flex;flex-direction:column;height:100vh}
header{flex-shrink:0;display:flex;align-items:center;justify-content:space-between;padding:0 16px;height:56px;background:var(--surface);border-bottom:1px solid var(--border)}
.h-left{display:flex;align-items:center;gap:10px}
.logo{font-size:26px;line-height:1}
.h-title{font-size:15px;font-weight:700;color:var(--text)}
.h-sub{font-size:11px;color:var(--muted)}
.h-right{display:flex;align-items:center;gap:10px}
.tts-toggle{display:flex;align-items:center;gap:5px;font-size:12px;color:var(--muted);cursor:pointer;padding:4px 10px;border-radius:14px;border:1px solid var(--border);transition:all .15s;user-select:none}
.tts-toggle.on{color:var(--green);border-color:var(--green);background:#0f2018}
.sdot{width:8px;height:8px;border-radius:50%;background:var(--green);flex-shrink:0}
.sdot.dead{background:var(--red)}
.sdot.loading{background:var(--gold);animation:pulse 1s infinite}
#chatWrap{flex:1;overflow-y:auto;padding:16px 0 8px;scroll-behavior:smooth}
#chatWrap::-webkit-scrollbar{width:4px}
#chatWrap::-webkit-scrollbar-thumb{background:#30363d;border-radius:2px}
.msg{display:flex;gap:10px;padding:6px 16px;animation:fadeIn .2s ease}
.msg.user{flex-direction:row-reverse}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.avatar{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;margin-top:2px}
.coach .avatar{background:var(--coach-bg);border:1px solid var(--coach-border)}
.user .avatar{background:var(--user-bg);border:1px solid var(--user-border)}
.bubble{max-width:min(480px,82vw)}
.bubble-inner{padding:11px 14px;border-radius:12px;font-size:14px;line-height:1.75;white-space:pre-wrap;word-break:break-word}
.coach .bubble-inner{background:var(--coach-bg);border:1px solid var(--coach-border);border-radius:4px 12px 12px 12px}
.user .bubble-inner{background:var(--user-bg);border:1px solid var(--user-border);border-radius:12px 4px 12px 12px}
.bubble-footer{display:flex;align-items:center;gap:6px;margin-top:4px;padding:0 2px}
.coach .bubble-footer{justify-content:flex-start}
.user .bubble-footer{justify-content:flex-end}
.bfoot-btn{background:none;border:none;cursor:pointer;color:var(--muted);font-size:13px;padding:2px 5px;border-radius:4px;transition:color .15s,background .15s;display:flex;align-items:center;gap:3px}
.bfoot-btn:hover{color:var(--text);background:var(--surface2)}
.bfoot-btn.playing{color:var(--green)}
.btime{font-size:10px;color:var(--muted)}
.typing-dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--green);margin:0 2px;animation:bounce .8s infinite}
.typing-dot:nth-child(2){animation-delay:.15s}
.typing-dot:nth-child(3){animation-delay:.3s}
@keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-6px)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
footer{flex-shrink:0;padding:10px 12px 14px;background:var(--surface);border-top:1px solid var(--border)}
.input-row{display:flex;align-items:flex-end;gap:8px}
.mic-btn{width:44px;height:44px;border-radius:50%;border:2px solid var(--border);background:var(--surface2);color:var(--muted);font-size:20px;cursor:pointer;flex-shrink:0;transition:all .2s;display:flex;align-items:center;justify-content:center;position:relative;-webkit-user-select:none;user-select:none;touch-action:none}
.mic-btn:hover{border-color:var(--green);color:var(--green)}
.mic-btn.listening{border-color:var(--red);color:var(--red);background:#1c0a09}
.mic-btn.listening::after{content:'';position:absolute;width:52px;height:52px;border-radius:50%;border:2px solid var(--red);animation:ripple 1s infinite}
@keyframes ripple{0%{transform:scale(1);opacity:.8}100%{transform:scale(1.4);opacity:0}}
#inputBox{flex:1;min-height:44px;max-height:120px;padding:10px 13px;background:var(--surface2);border:1px solid var(--border);border-radius:10px;color:var(--text);font-size:14px;line-height:1.5;resize:none;outline:none;font-family:inherit;transition:border-color .15s}
#inputBox::placeholder{color:var(--muted)}
#inputBox:focus{border-color:var(--green)}
.send-btn{width:44px;height:44px;border-radius:50%;border:none;background:var(--green);color:#0d1117;cursor:pointer;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:18px;transition:opacity .15s}
.send-btn:hover{opacity:.85}
.send-btn:disabled{opacity:.4;cursor:not-allowed}
.voice-hint{font-size:11px;color:var(--muted);text-align:center;margin-top:6px;height:14px;transition:color .2s}
.voice-hint.listening{color:var(--red)}
.suggestions{display:flex;flex-wrap:wrap;gap:6px;padding:8px 16px 4px}
.sug{background:var(--surface2);border:1px solid var(--border);border-radius:16px;padding:5px 12px;font-size:12px;color:var(--muted);cursor:pointer;transition:all .15s;white-space:nowrap}
.sug:hover{border-color:var(--green);color:var(--green)}
#toast{position:fixed;bottom:90px;left:50%;transform:translateX(-50%) translateY(20px);background:var(--surface2);border:1px solid var(--border);color:var(--text);padding:8px 18px;border-radius:20px;font-size:13px;opacity:0;transition:all .3s;pointer-events:none;z-index:99}
#toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
@supports(padding-bottom:env(safe-area-inset-bottom)){
  footer{padding-bottom:calc(14px + env(safe-area-inset-bottom))}
}
</style>
</head>
<body>
<div class="app">
<header>
  <div class="h-left">
    <span class="logo">💪</span>
    <div>
      <div class="h-title">筋膜枪使用教练</div>
      <div class="h-sub">安全 · 个性化 · 专业</div>
    </div>
  </div>
  <div class="h-right">
    <div class="tts-toggle on" id="ttsToggle" onclick="toggleAutoTTS()">
      <span>🔊</span><span id="ttsLabel">自动朗读</span>
    </div>
    <div class="sdot loading" id="sdot"></div>
  </div>
</header>

<div id="chatWrap"></div>

<div class="suggestions" id="suggestions">
  <span class="sug" onclick="fillSug(this)">练完腿，大腿酸</span>
  <span class="sug" onclick="fillSug(this)">久坐颈肩僵硬</span>
  <span class="sug" onclick="fillSug(this)">下背部酸痛</span>
  <span class="sug" onclick="fillSug(this)">跑步后小腿紧</span>
  <span class="sug" onclick="fillSug(this)">运动前热身激活</span>
</div>

<footer>
  <div class="input-row">
    <button class="mic-btn" id="micBtn" title="按住说话，松手发送">🎙</button>
    <textarea id="inputBox" placeholder="描述你的情况，例如：今天练完腿，大腿前侧很酸…" rows="1"
      oninput="autoResize(this)" onkeydown="handleKey(event)"></textarea>
    <button class="send-btn" id="sendBtn" onclick="sendMessage()">➤</button>
  </div>
  <div class="voice-hint" id="voiceHint"></div>
</footer>
</div>
<div id="toast"></div>

<script>
let conversation   = [];
let autoTTS        = true;
let isStreaming    = false;
let currentAudio   = null;
let recognition    = null;
let msgCounter     = 0;
let ttsAvailable   = true;
let voiceTriggered = false;
let isListening    = false;
let _pttReleased   = false;

window.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  setupVoice();
  showWelcome();
});

function showWelcome() {
  addMessage('coach',
`你好！我是你的筋膜枪使用教练 💪

我可以根据你的情况，给出安全、个性化的使用建议——
打哪里、用什么档位、打多久、怎么移动。

你可以告诉我：
• 今天做了什么运动，哪里酸？
• 长期坐办公室，颈肩/腰部不舒服？
• 想在运动前热身激活肌肉？

⚠️ 温馨提示：如有急性受伤、明显肿胀或剧痛，我会建议先就医而非使用筋膜枪。

请告诉我你现在的情况吧 👇`, true);
}

async function checkHealth() {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    document.getElementById('sdot').className = 'sdot' + (d.status === 'ok' ? '' : ' dead');
    ttsAvailable = d.tts !== false;
    if (!ttsAvailable) {
      document.getElementById('ttsLabel').textContent = '朗读不可用';
      document.getElementById('ttsToggle').classList.remove('on');
      autoTTS = false;
    }
  } catch { document.getElementById('sdot').className = 'sdot dead'; }
}

function addMessage(role, content, skipHistory) {
  const id = 'msg-' + (++msgCounter);
  const wrap = document.getElementById('chatWrap');
  const now = new Date().toLocaleTimeString('zh-CN', {hour:'2-digit',minute:'2-digit'});
  const isCoach = role === 'coach';
  const div = document.createElement('div');
  div.className = `msg ${isCoach ? 'coach' : 'user'}`;
  div.id = id;
  const footer = isCoach
    ? `<div class="bubble-footer">
        <button class="bfoot-btn" id="play-${id}" onclick="playTTSForMsg('${id}')">🔊 朗读</button>
        <span class="btime">${now}</span>
       </div>`
    : `<div class="bubble-footer"><span class="btime">${now}</span></div>`;
  div.innerHTML = `
    <div class="avatar">${isCoach ? '💪' : '👤'}</div>
    <div class="bubble">
      <div class="bubble-inner" id="inner-${id}">${escHtml(content)}</div>
      ${footer}
    </div>`;
  wrap.appendChild(div);
  scrollBottom();
  if (!skipHistory) conversation.push({role: isCoach ? 'assistant' : 'user', content});
  if (isCoach && autoTTS && ttsAvailable && !skipHistory) playTTS(cleanTTS(content), id);
  return id;
}

function addStreamingMessage() {
  const id = 'msg-' + (++msgCounter);
  const div = document.createElement('div');
  div.className = 'msg coach'; div.id = id;
  div.innerHTML = `<div class="avatar">💪</div>
    <div class="bubble"><div class="bubble-inner" id="inner-${id}">
      <span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>
    </div></div>`;
  document.getElementById('chatWrap').appendChild(div);
  scrollBottom();
  return id;
}

function finalizeStreamingMessage(id, fullText) {
  const inner = document.getElementById('inner-' + id);
  const msg   = document.getElementById(id);
  if (!inner || !msg) return;
  inner.innerHTML = escHtml(fullText);
  const now = new Date().toLocaleTimeString('zh-CN', {hour:'2-digit',minute:'2-digit'});
  const f = document.createElement('div');
  f.className = 'bubble-footer';
  f.innerHTML = `<button class="bfoot-btn" id="play-${id}" onclick="playTTSForMsg('${id}')">🔊 朗读</button>
    <span class="btime">${now}</span>`;
  msg.querySelector('.bubble').appendChild(f);
  scrollBottom();
}

async function sendMessage() {
  const input = document.getElementById('inputBox');
  const text  = input.value.trim();
  if (!text || isStreaming) return;
  input.value = ''; autoResize(input);
  const sugg = document.getElementById('suggestions');
  if (sugg) sugg.style.display = 'none';
  addMessage('user', text);
  const streamId = addStreamingMessage();
  isStreaming = true;
  document.getElementById('sendBtn').disabled = true;
  let fullText = '', buffer = '';
  try {
    const resp = await fetch('/chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({messages: conversation}),
    });
    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    const inner   = document.getElementById('inner-' + streamId);
    inner.innerHTML = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream:true});
      const lines = buffer.split('\n'); buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') break;
        try {
          const d = JSON.parse(raw);
          if (d.text) { fullText += d.text; inner.textContent = fullText; scrollBottom(); }
          if (d.error) { fullText = d.error; inner.textContent = fullText; }
        } catch {}
      }
    }
  } catch { fullText = '连接失败，请检查服务是否正常运行。'; }
  finalizeStreamingMessage(streamId, fullText || '（无响应）');
  conversation.push({role:'assistant', content: fullText});
  const shouldSpeak = ttsAvailable && fullText && (autoTTS || voiceTriggered);
  voiceTriggered = false;
  if (shouldSpeak) playTTS(cleanTTS(fullText), streamId);
  isStreaming = false;
  document.getElementById('sendBtn').disabled = false;
  input.focus();
}

async function playTTS(text, msgId) {
  if (!ttsAvailable || !text) return;
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }
  if (msgId) { const b = document.getElementById('play-'+msgId); if(b) b.classList.add('playing'); }
  try {
    const resp = await fetch('/tts', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text, voice:'zh-CN-XiaoxiaoNeural'}),
    });
    if (!resp.ok) { ttsAvailable = false; return; }
    const blob = await resp.blob();
    const url  = URL.createObjectURL(blob);
    currentAudio = new Audio(url);
    currentAudio.onended = () => {
      URL.revokeObjectURL(url);
      if (msgId) { const b = document.getElementById('play-'+msgId); if(b) b.classList.remove('playing'); }
    };
    currentAudio.play();
  } catch {
    if (msgId) { const b = document.getElementById('play-'+msgId); if(b) b.classList.remove('playing'); }
  }
}

function playTTSForMsg(msgId) {
  const inner = document.getElementById('inner-' + msgId);
  if (inner) playTTS(cleanTTS(inner.textContent || ''), msgId);
}

function toggleAutoTTS() {
  if (!ttsAvailable) { toast('TTS 服务不可用'); return; }
  autoTTS = !autoTTS;
  document.getElementById('ttsToggle').classList.toggle('on', autoTTS);
  document.getElementById('ttsLabel').textContent = autoTTS ? '自动朗读' : '手动朗读';
  if (!autoTTS && currentAudio) { currentAudio.pause(); currentAudio = null; }
}

function cleanTTS(text) {
  return text.replace(/【([^】]+)】/g,'$1').replace(/[*_`#~]/g,'')
    .replace(/https?:\/\/\S+/g,'').replace(/\n{3,}/g,'\n\n').trim().slice(0,1500);
}

// ── 按住说话 PTT ──────────────────────────────────────────────────────────────
function setupVoice() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  const btn = document.getElementById('micBtn');
  if (!SpeechRec) {
    btn.title = '浏览器不支持语音输入（推荐 Chrome/Edge）';
    btn.style.opacity = '0.4';
    setVoiceHint('语音输入需要 Chrome / Edge 浏览器', false);
    return;
  }
  btn.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    if (isStreaming) return;
    btn.setPointerCapture(e.pointerId);
    startVoice();
  });
  btn.addEventListener('pointerup',     releaseMic);
  btn.addEventListener('pointercancel', releaseMic);
  setVoiceHint('按住麦克风说话', false);

  recognition = new SpeechRec();
  recognition.lang = 'zh-CN';
  recognition.continuous = true;
  recognition.interimResults = true;

  recognition.onstart = () => {
    isListening = true; _pttReleased = false;
    btn.classList.add('listening'); btn.textContent = '🔴';
    setVoiceHint('录音中... 松手发送', true);
  };

  recognition.onresult = (e) => {
    let text = '';
    for (let i = 0; i < e.results.length; i++) text += e.results[i][0].transcript;
    const input = document.getElementById('inputBox');
    input.value = text; autoResize(input);
  };

  recognition.onerror = (e) => {
    stopVoice();
    if (e.error !== 'no-speech') toast('语音识别失败: ' + e.error);
  };

  recognition.onend = () => {
    const wasReleased = _pttReleased;
    stopVoice();
    if (wasReleased) {
      const text = document.getElementById('inputBox').value.trim();
      if (text) sendMessage();
    }
  };
}

function releaseMic() {
  if (!isListening) return;
  _pttReleased = true;
  try { recognition?.stop(); } catch {}
}

function startVoice() {
  if (isListening || isStreaming) return;
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }
  voiceTriggered = true;
  try { recognition.start(); } catch {}
}

function stopVoice() {
  isListening = false;
  const btn = document.getElementById('micBtn');
  btn.classList.remove('listening'); btn.textContent = '🎙';
  setVoiceHint('按住麦克风说话', false);
}

// ── helpers ──────────────────────────────────────────────────────────────────
function handleKey(e) { if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage();} }
function autoResize(el) { el.style.height='auto'; el.style.height=Math.min(el.scrollHeight,120)+'px'; }
function scrollBottom() { const w=document.getElementById('chatWrap'); w.scrollTop=w.scrollHeight; }
function escHtml(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function fillSug(el) {
  const input=document.getElementById('inputBox');
  input.value=el.textContent; autoResize(input); input.focus();
  document.getElementById('suggestions').style.display='none';
}
function setVoiceHint(text,isActive) {
  const el=document.getElementById('voiceHint'); if(!el) return;
  el.textContent=text; el.className='voice-hint'+(isActive?' listening':'');
}
function toast(msg) {
  const el=document.getElementById('toast'); el.textContent=msg;
  el.classList.add('show'); setTimeout(()=>el.classList.remove('show'),2800);
}
</script>
</body>
</html>
"""

@app.get("/debug/oci")
def debug_oci():
    import base64
    oci_dir  = os.path.expanduser("~/.oci")
    cfg_path = os.path.join(oci_dir, "config")
    key_path = os.path.join(oci_dir, "oci_api_key.pem")

    info = {}

    # env vars present?
    info["env_OCI_TENANCY"]      = bool(os.environ.get("OCI_TENANCY"))
    info["env_OCI_USER"]         = bool(os.environ.get("OCI_USER"))
    info["env_OCI_FINGERPRINT"]  = bool(os.environ.get("OCI_FINGERPRINT"))
    info["env_OCI_PRIVATE_KEY"]  = bool(os.environ.get("OCI_PRIVATE_KEY"))
    info["env_OCI_REGION"]       = os.environ.get("OCI_REGION", "(not set)")
    info["fingerprint_value"]    = os.environ.get("OCI_FINGERPRINT", "")[:40]  # partial only

    # config file
    info["config_exists"] = os.path.exists(cfg_path)
    if info["config_exists"]:
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw = f.read()
        lines = raw.splitlines()
        info["config_lines"] = len(lines)
        info["config_content_redacted"] = "\n".join(
            (l if not l.lower().startswith("key_file") else l)
            for l in lines
        )

    # key file
    info["key_file_exists"] = os.path.exists(key_path)
    if info["key_file_exists"]:
        raw_key = open(key_path, "rb").read()
        info["key_bytes"] = len(raw_key)
        info["key_has_crlf"] = b"\r\n" in raw_key
        text_key = raw_key.decode("utf-8", errors="replace")
        klines = text_key.splitlines()
        info["key_first_line"] = klines[0] if klines else ""
        info["key_last_line"]  = klines[-1] if klines else ""
        info["key_total_lines"] = len(klines)

    # try a live OCI call
    try:
        import oci as _oci
        cfg2 = _oci.config.from_file(cfg_path, "DEFAULT")
        _oci.config.validate_config(cfg2)
        info["oci_config_valid"] = True
        # Try a cheap API call: list compartments
        try:
            id_client = _oci.identity.IdentityClient(cfg2)
            resp = id_client.get_tenancy(cfg2["tenancy"])
            info["oci_api_test"] = "ok: tenancy=" + str(resp.data.name)
        except Exception as e_api:
            info["oci_api_test"] = "FAIL: " + str(e_api)[:300]
    except Exception as e_cfg:
        info["oci_config_valid"] = False
        info["oci_config_error"] = str(e_cfg)[:300]

    return info


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"[helth] http://localhost:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
