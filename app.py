"""'Y' — a single-file X look-alike where three NLP models reply to your tweet.

Self-contained: this one file is the whole app. It imports only third-party
modules (Flask / torch / transformers / scikit-learn) and reads the trained
artifacts from ./models — it does NOT import anything else from this project.

Run:  python app.py   ->   http://127.0.0.1:5057
(Only requirement besides the libraries: the ./models folder.)
"""
import pickle
import re
from pathlib import Path

import torch
import torch.nn as nn
from flask import Flask, Response, jsonify, request

MODELS_DIR = Path(__file__).resolve().parent / "models"

# --------------------------------------------------------------------------
# Text cleanup (same as the training pipeline)
# --------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+")
_USER_RE = re.compile(r"@\w+")
_WS_RE = re.compile(r"\s+")


def clean_text(t: str) -> str:
    t = _URL_RE.sub("", t)
    t = _USER_RE.sub("@user", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


# --------------------------------------------------------------------------
# BiLSTM architecture + tokenizer (must match how bilstm.pt was trained)
# --------------------------------------------------------------------------
MAX_LEN = 40
EMB_DIM = 100
HIDDEN = 128
_tok_re = re.compile(r"[a-z0-9@#']+")


def tokenize(t: str):
    return _tok_re.findall(t.lower())


def encode(t, vocab):
    ids = [vocab.get(w, 1) for w in tokenize(t)][:MAX_LEN]
    if not ids:
        ids = [1]
    return ids


class BiLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMB_DIM, HIDDEN, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(HIDDEN * 2, 1)

    def forward(self, x, lens):
        e = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lens, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(self.drop(h)).squeeze(1)


# --------------------------------------------------------------------------
# Loaders — read straight from ./models and return predict(text)->(label, conf)
# --------------------------------------------------------------------------
def load_classical():
    with open(MODELS_DIR / "classical.pkl", "rb") as f:
        clf = pickle.load(f)

    def predict(text):
        p = clf.predict_proba([text])[0]
        return int(p.argmax()), float(p.max())

    return predict


def load_bilstm():
    ckpt = torch.load(MODELS_DIR / "bilstm.pt", map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    model = BiLSTM(len(vocab))
    model.load_state_dict(ckpt["state"])
    model.eval()

    def predict(text):
        ids = torch.tensor([encode(text, vocab)])
        with torch.no_grad():
            logit = model(ids, torch.tensor([ids.shape[1]]))
            p = torch.sigmoid(logit).item()
        return (1, p) if p > 0.5 else (0, 1 - p)

    return predict


def load_distilbert():
    d = MODELS_DIR / "distilbert"
    if not d.exists():
        return None
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForSequenceClassification.from_pretrained(d)
    model.eval()

    def predict(text):
        e = tok(text, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            probs = model(**e).logits.softmax(-1)[0]
        i = int(probs.argmax())
        return i, float(probs[i])

    return predict


# --------------------------------------------------------------------------
# The page (inlined — no templates/ folder needed)
# --------------------------------------------------------------------------
INDEX_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Y</title>
<style>
  :root{
    --bg:#000; --text:#e7e9ea; --dim:#71767b; --border:#2f3336;
    --blue:#1d9bf0; --blue-h:#1a8cd8; --hover:rgba(231,233,234,.03);
    --card:#16181c; --green:#00ba7c; --pink:#f91880;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{background:var(--bg);color:var(--text);
    font-family:"TwitterChirp",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased;}
  a{color:inherit;text-decoration:none}
  .wrap{display:flex;justify-content:center;min-height:100vh;max-width:1265px;margin:0 auto}

  /* ---------- LEFT NAV ---------- */
  .left{width:275px;flex-shrink:0;padding:0 12px;display:flex;flex-direction:column;
    height:100vh;position:sticky;top:0}
  .logo{width:50px;height:50px;display:flex;align-items:center;justify-content:center;
    border-radius:50%;margin:4px 0;font-size:30px;font-weight:800}
  .logo:hover{background:var(--hover)}
  .nav{display:flex;flex-direction:column;gap:2px;margin-top:2px}
  .nav a{display:flex;align-items:center;gap:18px;padding:11px 12px;border-radius:9999px;
    font-size:20px;width:fit-content;transition:background .15s}
  .nav a:hover{background:var(--hover)}
  .nav a.active{font-weight:800}
  .nav svg{width:26px;height:26px;fill:var(--text)}
  .post-btn{background:var(--blue);color:#fff;border:none;border-radius:9999px;
    font-size:17px;font-weight:700;height:52px;width:90%;margin-top:18px;cursor:pointer}
  .post-btn:hover{background:var(--blue-h)}
  .account{margin-top:auto;margin-bottom:12px;display:flex;align-items:center;gap:10px;
    padding:11px;border-radius:9999px;cursor:pointer}
  .account:hover{background:var(--hover)}
  .account .av{width:40px;height:40px;border-radius:50%;background:linear-gradient(135deg,#1d9bf0,#794bc4);
    display:flex;align-items:center;justify-content:center;font-weight:800;color:#fff}
  .account .meta{line-height:1.2;flex:1;min-width:0}
  .account .meta b{font-size:15px}
  .account .meta span{color:var(--dim);font-size:14px;display:block}
  .account .dots{color:var(--text);font-weight:800}

  /* ---------- CENTER ---------- */
  .center{width:600px;flex-shrink:0;border-left:1px solid var(--border);
    border-right:1px solid var(--border);min-height:100vh}
  .topbar{position:sticky;top:0;z-index:5;backdrop-filter:blur(12px);
    background:rgba(0,0,0,.65);border-bottom:1px solid var(--border)}
  .topbar h1{font-size:20px;font-weight:800;padding:14px 16px 4px}
  .tabs{display:flex}
  .tab{flex:1;text-align:center;padding:16px 0 0;color:var(--dim);font-weight:600;
    font-size:15px;cursor:pointer;position:relative}
  .tab.active{color:var(--text);font-weight:700}
  .tab span{display:inline-block;padding-bottom:14px}
  .tab.active span{border-bottom:4px solid var(--blue);border-radius:9999px}
  .tab:hover{background:var(--hover)}

  /* compose */
  .compose{display:flex;gap:12px;padding:12px 16px;border-bottom:1px solid var(--border)}
  .compose .av{width:40px;height:40px;flex-shrink:0;border-radius:50%;
    background:linear-gradient(135deg,#1d9bf0,#794bc4);display:flex;align-items:center;
    justify-content:center;font-weight:800;color:#fff}
  .compose .right{flex:1}
  .compose textarea{width:100%;background:transparent;border:none;outline:none;color:var(--text);
    font-size:20px;resize:none;padding:8px 0;font-family:inherit;min-height:28px;overflow:hidden}
  .compose textarea::placeholder{color:var(--dim)}
  .compose .bar{display:flex;align-items:center;justify-content:space-between;
    border-top:1px solid var(--border);padding-top:10px;margin-top:6px}
  .compose .tools{display:flex;gap:4px}
  .compose .tools svg{width:20px;height:20px;fill:var(--blue)}
  .compose .tools .ic{width:34px;height:34px;border-radius:50%;display:flex;align-items:center;
    justify-content:center;cursor:pointer}
  .compose .tools .ic:hover{background:rgba(29,155,240,.1)}
  .send{background:var(--blue);color:#fff;border:none;border-radius:9999px;font-weight:700;
    font-size:15px;padding:0 18px;height:36px;cursor:pointer;opacity:.5;pointer-events:none}
  .send.on{opacity:1;pointer-events:auto}
  .send:hover{background:var(--blue-h)}

  /* tweet */
  .tweet{display:flex;gap:12px;padding:12px 16px;border-bottom:1px solid var(--border);
    cursor:pointer;transition:background .12s;position:relative}
  .tweet:hover{background:var(--hover)}
  .av{width:40px;height:40px;flex-shrink:0;border-radius:50%;display:flex;align-items:center;
    justify-content:center;font-size:20px}
  .av.user{background:linear-gradient(135deg,#1d9bf0,#794bc4);font-weight:800;color:#fff;font-size:16px}
  .tbody{flex:1;min-width:0}
  .thead{display:flex;align-items:center;gap:5px;font-size:15px;flex-wrap:wrap}
  .thead b{font-weight:700}
  .thead .verified{width:18px;height:18px;fill:var(--blue)}
  .thead .at,.thead .dot,.thead .time{color:var(--dim);font-weight:400}
  .ttext{font-size:15px;line-height:1.4;margin:2px 0 4px;white-space:pre-wrap;word-wrap:break-word}
  .actions{display:flex;justify-content:space-between;max-width:425px;margin-top:8px;color:var(--dim)}
  .act{display:flex;align-items:center;gap:6px;font-size:13px;cursor:pointer}
  .act svg{width:18px;height:18px;fill:var(--dim)}
  .act:hover{color:var(--blue)} .act:hover svg{fill:var(--blue)}
  .act.like:hover{color:var(--pink)} .act.like:hover svg{fill:var(--pink)}
  .act.rt:hover{color:var(--green)} .act.rt:hover svg{fill:var(--green)}

  /* thread line connecting a tweet to its replies */
  .tweet.has-thread .av{position:relative}
  .tweet.has-thread .av::after{content:"";position:absolute;top:48px;left:50%;
    width:2px;height:calc(100% - 28px);background:var(--border);transform:translateX(-50%)}
  .reply .av::after{display:none}
  .pill{display:inline-flex;align-items:center;gap:6px;font-size:13px;font-weight:700;
    padding:3px 10px;border-radius:9999px;margin-top:6px}
  .conf{height:6px;border-radius:9999px;background:var(--border);margin-top:8px;overflow:hidden;max-width:280px}
  .conf > i{display:block;height:100%;border-radius:9999px}
  .era{color:var(--dim);font-size:13px}
  .typing{color:var(--dim);font-size:14px;padding:14px 16px;border-bottom:1px solid var(--border)}
  .hint{color:var(--dim);text-align:center;padding:40px 24px;font-size:15px;line-height:1.5}

  /* ---------- RIGHT ---------- */
  .right{width:350px;flex-shrink:0;padding:0 0 0 28px;height:100vh;position:sticky;top:0}
  .search{position:sticky;top:0;background:var(--bg);padding:6px 0 12px}
  .search div{display:flex;align-items:center;gap:12px;background:#202327;border-radius:9999px;
    padding:11px 16px;color:var(--dim)}
  .search svg{width:20px;height:20px;fill:var(--dim)}
  .search input{background:transparent;border:none;outline:none;color:var(--text);font-size:15px;flex:1}
  .box{background:#16181c;border-radius:16px;margin-top:16px;overflow:hidden}
  .box h2{font-size:20px;font-weight:800;padding:12px 16px}
  .trend{padding:10px 16px;cursor:pointer}
  .trend:hover{background:var(--hover)}
  .trend .c{color:var(--dim);font-size:13px}
  .trend .t{font-weight:700;font-size:15px;margin:2px 0}
  .footer{color:var(--dim);font-size:13px;padding:16px;display:flex;flex-wrap:wrap;gap:8px 12px}

  @media(max-width:1100px){.right{display:none}}
  @media(max-width:680px){.left{width:68px}.left .nav a span,.logo+.nav .lbl{display:none}
    .nav a span{display:none}.post-btn{width:52px;font-size:0;position:relative}
    .post-btn::after{content:"+";font-size:28px}.account .meta,.account .dots{display:none}}
</style>
</head>
<body>
<div class="wrap">

  <!-- LEFT NAV -->
  <nav class="left">
    <div class="logo">Y</div>
    <div class="nav">
      <a class="active" href="#"><svg viewBox="0 0 24 24"><path d="M12 1.696L.622 8.807l1.06 1.696L3 9.679V19.5C3 20.881 4.119 22 5.5 22h13c1.381 0 2.5-1.119 2.5-2.5V9.679l1.318.824 1.06-1.696L12 1.696zM12 16.5c-1.933 0-3.5-1.567-3.5-3.5s1.567-3.5 3.5-3.5 3.5 1.567 3.5 3.5-1.567 3.5-3.5 3.5z"/></svg><span>Home</span></a>
      <a href="#"><svg viewBox="0 0 24 24"><path d="M10.25 3.75c-3.59 0-6.5 2.91-6.5 6.5s2.91 6.5 6.5 6.5c1.795 0 3.419-.726 4.596-1.904 1.178-1.177 1.904-2.801 1.904-4.596 0-3.59-2.91-6.5-6.5-6.5zm-8.5 6.5c0-4.694 3.806-8.5 8.5-8.5s8.5 3.806 8.5 8.5c0 1.986-.682 3.815-1.824 5.262l4.781 4.781-1.414 1.414-4.781-4.781c-1.447 1.142-3.276 1.824-5.262 1.824-4.694 0-8.5-3.806-8.5-8.5z"/></svg><span>Explore</span></a>
      <a href="#"><svg viewBox="0 0 24 24"><path d="M19.993 9.042C19.48 5.017 16.054 2 11.996 2s-7.49 3.021-7.999 7.051L2.866 18H7.1c.463 2.282 2.481 4 4.9 4s4.437-1.718 4.9-4h4.236l-1.143-8.958zM12 20c-1.306 0-2.417-.835-2.829-2h5.658c-.412 1.165-1.523 2-2.829 2zm-6.866-4l.847-6.698C6.364 6.272 8.941 4 11.996 4s5.627 2.268 6.013 5.295L18.864 16H5.134z"/></svg><span>Notifications</span></a>
      <a href="#"><svg viewBox="0 0 24 24"><path d="M1.998 5.5c0-1.381 1.119-2.5 2.5-2.5h15c1.381 0 2.5 1.119 2.5 2.5v13c0 1.381-1.119 2.5-2.5 2.5h-15c-1.381 0-2.5-1.119-2.5-2.5v-13zm2.5-.5c-.276 0-.5.224-.5.5v2.764l8 3.638 8-3.636V5.5c0-.276-.224-.5-.5-.5h-15zm15.5 5.463l-8 3.636-8-3.638V18.5c0 .276.224.5.5.5h15c.276 0 .5-.224.5-.5v-8.037z"/></svg><span>Messages</span></a>
      <a href="#"><svg viewBox="0 0 24 24"><path d="M19.75 2H4.25C3.01 2 2 3.01 2 4.25v15.5C2 20.99 3.01 22 4.25 22h15.5c1.24 0 2.25-1.01 2.25-2.25V4.25C22 3.01 20.99 2 19.75 2zM4.25 3.5h15.5c.413 0 .75.337.75.75v15.5c0 .412-.337.75-.75.75H4.25c-.413 0-.75-.338-.75-.75V4.25c0-.413.337-.75.75-.75zM12 6.025c-1.413 0-2.563 1.15-2.563 2.563S10.587 11.15 12 11.15s2.563-1.15 2.563-2.562S13.413 6.025 12 6.025zM7.687 18.04c.275-2.037 2.05-3.625 4.313-3.625s4.038 1.588 4.313 3.625H7.687z"/></svg><span>Profile</span></a>
      <a href="#"><svg viewBox="0 0 24 24"><path d="M3 12c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2zm9 2c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm7 0c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2z"/></svg><span>More</span></a>
    </div>
    <button class="post-btn" onclick="document.getElementById('ta').focus()">Post</button>
    <div class="account">
      <div class="av">Y</div>
      <div class="meta"><b>You</b><span>@you</span></div>
      <div class="dots">···</div>
    </div>
  </nav>

  <!-- CENTER -->
  <main class="center">
    <div class="topbar">
      <h1>Home</h1>
      <div class="tabs">
        <div class="tab active"><span>For you</span></div>
        <div class="tab"><span>Following</span></div>
      </div>
    </div>

    <div class="compose">
      <div class="av">Y</div>
      <div class="right">
        <textarea id="ta" rows="1" placeholder="What is happening?!"></textarea>
        <div class="bar">
          <div class="tools">
            <div class="ic"><svg viewBox="0 0 24 24"><path d="M3 5.5C3 4.119 4.119 3 5.5 3h13C19.881 3 21 4.119 21 5.5v13c0 1.381-1.119 2.5-2.5 2.5h-13C4.119 21 3 19.881 3 18.5v-13zM5.5 5c-.276 0-.5.224-.5.5v9.086l3-3 3 3 5-5 3 3V5.5c0-.276-.224-.5-.5-.5h-13zM19 15.414l-3-3-5 5-3-3-3 3V18.5c0 .276.224.5.5.5h13c.276 0 .5-.224.5-.5v-3.086zM9.75 7C8.784 7 8 7.784 8 8.75s.784 1.75 1.75 1.75 1.75-.784 1.75-1.75S10.716 7 9.75 7z"/></svg></div>
            <div class="ic"><svg viewBox="0 0 24 24"><path d="M3 5.5C3 4.119 4.119 3 5.5 3h13C19.881 3 21 4.119 21 5.5v13c0 1.381-1.119 2.5-2.5 2.5h-13C4.119 21 3 19.881 3 18.5v-13zM5.5 5c-.276 0-.5.224-.5.5v13c0 .276.224.5.5.5h13c.276 0 .5-.224.5-.5v-13c0-.276-.224-.5-.5-.5h-13zM18 10.711V9.25h-3.74v5.5h1.44v-1.719h1.7V11.57h-1.7v-.859H18zM11.79 9.25h1.44v5.5h-1.44v-5.5zm-3.07 1.375c.34 0 .77.172 1.02.43l1.029-.86c-.51-.601-1.28-.945-2.049-.945C7.11 9.25 6 10.453 6 12s1.11 2.75 2.72 2.75c.85 0 1.6-.34 2.05-.945v-2.149H8.38v1.032h1.1v.515c-.18.084-.44.171-.76.171-.85 0-1.46-.61-1.46-1.375s.61-1.374 1.46-1.374z"/></svg></div>
            <div class="ic"><svg viewBox="0 0 24 24"><path d="M8 9.5C8 8.119 8.672 7 9.5 7S11 8.119 11 9.5 10.328 12 9.5 12 8 10.881 8 9.5zm6.5 2.5c.828 0 1.5-1.119 1.5-2.5S15.328 7 14.5 7 13 8.119 13 9.5s.672 2.5 1.5 2.5zM12 16c-2.224 0-3.021-2.227-3.051-2.316l-1.897.633c.05.15 1.271 3.683 4.948 3.683s4.898-3.533 4.948-3.683l-1.896-.638C15.026 13.762 14.218 16 12 16zm10-4c0 5.514-4.486 10-10 10S2 17.514 2 12 6.486 2 12 2s10 4.486 10 10zm-2 0c0-4.411-3.589-8-8-8s-8 3.589-8 8 3.589 8 8 8 8-3.589 8-8z"/></svg></div>
            <div class="ic"><svg viewBox="0 0 24 24"><path d="M6 5c-1.105 0-2 .895-2 2s.895 2 2 2 2-.895 2-2-.895-2-2-2zM2 7c0-2.209 1.791-4 4-4s4 1.791 4 4-1.791 4-4 4-4-1.791-4-4zm20 1H12v2h10V8zM6 15c-1.105 0-2 .895-2 2s.895 2 2 2 2-.895 2-2-.895-2-2-2zm-4 2c0-2.209 1.791-4 4-4s4 1.791 4 4-1.791 4-4 4-4-1.791-4-4zm20 1H12v-2h10v2z"/></svg></div>
          </div>
          <button class="send" id="send">Post</button>
        </div>
      </div>
    </div>

    <div id="feed">
      <div class="hint" id="hint">Post a tweet and watch three generations of NLP models —<br>
      a 2010 classical model, a 2015 neural net, and a 2019 transformer —<br>
      reply with how they read your sentiment. 💚 / 💔</div>
    </div>
  </main>

  <!-- RIGHT -->
  <aside class="right">
    <div class="search"><div>
      <svg viewBox="0 0 24 24"><path d="M10.25 3.75c-3.59 0-6.5 2.91-6.5 6.5s2.91 6.5 6.5 6.5c1.795 0 3.419-.726 4.596-1.904 1.178-1.177 1.904-2.801 1.904-4.596 0-3.59-2.91-6.5-6.5-6.5zm-8.5 6.5c0-4.694 3.806-8.5 8.5-8.5s8.5 3.806 8.5 8.5c0 1.986-.682 3.815-1.824 5.262l4.781 4.781-1.414 1.414-4.781-4.781c-1.447 1.142-3.276 1.824-5.262 1.824-4.694 0-8.5-3.806-8.5-8.5z"/></svg>
      <input placeholder="Search"></div>
    </div>
    <div class="box">
      <h2>What's happening</h2>
      <div class="trend"><div class="c">NLP · Trending</div><div class="t">#Sentiment140</div><div class="c">1.6M posts</div></div>
      <div class="trend"><div class="c">Technology · Trending</div><div class="t">DistilBERT</div><div class="c">42.8K posts</div></div>
      <div class="trend"><div class="c">Trending</div><div class="t">TF-IDF</div><div class="c">12.1K posts</div></div>
      <div class="trend"><div class="c">Machine Learning · Trending</div><div class="t">BiLSTM</div><div class="c">8,290 posts</div></div>
    </div>
    <div class="box">
      <h2>Who to follow</h2>
      <div class="trend"><div class="t">📐 TF-IDF + LogReg</div><div class="c">@tfidf_lr</div></div>
      <div class="trend"><div class="t">🧠 BiLSTM</div><div class="c">@bilstm_net</div></div>
      <div class="trend"><div class="t">🤖 DistilBERT</div><div class="c">@distilbert</div></div>
    </div>
    <div class="footer">Terms of Service · Privacy Policy · Cookie Policy · Accessibility · Ads info · More · © 2026 Y Corp.</div>
  </aside>

</div>

<script>
const VERIFIED = '<svg class="verified" viewBox="0 0 24 24"><path d="M22.25 12c0-1.43-.88-2.67-2.19-3.34.46-1.39.2-2.9-.81-3.91s-2.52-1.27-3.91-.81c-.67-1.31-1.91-2.19-3.34-2.19s-2.67.88-3.33 2.19c-1.4-.46-2.91-.2-3.92.81s-1.26 2.52-.8 3.91c-1.31.67-2.2 1.91-2.2 3.34s.89 2.67 2.2 3.34c-.46 1.39-.21 2.9.8 3.91s2.52 1.26 3.91.81c.67 1.31 1.91 2.19 3.34 2.19s2.67-.88 3.34-2.19c1.39.45 2.9.2 3.91-.81s1.27-2.52.81-3.91c1.31-.67 2.19-1.91 2.19-3.34zm-11.71 4.2L6.8 12.46l1.41-1.42 2.26 2.26 4.8-5.23 1.47 1.36-6.2 6.77z"/></svg>';
const ICONS = {
  reply:'<svg viewBox="0 0 24 24"><path d="M1.751 10c0-4.42 3.584-8 8.005-8h4.366c4.49 0 8.129 3.64 8.129 8.13 0 2.96-1.607 5.68-4.196 7.11l-8.054 4.46v-3.69h-.067c-4.49.1-8.183-3.51-8.183-8.01z"/></svg>',
  rt:'<svg viewBox="0 0 24 24"><path d="M4.5 3.88l4.432 4.14-1.364 1.46L5.5 7.55V16c0 1.1.896 2 2 2H13v2H7.5c-2.209 0-4-1.79-4-4V7.55L1.432 9.48.068 8.02 4.5 3.88zM16.5 6H11V4h5.5c2.209 0 4 1.79 4 4v8.45l2.068-1.93 1.364 1.46-4.432 4.14-4.432-4.14 1.364-1.46 2.068 1.93V8c0-1.1-.896-2-2-2z"/></svg>',
  like:'<svg viewBox="0 0 24 24"><path d="M16.697 5.5c-1.222-.06-2.679.51-3.89 2.16l-.805 1.09-.806-1.09C9.984 6.01 8.526 5.44 7.304 5.5c-1.243.07-2.349.78-2.91 1.91-.552 1.12-.633 2.78.479 4.82 1.074 1.97 3.257 4.27 7.129 6.61 3.87-2.34 6.052-4.64 7.126-6.61 1.111-2.04 1.03-3.7.477-4.82-.561-1.13-1.666-1.84-2.908-1.91z"/></svg>',
  views:'<svg viewBox="0 0 24 24"><path d="M8.75 21V3h2v18h-2zM18 21V8.5h2V21h-2zM4 21l.004-10h2L6 21H4zm9.248 0v-7h2v7h-2z"/></svg>',
  bookmark:'<svg viewBox="0 0 24 24"><path d="M4 4.5C4 3.12 5.119 2 6.5 2h11C18.881 2 20 3.12 20 4.5v18.44l-8-5.71-8 5.71V4.5zM6.5 4c-.276 0-.5.22-.5.5v14.56l6-4.29 6 4.29V4.5c0-.28-.224-.5-.5-.5h-11z"/></svg>',
  share:'<svg viewBox="0 0 24 24"><path d="M12 2.59l5.7 5.7-1.41 1.42L13 6.41V16h-2V6.41l-3.3 3.3-1.41-1.42L12 2.59zM21 15l-.02 3.51c0 1.38-1.12 2.49-2.5 2.49H5.5C4.11 21 3 19.88 3 18.5V15h2v3.5c0 .28.22.5.5.5h12.98c.28 0 .5-.22.5-.5L19 15h2z"/></svg>'
};
function rnd(n){return Math.floor(Math.random()*n)}
function actionRow(){
  return `<div class="actions">
    <div class="act">${ICONS.reply}<span>${rnd(40)}</span></div>
    <div class="act rt">${ICONS.rt}<span>${rnd(120)}</span></div>
    <div class="act like">${ICONS.like}<span>${rnd(900)}</span></div>
    <div class="act">${ICONS.views}<span>${(Math.random()*40+1).toFixed(1)}K</span></div>
    <div class="act">${ICONS.bookmark}</div>
    <div class="act">${ICONS.share}</div>
  </div>`;
}
const esc = s => s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

const ta = document.getElementById('ta');
const send = document.getElementById('send');
const feed = document.getElementById('feed');
ta.addEventListener('input',()=>{
  ta.style.height='auto'; ta.style.height=ta.scrollHeight+'px';
  send.classList.toggle('on', ta.value.trim().length>0);
});
ta.addEventListener('keydown',e=>{ if((e.metaKey||e.ctrlKey)&&e.key==='Enter') post(); });
send.addEventListener('click', post);

function post(){
  const text = ta.value.trim();
  if(!text) return;
  ta.value=''; ta.style.height='auto'; send.classList.remove('on');
  postText(text);
}

async function postText(text){
  const hint = document.getElementById('hint'); if(hint) hint.remove();

  const userTweet = document.createElement('div');
  userTweet.className = 'tweet has-thread';
  userTweet.innerHTML = `
    <div class="av user">Y</div>
    <div class="tbody">
      <div class="thead"><b>You</b><span class="at">@you</span><span class="dot">·</span><span class="time">now</span></div>
      <div class="ttext">${esc(text)}</div>
      ${actionRow()}
    </div>`;

  const typing = document.createElement('div');
  typing.className = 'typing';
  typing.textContent = 'Models are reading your post…';

  feed.prepend(typing);
  feed.prepend(userTweet);

  let replies = [];
  try{
    const r = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    replies = (await r.json()).replies || [];
  }catch(e){ typing.textContent='Could not reach the models.'; return; }
  typing.remove();

  // insert replies right after the user tweet, threaded
  let anchor = userTweet;
  replies.forEach((m,i)=>{
    const last = i===replies.length-1;
    const el = document.createElement('div');
    el.className = 'tweet reply' + (last?'':' has-thread');
    const barColor = m.label==='positive' ? 'var(--green)' : 'var(--pink)';
    el.innerHTML = `
      <div class="av" style="background:${m.color}22">${m.avatar}</div>
      <div class="tbody">
        <div class="thead"><b>${m.name}</b>${VERIFIED}<span class="at">@${m.handle}</span><span class="dot">·</span><span class="time">now</span></div>
        <div class="era">Replying to <span style="color:var(--blue)">@you</span> · ${m.era}</div>
        <div class="ttext">${m.text}</div>
        <span class="pill" style="background:${barColor}22;color:${barColor}">${m.emoji} ${m.label} · ${m.confidence}% sure</span>
        <div class="conf"><i style="width:${m.confidence}%;background:${barColor}"></i></div>
        ${actionRow()}
      </div>`;
    anchor.after(el); anchor = el;
  });
}

// Seed one demo conversation on load so the idea is obvious at a glance.
// (Script is at end of <body>, so the DOM above already exists.)
postText("Just had the best pancakes ever at this little diner downtown. Going to be a great day!");
</script>
</body>
</html>'''


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
app = Flask(__name__)

print("Loading models (once)...")
MODELS = []  # order = NLP timeline
MODELS.append({
    "name": "TF-IDF + LogReg", "handle": "tfidf_lr", "era": "2010 · classical",
    "avatar": "📐", "color": "#1d9bf0", "fn": load_classical(),
})
MODELS.append({
    "name": "BiLSTM", "handle": "bilstm_net", "era": "2015 · neural",
    "avatar": "🧠", "color": "#00ba7c", "fn": load_bilstm(),
})
_db = load_distilbert()
if _db:
    MODELS.append({
        "name": "DistilBERT", "handle": "distilbert", "era": "2019 · transformer",
        "avatar": "🤖", "color": "#f91880", "fn": _db,
    })
print(f"Loaded {len(MODELS)} models.")

# Sentiment -> emoji + phrasing, keyed by (label, confidence band).
POS = ["🙂 reading this as positive.", "😊 this gives off good vibes.",
       "😄 definitely a positive one!", "🔥 loving the energy here."]
NEG = ["🙁 i'm reading this as negative.", "😕 this feels negative to me.",
       "😞 picking up bad vibes here.", "💀 oof, this one's negative."]


def reaction(label, conf):
    bucket = min(int((conf - 0.5) / 0.125), 3)  # 0..3 by confidence
    bucket = max(bucket, 0)
    return (POS if label == 1 else NEG)[bucket]


@app.route("/")
def home():
    return Response(INDEX_HTML, mimetype="text/html")


@app.route("/predict", methods=["POST"])
def predict():
    raw = (request.json or {}).get("text", "")
    text = clean_text(raw)
    replies = []
    for m in MODELS:
        if not text.strip():
            continue
        label, conf = m["fn"](text)
        replies.append({
            "name": m["name"], "handle": m["handle"], "era": m["era"],
            "avatar": m["avatar"], "color": m["color"],
            "label": "positive" if label == 1 else "negative",
            "emoji": "💚" if label == 1 else "💔",
            "confidence": round(conf * 100),
            "text": reaction(label, conf),
        })
    return jsonify({"replies": replies})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5057, debug=False)
