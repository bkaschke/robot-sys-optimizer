"""Build docs/USER_GUIDE.html from docs/USER_GUIDE.md (the Markdown file stays the source).

    python scripts/build_guide.py          # writes docs/USER_GUIDE.html
    python scripts/build_guide.py --check  # exit 1 if the HTML is out of date

The page is self-contained (inline CSS + a few lines of JS for the theme button and the search
box: v8 T4, filters the sidebar, highlights matches, Enter / Shift+Enter jump, Esc clears,
``/`` focuses it, ``?q=word`` searches on load), with a sidebar of sections, GitHub-style heading anchors (so the guide's own contents links
work), light/dark styling that follows the system with a manual toggle, and print styles.
The app serves it on --port + 1 and links to it from the top of the panel.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "USER_GUIDE.md"
OUT = ROOT / "docs" / "USER_GUIDE.html"

CSS = """
:root{--bg:#fff;--fg:#1d1f23;--muted:#5f6670;--line:#e2e5ea;--code:#f4f6f8;--accent:#3d6fc4;--side:#f8f9fb}
:root[data-theme=dark]{--bg:#16181c;--fg:#e6e8eb;--muted:#9aa0a8;--line:#2c3038;--code:#22262d;--accent:#7aa7f0;--side:#1b1e23}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#16181c;--fg:#e6e8eb;--muted:#9aa0a8;
--line:#2c3038;--code:#22262d;--accent:#7aa7f0;--side:#1b1e23}}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
nav{position:fixed;top:0;left:0;bottom:0;width:270px;overflow:auto;padding:20px 16px;background:var(--side);
border-right:1px solid var(--line);font-size:13.5px}
nav .title{font-weight:600;font-size:15px;margin-bottom:12px}nav a{display:block;color:var(--muted);
text-decoration:none;padding:3px 6px;border-radius:5px}nav a:hover{color:var(--fg);background:var(--line)}
nav a.h3{padding-left:20px;font-size:12.5px}
#theme{margin-top:14px;font:inherit;padding:4px 10px;border:1px solid var(--line);border-radius:6px;
background:var(--bg);color:var(--fg);cursor:pointer}
main{margin-left:270px;padding:24px 48px 80px;max-width:1060px}
h1{font-size:28px;margin:8px 0 12px}h2{font-size:22px;margin:40px 0 10px;padding-bottom:6px;
border-bottom:1px solid var(--line)}h3{font-size:17px;margin:26px 0 8px}h1,h2,h3{scroll-margin-top:12px}
a{color:var(--accent)}hr{border:0;border-top:1px solid var(--line);margin:28px 0}
code{background:var(--code);padding:1px 5px;border-radius:4px;font:13px/1.5 ui-monospace,Consolas,monospace}
pre{background:var(--code);padding:12px 14px;border-radius:8px;overflow:auto}pre code{padding:0;background:none}
table{border-collapse:collapse;width:100%;margin:10px 0 16px;font-size:14px;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:6px 9px;vertical-align:top;text-align:left}th{background:var(--side)}
blockquote{margin:12px 0;padding:4px 14px;border-left:3px solid var(--accent);color:var(--muted)}
.footer{color:var(--muted);font-size:12px;margin-top:40px}
@media (max-width:860px){nav{position:static;width:auto;border-right:0;border-bottom:1px solid var(--line)}
main{margin-left:0;padding:16px}}
@media print{nav{display:none}main{margin:0;max-width:none}a{color:inherit}}
#search{width:100%;font:inherit;padding:6px 9px;margin:0 0 6px;border:1px solid var(--line);border-radius:6px;
background:var(--bg);color:var(--fg)}#search:focus{outline:2px solid var(--accent);outline-offset:-1px}
#hits{color:var(--muted);font-size:12px;min-height:16px;margin-bottom:8px}
mark.hit{background:#ffe066;color:#1d1f23;border-radius:2px;padding:0 1px}mark.hit.cur{background:#ff9f1c}
nav a.hide{display:none}
@media print{mark.hit{background:none;color:inherit}}
"""

SEARCH_JS = """
(function(){var box=document.getElementById('search'),out=document.getElementById('hits'),
main=document.querySelector('main'),links=[].slice.call(document.querySelectorAll('nav a[href^="#"]')),
heads=[].slice.call(main.querySelectorAll('h1,h2,h3')),marks=[],cur=-1,timer=null;
function clear(){var parents=[];marks.forEach(function(m){var p=m.parentNode;p.replaceChild(document.createTextNode(m.textContent),m);
if(parents.indexOf(p)<0)parents.push(p)});parents.forEach(function(p){p.normalize()});marks=[];cur=-1;
links.forEach(function(a){a.classList.remove('hide')});out.textContent=''}
function sectionOf(node){var h=null;for(var i=0;i<heads.length;i++){
if(heads[i].compareDocumentPosition(node)&Node.DOCUMENT_POSITION_FOLLOWING)h=heads[i];else break}return h}
function run(q){clear();q=(q||'').trim();if(q.length<2)return;var lq=q.toLowerCase(),nodes=[],
w=document.createTreeWalker(main,NodeFilter.SHOW_TEXT,null);
while(w.nextNode())nodes.push(w.currentNode);
nodes.forEach(function(n){var t=n.nodeValue,lt=t.toLowerCase(),i=lt.indexOf(lq);if(i<0)return;
var frag=document.createDocumentFragment(),last=0;while(i>=0){frag.appendChild(document.createTextNode(t.slice(last,i)));
var m=document.createElement('mark');m.className='hit';m.textContent=t.slice(i,i+q.length);frag.appendChild(m);marks.push(m);
last=i+q.length;i=lt.indexOf(lq,last)}frag.appendChild(document.createTextNode(t.slice(last)));n.parentNode.replaceChild(frag,n)});
var ids={};marks.forEach(function(m){var h=m.closest('h1,h2,h3')||sectionOf(m);if(h&&h.id){ids[h.id]=1;
var k=heads.indexOf(h);if(h.tagName==='H3')for(var j=k;j>=0;j--)if(heads[j].tagName==='H2'){ids[heads[j].id]=1;break}}});
links.forEach(function(a){if(!ids[a.getAttribute('href').slice(1)])a.classList.add('hide')});
out.textContent=marks.length?'':'no matches';if(marks.length)go(0)}
function go(k){if(!marks.length)return;if(cur>=0)marks[cur].classList.remove('cur');cur=(k+marks.length)%marks.length;
marks[cur].classList.add('cur');marks[cur].scrollIntoView({block:'center'});
out.textContent=(cur+1)+' / '+marks.length+' · Enter = next, Shift+Enter = previous, Esc = clear'}
box.addEventListener('input',function(){clearTimeout(timer);timer=setTimeout(function(){run(box.value)},150)});
box.addEventListener('keydown',function(e){if(e.key==='Enter'){e.preventDefault();if(!marks.length)run(box.value);
else go(cur+(e.shiftKey?-1:1))}else if(e.key==='Escape'){box.value='';clear()}});
document.addEventListener('keydown',function(e){if(e.key==='/'&&document.activeElement!==box){e.preventDefault();box.focus()}});
var q=new URLSearchParams(location.search).get('q');if(q){box.value=q;run(q)}
window.rsoGuideSearch=function(q){run(q);return marks.length};
})();
"""

JS = """
(function(){var r=document.documentElement,k='rso-guide-theme',b=document.getElementById('theme');
try{var t=localStorage.getItem(k);if(t)r.setAttribute('data-theme',t)}catch(e){}
function dark(){var t=r.getAttribute('data-theme');return t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches}
function label(){b.textContent=dark()?'Light mode':'Dark mode'}label();
b.onclick=function(){var t=dark()?'light':'dark';r.setAttribute('data-theme',t);try{localStorage.setItem(k,t)}catch(e){}label()};
})();
"""


def slug(text: str) -> str:
    """GitHub-style heading anchor: lowercase, drop punctuation, spaces -> hyphens."""
    s = re.sub(r"[^\w\- ]", "", text.strip().lower())
    return s.replace(" ", "-")


def render(md_text: str) -> str:
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = md.parse(md_text)
    toc: list[tuple[int, str, str]] = []
    title = "User Guide"
    seen: dict[str, int] = {}
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open":
            text = "".join(
                c.content for c in (tokens[i + 1].children or []) if c.type in ("text", "code_inline")
            )
            sid = slug(text)
            if sid in seen:
                seen[sid] += 1
                sid = f"{sid}-{seen[sid]}"
            else:
                seen[sid] = 0
            tok.attrSet("id", sid)
            level = int(tok.tag[1])
            if level == 1:
                title = text
            elif level in (2, 3) and text != "Contents":
                toc.append((level, sid, text))
    body = md.renderer.render(tokens, md.options, {})
    nav = "".join(f'<a class="h{lv}" href="#{sid}">{html.escape(t)}</a>' for lv, sid, t in toc)
    return (
        "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>\n"
        f"<nav><div class='title'>{html.escape(title)}</div>"
        "<input id='search' type='search' placeholder='Search the guide  ( / )' aria-label='Search the guide'"
        " autocomplete='off'><div id='hits'></div>"
        f"{nav}"
        "<button id='theme' type='button'>Dark mode</button></nav>\n"
        f"<main>\n{body}\n<p class='footer'>Generated from docs/USER_GUIDE.md by scripts/build_guide.py.</p>"
        f"</main>\n<script>{JS}{SEARCH_JS}</script>\n</body></html>\n"
    )


def build() -> str:
    return render(SRC.read_text(encoding="utf-8"))


def main() -> int:
    page = build()
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != page:
            print("docs/USER_GUIDE.html is out of date: run python scripts/build_guide.py")
            return 1
        print("docs/USER_GUIDE.html is up to date")
        return 0
    OUT.write_text(page, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(page) // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
