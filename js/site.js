(function () {
  var root = document.documentElement;

  /* Theme */

  function storedTheme() {
    try { return localStorage.getItem("theme"); } catch (e) { return null; }
  }

  // Swap themes with every transition off, so colors don't smear. The toggle's
  // own icon cross-fade is exempt.
  function applyTheme(theme) {
    var style = document.createElement("style");
    style.textContent = "*:not(.swap-icon),*::before,*::after{transition:none !important}";
    document.head.appendChild(style);
    root.dataset.theme = theme;
    void document.body.offsetHeight;
    document.dispatchEvent(new CustomEvent("themechange", { detail: theme }));
    requestAnimationFrame(function () {
      requestAnimationFrame(function () { style.remove(); });
    });
  }

  var toggle = document.querySelector(".theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = root.dataset.theme === "dark" ? "light" : "dark";
      applyTheme(next);
      try { localStorage.setItem("theme", next); } catch (e) {}
    });
  }

  var mql = window.matchMedia("(prefers-color-scheme: dark)");
  mql.addEventListener("change", function (e) {
    if (!storedTheme()) applyTheme(e.matches ? "dark" : "light");
  });

  /* Copy link */

  document.querySelectorAll(".copy-link").forEach(function (button) {
    var status = button.parentElement.querySelector(".copy-status");
    var timer;
    button.addEventListener("click", function () {
      if (!navigator.clipboard) return;
      navigator.clipboard.writeText(button.dataset.url).then(function () {
        button.dataset.copied = "true";
        if (status) status.textContent = "Link copied";
        clearTimeout(timer);
        timer = setTimeout(function () {
          delete button.dataset.copied;
          if (status) status.textContent = "";
        }, 2000);
      });
    });
  });

  /* Heading anchors */

  document.querySelectorAll(".prose h2[id], .prose h3[id]").forEach(function (heading) {
    var link = document.createElement("a");
    link.className = "heading-anchor";
    link.href = "#" + heading.id;
    link.setAttribute("aria-label", "Link to this section");
    link.textContent = "#";
    heading.appendChild(link);
  });

  /* Syntax highlighting, only for code blocks that name a language */

  function highlight() {
    if (!window.hljs) return;
    // The bundled highlight.js is 10.5, which only has highlightBlock (renamed highlightElement in 10.7)
    var highlightOne = window.hljs.highlightElement || window.hljs.highlightBlock;
    document.querySelectorAll('pre code[class*="language-"]:not(.language-mermaid)').forEach(function (block) {
      highlightOne.call(window.hljs, block);
    });
  }
  if (window.hljs) highlight();
  else window.addEventListener("load", highlight);
})();

/* Hover notes: <abbr title="..."> in posts, from markdown abbreviations
   (*[term]: note) or inline HTML, shown in a styled tooltip */
(function () {
  var terms = document.querySelectorAll(".prose abbr[title]");
  if (!terms.length) return;

  var tip = document.createElement("div");
  tip.className = "tooltip";
  tip.id = "tooltip";
  tip.setAttribute("role", "tooltip");
  document.body.appendChild(tip);

  var current = null;
  var tapped = null;
  var GAP = 8;
  var EDGE = 8;

  function position(el) {
    // Anchor to the line the term starts on, even if it wraps
    var rect = el.getClientRects()[0] || el.getBoundingClientRect();
    var w = tip.offsetWidth;
    var h = tip.offsetHeight;
    var top = rect.top - h - GAP;
    var side = "top";
    if (top < EDGE) {
      top = rect.bottom + GAP;
      side = "bottom";
    }
    var left = rect.left + rect.width / 2 - w / 2;
    left = Math.max(EDGE, Math.min(left, window.innerWidth - w - EDGE));
    tip.style.top = top + "px";
    tip.style.left = left + "px";
    tip.dataset.side = side;
  }

  function show(el) {
    if (current && current !== el) current.removeAttribute("aria-describedby");
    current = el;
    tip.textContent = el.dataset.tip;
    position(el);
    tip.dataset.open = "";
    el.setAttribute("aria-describedby", "tooltip");
  }

  function hide() {
    if (!current) return;
    current.removeAttribute("aria-describedby");
    current = null;
    tapped = null;
    delete tip.dataset.open;
  }

  terms.forEach(function (el) {
    // Move the note off `title` so the browser's own tooltip doesn't double it
    el.dataset.tip = el.getAttribute("title");
    el.removeAttribute("title");
    el.tabIndex = 0;

    el.addEventListener("pointerenter", function (e) {
      if (e.pointerType === "mouse") show(el);
    });
    el.addEventListener("pointerleave", function (e) {
      if (e.pointerType === "mouse" && current === el) hide();
    });
    el.addEventListener("focus", function () { show(el); });
    el.addEventListener("blur", function () { if (current === el) hide(); });
    el.addEventListener("click", function (e) {
      // A tap or click toggles, so touch screens can open and close it
      e.stopPropagation();
      if (tapped === el) {
        hide();
      } else {
        show(el);
        tapped = el;
      }
    });
  });

  document.addEventListener("click", hide);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") hide();
  });
  window.addEventListener("scroll", hide, { passive: true });
  window.addEventListener("resize", hide);
})();

/* Reading progress */
(function () {
  var bar = document.querySelector(".progress");
  var article = document.querySelector(".post .prose");
  if (!bar || !article) return;
  var queued = false;
  function update() {
    queued = false;
    var rect = article.getBoundingClientRect();
    var total = rect.height - window.innerHeight * 0.5;
    var done = Math.min(1, Math.max(0, -rect.top / Math.max(total, 1)));
    bar.style.transform = "scaleX(" + done + ")";
  }
  function queue() {
    if (!queued) {
      queued = true;
      requestAnimationFrame(update);
    }
  }
  window.addEventListener("scroll", queue, { passive: true });
  window.addEventListener("resize", queue);
  update();
})();

/* Table of contents, built from the post's h2 and h3 headings */
(function () {
  var side = document.querySelector(".toc-side");
  var inline = document.querySelector(".toc-inline");
  if (!side || !inline) return;
  var headings = Array.prototype.slice.call(
    document.querySelectorAll(".post .prose h2[id], .post .prose h3[id]")
  );
  if (headings.length < 3) return;

  function buildList() {
    var ul = document.createElement("ul");
    headings.forEach(function (h) {
      var li = document.createElement("li");
      li.className = "toc-" + h.tagName.toLowerCase();
      var a = document.createElement("a");
      a.href = "#" + h.id;
      // Heading text without the "#" anchor link appended to it
      var clone = h.cloneNode(true);
      clone.querySelectorAll(".heading-anchor").forEach(function (n) { n.remove(); });
      a.textContent = clone.textContent.trim();
      a.dataset.target = h.id;
      li.appendChild(a);
      ul.appendChild(li);
    });
    return ul;
  }

  side.querySelector("nav").appendChild(buildList());
  inline.querySelector("nav").appendChild(buildList());
  side.hidden = false;
  inline.hidden = false;

  inline.addEventListener("click", function (e) {
    if (e.target.closest("a")) inline.open = false;
  });

  var links = side.querySelectorAll("a");
  var active = null;
  var queued = false;
  function spy() {
    queued = false;
    var current = headings[0];
    for (var i = 0; i < headings.length; i++) {
      if (headings[i].getBoundingClientRect().top <= window.innerHeight * 0.3) current = headings[i];
      else break;
    }
    if (current === active) return;
    active = current;
    links.forEach(function (a) {
      if (a.dataset.target === current.id) a.setAttribute("aria-current", "location");
      else a.removeAttribute("aria-current");
    });
  }
  window.addEventListener("scroll", function () {
    if (!queued) {
      queued = true;
      requestAnimationFrame(spy);
    }
  }, { passive: true });
  spy();
})();

/* Mermaid diagrams: ```mermaid code blocks, loaded only when a post has one */
(function () {
  var blocks = document.querySelectorAll("pre > code.language-mermaid");
  if (!blocks.length) return;

  var figures = [];
  blocks.forEach(function (code) {
    var figure = document.createElement("figure");
    figure.className = "diagram";
    figure.dataset.source = code.textContent;
    code.parentElement.replaceWith(figure);
    figures.push(figure);
  });

  var mermaidReady = import("https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.esm.min.mjs")
    .then(function (m) { return m.default; });

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function render() {
    mermaidReady.then(function (mermaid) {
      mermaid.initialize({
        startOnLoad: false,
        theme: "base",
        fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
        themeVariables: {
          fontSize: "14px",
          background: "transparent",
          primaryColor: cssVar("--bg"),
          primaryTextColor: cssVar("--fg"),
          primaryBorderColor: cssVar("--border-strong"),
          secondaryColor: cssVar("--surface"),
          tertiaryColor: cssVar("--surface"),
          lineColor: cssVar("--fg-muted"),
          textColor: cssVar("--fg"),
          mainBkg: cssVar("--bg"),
          clusterBkg: cssVar("--surface"),
          clusterBorder: cssVar("--border-strong"),
          edgeLabelBackground: cssVar("--surface"),
          noteBkgColor: cssVar("--accent-soft"),
          noteBorderColor: cssVar("--accent"),
          noteTextColor: cssVar("--fg")
        }
      });
      figures.forEach(function (figure, i) {
        mermaid.render("diagram-" + i + "-" + Date.now(), figure.dataset.source).then(function (out) {
          figure.innerHTML = out.svg;
          // Mermaid sizes the SVG to fill its box, which shrinks wide diagrams
          // until the labels are unreadable. Use its natural width instead.
          var svg = figure.querySelector("svg");
          var box = svg && svg.viewBox && svg.viewBox.baseVal;
          if (box && box.width) {
            svg.style.width = box.width + "px";
            svg.style.maxWidth = "none";
          }
        }).catch(function (err) {
          figure.textContent = "Diagram error: " + err.message;
        });
      });
    });
  }

  render();
  document.addEventListener("themechange", render);
})();

/* Math: $$...$$ in markdown, typeset with KaTeX only when a post has math */
(function () {
  var prose = document.querySelector(".prose");
  if (!prose) return;
  var text = prose.textContent;
  if (text.indexOf("\\(") === -1 && text.indexOf("\\[") === -1) return;

  var css = document.createElement("link");
  css.rel = "stylesheet";
  css.href = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css";
  document.head.appendChild(css);

  import("https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.mjs").then(function (m) {
    m.default(prose, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false }
      ],
      throwOnError: false
    });
  });
})();
