(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const t = (key) => (window.NVC_I18N[state.lang] || window.NVC_I18N.en)[key] || key;

  const state = {
    lang: localStorage.getItem("nvc-lang") || (navigator.language.startsWith("ru") ? "ru" : "en"),
    theme: localStorage.getItem("nvc-theme") || "dark",
    models: [],
    separationModels: [],
    helperRows: [
      { path: "", name: "", id: 0, repeat: 1 },
      { path: "", name: "", id: 1, repeat: 1 },
    ],
    helperPage: 0,
    pageSize: 10,
  };

  function applyI18n() {
    $$("[data-i18n]").forEach((el) => {
      if (el.classList.contains("drop-name") && el.dataset.empty === "false") return;
      el.textContent = t(el.dataset.i18n);
    });
    // Re-translate <option> placeholders that fillSelect/fillPretrained created.
    $$("select option[data-i18n]").forEach((opt) => {
      opt.textContent = t(opt.dataset.i18n);
    });
    const status = $("#hStatus");
    if (status) status.dataset.placeholder = t("statusEmpty");
    $("#themeToggle").setAttribute("aria-label", state.theme === "dark" ? t("themeToLight") : t("themeToDark"));
    document.documentElement.lang = state.lang === "ru" ? "ru" : "en";
    $("#langSelect").value = state.lang;
    $("#langSelect")._syncSelect?.();
    applySeparationModels();
    $$("[data-drop-for]").forEach(refreshDrop);
  }

  function applyTheme() {
    document.documentElement.dataset.theme = state.theme;
    document.querySelector('meta[name="theme-color"]').setAttribute("content", state.theme === "dark" ? "#000000" : "#ffffff");
    document.documentElement.style.colorScheme = state.theme;
    applyI18n();
  }

  function bindRanges() {
    $$("input[type=range]").forEach((input) => {
      const out = input.nextElementSibling;
      const sync = () => {
        if (out && out.tagName === "OUTPUT") out.value = input.value;
        const min = Number(input.min || 0);
        const max = Number(input.max || 100);
        const percent = max > min ? ((Number(input.value) - min) / (max - min)) * 100 : 0;
        input.style.setProperty("--val", `${Math.min(100, Math.max(0, percent))}%`);
      };
      input.addEventListener("input", sync);
      sync();
    });
  }

  function radio(name) {
    const el = document.querySelector(`input[name="${name}"]:checked`);
    return el ? el.value : "";
  }

  function setRadio(name, value) {
    const el = document.querySelector(`input[name="${name}"][value="${value}"]`);
    if (el) el.checked = true;
  }

  function fillSelect(sel, items, current) {
    const keep = current ?? sel.value;
    sel.innerHTML = "";
    items.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = typeof item === "object" ? item.value : item;
      opt.textContent = typeof item === "object" ? item.label : item;
      sel.appendChild(opt);
    });
    if (keep && items.some((item) => (typeof item === "object" ? item.value : item) === keep)) sel.value = keep;
    else if (items.length) sel.selectedIndex = 0;
  }

  function enhanceSelects(root = document) {
    $$(`select:not(.player-format)`, root).forEach((select) => {
      if (select.dataset.selectReady) return;
      select.dataset.selectReady = "true";

      const wrap = document.createElement("div");
      wrap.className = "sel";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sel-btn";
      button.setAttribute("aria-haspopup", "listbox");
      button.setAttribute("aria-expanded", "false");
      const value = document.createElement("span");
      value.className = "sel-value";
      const chevron = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      chevron.classList.add("chev");
      chevron.setAttribute("viewBox", "0 0 16 16");
      chevron.setAttribute("aria-hidden", "true");
      chevron.innerHTML = '<path d="m4 6 4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>';
      button.append(value, chevron);

      const list = document.createElement("div");
      list.className = "sel-list";
      list.setAttribute("role", "listbox");
      list.hidden = true;

      select.parentNode.insertBefore(wrap, select);
      wrap.append(button, list, select);
      select.classList.add("sel-native");
      select.setAttribute("aria-hidden", "true");
      select.tabIndex = -1;

      let closeTimer = 0;
      const close = (instant = false) => {
        window.clearTimeout(closeTimer);
        wrap.classList.remove("open");
        button.setAttribute("aria-expanded", "false");
        if (instant || list.hidden) {
          wrap.classList.remove("closing");
          list.hidden = true;
          return;
        }
        wrap.classList.add("closing");
        closeTimer = window.setTimeout(() => {
          wrap.classList.remove("closing");
          list.hidden = true;
        }, 120);
      };

      const sync = () => {
        value.textContent = select.selectedOptions[0]?.textContent || "";
        button.disabled = select.disabled;
        button.setAttribute("aria-label", select.getAttribute("aria-label") || value.textContent);
        list.innerHTML = "";
        [...select.options].forEach((option, index) => {
          const item = document.createElement("button");
          item.type = "button";
          item.className = "sel-item";
          item.setAttribute("role", "option");
          item.setAttribute("aria-selected", option.selected ? "true" : "false");
          item.classList.toggle("is-active", option.selected);
          item.disabled = option.disabled;
          item.textContent = option.textContent;
          item.addEventListener("click", () => {
            select.selectedIndex = index;
            select.dispatchEvent(new Event("change", { bubbles: true }));
            sync();
            close();
            button.focus();
          });
          list.appendChild(item);
        });
      };

      const open = () => {
        if (button.disabled) return;
        $$(".sel.open").forEach((other) => {
          if (other !== wrap) other._closeSelect?.();
        });
        sync();
        window.clearTimeout(closeTimer);
        wrap.classList.remove("closing");
        list.hidden = false;
        wrap.classList.add("open");
        button.setAttribute("aria-expanded", "true");
      };

      wrap._closeSelect = close;
      select._syncSelect = sync;
      button.addEventListener("click", () => wrap.classList.contains("open") ? close() : open());
      button.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
          close();
          return;
        }
        if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const last = Math.max(0, select.options.length - 1);
        let index = select.selectedIndex;
        if (event.key === "ArrowDown") index = Math.min(last, index + 1);
        if (event.key === "ArrowUp") index = Math.max(0, index - 1);
        if (event.key === "Home") index = 0;
        if (event.key === "End") index = last;
        select.selectedIndex = index;
        select.dispatchEvent(new Event("change", { bubbles: true }));
        sync();
        open();
      });
      select.addEventListener("change", sync);
      new MutationObserver(sync).observe(select, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ["disabled", "label"],
      });
      sync();
    });

    if (!document.documentElement.dataset.selectDismissReady) {
      document.documentElement.dataset.selectDismissReady = "true";
      document.addEventListener("pointerdown", (event) => {
        $$(".sel.open").forEach((wrap) => {
          if (!wrap.contains(event.target)) wrap._closeSelect?.();
        });
      });
      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") $$(".sel.open").forEach((wrap) => wrap._closeSelect?.());
      });
    }
  }

  const logPatterns = [
    [/^失败\n/, () => t("failed") + "\n"],
    [/正在准备 PyMSS 分离任务/g, () => t("stPreparing")],
    [/正在处理音频/g, () => t("stProcessingAudio")],
    [/文件处理结束/g, () => t("stFileDone")],
    [/文件 (\d+)\/(\d+) · (.*?) · ([\d.]+)\/([\d.]+) 秒/g, (m, a, b, c, d, e) => t("stFileProgress").replace("%s", a).replace("%s", b).replace("%s", c).replace("%s", d).replace("%s", e)],
    [/分离完成：成功 (\d+)，失败 (\d+)/g, (m, a, b) => t("stSepDone").replace("%s", a).replace("%s", b)],
    [/PyMSS 分离任务已停止。?/g, () => t("stSepStopped")],
    [/PyMSS 分离任务失败/g, () => t("stSepFailed")],
    [/(^|\n)(.*?) 模型已加载 \| 参数精度 (\w+)/g, (m, br, dev, dtype) => br + t("stModelLoaded").replace("%s", dev).replace("%s", dtype)],
    [/\[(\d+)\/(\d+)\] 开始处理 (.*)/g, (m, a, b, c) => t("stFileStart").replace("%s", a).replace("%s", b).replace("%s", c)],
    [/(.*) -> 成功 \| (\w+) \| 推理 ([\d.]+)s \| 编码 ([\d.]+)s/g, (m, f, dt, a, b) => t("stFileOkDtype").replace("%s", f).replace("%s", dt).replace("%s", a).replace("%s", b)],
    [/(.*) -> 成功 \| 推理 ([\d.]+)s \| 编码 ([\d.]+)s/g, (m, f, a, b) => t("stFileOk").replace("%s", f).replace("%s", a).replace("%s", b)],
    [/-> 失败/g, () => "-> " + t("stFileFail")],
    [/DirectML FP16 路径不兼容，准备改用 FP32/g, () => t("stFp32")],
    [/子进程失败/g, () => t("stSubprocFail")],
    [/当前没有正在运行的 PyMSS 任务。/g, () => t("stNoTask")],
    [/已请求停止 PyMSS 分离任务。/g, () => t("stStopReq")],
    [/已有 PyMSS 分离任务正在运行，请先停止当前任务。/g, () => t("stBusy")],
    [/没有找到可处理的音频文件/g, () => t("stNoFiles")],
    [/输出文件夹不能为空/g, () => t("stNoOutDir")],
    [/音频编码没有生成有效文件/g, () => t("stBadEncode")],
    [/模型缺少输出 stem/g, () => t("stMissingStem")],
    [/模型输出包含 NaN\/Inf/g, () => t("stNan")],
    [/项目内置的 pymss 运行库加载失败/g, () => t("stRuntimeFail")],
  ];

  function translateLog(text) {
    let out = String(text ?? "");
    for (const [pattern, fn] of logPatterns) out = out.replace(pattern, fn);
    return out;
  }

  function stemLabel(suffix) {
    const key = "stem_" + suffix;
    const value = t(key);
    return value === key ? suffix : value;
  }

  function renderSeparationResults(files) {
    const box = $("#pResults");
    if (!box) return;
    box.innerHTML = "";
    if (!files || !files.length) {
      box.hidden = true;
      return;
    }
    box.hidden = false;
    const title = document.createElement("span");
    title.className = "log-label";
    title.textContent = t("resultsLabel");
    box.appendChild(title);
    files.forEach((item) => {
      const group = document.createElement("div");
      group.className = "result-group";
      if (files.length > 1 && item.name) {
        const name = document.createElement("span");
        name.className = "result-name";
        name.textContent = item.name;
        group.appendChild(name);
      }
      [
        [item.primary, item.primary_suffix],
        [item.secondary, item.secondary_suffix],
      ].forEach(([path, suffix]) => {
        if (!path) return;
        const label = document.createElement("span");
        label.className = "result-label";
        label.textContent = stemLabel(suffix);
        const audio = document.createElement("audio");
        audio.preload = "metadata";
        audio.dataset.downloadFormats = "wav,mp3,flac";
        audio.dataset.downloadName = path.split(/[\\/]/).pop();
        audio.src = "/api/file?path=" + encodeURIComponent(path);
        group.append(label, audio);
      });
      box.appendChild(group);
    });
    enhanceAudioPlayers(box);
  }

  const separationLabelKeys = {
    "去混响": "sepDereverb",
    "去混响（激进）": "sepDereverbAggressive",
    "去伴奏": "sepVocals",
    "去伴奏（激进）": "sepVocalsAggressive",
    "提主旋律": "sepMainVocal",
  };

  function applySeparationModels() {
    const select = $("#pModel");
    if (!select || !state.separationModels.length) return;
    const current = select.value;
    fillSelect(select, state.separationModels.map((value) => ({
      value,
      label: separationLabelKeys[value] ? t(separationLabelKeys[value]) : value,
    })), current);
  }

  function applyIndexChoices(choices, selected) {
    const select = $("#inferIndex");
    if (!select) return;
    const values = [...new Set([...(choices || []), selected].filter(Boolean))];
    const items = [];
    if (values.length) {
      // Always offer the "do not use index" option first, and keep it as the
      // default selection even after a new index is imported.
      items.push({ value: "", label: t("noIndex"), i18n: "noIndex" });
    }
    values.forEach((value) => {
      items.push({ value, label: value.split(/[\\/]/).pop() });
    });
    fillSelect(select, items.length ? items : [""], "");
    if (items.length && select.options[0]) {
      select.options[0].dataset.i18n = "noIndex";
      select.options[0].textContent = t("noIndex");
    }
  }

  async function api(url, options) {
    const res = await fetch(url, options);
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || t("failed"));
    return data;
  }

  async function readSSE(url, form, onEvent, startBtn, stopBtn) {
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.hidden = false;
    try {
      const res = await fetch(url, { method: "POST", body: form });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const chunks = buf.split("\n\n");
        buf = chunks.pop();
        for (const chunk of chunks) {
          const line = chunk.split("\n").find((row) => row.startsWith("data: "));
          if (!line) continue;
          const event = JSON.parse(line.slice(6));
          onEvent(event);
          if (event.start_visible === false && stopBtn) stopBtn.hidden = false;
          if (event.stop_visible === false && stopBtn) stopBtn.hidden = true;
          if (event.done) return event;
        }
      }
    } finally {
      if (startBtn) startBtn.disabled = false;
      if (stopBtn) stopBtn.hidden = true;
    }
  }

  function trainForm() {
    const data = new FormData();
    data.set("exp", $("#expName").value);
    data.set("sr", radio("sr") || "40k");
    data.set("if_f0", radio("ifF0"));
    data.set("version", radio("ver"));
    data.set("n_p", $("#cpuWorkers").value);
    data.set("mode", radio("mode"));
    data.set("trainset", $("#trainset").value);
    data.set("spk_id", $("#spkId").value);
    data.set("gpus", $("#hubertGpus").value);
    data.set("gpus_rmvpe", $("#rmvpeGpus").value);
    data.set("f0_method", radio("tf0"));
    data.set("save_epoch", $("#saveEvery").value);
    data.set("total_epoch", $("#totalEpoch").value);
    data.set("batch_size", $("#batchSize").value);
    data.set("save_latest", radio("latest"));
    data.set("cache_gpu", radio("cache"));
    data.set("save_every", radio("small"));
    data.set("pretrained_g", $("#preG").value);
    data.set("pretrained_d", $("#preD").value);
    data.set("train_gpus", $("#trainGpus").value);
    data.set("embedder", $("#embedder")?.value || "hubert_base");
    data.set("noise_reduction", radio("noiseReduce") === "1" ? "true" : "false");
    data.set("reduction_strength", $("#reduceStrength")?.value || 0.75);
    data.set("ms_mel", radio("msMel") === "1" ? "true" : "false");
    data.set("gradient_checkpointing", radio("gradCkpt") === "1" ? "true" : "false");
    data.set("bf16", radio("bf16") === "1" ? "true" : "false");
    return data;
  }

  function renderHelper() {
    const total = Math.max(1, Math.ceil(state.helperRows.length / state.pageSize));
    state.helperPage = Math.max(0, Math.min(state.helperPage, total - 1));
    $("#hPage").textContent = `${state.helperPage + 1} / ${total} · ${state.helperRows.length}`;
    const start = state.helperPage * state.pageSize;
    const slice = state.helperRows.slice(start, start + state.pageSize);
    $("#hRows").innerHTML = slice.map((row, idx) => `
      <div class="row" data-index="${start + idx}">
        <label><span>${t("path")}</span><input data-k="path" value="${escapeAttr(row.path)}" /></label>
        <label><span>${t("name")}</span><input data-k="name" value="${escapeAttr(row.name)}" /></label>
        <label><span>${t("id")}</span><input data-k="id" type="number" min="0" max="109" value="${row.id ?? ""}" /></label>
        <label><span>${t("repeat")}</span><input data-k="repeat" type="number" min="1" value="${row.repeat ?? ""}" /></label>
      </div>
    `).join("");
    $$("#hRows .row input").forEach((input) => {
      input.addEventListener("input", () => {
        const index = Number(input.closest(".row").dataset.index);
        const key = input.dataset.k;
        state.helperRows[index][key] = input.type === "number" ? Number(input.value) : input.value;
      });
    });
  }

  function escapeAttr(value) {
    return String(value ?? "").replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
  }

  function applyModels(models) {
    state.models = models || [];
    fillSelect($("#inferModel"), state.models);
  }

  async function loadPretrainedChoices(preferredG, preferredD) {
    const query = new URLSearchParams({
      sr: radio("sr") || "40k",
      if_f0: radio("ifF0") === "1" ? "1" : "0",
      version: radio("ver") || "v2",
    });
    const data = await api(`/api/train/pretrained?${query}`);
    const fillPretrained = (selector, choices, preferred) => {
      const select = $(selector);
      fillSelect(select, choices || [""], preferred ?? select.value);
      const empty = [...select.options].find((option) => option.value === "");
      if (empty) {
        empty.dataset.i18n = "noPretrained";
        empty.textContent = t("noPretrained");
      }
    };
    fillPretrained("#preG", data.generator, preferredG);
    fillPretrained("#preD", data.discriminator, preferredD);
  }

  async function loadFaq() {
    const data = await api(`/api/faq?lang=${state.lang}`);
    const body = $("#faqBody");
    if (window.marked) {
      body.innerHTML = window.marked.parse(data.markdown || "");
    } else {
      body.textContent = data.markdown || "";
    }
  }

  function refreshDrop(zone) {
    const input = $("#" + zone.dataset.dropFor);
    const label = zone.querySelector(".drop-name");
    if (!input || !label) return;
    const files = [...input.files];
    const clear = zone.querySelector(".drop-clear");
    if (clear) {
      clear.hidden = !files.length;
      clear.setAttribute("aria-label", t(files.length > 1 ? "clearFiles" : "clearFile"));
    }
    if (!files.length) {
      label.dataset.empty = "true";
      label.textContent = t("noFile");
    } else {
      label.dataset.empty = "false";
      label.textContent = files.length === 1 ? files[0].name : t("filesSelected").replace("%s", files.length);
    }
    syncAudioPreview(zone, files);
  }

  function syncAudioPreview(zone, files) {
    const preview = zone.dataset.previewFor ? $("#" + zone.dataset.previewFor) : null;
    if (!preview) return;
    if (zone._previewUrl) {
      URL.revokeObjectURL(zone._previewUrl);
      zone._previewUrl = "";
    }
    const audio = files.find((file) => file.type.startsWith("audio/") || /\.(aac|flac|m4a|mp3|ogg|opus|wav)$/i.test(file.name));
    if (!audio) {
      preview.hidden = true;
      preview.removeAttribute("src");
      delete preview.dataset.downloadName;
      preview.load();
      return;
    }
    zone._previewUrl = URL.createObjectURL(audio);
    preview.src = zone._previewUrl;
    preview.dataset.downloadName = audio.name;
    preview.hidden = false;
    preview.setAttribute("aria-label", t("previewAudio"));
    preview.load();
  }

  function fmtTime(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) seconds = 0;
    const min = Math.floor(seconds / 60);
    const sec = Math.floor(seconds % 60);
    return `${min}:${String(sec).padStart(2, "0")}`;
  }

  function enhanceAudioPlayers(root = document) {
    $$("audio", root).forEach((audio) => {
      if (audio.dataset.playerReady) return;
      audio.dataset.playerReady = "true";
      const wrap = document.createElement("div");
      wrap.className = "player";
      wrap.hidden = audio.hidden;
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "player-toggle";
      toggle.setAttribute("aria-label", t("play"));
      toggle.innerHTML =
        '<svg class="i-play" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z" fill="currentColor"/></svg>' +
        '<svg class="i-pause" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h4v14H7zM13 5h4v14h-4z" fill="currentColor"/></svg>';
      const track = document.createElement("div");
      track.className = "player-track";
      track.setAttribute("role", "slider");
      track.setAttribute("aria-label", t("seek"));
      track.tabIndex = 0;
      const fill = document.createElement("div");
      fill.className = "player-fill";
      track.appendChild(fill);
      const time = document.createElement("span");
      time.className = "player-time";
      time.textContent = "0:00 / 0:00";
      const volume = document.createElement("label");
      volume.className = "player-volume";
      volume.title = t("volume");
      volume.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 10v4h4l5 4V6L8 10H4zm12.5-1.5a5 5 0 0 1 0 7m2-9a8 8 0 0 1 0 11" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
      const volumeInput = document.createElement("input");
      volumeInput.type = "range";
      volumeInput.className = "player-volume-input";
      volumeInput.min = "0";
      volumeInput.max = "1";
      volumeInput.step = "0.01";
      volumeInput.value = String(audio.volume);
      volumeInput.setAttribute("aria-label", t("volume"));
      volume.appendChild(volumeInput);
      const download = document.createElement("a");
      download.className = "player-download";
      download.setAttribute("aria-label", t("download"));
      download.title = t("download");
      download.hidden = true;
      download.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11m0 0 4.5-4.5M12 15l-4.5-4.5M5 19h14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';
      const formats = (audio.dataset.downloadFormats || "").split(",").map((f) => f.trim()).filter(Boolean);
      let formatSelect = null;
      if (formats.length) {
        formatSelect = document.createElement("select");
        formatSelect.className = "player-format";
        formatSelect.setAttribute("aria-label", t("downloadFormat"));
        formatSelect.hidden = true;
        formats.forEach((format) => {
          const option = document.createElement("option");
          option.value = format;
          option.textContent = format;
          formatSelect.appendChild(option);
        });
      }
      audio.parentNode.insertBefore(wrap, audio);
      wrap.append(toggle, track, time, volume, ...(formatSelect ? [formatSelect] : []), download, audio);

      const downloadName = (src) => {
        if (audio.dataset.downloadName) return audio.dataset.downloadName;
        try {
          const url = new URL(src, location.href);
          const served = url.searchParams.get("path");
          const base = (served || url.pathname).split(/[\\/]/).pop();
          if (base) return base;
        } catch { /* ignore */ }
        return "audio.wav";
      };
      const syncDownload = () => {
        const src = audio.currentSrc || audio.src;
        download.hidden = !src;
        if (formatSelect) formatSelect.hidden = !src;
        if (!src) return;
        let href = src;
        let name = downloadName(src);
        const format = formatSelect ? formatSelect.value : "";
        if (format && !src.startsWith("blob:")) {
          const url = new URL(src, location.href);
          url.searchParams.set("format", format);
          href = url.pathname + url.search;
          name = name.replace(/\.[^.]+$/, "") + "." + format;
        }
        download.href = href;
        download.setAttribute("download", name);
      };
      if (formatSelect) formatSelect.addEventListener("change", syncDownload);

      const sync = () => {
        const dur = audio.duration;
        const ratio = Number.isFinite(dur) && dur > 0 ? audio.currentTime / dur : 0;
        fill.style.width = `${ratio * 100}%`;
        track.setAttribute("aria-valuenow", String(Math.round(ratio * 100)));
        time.textContent = `${fmtTime(audio.currentTime)} / ${fmtTime(dur)}`;
      };
      audio.addEventListener("timeupdate", sync);
      audio.addEventListener("loadedmetadata", sync);
      audio.addEventListener("durationchange", sync);
      audio.addEventListener("load", syncDownload);
      new MutationObserver(syncDownload).observe(audio, { attributes: true, attributeFilter: ["src"] });
      syncDownload();
      audio.addEventListener("play", () => {
        wrap.classList.add("is-playing");
        toggle.setAttribute("aria-label", t("pause"));
      });
      const onStop = () => {
        wrap.classList.remove("is-playing");
        toggle.setAttribute("aria-label", t("play"));
      };
      audio.addEventListener("pause", onStop);
      audio.addEventListener("ended", onStop);

      const syncVolume = () => {
        const level = audio.muted ? 0 : audio.volume;
        volumeInput.value = String(level);
        volumeInput.style.setProperty("--volume", `${level * 100}%`);
      };
      volumeInput.addEventListener("input", () => {
        audio.muted = false;
        audio.volume = Number(volumeInput.value);
      });
      audio.addEventListener("volumechange", syncVolume);
      syncVolume();

      toggle.addEventListener("click", () => {
        if (audio.paused) audio.play();
        else audio.pause();
      });

      const seek = (event) => {
        if (!Number.isFinite(audio.duration) || !audio.duration) return;
        const rect = track.getBoundingClientRect();
        const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
        audio.currentTime = ratio * audio.duration;
      };
      track.addEventListener("pointerdown", (event) => {
        track.setPointerCapture(event.pointerId);
        seek(event);
        const move = (e) => seek(e);
        track.addEventListener("pointermove", move);
        track.addEventListener("pointerup", () => track.removeEventListener("pointermove", move), { once: true });
      });
      track.addEventListener("keydown", (event) => {
        if (!Number.isFinite(audio.duration) || !audio.duration) return;
        const step = event.key === "ArrowRight" ? 5 : event.key === "ArrowLeft" ? -5 : 0;
        if (!step) return;
        event.preventDefault();
        audio.currentTime = Math.min(audio.duration, Math.max(0, audio.currentTime + step));
      });

      new MutationObserver(() => {
        wrap.hidden = audio.hidden;
        if (audio.hidden && !audio.paused) audio.pause();
      }).observe(audio, { attributes: true, attributeFilter: ["hidden"] });
      sync();
    });
  }

  function clearDrop(inputId) {
    const input = $("#" + inputId);
    if (!input) return;
    input.value = "";
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function bindDrops() {
    $$("[data-drop-for]").forEach((zone) => {
      const input = $("#" + zone.dataset.dropFor);
      if (!input) return;
      zone.addEventListener("click", (event) => {
        if (event.target.closest(".drop-clear, .player, button, input, canvas")) return;
        if (event.target === input) return;
        event.preventDefault();
        input.click();
      });
      zone.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          input.click();
        }
      });
      zone.addEventListener("dragover", (event) => {
        event.preventDefault();
        zone.classList.add("is-over");
      });
      zone.addEventListener("dragleave", () => zone.classList.remove("is-over"));
      zone.addEventListener("drop", (event) => {
        event.preventDefault();
        zone.classList.remove("is-over");
        if (event.dataTransfer?.files?.length) input.files = event.dataTransfer.files;
        input.dispatchEvent(new Event("change", { bubbles: true }));
        refreshDrop(zone);
      });
      const clear = zone.querySelector(".drop-clear");
      if (clear) {
        clear.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          clearDrop(clear.dataset.clearFor || input.id);
        });
      }
      input.addEventListener("change", () => refreshDrop(zone));
      zone.tabIndex = 0;
      refreshDrop(zone);
    });
  }

  async function uploadTrainDataset() {
    const file = $("#trainsetZip").files[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".zip")) {
      $("#preLog").textContent = t("datasetZipOnly");
      return;
    }
    const form = new FormData();
    form.set("dataset", file);
    $("#preLog").textContent = t("working");
    try {
      const data = await api("/api/train/dataset", { method: "POST", body: form });
      $("#trainset").value = data.path || "";
      $("#preLog").textContent = `${t("datasetReady")} ${data.path || ""}`;
    } catch (error) {
      $("#preLog").textContent = error.message;
    }
  }

  function syncLibraryKind() {
    const pretrained = radio("libKind") === "pretrained";
    $("#libUrlWrap").hidden = pretrained;
    $("#libPretrainedFields").hidden = !pretrained;
  }

  async function showLibraryResult(data) {
    if (data.models) {
      applyModels(data.models);
      if ($("#inferModel").value) await selectVoice();
    }
    if (data.extracted && data.extracted.length) {
      $("#libLog").textContent = `${t("done")} ${data.name || ""}\n${data.extracted.join("\n")}`;
    } else if (data.paths && data.paths.length) {
      $("#libLog").textContent = `${t("done")}\n${data.paths.join("\n")}`;
    } else {
      $("#libLog").textContent = `${t("done")} ${data.path || ""}`;
    }
  }

  async function selectVoice() {
    const model = $("#inferModel").value;
    if (!model) {
      applyIndexChoices([], "");
      return false;
    }
    try {
      const data = await api("/api/infer/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          protect0: $("#sProtect").value,
          protect1: $("#bProtect").value,
        }),
      });
      const slider = data.speaker_slider || {};
      const drop = data.speaker_dropdown || {};
      $("#speakerSliderWrap").hidden = slider.visible === false;
      $("#speakerDropWrap").hidden = drop.visible === false;
      if (slider.value != null) $("#speakerId").value = slider.value;
      if (slider.maximum != null) $("#speakerId").max = slider.maximum;
      fillSelect($("#speakerNamed"), drop.choices || []);
      if (drop.value) $("#speakerNamed").value = drop.value;
      applyIndexChoices(data.index_choices, data.index || data.index_batch || "");
      return true;
    } catch (error) {
      applyIndexChoices([], "");
      $("#sLog").textContent = error.message;
      return false;
    }
  }

  async function updateSpeakerIndex() {
    const data = await api("/api/infer/speaker-index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: $("#inferModel").value,
        speaker: $("#speakerId").value,
        speaker_label: $("#speakerNamed").value,
      }),
    });
    applyIndexChoices(data.index_choices, data.index || data.index_batch || "");
  }

  async function init() {
    applyTheme();
    bindRanges();
    bindDrops();
    enhanceAudioPlayers();
    enhanceSelects();
    const boot = await api("/api/bootstrap");
    $("#deviceReadout").textContent = `${boot.device} · ${boot.dtype}`;
    $("#gpuInfo").value = boot.gpu_info || "";
    $("#hubertGpus").value = boot.feature_gpus || "";
    $("#rmvpeGpus").value = boot.feature_gpus || "";
    $("#trainGpus").value = boot.gpus || "";
    $("#cpuWorkers").max = Math.max(1, boot.n_cpu || 8);
    $("#cpuWorkers").value = Math.ceil((boot.n_cpu || 4) / 1.5);
    $("#cpuWorkers").dispatchEvent(new Event("input"));
    $("#batchSize").value = boot.default_batch_size || 4;
    $("#batchSize").dispatchEvent(new Event("input"));
    setRadio("tf0", boot.default_f0 || "rmvpe");
    $("#rmvpeWrap").hidden = !boot.f0_gpu_visible;
    applyModels(boot.models);
    state.separationModels = boot.pymss_models || [];
    applySeparationModels();
    $("#pInfo").value = boot.pymss_info || "";
    await loadPretrainedChoices(boot.pretrained_g, boot.pretrained_d);
    renderHelper();
    await loadFaq();
    if ($("#inferModel").value) await selectVoice();
  }

  const rail = $(".rail");
  const tabInk = document.createElement("span");
  tabInk.className = "tab-ink";
  tabInk.setAttribute("aria-hidden", "true");
  rail.prepend(tabInk);
  rail.classList.add("has-tab-ink");

  function moveTabInk(tab, instant = false) {
    if (!tab) return;
    if (instant) tabInk.style.transition = "none";
    tabInk.style.top = `${tab.offsetTop}px`;
    tabInk.style.left = `${tab.offsetLeft}px`;
    tabInk.style.width = `${tab.offsetWidth}px`;
    tabInk.style.height = `${tab.offsetHeight}px`;
    if (instant) requestAnimationFrame(() => { tabInk.style.transition = ""; });
  }

  let tabSwitching = false;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  $$(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tabs = $$(".tab");
      const current = $(".tab.is-active");
      if (current === btn || tabSwitching) return;
      const outgoing = $(".panel.is-active");
      const incoming = $(`.panel[data-panel="${btn.dataset.tab}"]`);
      const backwards = tabs.indexOf(btn) < tabs.indexOf(current);
      tabs.forEach((el) => {
        el.classList.toggle("is-active", el === btn);
        el.toggleAttribute("aria-current", el === btn);
      });
      moveTabInk(btn);

      if (reducedMotion.matches || !outgoing || !incoming) {
        $$(".panel").forEach((panel) => panel.classList.toggle("is-active", panel === incoming));
        $("#workspace").scrollTo({ top: 0, behavior: "auto" });
        return;
      }

      tabSwitching = true;
      outgoing.classList.toggle("back", backwards);
      outgoing.classList.add("panel-leave");
      window.setTimeout(() => {
        outgoing.classList.remove("is-active", "panel-leave", "back");
        incoming.classList.add("is-active", "panel-enter");
        incoming.classList.toggle("back", backwards);
        $("#workspace").scrollTo({ top: 0, behavior: "smooth" });
        window.setTimeout(() => {
          incoming.classList.remove("panel-enter", "back");
          tabSwitching = false;
        }, 320);
      }, 180);
    });
  });
  requestAnimationFrame(() => moveTabInk($(".tab.is-active"), true));
  window.addEventListener("resize", () => moveTabInk($(".tab.is-active"), true));

  $$(".subtab").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".subtab").forEach((el) => el.classList.toggle("is-active", el === btn));
      $$(".subpanel").forEach((panel) => panel.classList.toggle("is-active", panel.dataset.subpanel === btn.dataset.sub));
    });
  });

  $("#langSelect").addEventListener("change", async () => {
    state.lang = $("#langSelect").value;
    localStorage.setItem("nvc-lang", state.lang);
    applyI18n();
    renderHelper();
    await loadFaq();
  });

  $("#themeToggle").addEventListener("click", () => {
    state.theme = state.theme === "dark" ? "light" : "dark";
    localStorage.setItem("nvc-theme", state.theme);
    applyTheme();
  });

  $("#refreshModels").addEventListener("click", async () => {
    applyModels((await api("/api/models")).models);
    await selectVoice();
  });
  $("#unloadModel").addEventListener("click", async () => {
    applyModels((await api("/api/models/unload", { method: "POST" })).models);
    $("#inferModel").value = "";
    applyIndexChoices([], "");
  });
  $("#runLib")?.addEventListener("click", async () => {
    const kind = radio("libKind");
    const file = $("#libFile").files[0];
    const url = $("#libUrl").value.trim();
    const gUrl = $("#libGUrl").value.trim();
    const dUrl = $("#libDUrl").value.trim();
    if (!file && kind === "zip" && !url) {
      $("#libLog").textContent = t("libNeedUrl");
      return;
    }
    if (!file && kind === "pretrained" && !gUrl && !dUrl) {
      $("#libLog").textContent = t("libNeedPretrained");
      return;
    }
    $("#libLog").textContent = t("working");
    try {
      let data;
      if (file) {
        const form = new FormData();
        form.set("file", file);
        form.set("kind", kind);
        data = await api("/api/library/upload", { method: "POST", body: form });
      } else if (kind === "pretrained") {
        data = await api("/api/library/import-pretrained", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ g_url: gUrl, d_url: dUrl, source: radio("libSrc") }),
        });
      } else {
        data = await api("/api/library/import", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url, source: radio("libSrc"), kind }),
        });
      }
      await showLibraryResult(data);
    } catch (error) {
      $("#libLog").textContent = error.message;
    }
  });
  $$('input[name="libKind"]').forEach((input) => input.addEventListener("change", syncLibraryKind));
  syncLibraryKind();
  $("#inferModel").addEventListener("change", selectVoice);
  $("#speakerId").addEventListener("change", updateSpeakerIndex);
  $("#speakerNamed").addEventListener("change", updateSpeakerIndex);
  $("#trainsetZip").addEventListener("change", uploadTrainDataset);
  $("#refreshPretrained").addEventListener("click", () => loadPretrainedChoices($("#preG").value, $("#preD").value));

  $("#runSingle").addEventListener("click", async () => {
    if (!$("#sAudio").files[0]) return ($("#sLog").textContent = t("chooseAudio"));
    if (!$("#inferModel").value) return ($("#sLog").textContent = t("chooseVoice"));
    const data = new FormData();
    data.set("model", $("#inferModel").value);
    data.set("speaker", $("#speakerId").value);
    data.set("speaker_label", $("#speakerNamed").value);
    data.set("pitch", $("#sPitch").value);
    data.set("f0_method", radio("sF0"));
    data.set("index_path", $("#inferIndex").value);
    data.set("index_rate", $("#sIndexRate").value);
    data.set("resample_sr", $("#sResample").value);
    data.set("rms_mix_rate", $("#sRms").value);
    data.set("protect", $("#sProtect").value);
    data.set("audio", $("#sAudio").files[0]);
    $("#sLog").textContent = t("working");
    try {
      const res = await api("/api/infer/single", { method: "POST", body: data });
      $("#sLog").textContent = translateLog(res.status || t("done"));
      if (res.audio) {
        $("#sOut").hidden = false;
        $("#sOut").src = res.audio;
      }
    } catch (error) {
      $("#sLog").textContent = error.message;
    }
  });

  $("#runBatch").addEventListener("click", async () => {
    const data = new FormData();
    data.set("model", $("#inferModel").value);
    data.set("speaker", $("#speakerId").value);
    data.set("speaker_label", $("#speakerNamed").value);
    data.set("pitch", $("#bPitch").value);
    data.set("output_dir", $("#bOut").value);
    data.set("index_path", $("#inferIndex").value);
    data.set("f0_method", radio("bF0"));
    data.set("format", radio("bFmt"));
    data.set("resample_sr", $("#bResample").value);
    data.set("rms_mix_rate", $("#bRms").value);
    data.set("protect", $("#bProtect").value);
    data.set("index_rate", $("#bIndexRate").value);
    data.set("input_dir", $("#bIn").value);
    [...$("#bFiles").files].forEach((file) => data.append("files", file));
    $("#bLog").textContent = t("working");
    await readSSE("/api/infer/batch", data, (event) => {
      if (event.text) $("#bLog").textContent = translateLog(event.text);
    }, $("#runBatch"));
  });

  $("#pModel").addEventListener("change", async () => {
    $("#pInfo").value = (await api(`/api/pymss/info?model=${encodeURIComponent($("#pModel").value)}`)).info;
  });
  $("#runSep").addEventListener("click", async () => {
    const data = new FormData();
    data.set("model", $("#pModel").value);
    data.set("input_dir", $("#pIn").value);
    data.set("vocal_dir", $("#pVocal").value);
    data.set("residual_dir", $("#pResidual").value);
    data.set("format", radio("pFmt"));
    [...$("#pFiles").files].forEach((file) => data.append("files", file));
    renderSeparationResults(null);
    $("#pLog").textContent = t("working");
    $("#pProgress").hidden = false;
    await readSSE("/api/separate", data, (event) => {
      if (event.text) $("#pLog").textContent = translateLog(event.text);
      if (event.progress) {
        $("#pProgress").hidden = false;
        $("#pProgress").innerHTML = translateLog(event.progress);
      }
      if (event.files) renderSeparationResults(event.files);
    }, $("#runSep"), $("#stopSep"));
  });
  $("#stopSep").addEventListener("click", async () => {
    const data = await api("/api/separate/stop", { method: "POST" });
    $("#pLog").textContent = translateLog(data.text || "");
    if (data.progress) $("#pProgress").innerHTML = translateLog(data.progress);
  });

  $$("input[name=mode]").forEach((el) => el.addEventListener("change", async () => {
    const data = await api("/api/train/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: radio("mode") }),
    });
    const multi = radio("mode") === "多说话人";
    $("#multiHelp").hidden = !multi;
    $("#spkWrap").hidden = data.speaker?.visible === false || multi;
    $("#trainsetLabel").textContent = multi ? t("multiFolder") : t("trainFolder");
    if (data.folder?.placeholder) $("#trainset").placeholder = data.folder.placeholder;
    if (data.folder?.value === "") $("#trainset").value = "";
  }));

  async function syncTrainPaths() {
    const body = { sr: radio("sr"), if_f0: radio("ifF0") === "1", version: radio("ver") };
    const data = await api("/api/train/version", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    await loadPretrainedChoices(data.pretrained_g, data.pretrained_d);
    const sr = data.sr || {};
    $("#sr32wrap").hidden = !(sr.choices || []).includes("32k");
    if (sr.value) setRadio("sr", sr.value);
    const f0 = await api("/api/train/f0", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("#rmvpeWrap").hidden = f0.rmvpe_visible === false;
  }
  $$("input[name=sr], input[name=ifF0], input[name=ver]").forEach((el) => el.addEventListener("change", syncTrainPaths));

  const bindJob = (runId, stopId, url, stopUrl, logId) => {
    $(runId).addEventListener("click", async () => {
      $(logId).textContent = t("working");
      await readSSE(url, trainForm(), (event) => {
        if (event.text) $(logId).textContent = event.text;
        if (event.models) applyModels(event.models);
      }, $(runId), $(stopId));
    });
    $(stopId).addEventListener("click", async () => {
      const data = await api(stopUrl, { method: "POST" });
      if (data.text) $(logId).textContent = data.text;
    });
  };
  bindJob("#runPre", "#stopPre", "/api/train/preprocess", "/api/train/preprocess/stop", "#preLog");
  bindJob("#runExt", "#stopExt", "/api/train/extract", "/api/train/extract/stop", "#extLog");
  bindJob("#runTrain", "#stopTrain", "/api/train/model", "/api/train/model/stop", "#trainLog");
  bindJob("#runIndex", "#stopIndex", "/api/train/index", "/api/train/index/stop", "#trainLog");
  bindJob("#runAll", "#stopAll", "/api/train/oneclick", "/api/train/oneclick/stop", "#trainLog");

  $("#expName").addEventListener("input", () => { $("#hExp").value = $("#expName").value; });
  $("#hExp").addEventListener("input", () => { $("#expName").value = $("#hExp").value; });
  $("#hPrev").addEventListener("click", () => { state.helperPage -= 1; renderHelper(); });
  $("#hNext").addEventListener("click", () => { state.helperPage += 1; renderHelper(); });
  $("#hAdd").addEventListener("click", () => {
    if (state.helperRows.length >= 110) return;
    const used = new Set(state.helperRows.map((row) => Number(row.id)));
    const nextId = [...Array(110).keys()].find((id) => !used.has(id)) ?? 0;
    state.helperRows.push({ path: "", name: "", id: nextId, repeat: 1 });
    state.helperPage = Math.floor((state.helperRows.length - 1) / state.pageSize);
    renderHelper();
  });
  $("#hDel").addEventListener("click", () => {
    if (state.helperRows.length <= 2) return;
    state.helperRows.pop();
    renderHelper();
  });
  $("#hSubmit").addEventListener("click", async () => {
    const data = await api("/api/helper/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ exp: $("#hExp").value, rows: state.helperRows }),
    });
    $("#hStatus").innerHTML = data.status || t("done");
  });

  $("#runMerge").addEventListener("click", async () => {
    $("#mLog").textContent = (await api("/api/ckpt/merge", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        a: $("#mA").value, b: $("#mB").value, alpha: $("#mAlpha").value,
        sr: radio("mSr"), if_f0: radio("mF0"), info: $("#mInfo").value,
        name: $("#mName").value, version: radio("mVer"),
      }),
    })).text;
  });
  $("#runMod").addEventListener("click", async () => {
    $("#cLog").textContent = (await api("/api/ckpt/modify", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: $("#cPath").value, info: $("#cInfo").value, name: $("#cName").value }),
    })).text;
  });
  $("#runShow").addEventListener("click", async () => {
    $("#sInfo").textContent = (await api("/api/ckpt/show", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: $("#sPath").value }),
    })).text;
  });
  $("#ePath").addEventListener("change", async () => {
    const data = await api("/api/ckpt/extract-info", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: $("#ePath").value }),
    });
    if (typeof data.sr === "string") setRadio("eSr", data.sr);
    if (typeof data.f0 === "string") setRadio("eF0", data.f0);
    if (typeof data.version === "string") setRadio("eVer", data.version);
  });
  $("#runExtCkpt").addEventListener("click", async () => {
    $("#eLog").textContent = (await api("/api/ckpt/extract", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        path: $("#ePath").value, name: $("#eName").value, sr: radio("eSr"),
        if_f0: radio("eF0"), info: $("#eInfo").value, version: radio("eVer"),
      }),
    })).text;
  });

  init().catch((error) => {
    $("#deviceReadout").textContent = error.message;
  });
})();
