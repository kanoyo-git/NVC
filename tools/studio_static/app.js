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
    $("#themeToggle").setAttribute("aria-label", state.theme === "dark" ? t("themeToLight") : t("themeToDark"));
    document.documentElement.lang = state.lang === "ru" ? "ru" : "en";
    $("#langSelect").value = state.lang;
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
      if (!out || out.tagName !== "OUTPUT") return;
      const sync = () => { out.value = input.value; };
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
    const values = [...new Set((choices || []).filter(Boolean))];
    fillSelect(select, values.length ? values : [""], selected || "");
    if (!values.length && select.options[0]) select.options[0].textContent = t("noIndex");
    if (selected && values.includes(selected)) select.value = selected;
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
      if (empty) empty.textContent = t("noPretrained");
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
    if (!files.length) {
      label.dataset.empty = "true";
      label.textContent = t("noFile");
      return;
    }
    label.dataset.empty = "false";
    label.textContent = files.length === 1 ? files[0].name : t("filesSelected").replace("%s", files.length);
  }

  function bindDrops() {
    $$("[data-drop-for]").forEach((zone) => {
      const input = $("#" + zone.dataset.dropFor);
      if (!input) return;
      zone.addEventListener("click", () => input.click());
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

  async function selectVoice() {
    const model = $("#inferModel").value;
    if (!model) return;
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

  $$(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".tab").forEach((el) => {
        el.classList.toggle("is-active", el === btn);
        el.toggleAttribute("aria-current", el === btn);
      });
      $$(".panel").forEach((panel) => panel.classList.toggle("is-active", panel.dataset.panel === btn.dataset.tab));
    });
  });

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
  });
  $("#unloadModel").addEventListener("click", async () => {
    applyModels((await api("/api/models/unload", { method: "POST" })).models);
  });
  $("#runLib")?.addEventListener("click", async () => {
    const url = $("#libUrl").value.trim();
    if (!url) {
      $("#libLog").textContent = t("libNeedUrl");
      return;
    }
    $("#libLog").textContent = t("working");
    try {
      const data = await api("/api/library/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url,
          source: radio("libSrc"),
          kind: radio("libKind"),
          filename: $("#libName").value.trim(),
        }),
      });
      if (data.models) applyModels(data.models);
      if (data.extracted && data.extracted.length) {
        $("#libLog").textContent = `${t("done")} ${data.name}\n${data.extracted.join("\n")}`;
      } else {
        $("#libLog").textContent = `${t("done")} ${data.path || ""}`;
      }
    } catch (error) {
      $("#libLog").textContent = error.message;
    }
  });
  $("#inferModel").addEventListener("change", selectVoice);
  $("#speakerId").addEventListener("change", updateSpeakerIndex);
  $("#speakerNamed").addEventListener("change", updateSpeakerIndex);
  $("#trainsetZip").addEventListener("change", uploadTrainDataset);
  $("#refreshPretrained").addEventListener("click", () => loadPretrainedChoices($("#preG").value, $("#preD").value));

  $("#runSingle").addEventListener("click", async () => {
    if (!$("#sAudio").files[0]) return ($("#sLog").textContent = t("chooseAudio"));
    const data = new FormData();
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
    const res = await api("/api/infer/single", { method: "POST", body: data });
    $("#sLog").textContent = res.status || t("done");
    if (res.audio) {
      $("#sOut").hidden = false;
      $("#sOut").src = res.audio;
    }
  });

  $("#runBatch").addEventListener("click", async () => {
    const data = new FormData();
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
      if (event.text) $("#bLog").textContent = event.text;
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
    $("#pLog").textContent = t("working");
    $("#pProgress").hidden = false;
    await readSSE("/api/separate", data, (event) => {
      if (event.text) $("#pLog").textContent = event.text;
      if (event.progress) {
        $("#pProgress").hidden = false;
        $("#pProgress").innerHTML = event.progress;
      }
    }, $("#runSep"), $("#stopSep"));
  });
  $("#stopSep").addEventListener("click", async () => {
    const data = await api("/api/separate/stop", { method: "POST" });
    $("#pLog").textContent = data.text || "";
    if (data.progress) $("#pProgress").innerHTML = data.progress;
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
