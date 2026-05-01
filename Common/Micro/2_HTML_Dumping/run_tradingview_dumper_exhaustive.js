/**
 * Integrated TradingView Financials dump pipeline — EXHAUSTIVE variant.
 * For finstatement tabs 1 (Statements), 2 (Statistics), 4 (Earnings), iterates FY / FH / FQ
 * (Annual / Semiannual / Quarterly) and dumps each. Other tabs match the base dumper.
 * On Earnings (fin 4), two timeframe tablists exist (upper square-tabs + lower #financials-earnings-tabs);
 * both are switched for each FY/FH/FQ step so EPS and Revenue sections stay in sync.
 *
 * Flow (matches Assemble_modules.txt):
 *   1) Main tab → Financials
 *   2) Finstatement indices 1–5 only (skip 0 Overview)
 *   3) If index === 1 (Statements): sub-loop Income / Balance / Cash flow (0–2)
 *   4) Timeframe → Annual
 *   5) Expand all sub-tab arrows
 *   6) Download HTML dumps (no timestamp in .html names). After the run, downloads YYMMDD_HHMM.txt
 *      with a summary of dumps and errors.
 *
 * Usage:
 *   await runTradingViewDumperPipelineExhaustive()
 *   await runTradingViewDumperPipelineExhaustive({ delayAfterNavMs: 600 })
 */
(() => {
  const delay = (ms) => new Promise((r) => setTimeout(r, ms));

  function formatReportTimestampYYMMDD_HHMM(d = new Date()) {
    const yy = String(d.getFullYear()).slice(-2);
    const MM = String(d.getMonth() + 1).padStart(2, "0");
    const DD = String(d.getDate()).padStart(2, "0");
    const HH = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    return `${yy}${MM}${DD}_${HH}${mm}`;
  }

  function downloadBlobAsFile(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType || "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      URL.revokeObjectURL(a.href);
      a.remove();
    }, 1000);
  }

  function buildDumpReportText({ errors, results, ok }) {
    const label = formatReportTimestampYYMMDD_HHMM();
    const lines = [];
    lines.push("TradingView Dumper (exhaustive FY/FH/FQ on fin 1,2,4) — run report");
    lines.push(`Report title (filename): ${label}.txt`);
    lines.push(`Finished (ISO): ${new Date().toISOString()}`);
    lines.push(`Page URL: ${typeof location !== "undefined" ? location.href : ""}`);
    lines.push("");
    lines.push("=== Summary ===");
    lines.push(`Pipeline OK (no recorded errors): ${ok}`);
    lines.push(`Error count: ${errors.length}`);
    lines.push(`Dump attempts: ${results.length}`);
    lines.push("");

    lines.push("=== Errors ===");
    if (!errors.length) {
      lines.push("(none)");
    } else {
      errors.forEach((e, i) => {
        lines.push(`--- Error ${i + 1} ---`);
        try {
          lines.push(JSON.stringify(e, null, 2));
        } catch {
          lines.push(String(e));
        }
      });
    }
    lines.push("");

    lines.push("=== Dumps (what was saved) ===");
    results.forEach((r, i) => {
      const fin = r.finIdx;
      const stmt = r.stmtIdx;
      const tfId = r.timeframeId != null ? r.timeframeId : null;
      const labelRow =
        stmt != null ? `finIdx=${fin} statementsSubIdx=${stmt}` : `finIdx=${fin} (no statements sub-loop)`;
      lines.push(`--- ${i + 1}. ${labelRow}${tfId != null ? ` timeframeId=${tfId}` : ""} ---`);
      if (r.dump && r.dump.filename) {
        if (r.dump.duplicateSkipped) {
          lines.push(`  (skipped duplicate basename — no second download)`);
        }
        lines.push(`  filename: ${r.dump.filename}`);
        lines.push(`  ticker: ${r.dump.ticker}`);
        lines.push(`  finStatementIndexed: ${r.dump.finStatementIndexed}`);
        lines.push(`  statementsSubtabIndexed: ${r.dump.statementsSubtabIndexed ?? "n/a"}`);
        lines.push(`  timeframe: ${r.dump.timeframe}`);
      } else {
        lines.push("  (no HTML file — dump failed or missing)");
      }
    });

    return { text: lines.join("\n"), label };
  }

  // --- 00a: Main tab → Financials (check_set_main_tab_financials.js) ---
  const MAIN_TAB_TARGET_LABEL = "Financials";
  const MAIN_TAB_CANDIDATE_SELECTORS = [
    '[data-name="underline-anchor-buttons"] [role="tablist"]',
  ];

  function mainTabFindContainer() {
    for (const selector of MAIN_TAB_CANDIDATE_SELECTORS) {
      const el = document.querySelector(selector);
      if (!el) continue;
      if (el.getAttribute("role") !== "tablist") {
        const nestedTablist = el.querySelector('[role="tablist"]');
        if (nestedTablist) return { container: nestedTablist, matchedSelector: selector };
      }
      return { container: el, matchedSelector: selector };
    }
    return { container: null, matchedSelector: null };
  }

  function mainTabIsSelected(tab) {
    if (tab.getAttribute("aria-selected") === "true") return true;
    return Array.from(tab.classList).some((cls) => cls.startsWith("selected-"));
  }

  function mainTabGetLabel(tab) {
    return (tab.textContent || "").trim();
  }

  function mainTabWaitForTarget(container, targetLabel, timeoutMs = 4000) {
    return new Promise((resolve) => {
      const start = Date.now();
      const timer = setInterval(() => {
        const tabs = Array.from(container.querySelectorAll('a[role="tab"], button[role="tab"]'));
        const selected = tabs.find((tab) => mainTabIsSelected(tab));
        const done =
          (selected && mainTabGetLabel(selected).toLowerCase() === targetLabel.toLowerCase()) ||
          Date.now() - start >= timeoutMs;
        if (done) {
          clearInterval(timer);
          resolve(
            Boolean(selected && mainTabGetLabel(selected).toLowerCase() === targetLabel.toLowerCase())
          );
        }
      }, 100);
    });
  }

  async function checkAndSetFinancials() {
    const { container, matchedSelector } = mainTabFindContainer();
    if (!container) {
      const result = {
        ok: false,
        changed: false,
        reason: "Could not find main tab container.",
        attemptedSelectors: MAIN_TAB_CANDIDATE_SELECTORS,
      };
      console.warn(result.reason);
      return result;
    }

    const tabs = Array.from(container.querySelectorAll('a[role="tab"], button[role="tab"]'));
    if (!tabs.length) {
      const result = {
        ok: false,
        changed: false,
        reason: "No tabs found in main tab container.",
        matchedSelector,
      };
      console.warn(result.reason);
      return result;
    }

    const targetTab = tabs.find(
      (tab) => mainTabGetLabel(tab).toLowerCase() === MAIN_TAB_TARGET_LABEL.toLowerCase()
    );
    if (!targetTab) {
      const result = {
        ok: false,
        changed: false,
        reason: `Could not find target tab "${MAIN_TAB_TARGET_LABEL}".`,
        matchedSelector,
      };
      console.warn(result.reason);
      return result;
    }

    let changed = false;
    if (!mainTabIsSelected(targetTab)) {
      targetTab.click();
      changed = true;
      await mainTabWaitForTarget(container, MAIN_TAB_TARGET_LABEL);
    }

    const finalTabs = Array.from(container.querySelectorAll('a[role="tab"], button[role="tab"]'));
    const selectedAfter = finalTabs.find((t) => mainTabIsSelected(t)) || null;

    const result = {
      ok: Boolean(
        selectedAfter && mainTabGetLabel(selectedAfter).toLowerCase() === MAIN_TAB_TARGET_LABEL.toLowerCase()
      ),
      changed,
      matchedSelector,
      selectedAfter: selectedAfter ? { label: mainTabGetLabel(selectedAfter) } : null,
    };

    if (result.ok) {
      console.log(`Main tab is now: ${MAIN_TAB_TARGET_LABEL}`);
    } else {
      console.warn(`Could not confirm "${MAIN_TAB_TARGET_LABEL}" is selected.`);
    }
    return result;
  }

  // --- 00b: Finstatement tabs (set_selected_finstatement.js) ---
  const INDEX_TO_TAB_ID = {
    0: "overview",
    1: "statements",
    2: "statistics",
    3: "dividends",
    4: "earnings",
    5: "revenue",
  };

  const FIN_FALLBACK_SELECTORS = [
    '[data-name="round-tabs-anchors"] [role="tablist"]',
    '[data-name="round-tabs-anchors"]',
    ".scrollWrap-vgCB17hK [role='tablist']",
    "[role='tablist']",
  ];

  function finFindBestContainer() {
    const byId = document.getElementById("financials-tabs");
    if (byId?.getAttribute("role") === "tablist") {
      return { container: byId, matchedSelector: "#financials-tabs" };
    }
    for (const el of document.querySelectorAll(
      '[data-name="round-tabs-anchors"] [role="tablist"]'
    )) {
      if (el.id === "financials-tabs") {
        return {
          container: el,
          matchedSelector: '[data-name="round-tabs-anchors"] [role="tablist"]#financials-tabs',
        };
      }
    }
    for (const selector of FIN_FALLBACK_SELECTORS) {
      const el = document.querySelector(selector);
      if (!el) continue;
      if (el.getAttribute("role") !== "tablist") {
        const nestedTablist = el.querySelector('[role="tablist"]');
        if (nestedTablist) return { container: nestedTablist, matchedSelector: selector };
      }
      return { container: el, matchedSelector: selector };
    }
    return { container: null, matchedSelector: null };
  }

  function finGetLabel(tab) {
    return (
      tab?.getAttribute("data-overflow-tooltip-text")?.trim() ||
      tab?.textContent?.trim() ||
      tab?.id ||
      null
    );
  }

  function finIsSelected(tab) {
    if (!tab) return false;
    if (tab.getAttribute("aria-selected") === "true") return true;
    return Array.from(tab.classList).some((cls) => cls.startsWith("selected-"));
  }

  function finGetSelectedTab(container) {
    const tabs = Array.from(container.querySelectorAll('a[role="tab"], button[role="tab"]'));
    return tabs.find((tab) => finIsSelected(tab)) || null;
  }

  function finWaitForSelection(container, expectedId, timeoutMs = 4000) {
    return new Promise((resolve) => {
      const start = Date.now();
      const timer = setInterval(() => {
        const selected = finGetSelectedTab(container);
        const done = selected?.id === expectedId || Date.now() - start >= timeoutMs;
        if (done) {
          clearInterval(timer);
          resolve(selected?.id === expectedId);
        }
      }, 100);
    });
  }

  async function setSelectedFinStatement(targetIndex) {
    const parsed = Number(targetIndex);
    if (!Number.isInteger(parsed) || !(parsed in INDEX_TO_TAB_ID)) {
      const result = {
        ok: false,
        changed: false,
        reason: "Invalid argument. Use integer 0..5.",
        validMap: INDEX_TO_TAB_ID,
      };
      console.warn(result.reason, result.validMap);
      return result;
    }

    const targetId = INDEX_TO_TAB_ID[parsed];
    const { container, matchedSelector } = finFindBestContainer();
    if (!container) {
      const result = {
        ok: false,
        changed: false,
        reason: "Could not find financial statement tab container.",
        attemptedSelectors: ["#financials-tabs", ...FIN_FALLBACK_SELECTORS],
      };
      console.warn(result.reason);
      return result;
    }

    const targetBtn =
      container.querySelector(`#${targetId}[role="tab"]`) ||
      container.querySelector(`[role="tab"]#${targetId}`);
    if (!targetBtn) {
      const result = {
        ok: false,
        changed: false,
        reason: `Target tab not found: ${targetId}`,
        matchedSelector,
      };
      console.warn(result.reason);
      return result;
    }

    let changed = false;
    if (!finIsSelected(targetBtn)) {
      targetBtn.click();
      changed = true;
      await finWaitForSelection(container, targetId);
    }

    const selectedAfter = finGetSelectedTab(container);
    const result = {
      ok: selectedAfter?.id === targetId,
      changed,
      matchedSelector,
      requestedIndex: parsed,
      requestedTabId: targetId,
      after: { id: selectedAfter?.id || null, label: finGetLabel(selectedAfter) },
    };

    if (result.ok) {
      console.log(`Financial statement selected: ${result.after.label} (${result.after.id})`);
    } else {
      console.warn("Selection may not have updated as expected.", result);
    }
    return result;
  }

  // --- 00c: Statements sub-tabs (check_set_statements_subtab.js) ---
  const INDEX_TO_STMT_TAB = {
    0: { domId: "income statements", tooltip: "Income statement" },
    1: { domId: "balance sheet", tooltip: "Balance sheet" },
    2: { domId: "cash flow", tooltip: "Cash flow" },
  };

  const STATEMENTS_TABLIST_SELECTOR = "#financials-page-statements-tabs";

  function stmtFindContainer() {
    const el = document.querySelector(STATEMENTS_TABLIST_SELECTOR);
    if (!el || el.getAttribute("role") !== "tablist") return null;
    return el;
  }

  function stmtTabByDomId(container, domId) {
    return (
      Array.from(container.querySelectorAll('[role="tab"]')).find((t) => t.id === domId) || null
    );
  }

  function stmtIsSelected(tab) {
    if (!tab) return false;
    if (tab.getAttribute("aria-selected") === "true") return true;
    return Array.from(tab.classList).some((cls) => cls.startsWith("selected-"));
  }

  function stmtGetLabel(tab) {
    return (
      tab?.getAttribute("data-overflow-tooltip-text")?.trim() ||
      tab?.textContent?.trim() ||
      tab?.id ||
      null
    );
  }

  function stmtGetSelectedTab(container) {
    const tabs = Array.from(container.querySelectorAll('a[role="tab"], button[role="tab"]'));
    return tabs.find((tab) => stmtIsSelected(tab)) || null;
  }

  function stmtWaitForSelection(container, expectedDomId, timeoutMs = 4000) {
    return new Promise((resolve) => {
      const start = Date.now();
      const timer = setInterval(() => {
        const selected = stmtGetSelectedTab(container);
        const done = selected?.id === expectedDomId || Date.now() - start >= timeoutMs;
        if (done) {
          clearInterval(timer);
          resolve(selected?.id === expectedDomId);
        }
      }, 100);
    });
  }

  async function setSelectedStatementsSubtab(targetIndex) {
    const parsed = Number(targetIndex);
    if (!Number.isInteger(parsed) || !(parsed in INDEX_TO_STMT_TAB)) {
      const result = {
        ok: false,
        changed: false,
        reason: "Invalid argument. Use integer 0 (Income), 1 (Balance), 2 (Cash flow).",
        validMap: INDEX_TO_STMT_TAB,
      };
      console.warn(result.reason, result.validMap);
      return result;
    }

    const { domId, tooltip } = INDEX_TO_STMT_TAB[parsed];
    const container = stmtFindContainer();
    if (!container) {
      const result = {
        ok: false,
        changed: false,
        reason: `Tablist not found: ${STATEMENTS_TABLIST_SELECTOR}`,
      };
      console.warn(result.reason);
      return result;
    }

    const targetTab = stmtTabByDomId(container, domId);
    if (!targetTab) {
      const result = {
        ok: false,
        changed: false,
        reason: `Target sub-tab not found for id="${domId}".`,
        requestedIndex: parsed,
        requestedTooltip: tooltip,
      };
      console.warn(result.reason);
      return result;
    }

    let changed = false;
    if (!stmtIsSelected(targetTab)) {
      targetTab.click();
      changed = true;
      await stmtWaitForSelection(container, domId);
    }

    const after = stmtGetSelectedTab(container);
    const result = {
      ok: after?.id === domId,
      changed,
      requestedIndex: parsed,
      requestedDomId: domId,
      after: { id: after?.id || null, label: stmtGetLabel(after) },
    };

    if (result.ok) {
      console.log(`Statements sub-tab: ${result.after.label}`);
    } else {
      console.warn("Selection may not have updated as expected.", result);
    }
    return result;
  }

  // --- 00d: Timeframe annual (set_timeframe_annual.js) ---
  // Some Financials tabs (e.g. Dividends, Revenue — finIdx 3 & 5) omit FY/FQ UI; treat as skipped, not error.
  const TF_CANDIDATE_SELECTORS = [
    '[data-name="square-tabs-buttons"] [role="tablist"]',
    '[data-name="square-tabs-buttons"]',
    ".scrollWrap-mf1FlhVw [role='tablist']",
  ];

  function tfFindBestContainer() {
    for (const selector of TF_CANDIDATE_SELECTORS) {
      const el = document.querySelector(selector);
      if (!el) continue;
      if (el.getAttribute("role") !== "tablist") {
        const nestedTablist = el.querySelector('[role="tablist"]');
        if (nestedTablist) return { container: nestedTablist, matchedSelector: selector };
      }
      return { container: el, matchedSelector: selector };
    }
    return { container: null, matchedSelector: null };
  }

  /** Lower Earnings strip (Revenue vs EPS): separate FY/FH/FQ control from the upper tablist. */
  const EARNINGS_SECONDARY_TABLIST_ID = "financials-earnings-tabs";

  /**
   * Tablists to drive on Earnings, in order: upper (tfFindBestContainer) then lower (#financials-earnings-tabs).
   * Lower may omit FH (Semiannual); that strip is skipped for that id only.
   */
  function tfGetEarningsTablistContainers() {
    const { container: primary } = tfFindBestContainer();
    const secondary = document.getElementById(EARNINGS_SECONDARY_TABLIST_ID);
    const secondaryOk =
      secondary &&
      secondary.getAttribute("role") === "tablist" &&
      secondary !== primary;

    const out = [];
    if (primary) out.push({ container: primary, role: "upper" });
    if (secondaryOk) out.push({ container: secondary, role: "lower" });
    return out;
  }

  function tfGetSelectedButton(container) {
    let selectedBtn = container.querySelector('button[role="tab"][aria-selected="true"]');
    if (selectedBtn) return selectedBtn;
    const tabs = container.querySelectorAll('button[role="tab"]');
    return Array.from(tabs).find((btn) =>
      Array.from(btn.classList).some((cls) => cls.startsWith("selected-"))
    );
  }

  function tfGetTimeframe(btn) {
    if (!btn) return null;
    return (
      btn.getAttribute("data-overflow-tooltip-text")?.trim() ||
      btn.textContent?.trim() ||
      btn.id ||
      null
    );
  }

  function tfFindAnnualButton(container) {
    let annualBtn = container.querySelector('button[role="tab"]#FY');
    if (annualBtn) return annualBtn;
    const tabs = container.querySelectorAll('button[role="tab"]');
    annualBtn = Array.from(tabs).find((btn) => {
      const tip = btn.getAttribute("data-overflow-tooltip-text")?.trim().toLowerCase();
      const text = btn.textContent?.trim().toLowerCase();
      return tip === "annual" || text === "annual";
    });
    return annualBtn || null;
  }

  function tfWaitForAnnualSelected(container, timeoutMs = 3000) {
    return new Promise((resolve) => {
      const start = Date.now();
      const timer = setInterval(() => {
        const selected = tfGetSelectedButton(container);
        const timeframe = tfGetTimeframe(selected)?.toLowerCase();
        if (timeframe === "annual" || Date.now() - start >= timeoutMs) {
          clearInterval(timer);
          resolve(timeframe === "annual");
        }
      }, 100);
    });
  }

  async function setTimeframeAnnual() {
    const { container, matchedSelector } = tfFindBestContainer();
    if (!container) {
      const result = {
        ok: true,
        skipped: true,
        changed: false,
        reason:
          "No timeframe tablist on this page (normal for some Financials tabs, e.g. Dividends / Revenue).",
        attemptedSelectors: TF_CANDIDATE_SELECTORS,
      };
      console.info("[TradingViewDumper]", result.reason);
      return result;
    }

    const selectedBefore = tfGetSelectedButton(container);
    const timeframeBefore = tfGetTimeframe(selectedBefore);
    const annualBtn = tfFindAnnualButton(container);

    if (!annualBtn) {
      const result = {
        ok: true,
        skipped: true,
        changed: false,
        reason: "Timeframe tablist present but no Annual (FY) control found.",
        matchedSelector,
        timeframeBefore,
      };
      console.info("[TradingViewDumper]", result.reason);
      return result;
    }

    const alreadyAnnual =
      annualBtn.getAttribute("aria-selected") === "true" ||
      tfGetTimeframe(annualBtn)?.toLowerCase() === (timeframeBefore || "").toLowerCase();

    if (!alreadyAnnual) {
      annualBtn.click();
      await tfWaitForAnnualSelected(container);
    }

    const selectedAfter = tfGetSelectedButton(container);
    const timeframeAfter = tfGetTimeframe(selectedAfter);
    const isAnnualNow = (timeframeAfter || "").toLowerCase() === "annual";

    const result = {
      ok: isAnnualNow,
      skipped: false,
      changed: !alreadyAnnual,
      matchedSelector,
      timeframeBefore: timeframeBefore || null,
      timeframeAfter: timeframeAfter || null,
      selectedButtonId: selectedAfter?.id || null,
    };

    if (result.ok) {
      console.log(`Timeframe is now: ${result.timeframeAfter}`);
    } else {
      console.warn("Attempted to switch to Annual, but current selection is:", result.timeframeAfter);
    }
    return result;
  }

  /** Square-tab button ids: FY Annual, FH Semiannual, FQ Quarterly (see CheckTimeframe samples). */
  const TIMEFRAME_BUTTON_IDS = ["FY", "FH", "FQ"];

  function tfWaitForSelectedId(container, buttonId, timeoutMs = 4000) {
    return new Promise((resolve) => {
      const start = Date.now();
      const timer = setInterval(() => {
        const selected = tfGetSelectedButton(container);
        const done = selected?.id === buttonId || Date.now() - start >= timeoutMs;
        if (done) {
          clearInterval(timer);
          resolve(selected?.id === buttonId);
        }
      }, 100);
    });
  }

  async function setTimeframeByButtonIdSingle(buttonId) {
    const { container, matchedSelector } = tfFindBestContainer();
    if (!container) {
      return {
        ok: true,
        skipped: true,
        changed: false,
        reason: "No timeframe tablist.",
        buttonId,
      };
    }
    const btn = container.querySelector(`button[role="tab"]#${buttonId}`);
    if (!btn) {
      return {
        ok: true,
        skipped: true,
        changed: false,
        reason: `No timeframe button #${buttonId}.`,
        matchedSelector,
        buttonId,
      };
    }
    const already = btn.getAttribute("aria-selected") === "true";
    if (!already) {
      btn.click();
      await tfWaitForSelectedId(container, buttonId);
    }
    const selectedAfter = tfGetSelectedButton(container);
    const ok = selectedAfter?.id === buttonId;
    return {
      ok,
      skipped: false,
      changed: !already,
      buttonId,
      label: tfGetTimeframe(selectedAfter),
      matchedSelector,
    };
  }

  /**
   * Earnings: set the same FY/FH/FQ on upper and lower tablists (lower may not expose all ids).
   * Failure if the first strip cannot select buttonId; lower missing buttonId is skipped, not an error.
   */
  async function setTimeframeByButtonIdEarningsDual(buttonId) {
    const strips = tfGetEarningsTablistContainers();
    if (!strips.length) {
      return {
        ok: true,
        skipped: true,
        changed: false,
        reason: "No timeframe tablist (Earnings).",
        buttonId,
        earningsDual: true,
      };
    }

    let anyChanged = false;
    let primaryOk = true;
    let primaryLabel = null;
    const perStrip = [];

    for (let i = 0; i < strips.length; i++) {
      const { container, role } = strips[i];
      const isPrimaryStrip = i === 0;

      const btn = container.querySelector(`button[role="tab"]#${buttonId}`);
      if (!btn) {
        if (isPrimaryStrip) {
          primaryOk = false;
          perStrip.push({ role, skipped: true, reason: `no #${buttonId}` });
        } else {
          const id = container.id || EARNINGS_SECONDARY_TABLIST_ID;
          console.info(
            `[TradingViewDumperExhaustive] Earnings ${role} strip (#${id}): no #${buttonId} — skip (e.g. Semiannual only on upper).`
          );
          perStrip.push({ role, skipped: true, reason: `no #${buttonId}` });
        }
        continue;
      }

      const already = btn.getAttribute("aria-selected") === "true";
      if (!already) {
        btn.click();
        await tfWaitForSelectedId(container, buttonId);
        anyChanged = true;
        await delay(80);
      }

      const selectedAfter = tfGetSelectedButton(container);
      const stripOk = selectedAfter?.id === buttonId;
      const label = tfGetTimeframe(selectedAfter);
      if (isPrimaryStrip) {
        primaryOk = stripOk;
        primaryLabel = label;
      } else if (!stripOk) {
        primaryOk = false;
      }
      perStrip.push({ role, ok: stripOk, buttonId: selectedAfter?.id, label });
    }

    const upperEntry = strips.find((s) => s.role === "upper");
    const upperContainer = upperEntry?.container ?? strips[0]?.container;

    return {
      ok: primaryOk,
      skipped: false,
      changed: anyChanged,
      buttonId,
      label: primaryLabel ?? tfGetTimeframe(tfGetSelectedButton(upperContainer || strips[0]?.container)),
      matchedSelector: `${EARNINGS_SECONDARY_TABLIST_ID}-dual`,
      earningsDual: true,
      perStrip,
    };
  }

  /**
   * @param {string} buttonId FY | FH | FQ
   * @param {number} [finIdx] When 4 (Earnings), updates upper + #financials-earnings-tabs.
   */
  async function setTimeframeByButtonId(buttonId, finIdx) {
    if (finIdx === 4) {
      return setTimeframeByButtonIdEarningsDual(buttonId);
    }
    return setTimeframeByButtonIdSingle(buttonId);
  }

  // --- 001: Open all sub-tabs (open_all_subtabs.js) — async wait for completion ---
  /**
   * Financials tree expanders use paired CSS-module classes: arrow-{hash} / opened-{hash}.
   * The hash changes on each TV build (e.g. C9MdAMrq → v0BbAiJS); match by prefix, not fixed names.
   */
  function queryClosedExpandArrowElements() {
    const closed = [];
    for (const el of document.querySelectorAll("[class*='arrow-']")) {
      let hash = null;
      for (const c of el.classList) {
        if (c.startsWith("arrow-") && c.length > "arrow-".length) {
          hash = c.slice("arrow-".length);
          break;
        }
      }
      if (!hash) continue;
      if (el.classList.contains(`opened-${hash}`)) continue;
      closed.push(el);
    }
    return closed;
  }

  async function expandAllSubTabsAsync() {
    const deadline = Date.now() + 60000;
    while (Date.now() < deadline) {
      const closed = queryClosedExpandArrowElements();
      if (closed.length === 0) {
        console.log("All levels expanded.");
        return { ok: true };
      }
      closed.forEach((el) => el.click());
      await delay(300);
    }
    const err = { ok: false, reason: "expandAll timeout after 60s" };
    console.warn(err.reason);
    return err;
  }

  // --- 002: Dump HTML (DumpHtml.js) ---
  const TAB_ID_TO_INDEX = Object.fromEntries(
    Object.entries(INDEX_TO_TAB_ID).map(([k, v]) => [v, Number(k)])
  );

  const DUMP_STMT_DOMID_TO_INDEX = {
    "income statements": 0,
    "balance sheet": 1,
    "cash flow": 2,
  };

  function dumpSafePart(s) {
    return String(s || "unknown")
      .replace(/[<>:"/\\|?*\x00-\x1F]/g, "_")
      .replace(/\s+/g, "_")
      .slice(0, 120);
  }

  function dumpGetTickerFromPath() {
    const parts = location.pathname.split("/").filter(Boolean);
    const i = parts.indexOf("symbols");
    return i >= 0 && parts[i + 1] ? parts[i + 1] : "unknown_ticker";
  }

  function dumpGetSelectedMainTab() {
    const el = document.querySelector(
      '[data-name="underline-anchor-buttons"] [role="tab"][aria-selected="true"]'
    );
    return el?.textContent?.trim() || "unknownMainTab";
  }

  function dumpGetFinStatementTabEl() {
    return document.querySelector('#financials-tabs [role="tab"][aria-selected="true"]');
  }

  function dumpGetSelectedFinStatement() {
    const el = dumpGetFinStatementTabEl();
    return (
      el?.getAttribute("data-overflow-tooltip-text")?.trim() ||
      el?.textContent?.trim() ||
      "unknownFinStatement"
    );
  }

  function dumpGetFinStatementIndexedPart() {
    const el = dumpGetFinStatementTabEl();
    const label =
      el?.getAttribute("data-overflow-tooltip-text")?.trim() ||
      el?.textContent?.trim() ||
      "unknownFinStatement";
    const tabId = el?.id;
    const idx =
      tabId != null && TAB_ID_TO_INDEX[tabId] !== undefined ? TAB_ID_TO_INDEX[tabId] : null;
    const labelSafe = dumpSafePart(label);
    if (idx === null) return labelSafe;
    return `${idx}-${labelSafe}`;
  }

  function dumpGetSelectedStatementsSubtab() {
    const tablist = document.getElementById("financials-page-statements-tabs");
    if (!tablist || tablist.getAttribute("role") !== "tablist") return "";
    const el = tablist.querySelector('[role="tab"][aria-selected="true"]');
    return (
      el?.getAttribute("data-overflow-tooltip-text")?.trim() ||
      el?.textContent?.trim() ||
      ""
    );
  }

  function dumpGetStatementsSubtabIndexedPart() {
    const tablist = document.getElementById("financials-page-statements-tabs");
    if (!tablist || tablist.getAttribute("role") !== "tablist") return "NA";
    const el = tablist.querySelector('[role="tab"][aria-selected="true"]');
    if (!el) return "NA";
    const label =
      el.getAttribute("data-overflow-tooltip-text")?.trim() || el.textContent?.trim() || "";
    const domId = el.id;
    const idx =
      domId && DUMP_STMT_DOMID_TO_INDEX[domId] !== undefined ? DUMP_STMT_DOMID_TO_INDEX[domId] : null;
    const labelSafe = dumpSafePart(label || "unknownStmtSub");
    if (idx === null) return labelSafe;
    return `${idx}-${labelSafe}`;
  }

  /**
   * Must use the same timeframe tablist as setTimeframeByButtonId (tfFindBestContainer).
   * A global querySelector can hit a hidden duplicate [data-name="square-tabs-buttons"] and
   * always read "Annual", so filenames repeat Annual even after switching to FH/FQ.
   */
  function dumpGetSelectedTimeframe() {
    const { container } = tfFindBestContainer();
    const el = container ? tfGetSelectedButton(container) : null;
    const fallback =
      el ||
      document.querySelector(
        '[data-name="square-tabs-buttons"] button[role="tab"][aria-selected="true"]'
      );
    return (
      fallback?.getAttribute("data-overflow-tooltip-text")?.trim() ||
      fallback?.textContent?.trim() ||
      "unknownTimeframe"
    );
  }

  /** Reset at each pipeline run; avoids accidental duplicate download with same basename. */
  let lastDumpBasename = null;

  function dumpCurrentHtml() {
    const ticker = dumpSafePart(dumpGetTickerFromPath());
    const mainTab = dumpSafePart(dumpGetSelectedMainTab());
    const finRaw = dumpGetSelectedFinStatement();
    const fin = dumpGetFinStatementIndexedPart();
    const stmtSubRaw = dumpGetSelectedStatementsSubtab();
    const stmtSeg = dumpGetStatementsSubtabIndexedPart();
    const tf = dumpSafePart(dumpGetSelectedTimeframe());

    const html = "<!doctype html>\n" + document.documentElement.outerHTML;
    const filename = `[${ticker}]${fin}(${stmtSeg})_${tf}.html`;

    if (filename === lastDumpBasename) {
      console.warn(
        "[TradingViewDumperExhaustive] Skipping duplicate basename (same as previous dump):",
        filename
      );
      return {
        filename,
        duplicateSkipped: true,
        ticker,
        mainTab,
        finStatement: dumpSafePart(finRaw),
        finStatementIndexed: fin,
        statementsSubtab: stmtSubRaw ? dumpSafePart(stmtSubRaw) : null,
        statementsSubtabIndexed: stmtSeg !== "N/A" ? stmtSeg : null,
        timeframe: tf,
        url: location.href,
      };
    }
    lastDumpBasename = filename;

    const blob = new Blob([html], { type: "text/html;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      URL.revokeObjectURL(a.href);
      a.remove();
    }, 1000);

    console.log("Saved HTML:", filename);
    return {
      filename,
      duplicateSkipped: false,
      ticker,
      mainTab,
      finStatement: dumpSafePart(finRaw),
      finStatementIndexed: fin,
      statementsSubtab: stmtSubRaw ? dumpSafePart(stmtSubRaw) : null,
      statementsSubtabIndexed: stmtSeg !== "N/A" ? stmtSeg : null,
      timeframe: tf,
      url: location.href,
    };
  }

  /**
   * @param {{ delayAfterNavMs?: number, delayBetweenDumpsMs?: number }} [options]
   */
  async function runTradingViewDumperPipelineExhaustive(options) {
    const delayAfterNavMs = options?.delayAfterNavMs ?? 450;
    const delayBetweenDumpsMs = options?.delayBetweenDumpsMs ?? 400;

    const errors = [];
    const results = [];

    const pushErr = (entry) => {
      errors.push(entry);
      console.warn("[TradingViewDumperExhaustive]", entry);
    };

    const FIN_EXHAUSTIVE_TF = new Set([1, 2, 4]);

    lastDumpBasename = null;

    console.log(
      "[TradingViewDumperExhaustive] Starting pipeline (fin 1–5; Statements: sub×timeframes; Statistics & Earnings: FY/FH/FQ each)…"
    );

    const rMain = await checkAndSetFinancials();
    if (!rMain.ok) pushErr({ step: "1_mainTabFinancials", detail: rMain });
    await delay(delayAfterNavMs);

    for (let finIdx = 1; finIdx <= 5; finIdx++) {
      const rFin = await setSelectedFinStatement(finIdx);
      if (!rFin.ok) pushErr({ step: "2_setFinStatement", finIdx, detail: rFin });
      await delay(delayAfterNavMs);

      const useTfLoop = FIN_EXHAUSTIVE_TF.has(finIdx);

      if (finIdx === 1) {
        for (let stmtIdx = 0; stmtIdx <= 2; stmtIdx++) {
          const rStmt = await setSelectedStatementsSubtab(stmtIdx);
          if (!rStmt.ok) pushErr({ step: "3_statementsSubtab", finIdx, stmtIdx, detail: rStmt });
          await delay(delayAfterNavMs);

          for (const tfId of TIMEFRAME_BUTTON_IDS) {
            const rTf = await setTimeframeByButtonId(tfId, finIdx);
            if (!rTf.ok && !rTf.skipped) {
              pushErr({ step: "4_timeframeById", finIdx, stmtIdx, timeframeId: tfId, detail: rTf });
            }
            await delay(350);

            const rExp = await expandAllSubTabsAsync();
            if (!rExp.ok) pushErr({ step: "5_expandAll", finIdx, stmtIdx, timeframeId: tfId, detail: rExp });
            await delay(200);

            let dump;
            try {
              dump = dumpCurrentHtml();
            } catch (e) {
              pushErr({ step: "6_dumpHtml", finIdx, stmtIdx, timeframeId: tfId, detail: String(e) });
              dump = null;
            }
            results.push({ finIdx, stmtIdx, timeframeId: tfId, dump });
            await delay(delayBetweenDumpsMs);
          }
        }
      } else if (useTfLoop) {
        for (const tfId of TIMEFRAME_BUTTON_IDS) {
          const rTf = await setTimeframeByButtonId(tfId, finIdx);
          if (!rTf.ok && !rTf.skipped) {
            pushErr({ step: "4_timeframeById", finIdx, timeframeId: tfId, detail: rTf });
          }
          await delay(350);

          const rExp = await expandAllSubTabsAsync();
          if (!rExp.ok) pushErr({ step: "5_expandAll", finIdx, timeframeId: tfId, detail: rExp });
          await delay(200);

          let dump;
          try {
            dump = dumpCurrentHtml();
          } catch (e) {
            pushErr({ step: "6_dumpHtml", finIdx, timeframeId: tfId, detail: String(e) });
            dump = null;
          }
          results.push({ finIdx, stmtIdx: null, timeframeId: tfId, dump });
          await delay(delayBetweenDumpsMs);
        }
      } else {
        const rTf = await setTimeframeAnnual();
        if (!rTf.ok && !rTf.skipped) {
          pushErr({ step: "4_timeframeAnnual", finIdx, detail: rTf });
        }
        await delay(200);

        const rExp = await expandAllSubTabsAsync();
        if (!rExp.ok) pushErr({ step: "5_expandAll", finIdx, detail: rExp });
        await delay(200);

        let dump;
        try {
          dump = dumpCurrentHtml();
        } catch (e) {
          pushErr({ step: "6_dumpHtml", finIdx, detail: String(e) });
          dump = null;
        }
        results.push({ finIdx, stmtIdx: null, dump });
        await delay(delayBetweenDumpsMs);
      }
    }

    console.log("--- TradingViewDumperExhaustive: done ---");
    console.log("Errors count:", errors.length);
    if (errors.length) console.table(errors);
    else console.log("No errors recorded.");

    await delay(400);
    const { text: reportText, label: reportLabel } = buildDumpReportText({
      errors,
      results,
      ok: errors.length === 0,
    });
    const reportFilename = `${reportLabel}.txt`;
    downloadBlobAsFile(reportFilename, reportText, "text/plain;charset=utf-8");
    console.log("[TradingViewDumperExhaustive] Report saved:", reportFilename);

    return {
      errors,
      results,
      ok: errors.length === 0,
      reportFile: reportFilename,
      reportLabel,
    };
  }

  window.runTradingViewDumperPipelineExhaustive = runTradingViewDumperPipelineExhaustive;
  window.TradingViewDumperExhaustive = {
    runTradingViewDumperPipelineExhaustive,
    checkAndSetFinancials,
    setSelectedFinStatement,
    setSelectedStatementsSubtab,
    setTimeframeAnnual,
    setTimeframeByButtonId,
    setTimeframeByButtonIdSingle,
    setTimeframeByButtonIdEarningsDual,
    tfGetEarningsTablistContainers,
    TIMEFRAME_BUTTON_IDS,
    expandAllSubTabsAsync,
    queryClosedExpandArrowElements,
    dumpCurrentHtml,
  };

  console.log(
    "TradingViewDumperExhaustive loaded. Run: await runTradingViewDumperPipelineExhaustive()"
  );
})();
