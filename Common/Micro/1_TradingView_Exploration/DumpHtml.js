(() => {
  const INDEX_TO_TAB_ID = {
    0: "overview",
    1: "statements",
    2: "statistics",
    3: "dividends",
    4: "earnings",
    5: "revenue",
  };

  const TAB_ID_TO_INDEX = Object.fromEntries(
    Object.entries(INDEX_TO_TAB_ID).map(([k, v]) => [v, Number(k)])
  );

  const STMT_DOMID_TO_INDEX = {
    "income statements": 0,
    "balance sheet": 1,
    "cash flow": 2,
  };

  function safePart(s) {
    return String(s || "unknown")
      .replace(/[<>:"/\\|?*\x00-\x1F]/g, "_")
      .replace(/\s+/g, "_")
      .slice(0, 120);
  }

  function getTickerFromPath() {
    const parts = location.pathname.split("/").filter(Boolean);
    // e.g. /symbols/SGX-A17U/financials-overview/
    const i = parts.indexOf("symbols");
    return i >= 0 && parts[i + 1] ? parts[i + 1] : "unknown_ticker";
  }

  function getSelectedMainTab() {
    const el = document.querySelector('[data-name="underline-anchor-buttons"] [role="tab"][aria-selected="true"]');
    return el?.textContent?.trim() || "unknownMainTab";
  }

  function getSelectedFinStatementTabEl() {
    return document.querySelector(
      '#financials-tabs [role="tab"][aria-selected="true"]'
    );
  }

  function getSelectedFinStatement() {
    const el = getSelectedFinStatementTabEl();
    return (
      el?.getAttribute("data-overflow-tooltip-text")?.trim() ||
      el?.textContent?.trim() ||
      "unknownFinStatement"
    );
  }

  /** e.g. "1-Statements" — index from INDEX_TO_TAB_ID / tab id. */
  function getFinStatementIndexedPart() {
    const el = getSelectedFinStatementTabEl();
    const label =
      el?.getAttribute("data-overflow-tooltip-text")?.trim() ||
      el?.textContent?.trim() ||
      "unknownFinStatement";
    const tabId = el?.id;
    const idx =
      tabId != null && TAB_ID_TO_INDEX[tabId] !== undefined
        ? TAB_ID_TO_INDEX[tabId]
        : null;
    const labelSafe = safePart(label);
    if (idx === null) return labelSafe;
    return `${idx}-${labelSafe}`;
  }

  /** Income / Balance / Cash flow row (Statements view only); empty if not present. */
  function getSelectedStatementsSubtab() {
    const tablist = document.getElementById("financials-page-statements-tabs");
    if (!tablist || tablist.getAttribute("role") !== "tablist") {
      return "";
    }
    const el = tablist.querySelector('[role="tab"][aria-selected="true"]');
    return (
      el?.getAttribute("data-overflow-tooltip-text")?.trim() ||
      el?.textContent?.trim() ||
      ""
    );
  }

  /** e.g. "0-Income_statement" — index from STMT_DOMID_TO_INDEX; "N/A" if no sub-row. */
  function getStatementsSubtabIndexedPart() {
    const tablist = document.getElementById("financials-page-statements-tabs");
    if (!tablist || tablist.getAttribute("role") !== "tablist") {
      return "NA";
    }
    const el = tablist.querySelector('[role="tab"][aria-selected="true"]');
    if (!el) return "NA";
    const label =
      el.getAttribute("data-overflow-tooltip-text")?.trim() ||
      el.textContent?.trim() ||
      "";
    const domId = el.id;
    const idx =
      domId && STMT_DOMID_TO_INDEX[domId] !== undefined
        ? STMT_DOMID_TO_INDEX[domId]
        : null;
    const labelSafe = safePart(label || "unknownStmtSub");
    if (idx === null) return labelSafe;
    return `${idx}-${labelSafe}`;
  }

  function getSelectedTimeframe() {
    const el = document.querySelector('[data-name="square-tabs-buttons"] button[role="tab"][aria-selected="true"]');
    return (
      el?.getAttribute("data-overflow-tooltip-text")?.trim() ||
      el?.textContent?.trim() ||
      "unknownTimeframe"
    );
  }

  function dumpCurrentHtml() {
    const ticker = safePart(getTickerFromPath());
    const mainTab = safePart(getSelectedMainTab());
    const finRaw = getSelectedFinStatement();
    const fin = getFinStatementIndexedPart();
    const stmtSubRaw = getSelectedStatementsSubtab();
    const stmtSeg = getStatementsSubtabIndexedPart();
    const tf = safePart(getSelectedTimeframe());
    // const path = safePart(location.pathname.replaceAll("/", "__"));
    const ts = new Date().toISOString().replace(/[:.]/g, "-");

    const html = "<!doctype html>\n" + document.documentElement.outerHTML;
    const filename = `[${ticker}]_${fin}(${stmtSeg})_${tf}[${ts}].html`;

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
      ticker,
      mainTab,
      finStatement: safePart(finRaw),
      finStatementIndexed: fin,
      statementsSubtab: stmtSubRaw ? safePart(stmtSubRaw) : null,
      statementsSubtabIndexed: stmtSeg !== "N/A" ? stmtSeg : null,
      timeframe: tf,
      url: location.href,
    };
  }

  return dumpCurrentHtml();
})();
