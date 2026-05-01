(() => {
  // Paste into browser console on the Statements finstatement page, then call:
  //   getSelectedStatementsSubtab()           // detect only
  //   setSelectedStatementsSubtab(0)        // Income statement
  //   setSelectedStatementsSubtab(1)        // Balance sheet
  //   setSelectedStatementsSubtab(2)        // Cash flow
  // Future: loop 0..2 and call setSelectedStatementsSubtab(i) between dumps.

  /** @type {Record<number, { domId: string; tooltip: string }>} */
  const INDEX_TO_TAB = {
    0: { domId: "income statements", tooltip: "Income statement" },
    1: { domId: "balance sheet", tooltip: "Balance sheet" },
    2: { domId: "cash flow", tooltip: "Cash flow" },
  };

  // Unique to Statements row; avoids clash with #financials-tabs (see ref.txt / Assemble_modules).
  const STATEMENTS_TABLIST_SELECTOR = "#financials-page-statements-tabs";

  function findContainer() {
    const el = document.querySelector(STATEMENTS_TABLIST_SELECTOR);
    if (!el || el.getAttribute("role") !== "tablist") {
      return null;
    }
    return el;
  }

  function getLabel(tab) {
    return (
      tab?.getAttribute("data-overflow-tooltip-text")?.trim() ||
      tab?.textContent?.trim() ||
      tab?.id ||
      null
    );
  }

  function isSelected(tab) {
    if (!tab) return false;
    if (tab.getAttribute("aria-selected") === "true") return true;
    return Array.from(tab.classList).some((cls) => cls.startsWith("selected-"));
  }

  function tabByDomId(container, domId) {
    // IDs contain spaces ("income statements", etc.); match by property, not #id
    return (
      Array.from(container.querySelectorAll('[role="tab"]')).find((t) => t.id === domId) ||
      null
    );
  }

  function getAllSubtabStates(container) {
    const tabs = Array.from(container.querySelectorAll('a[role="tab"], button[role="tab"]'));
    return tabs.map((tab, i) => ({
      index: i,
      id: tab.id || null,
      label: getLabel(tab),
      selected: isSelected(tab),
      ariaSelected: tab.getAttribute("aria-selected"),
    }));
  }

  function getSelectedTab(container) {
    const tabs = Array.from(container.querySelectorAll('a[role="tab"], button[role="tab"]'));
    return tabs.find((tab) => isSelected(tab)) || null;
  }

  function waitForSelection(container, expectedDomId, timeoutMs = 4000) {
    return new Promise((resolve) => {
      const start = Date.now();
      const timer = setInterval(() => {
        const selected = getSelectedTab(container);
        const done = selected?.id === expectedDomId || Date.now() - start >= timeoutMs;
        if (done) {
          clearInterval(timer);
          resolve(selected?.id === expectedDomId);
        }
      }, 100);
    });
  }

  function getSelectedStatementsSubtab() {
    const container = findContainer();
    if (!container) {
      const result = {
        ok: false,
        reason: `Tablist not found: ${STATEMENTS_TABLIST_SELECTOR} (open Financials → Statements first).`,
      };
      console.warn(result.reason);
      return result;
    }

    const states = getAllSubtabStates(container);
    const selected = states.find((s) => s.selected) || null;
    console.table(states);

    return {
      ok: true,
      matchedSelector: STATEMENTS_TABLIST_SELECTOR,
      selected,
      tabs: states,
    };
  }

  async function setSelectedStatementsSubtab(targetIndex) {
    const parsed = Number(targetIndex);
    if (!Number.isInteger(parsed) || !(parsed in INDEX_TO_TAB)) {
      const result = {
        ok: false,
        changed: false,
        reason: "Invalid argument. Use integer 0 (Income), 1 (Balance), 2 (Cash flow).",
        validMap: INDEX_TO_TAB,
      };
      console.warn(result.reason, result.validMap);
      return result;
    }

    const { domId, tooltip } = INDEX_TO_TAB[parsed];
    const container = findContainer();
    if (!container) {
      const result = {
        ok: false,
        changed: false,
        reason: `Tablist not found: ${STATEMENTS_TABLIST_SELECTOR}`,
      };
      console.warn(result.reason);
      return result;
    }

    const before = getSelectedTab(container);
    const beforeInfo = {
      id: before?.id || null,
      label: getLabel(before),
    };

    const targetTab = tabByDomId(container, domId);
    if (!targetTab) {
      const result = {
        ok: false,
        changed: false,
        reason: `Target sub-tab not found for id="${domId}".`,
        requestedIndex: parsed,
        requestedTooltip: tooltip,
        before: beforeInfo,
      };
      console.warn(result.reason);
      return result;
    }

    let changed = false;
    if (!isSelected(targetTab)) {
      targetTab.click();
      changed = true;
      await waitForSelection(container, domId);
    }

    const after = getSelectedTab(container);
    const afterInfo = {
      id: after?.id || null,
      label: getLabel(after),
    };

    const result = {
      ok: after?.id === domId,
      changed,
      matchedSelector: STATEMENTS_TABLIST_SELECTOR,
      requestedIndex: parsed,
      requestedDomId: domId,
      before: beforeInfo,
      after: afterInfo,
    };

    if (result.ok) {
      console.log(`Statements sub-tab: ${afterInfo.label}`);
    } else {
      console.warn("Selection may not have updated as expected.", result);
    }

    console.table(getAllSubtabStates(container));
    return result;
  }

  window.getSelectedStatementsSubtab = getSelectedStatementsSubtab;
  window.setSelectedStatementsSubtab = setSelectedStatementsSubtab;

  console.log(
    "Ready. getSelectedStatementsSubtab() | setSelectedStatementsSubtab(0|1|2)"
  );
})();

/*
  --- Task B: selector conflicts with Assemble_modules (0a–2) ---

  Problem: both the main Financials row (#financials-tabs) and this Statements
  row use data-name="round-tabs-anchors". document.querySelector('[data-name="round-tabs-anchors"] [role="tablist"]')
  returns only the FIRST match in document order — wrong if both exist.

  This script scopes ONLY to #financials-page-statements-tabs (unique per ref).

  Recommended hardening elsewhere:
  - 0b set_selected_finstatement.js: prefer #financials-tabs when present, then
    fallback to data-name wrapper; never rely on generic round-tabs-anchors alone
    on pages that can show both tablists.
  - 002 DumpHtml.js: resolve finstatement label via #financials-tabs [aria-selected=true],
    not generic round-tabs-anchors; add optional #financials-page-statements-tabs
    for Statements sub-row when dumping that page.
*/
