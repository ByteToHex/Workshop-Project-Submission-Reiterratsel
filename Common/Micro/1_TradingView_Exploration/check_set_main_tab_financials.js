(() => {
  // Paste this entire block into browser console on the target page.
  const TARGET_LABEL = "Financials";
  const CANDIDATE_SELECTORS = [
    // "#symbol-page-tabs",
    '[data-name="underline-anchor-buttons"] [role="tablist"]',
    // '[data-name="underline-anchor-buttons"]',
    // ".scroll-wrap-SmxgjhBJ [role='tablist']",
    // "[role='tablist']",
  ];

  function findBestContainer() {
    for (const selector of CANDIDATE_SELECTORS) {
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

  function isSelected(tab) {
    if (tab.getAttribute("aria-selected") === "true") return true;
    return Array.from(tab.classList).some((cls) => cls.startsWith("selected-"));
  }

  function getLabel(tab) {
    return (tab.textContent || "").trim();
  }

  function getTabStates(tabs) {
    return tabs.map((tab) => ({
      label: getLabel(tab),
      id: tab.id || null,
      href: tab.getAttribute("href"),
      selected: isSelected(tab),
      ariaSelected: tab.getAttribute("aria-selected"),
    }));
  }

  function waitForTargetSelected(container, targetLabel, timeoutMs = 4000) {
    return new Promise((resolve) => {
      const start = Date.now();
      const timer = setInterval(() => {
        const tabs = Array.from(container.querySelectorAll('a[role="tab"], button[role="tab"]'));
        const selected = tabs.find((tab) => isSelected(tab));
        const done =
          (selected && getLabel(selected).toLowerCase() === targetLabel.toLowerCase()) ||
          Date.now() - start >= timeoutMs;
        if (done) {
          clearInterval(timer);
          resolve(Boolean(selected && getLabel(selected).toLowerCase() === targetLabel.toLowerCase()));
        }
      }, 100);
    });
  }

  async function checkAndSetFinancials() {
    const { container, matchedSelector } = findBestContainer();
    if (!container) {
      const result = {
        ok: false,
        changed: false,
        reason: "Could not find main tab container.",
        attemptedSelectors: CANDIDATE_SELECTORS,
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

    const initialStates = getTabStates(tabs);
    console.table(initialStates);

    const selectedBefore = initialStates.find((t) => t.selected) || null;
    const targetTab = tabs.find((tab) => getLabel(tab).toLowerCase() === TARGET_LABEL.toLowerCase());

    if (!targetTab) {
      const result = {
        ok: false,
        changed: false,
        reason: `Could not find target tab "${TARGET_LABEL}".`,
        matchedSelector,
        selectedBefore,
        tabs: initialStates,
      };
      console.warn(result.reason);
      return result;
    }

    let changed = false;
    if (!isSelected(targetTab)) {
      targetTab.click();
      changed = true;
      await waitForTargetSelected(container, TARGET_LABEL);
    }

    const finalTabs = Array.from(container.querySelectorAll('a[role="tab"], button[role="tab"]'));
    const finalStates = getTabStates(finalTabs);
    const selectedAfter = finalStates.find((t) => t.selected) || null;

    const result = {
      ok: Boolean(selectedAfter && selectedAfter.label.toLowerCase() === TARGET_LABEL.toLowerCase()),
      changed,
      matchedSelector,
      selectedBefore,
      selectedAfter,
      tabs: finalStates,
    };

    if (result.ok) {
      console.log(`Main tab is now: ${selectedAfter.label}`);
    } else {
      console.warn(`Could not confirm "${TARGET_LABEL}" is selected.`);
    }

    console.table(finalStates);
    return result;
  }

  return checkAndSetFinancials();
})();
