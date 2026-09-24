// HOW IT WORKS: a non-modal drawer on the right. The controls stay usable
// while it is open. Closes with ✕, Esc, or the tool's button again.
export function initDrawer() {
  const drawer = document.getElementById("drawer");
  const bodies = [...drawer.querySelectorAll("[data-drawer-tool]")];
  const buttons = [...document.querySelectorAll("[data-how]")];
  const buttonFor = (tool) => buttons.find((b) => b.closest("[data-tool]").dataset.tool === tool);
  let openTool = null;

  function sync() {
    drawer.classList.toggle("is-open", openTool !== null);
    buttons.forEach((b) => b.setAttribute("aria-expanded", String(b === buttonFor(openTool))));
  }

  function open(tool) {
    openTool = tool;
    bodies.forEach((body) => {
      body.hidden = body.dataset.drawerTool !== tool;
    });
    drawer.scrollTop = 0;
    sync();
  }

  function close() {
    const tool = openTool;
    openTool = null;
    sync();
    // Don't leave focus inside a drawer that is sliding away.
    if (drawer.contains(document.activeElement) && buttonFor(tool)) buttonFor(tool).focus();
  }

  drawer.querySelector("[data-drawer-close]").addEventListener("click", close);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && openTool !== null) close();
  });

  return {
    toggle: (tool) => (openTool === tool ? close() : open(tool)),
    // Follow the tabs: show the new tool's text, or close on Backstage.
    viewChanged(view) {
      if (openTool === null) return;
      if (bodies.some((body) => body.dataset.drawerTool === view)) open(view);
      else close();
    },
  };
}
