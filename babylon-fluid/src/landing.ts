// @ts-ignore
import Globe from 'globe.gl';
import * as THREE from 'three';

// ─────────────────────────────────────────────
// 1. GLACIER DATA
// ─────────────────────────────────────────────
interface GlacierEntry {
  id: string;
  name: string;
  region: string;
  lat: number;
  lon: number;
  href: string;
}

const GLACIERS: GlacierEntry[] = [
  {
    id: 'south-lhonak',
    name: 'South Lhonak Lake',
    region: 'North Sikkim, India',
    lat: 27.77,
    lon: 88.22,
    href: './command-center.html'
  }
].sort((a, b) => a.name.localeCompare(b.name));

// ─────────────────────────────────────────────
// 2. STAR FIELD (full-page canvas 2D)
// ─────────────────────────────────────────────
(function () {
  const c = document.getElementById('stars') as HTMLCanvasElement;
  const ctx = c.getContext('2d')!;
  const resize = () => { c.width = window.innerWidth; c.height = window.innerHeight; };
  resize();
  window.addEventListener('resize', resize);

  const stars = Array.from({ length: 320 }, () => ({
    x: Math.random(),
    y: Math.random(),
    r: Math.random() * 1.1 + 0.15,
    phase: Math.random() * Math.PI * 2
  }));

  let t = 0;
  (function draw() {
    ctx.clearRect(0, 0, c.width, c.height);
    t += 0.009;
    stars.forEach(s => {
      const alpha = 0.20 + 0.65 * (0.5 + 0.5 * Math.sin(t + s.phase));
      ctx.beginPath();
      ctx.arc(s.x * c.width, s.y * c.height, s.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(210,230,255,${alpha})`;
      ctx.fill();
    });
    requestAnimationFrame(draw);
  })();
})();

// ─────────────────────────────────────────────
// 3. SEARCH / DROPDOWN
// ─────────────────────────────────────────────
const searchInput = document.getElementById('search-input') as HTMLInputElement;
const dropdownBtn = document.getElementById('dropdown-btn') as HTMLButtonElement;
const glacierListEl = document.getElementById('glacier-list') as HTMLDivElement;

let listOpen = false;

function buildList(query: string) {
  const q = query.trim().toLowerCase();
  const filtered = q
    ? GLACIERS.filter(g => g.name.toLowerCase().includes(q) || g.region.toLowerCase().includes(q))
    : GLACIERS;

  glacierListEl.innerHTML = '';

  if (!filtered.length) {
    const el = document.createElement('div');
    el.className = 'glacier-item no-result';
    el.textContent = 'No glaciers found';
    glacierListEl.appendChild(el);
    return;
  }

  filtered.forEach(g => {
    const el = document.createElement('div');
    el.className = 'glacier-item';
    el.innerHTML = `<div class="gi-name">${g.name}</div><div class="gi-region">${g.region}</div>`;
    el.addEventListener('click', () => {
      searchInput.value = g.name;
      closeList();
      flyToGlacier(g);
    });
    glacierListEl.appendChild(el);
  });
}

function openList() {
  listOpen = true;
  glacierListEl.classList.add('open');
  dropdownBtn.classList.add('open');
  buildList(searchInput.value);
}

function closeList() {
  listOpen = false;
  glacierListEl.classList.remove('open');
  dropdownBtn.classList.remove('open');
}

dropdownBtn.addEventListener('click', e => { e.stopPropagation(); listOpen ? closeList() : openList(); });
searchInput.addEventListener('input', () => { if (!listOpen) openList(); else buildList(searchInput.value); });
searchInput.addEventListener('focus', () => { if (!listOpen) openList(); });
document.addEventListener('click', e => {
  if (!(document.getElementById('search-wrap') as HTMLElement).contains(e.target as Node)) closeList();
});

// ─────────────────────────────────────────────
// 4. SHARED REFS (needed by flyToGlacier + Globe init)
// ─────────────────────────────────────────────
const globeContainer = document.getElementById('globe-inner') as HTMLDivElement;
const overlay = document.getElementById('zoom-overlay') as HTMLDivElement;

let _world: any = null;
let _controls: any = null;
let autoSpinStopped = false;

// ─────────────────────────────────────────────
// 5. FLY TO GLACIER — 3-stage cinematic zoom
// ─────────────────────────────────────────────
function flyToGlacier(g: GlacierEntry) {
  autoSpinStopped = true;
  if (_controls) _controls.autoRotate = false;

  // Light up the selected marker
  const markerEl = markerEls.get(g.id);
  if (markerEl) markerEl.classList.add('marker-selected');

  // Stage 1 — smooth global pan + approach (0 → 2000ms)
  if (_world) _world.pointOfView({ lat: g.lat, lng: g.lon, altitude: 1.4 }, 2000);

  // Stage 2 — zoom in toward marker (2000 → 3600ms) — altitude 0.55 avoids atmosphere glare
  setTimeout(() => {
    if (_world) _world.pointOfView({ lat: g.lat, lng: g.lon, altitude: 0.55 }, 1600);
  }, 2000);

  // Stage 3 — navigate directly after zoom lands (no flash)
  setTimeout(() => {
    window.location.href = g.href;
  }, 3500);
}

// Expose so dropdown items built before Globe init can call it
(window as any).__flyToGlacier = flyToGlacier;

// Track marker elements by glacier id for visual feedback
const markerEls = new Map<string, HTMLElement>();

function makeMarkerEl(g: GlacierEntry): HTMLElement {
  const el = document.createElement('div');
  el.className = 'glacier-marker';
  el.innerHTML = `
    <div class="marker-ring"></div>
    <div class="marker-ring"></div>
    <div class="marker-dot"></div>
    <div class="marker-label">${g.name}</div>
  `;
  el.addEventListener('click', (e) => {
    e.stopPropagation();
    flyToGlacier(g);
  });
  markerEls.set(g.id, el);
  return el;
}

// ─────────────────────────────────────────────
// 7. GLOBE.GL
// ─────────────────────────────────────────────
const world = Globe({ animateIn: true })
  .globeImageUrl('//cdn.jsdelivr.net/npm/three-globe/example/img/earth-blue-marble.jpg')
  .bumpImageUrl('//cdn.jsdelivr.net/npm/three-globe/example/img/earth-topology.png')
  .backgroundColor('rgba(0,0,0,0)')
  .showAtmosphere(false)              // completely remove blue/teal atmosphere mesh
  // Large HTML markers with 56px clickable area
  .htmlElementsData(GLACIERS)
  .htmlLat((d: any) => d.lat)
  .htmlLng((d: any) => d.lon)
  .htmlAltitude(0.01)
  .htmlElement((d: any) => makeMarkerEl(d as GlacierEntry))
  (globeContainer);

_world = world;

// Globe material — subtle bump only, no specular glare
const globeMaterial = world.globeMaterial();
globeMaterial.bumpScale = 10;
globeMaterial.shininess = 0;

// ─────────────────────────────────────────────
// 8. AUTO-ROTATE — stops on first globe click
// ─────────────────────────────────────────────
const controls = world.controls();
_controls = controls;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;
controls.enableZoom = false;

globeContainer.addEventListener('mousedown', () => {
  if (!autoSpinStopped) {
    autoSpinStopped = true;
    controls.autoRotate = false;
  }
}, { capture: true });

globeContainer.addEventListener('touchstart', () => {
  if (!autoSpinStopped) {
    autoSpinStopped = true;
    controls.autoRotate = false;
  }
}, { capture: true, passive: true });

// ─────────────────────────────────────────────
// 9. INITIAL VIEW — India / South Asia facing camera
// ─────────────────────────────────────────────
// Point of view: India / South Asia, slightly west so globe centres in the visible area
world.pointOfView({ lat: 22, lng: 80, altitude: 2.2 });

// ─────────────────────────────────────────────
// 10. RESIZE
// ─────────────────────────────────────────────
window.addEventListener('resize', () => {
  world.width(globeContainer.clientWidth);
  world.height(globeContainer.clientHeight);
});

// ─────────────────────────────────────────────
// 11. BFCACHE — reset globe when user presses Back
// ─────────────────────────────────────────────
// When the browser restores this page from back-forward cache, the globe
// camera is frozen at the zoomed-in position. Reset it to the initial view.
window.addEventListener('pageshow', (e: PageTransitionEvent) => {
  if (e.persisted) {
    // Clear any selected marker state
    markerEls.forEach(el => el.classList.remove('marker-selected'));
    // Reset auto-spin
    autoSpinStopped = false;
    if (_controls) {
      _controls.autoRotate = true;
    }
    // Snap back to initial view instantly, then animate to full altitude
    if (_world) {
      _world.pointOfView({ lat: 22, lng: 80, altitude: 2.2 }, 1200);
    }
  }
});
