import * as BABYLON from "babylonjs";
// window.L is set by the synchronous /leaflet.js script tag in simulation.html

class Playground {
  public static async CreateScene(engine: BABYLON.Engine, canvas: HTMLCanvasElement): Promise<BABYLON.Scene> {
    var scene = new BABYLON.Scene(engine);
    scene.clearColor = new BABYLON.Color4(0.08, 0.09, 0.13, 1);

    var camera = new BABYLON.ArcRotateCamera("camera", Math.PI / 4, Math.PI / 3, 150, BABYLON.Vector3.Zero(), scene);
    camera.attachControl(canvas, true);
    camera.lowerRadiusLimit = 30;
    camera.upperRadiusLimit = 250;

    const sunLight = new BABYLON.DirectionalLight("sunLight", new BABYLON.Vector3(-0.5, -1.0, -0.3).normalize(), scene);
    sunLight.intensity = 1.4;
    sunLight.diffuse = new BABYLON.Color3(1.0, 0.95, 0.85);

    var hemiLight = new BABYLON.HemisphericLight("hemiLight", new BABYLON.Vector3(0, 1, 0), scene);
    hemiLight.intensity = 0.5;
    hemiLight.groundColor = new BABYLON.Color3(0.1, 0.1, 0.15);

    // =========================================================
    // 1. HARDCODED METADATA
    // =========================================================
    const h_norm_max_m = 20.0;
    const sim_duration_s = 7200;

    const terrainSize = 100;
    const subdivisions = 256;

    // =========================================================
    // 2. LOAD TEXTURES
    // =========================================================
    const satelliteTexture = new BABYLON.Texture("/satellite.jpg", scene);
    const floodTexture = new BABYLON.Texture("/flood_packed.png", scene, false, false, BABYLON.Texture.BILINEAR_SAMPLINGMODE);

    const normalTexture = new BABYLON.Texture("https://assets.babylonjs.com/textures/waterbump.png", scene);
    normalTexture.wrapU = BABYLON.Texture.WRAP_ADDRESSMODE;
    normalTexture.wrapV = BABYLON.Texture.WRAP_ADDRESSMODE;

    // =========================================================
    // 3. TERRAIN SHADER 
    // =========================================================
    BABYLON.Effect.ShadersStore["terrainVertexShader"] = `
      precision highp float;
      attribute vec3 position;
      attribute vec2 uv;
      uniform mat4 worldViewProjection;
      varying vec2 vUV;
      void main() {
        vUV = uv;
        gl_Position = worldViewProjection * vec4(position, 1.0);
      }
    `;

    BABYLON.Effect.ShadersStore["terrainFragmentShader"] = `
      precision highp float;
      varying vec2 vUV;
      uniform sampler2D satellite;
      uniform sampler2D floodMask;
      uniform float floodProgress;

      void main() {
        vec2 fUV = vec2(vUV.x, 1.0 - vUV.y);
        vec3 land = texture2D(satellite, vUV).rgb;
        
        // HORIZONTAL DILATION (Tuned down to a perfect middle ground)
        float rad = 0.009; 
        float diag = rad * 0.707;
        float halfRad = rad * 0.5;
        
        vec4 c  = texture2D(floodMask, fUV);
        vec4 l  = texture2D(floodMask, fUV + vec2(-rad, 0.0));
        vec4 r  = texture2D(floodMask, fUV + vec2(rad, 0.0));
        vec4 u  = texture2D(floodMask, fUV + vec2(0.0, rad));
        vec4 d  = texture2D(floodMask, fUV + vec2(0.0, -rad));
        vec4 tl = texture2D(floodMask, fUV + vec2(-diag, diag));
        vec4 tr = texture2D(floodMask, fUV + vec2(diag, diag));
        vec4 bl = texture2D(floodMask, fUV + vec2(-diag, -diag));
        vec4 br = texture2D(floodMask, fUV + vec2(diag, -diag));
        
        vec4 ml = texture2D(floodMask, fUV + vec2(-halfRad, 0.0));
        vec4 mr = texture2D(floodMask, fUV + vec2(halfRad, 0.0));
        vec4 mu = texture2D(floodMask, fUV + vec2(0.0, halfRad));
        vec4 md = texture2D(floodMask, fUV + vec2(0.0, -halfRad));

        float arrivalTime = min(c.g, min(min(l.g, r.g), min(u.g, d.g)));
        arrivalTime = min(arrivalTime, min(min(tl.g, tr.g), min(bl.g, br.g)));
        arrivalTime = min(arrivalTime, min(min(ml.g, mr.g), min(mu.g, md.g)));

        float isActive = 0.0;
        if (arrivalTime < 0.99 && floodProgress >= arrivalTime) {
            isActive = 1.0;
        }

        vec3 wetLand = land * vec3(0.50, 0.55, 0.62);
        vec3 finalLand = mix(land, wetLand, isActive * 0.75);
        gl_FragColor = vec4(finalLand, 1.0);
      }
    `;

    const terrainMaterial = new BABYLON.ShaderMaterial("terrainShader", scene,
      { vertex: "terrain", fragment: "terrain" },
      { attributes: ["position", "uv"], uniforms: ["worldViewProjection", "floodProgress"] }
    );
    terrainMaterial.setTexture("satellite", satelliteTexture);
    terrainMaterial.setTexture("floodMask", floodTexture);

    // =========================================================
    // 4. WATER SHADER
    // =========================================================
    BABYLON.Effect.ShadersStore["waterVertexShader"] = `
      precision highp float;
      attribute vec3 position;
      attribute vec2 uv;
      
      uniform mat4 worldViewProjection;
      uniform mat4 world;
      uniform sampler2D floodMask;
      uniform float floodProgress;
      uniform float hNormMaxM;
      
      varying vec2 vUV;
      varying vec3 vPositionW;
      
      void main() {
        vUV = uv;
        vec2 fUV = vec2(uv.x, 1.0 - vUV.y);
        
        // HORIZONTAL DILATION (Tuned down to a perfect middle ground)
        float rad = 0.009; 
        float diag = rad * 0.707;
        float halfRad = rad * 0.5;
        
        vec4 c  = texture2D(floodMask, fUV);
        vec4 l  = texture2D(floodMask, fUV + vec2(-rad, 0.0));
        vec4 r  = texture2D(floodMask, fUV + vec2(rad, 0.0));
        vec4 u  = texture2D(floodMask, fUV + vec2(0.0, rad));
        vec4 d  = texture2D(floodMask, fUV + vec2(0.0, -rad));
        vec4 tl = texture2D(floodMask, fUV + vec2(-diag, diag));
        vec4 tr = texture2D(floodMask, fUV + vec2(diag, diag));
        vec4 bl = texture2D(floodMask, fUV + vec2(-diag, -diag));
        vec4 br = texture2D(floodMask, fUV + vec2(diag, -diag));

        vec4 ml = texture2D(floodMask, fUV + vec2(-halfRad, 0.0));
        vec4 mr = texture2D(floodMask, fUV + vec2(halfRad, 0.0));
        vec4 mu = texture2D(floodMask, fUV + vec2(0.0, halfRad));
        vec4 md = texture2D(floodMask, fUV + vec2(0.0, -halfRad));

        float arrivalTime = min(c.g, min(min(l.g, r.g), min(u.g, d.g)));
        arrivalTime = min(arrivalTime, min(min(tl.g, tr.g), min(bl.g, br.g)));
        arrivalTime = min(arrivalTime, min(min(ml.g, mr.g), min(mu.g, md.g)));

        float depth = max(c.r, max(max(l.r, r.r), max(u.r, d.r)));
        depth = max(depth, max(max(tl.r, tr.r), max(bl.r, br.r)));
        depth = max(depth, max(max(ml.r, mr.r), max(mu.r, md.r)));
        
        float targetY = position.y;
        
        if (arrivalTime >= 0.99 || floodProgress < arrivalTime) {
            targetY -= 0.5; 
        } else {
            float rise = smoothstep(0.0, 0.05, floodProgress - arrivalTime);
            // Height multiplier kept identical
            targetY += (depth * hNormMaxM * 0.15 + 0.6) * rise;
        }

        vec3 displacedPos = vec3(position.x, targetY, position.z);
        vPositionW = (world * vec4(displacedPos, 1.0)).xyz;
        gl_Position = worldViewProjection * vec4(displacedPos, 1.0);
      }
    `;

    BABYLON.Effect.ShadersStore["waterFragmentShader"] = `
      precision highp float;
      varying vec2 vUV;
      varying vec3 vPositionW;

      uniform sampler2D floodMask;
      uniform sampler2D normalMap;

      uniform float time;
      uniform float floodProgress;
      uniform vec3  cameraPosition;
      uniform vec3  lightDir;
      uniform vec2  flowDir;

      void main() {
        vec2 fUV = vec2(vUV.x, 1.0 - vUV.y);
        
        // HORIZONTAL DILATION (Tuned down to a perfect middle ground)
        float rad = 0.009; 
        float diag = rad * 0.707;
        float halfRad = rad * 0.5;
        
        vec4 c  = texture2D(floodMask, fUV);
        vec4 l  = texture2D(floodMask, fUV + vec2(-rad, 0.0));
        vec4 r  = texture2D(floodMask, fUV + vec2(rad, 0.0));
        vec4 u  = texture2D(floodMask, fUV + vec2(0.0, rad));
        vec4 d  = texture2D(floodMask, fUV + vec2(0.0, -rad));
        vec4 tl = texture2D(floodMask, fUV + vec2(-diag, diag));
        vec4 tr = texture2D(floodMask, fUV + vec2(diag, diag));
        vec4 bl = texture2D(floodMask, fUV + vec2(-diag, -diag));
        vec4 br = texture2D(floodMask, fUV + vec2(diag, -diag));

        vec4 ml = texture2D(floodMask, fUV + vec2(-halfRad, 0.0));
        vec4 mr = texture2D(floodMask, fUV + vec2(halfRad, 0.0));
        vec4 mu = texture2D(floodMask, fUV + vec2(0.0, halfRad));
        vec4 md = texture2D(floodMask, fUV + vec2(0.0, -halfRad));

        float arrivalTime = min(c.g, min(min(l.g, r.g), min(u.g, d.g)));
        arrivalTime = min(arrivalTime, min(min(tl.g, tr.g), min(bl.g, br.g)));
        arrivalTime = min(arrivalTime, min(min(ml.g, mr.g), min(mu.g, md.g)));

        float depth = max(c.r, max(max(l.r, r.r), max(u.r, d.r)));
        depth = max(depth, max(max(tl.r, tr.r), max(bl.r, br.r)));
        depth = max(depth, max(max(ml.r, mr.r), max(mu.r, md.r)));
        
        if (arrivalTime >= 0.99 || floodProgress < arrivalTime) {
            discard; 
        }

        float waterDepth = smoothstep(0.005, 0.4, depth);

        vec2 flow1 = flowDir * time * 0.55;
        vec2 uv1   = vUV * 9.0 + flow1;
        vec2 lateralDir = vec2(-flowDir.y, flowDir.x);
        vec2 flow2 = flowDir * time * 0.30 + lateralDir * time * 0.08;
        vec2 uv2   = vUV * 15.0 + flow2;

        vec3 n1 = texture2D(normalMap, uv1).rgb * 2.0 - 1.0;
        vec3 n2 = texture2D(normalMap, uv2).rgb * 2.0 - 1.0;

        vec2 blendedXY  = (n1.xy * 0.65 + n2.xy * 0.35) * clamp(waterDepth * 1.3, 0.0, 1.0);
        vec3 worldNormal = normalize(vec3(blendedXY.x, 1.8, blendedXY.y));

        vec3 viewDir  = normalize(cameraPosition - vPositionW);
        vec3 halfVec  = normalize(lightDir + viewDir);
        float NdotH   = max(0.0, dot(worldNormal, halfVec));
        float specular = pow(NdotH, 96.0) * 2.0;
        float diffuse  = max(0.0, dot(worldNormal, lightDir));

        vec3 shallowColor = vec3(0.18, 0.45, 0.70);
        vec3 deepColor    = vec3(0.02, 0.25, 0.75);
        vec3 baseWater    = mix(shallowColor, deepColor, waterDepth);
        
        baseWater += vec3(0.05, 0.1, 0.15) * diffuse;
        baseWater += vec3(0.95, 0.97, 1.0) * specular;

        float leadingEdge = smoothstep(floodProgress - 0.04, floodProgress, arrivalTime);
        baseWater = mix(baseWater, vec3(0.9, 0.95, 1.0), leadingEdge * 0.8);

        float alpha = mix(0.85, 0.98, waterDepth);
        gl_FragColor = vec4(baseWater, clamp(alpha, 0.0, 1.0));
      }
    `;

    const waterMaterial = new BABYLON.ShaderMaterial("waterShader", scene,
      { vertex: "water", fragment: "water" },
      {
        attributes: ["position", "uv"],
        uniforms: [
          "worldViewProjection", "world",
          "time", "floodProgress", "hNormMaxM",
          "cameraPosition", "lightDir", "flowDir"
        ]
      }
    );
    waterMaterial.setTexture("floodMask", floodTexture);
    waterMaterial.setTexture("normalMap", normalTexture);
    waterMaterial.setVector3("lightDir", new BABYLON.Vector3(0.5, 1.0, 0.3).normalize());
    waterMaterial.setVector2("flowDir", new BABYLON.Vector2(0.707, 0.707));
    waterMaterial.setFloat("hNormMaxM", h_norm_max_m);

    waterMaterial.backFaceCulling = false;
    waterMaterial.alphaMode = BABYLON.Constants.ALPHA_COMBINE;
    waterMaterial.transparencyMode = BABYLON.Material.MATERIAL_ALPHABLEND;

    // =========================================================
    // 5. CREATE MESHES
    // =========================================================
    const terrain = BABYLON.MeshBuilder.CreateGroundFromHeightMap(
      "terrain",
      "/heightmap.jpg",
      { width: terrainSize, height: terrainSize, subdivisions: subdivisions, maxHeight: 20, minHeight: 0, onReady: (mesh) => mesh.freezeWorldMatrix() },
      scene
    );
    terrain.material = terrainMaterial;

    const waterMesh = BABYLON.MeshBuilder.CreateGroundFromHeightMap(
      "waterSurface",
      "/heightmap.jpg",
      { width: terrainSize, height: terrainSize, subdivisions: subdivisions, maxHeight: 20, minHeight: 0, onReady: (mesh) => mesh.freezeWorldMatrix() },
      scene
    );
    waterMesh.material = waterMaterial;

    scene.setRenderingAutoClearDepthStencil(1, false, false, false);
    terrain.renderingGroupId = 0;
    waterMesh.renderingGroupId = 1;

    // =========================================================
    // 6. UI DASHBOARD & RENDER LOOP
    // =========================================================

    let time = 0;
    let floodProgress = 0.0;
    let isPlaying = false;
    let playbackSpeed = 1;
    let infraRevealed = false;   // guard so infra panel reveals only once
    let leafletMap: any = null;  // Leaflet map instance (initialized once)
    let cachedMetadata: any = null; // cached full JSON for showDamageMap

    // Grab basic UI elements
    const slider = document.getElementById("timeSlider") as HTMLInputElement;
    const playBtn = document.getElementById("playBtn") as HTMLButtonElement;
    const resetBtn = document.getElementById("resetBtn") as HTMLButtonElement;
    const speedSelect = document.getElementById("speedSelect") as HTMLSelectElement;
    const impactText = document.getElementById("impactText") as HTMLSpanElement;

    // --- POINTS OF INTEREST CAMERA LOGIC ---
    // DEM: EPSG:4326, bounds lon 88.15–88.70, lat 27.55–27.95
    // Babylon ground: X = (lon−88.15)/0.55×100−50, Z = (lat−27.55)/0.40×100−50
    // Only locations INSIDE the terrain grid (−50..+50) are shown.
    // Mangan (Z=−63) and Dikchu (Z=−86) are outside → removed.
    const locations: Record<string, { target: BABYLON.Vector3, alpha: number, beta: number, radius: number }> = {
      overview: {
        // Default camera — full terrain view (matches initial scene setup)
        target: BABYLON.Vector3.Zero(),
        alpha: Math.PI / 4, beta: Math.PI / 3, radius: 150
      },
      lake: {
        // South Lhonak Lake — 27.913°N 88.199°E → X=-41 Z=+41
        target: new BABYLON.Vector3(-41, 6, 41),
        alpha: Math.PI / 4, beta: Math.PI / 3.2, radius: 22
      },
      upperValley: {
        // Upper Teesta Valley — 27.820°N 88.280°E → X=-26 Z=+18
        target: new BABYLON.Vector3(-26, 3, 18),
        alpha: Math.PI / 4, beta: Math.PI / 3.5, radius: 32
      },
      chungthang: {
        // Chungthang Dam — 27.598°N 88.651°E → X=+41 Z=-38
        target: new BABYLON.Vector3(41, 0, -38),
        alpha: Math.PI * 0.6, beta: Math.PI / 4, radius: 38
      }
    };

    // Attach click listeners to the HTML buttons
    const poiButtons = document.querySelectorAll(".poi-btn");
    poiButtons.forEach(btn => {
      btn.addEventListener("click", (e) => {
        const targetId = (e.currentTarget as HTMLElement).getAttribute("data-target");
        if (targetId && locations[targetId]) {
          const loc = locations[targetId];

          // Animate the camera smoothly to the new location
          BABYLON.Animation.CreateAndStartAnimation("camMove", camera, "target", 60, 60, camera.target, loc.target, 2, new BABYLON.CubicEase());
          BABYLON.Animation.CreateAndStartAnimation("camRadius", camera, "radius", 60, 60, camera.radius, loc.radius, 2, new BABYLON.CubicEase());
          BABYLON.Animation.CreateAndStartAnimation("camAlpha", camera, "alpha", 60, 60, camera.alpha, loc.alpha, 2, new BABYLON.CubicEase());
          BABYLON.Animation.CreateAndStartAnimation("camBeta", camera, "beta", 60, 60, camera.beta, loc.beta, 2, new BABYLON.CubicEase());
        }
      });
    });
    // --- END POI LOGIC ---

    // =========================================================
    // ETA Tracker — live status under each location name
    // =========================================================
    // Impact thresholds in simulation seconds (sim_duration = 7200s)
    const ETA_THRESHOLDS: Record<string, number> = {
      lake: 0,           // breach origin — already impacted
      upperValley: 1200, // ~20 min
      chungthang: 3600,  // ~60 min
    };
    const RECEDE_OFFSET = 1800; // seconds after impact before 'ISOLATED' label

    const updateETA = (simTimeSec: number) => {
      for (const [id, impactSec] of Object.entries(ETA_THRESHOLDS)) {
        const el = document.getElementById(`eta-${id}`);
        if (!el) continue;
        if (id === 'lake') continue; // always shows BREACH ORIGIN
        const remaining = impactSec - simTimeSec;
        if (remaining > 0) {
          const mins = Math.ceil(remaining / 60);
          el.textContent = `ETA: ${mins} min`;
          el.style.color = remaining < 900 ? '#f97316' : '#facc15';
          el.style.animation = '';
        } else if (simTimeSec < impactSec + RECEDE_OFFSET) {
          el.textContent = '⚡ IMPACTED';
          el.style.color = '#ef4444';
          el.style.animation = 'etaPulse 1s ease-in-out infinite';
        } else {
          const impactMins = Math.round(impactSec / 60);
          el.textContent = `REACHED IN: ${impactMins} MINS`;
          el.style.color = '#f87171'; // A lighter red/coral color to indicate it has already hit
          el.style.animation = '';
        }
      }
    };

    // Grab our new Telemetry/Damage Report elements
    const timeText = document.getElementById("timeText");
    const areaText = document.getElementById("areaText");
    const volText = document.getElementById("volText");
    const flowText = document.getElementById("flowText");

    // Fetch the metadata for telemetry loop AND infrastructure damage panel
    let telemetryData: TelemetryPoint[] = [];
    fetch("/glof_metadata.json")
      .then(res => res.json())
      .then(data => {
        telemetryData = data.telemetry || [];

        // ── Infrastructure damage panel ──────────────────────────────────
        const infra = data.infrastructure_damage_summary;
        if (infra) {
          const set = (id: string, val: number | string) => {
            const el = document.getElementById(id);
            if (el) el.textContent = String(val);
          };
          const setZero = (rowId: string, val: number) => {
            const row = document.getElementById(rowId);
            if (row) row.classList.toggle('zero', val === 0);
          };

          set('infra-roads', infra.roads_destroyed_km?.toFixed(2) ?? '0');
          set('infra-bridges', infra.bridges_destroyed ?? 0);
          set('infra-power', infra.power_facilities_destroyed ?? 0);
          set('infra-dams', infra.dams_at_risk ?? 0);
          set('infra-hospitals', infra.hospitals_destroyed ?? 0);
          set('infra-schools', infra.schools_destroyed ?? 0);

          setZero('infra-row-roads', infra.roads_destroyed_km ?? 0);
          setZero('infra-row-bridges', infra.bridges_destroyed ?? 0);
          setZero('infra-row-power', infra.power_facilities_destroyed ?? 0);
          setZero('infra-row-dams', infra.dams_at_risk ?? 0);
          setZero('infra-row-hospitals', infra.hospitals_destroyed ?? 0);
          setZero('infra-row-schools', infra.schools_destroyed ?? 0);

          // Critical assets tags
          const list = document.getElementById('infra-assets-list')!;
          const noAssets = document.getElementById('infra-no-assets');
          const assets: string[] = infra.critical_assets_at_risk ?? [];
          if (assets.length > 0) {
            if (noAssets) noAssets.remove();
            assets.forEach(name => {
              const tag = document.createElement('span');
              tag.className = 'asset-tag';
              tag.textContent = name;
              list.appendChild(tag);
            });
          } else {
            if (noAssets) noAssets.textContent = 'None identified';
          }
        }
        // ─────────────────────────────────────────────────────────────────
        cachedMetadata = data; // store for showDamageMap()
        (window as any)._glofMeta = data; // expose for plain-script showDamageMap
      });

    // =========================================================
    // showDamageMap() — Post-Simulation 2D Leaflet Damage Map
    // =========================================================


    // Expose as window globals for inline onclick handlers
    // (window as any).showDamageMap = showDamageMap;
    // (window as any).hideDamageMap = hideDamageMap;

    speedSelect.onchange = () => { playbackSpeed = parseFloat(speedSelect.value); };

    playBtn.onclick = () => { isPlaying = !isPlaying; playBtn.innerText = isPlaying ? "Pause" : "Play"; };
    resetBtn.onclick = () => {
      floodProgress = 0;
      slider.value = "0";
      isPlaying = false;
      playBtn.innerText = "Play";
      // Hide infra panel again so it reveals fresh on the next playthrough
      infraRevealed = false;
      const infraPanel = document.getElementById("infraPanel");
      if (infraPanel) infraPanel.classList.remove("infra-visible");

      // YAHAN CHANGE KIYA HAI: Window function call kar rahe hain
      if (typeof (window as any).hideDamageMap === "function") {
        (window as any).hideDamageMap();
      }
    };
    slider.oninput = () => { isPlaying = false; playBtn.innerText = "Play"; floodProgress = parseFloat(slider.value) / 100.0; };

    // Helper function to find the closest telemetry data point based on the slider
    const updateTelemetryUI = (progress: number) => {
      if (!telemetryData.length || !timeText || !areaText || !volText || !flowText) return;

      const currentSimTime = progress * sim_duration_s;

      // Find the closest data point in the array
      let closest = telemetryData[0];
      let minDiff = Math.abs(closest.time_sec - currentSimTime);

      for (const pt of telemetryData) {
        const diff = Math.abs(pt.time_sec - currentSimTime);
        if (diff < minDiff) {
          closest = pt;
          minDiff = diff;
        }
      }

      // Update the HTML text
      timeText.innerText = closest.time_sec.toFixed(0);
      areaText.innerText = closest.inundated_area_km2.toFixed(2);
      volText.innerText = closest.active_volume_m3.toLocaleString('en-US', { maximumFractionDigits: 0 });
      flowText.innerText = closest.peak_discharge_m3s.toLocaleString('en-US', { maximumFractionDigits: 0 });
    };

    scene.onBeforeRenderObservable.add(() => {
      const delta = engine.getDeltaTime() * 0.001;
      time += delta;

      terrainMaterial.setFloat("floodProgress", floodProgress);
      waterMaterial.setFloat("time", time);
      waterMaterial.setFloat("floodProgress", floodProgress);
      waterMaterial.setVector3("cameraPosition", camera.position);

      if (isPlaying) {
        floodProgress += delta * 0.03 * playbackSpeed;
        if (floodProgress <= 1.0) {
          slider.value = (floodProgress * 100).toString();
        } else {
          isPlaying = false;
          playBtn.innerText = "Play";
        }
      }

      // Update the side panel text
      if (floodProgress < 0.2) {
        impactText.innerText = "Contained";
        impactText.style.color = "#4ade80";
      } else if (floodProgress < 1.0) {
        impactText.innerText = "Critical Warning — Valley Flooding";
        impactText.style.color = "#facc15";
      } else {
        impactText.innerText = "Disaster — Full Inundation";
        impactText.style.color = "#ef4444";
        // Reveal infrastructure panel and damage map once when flood is complete
        if (!infraRevealed) {
          infraRevealed = true;
          const infraPanel = document.getElementById("infraPanel");
          if (infraPanel) infraPanel.classList.add("infra-visible");

          // YAHAN CHANGE KIYA HAI: Window function call kar rahe hain
          setTimeout(() => {
            if (typeof (window as any).showDamageMap === "function") {
              (window as any).showDamageMap();
            }
          }, 1200);
        }
      }

      // Feed the progress into our telemetry and ETA updaters
      const simTimeSec = floodProgress * sim_duration_s;
      updateTelemetryUI(floodProgress);
      updateETA(simTimeSec);
    });

    return scene;
  }
}

const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
const engine = new BABYLON.Engine(canvas, true);
Playground.CreateScene(engine, canvas).then((scene) => { engine.runRenderLoop(() => scene.render()); });
window.addEventListener("resize", () => engine.resize());