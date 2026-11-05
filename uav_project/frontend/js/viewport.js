// viewport.js
// Advanced Three.js UAV Viewport with:
// - True Center-Aligned Camera Framing (UAV stays rock-steady in the middle)
// - Built-in Mouse/Touch Orbit Controls (360° rotate, pan, scroll zoom)
// - See-Through / X-Ray Mode (revealing internal payload bay, moving CG, AC, & wing spar)
// - Dynamic Propeller & Control Surface Deflections
// - Camera Presets (Center, Chase, Free Orbit, Top, Side, Front)
// - Canvas Snapshot Capture

class AircraftViewport {
  constructor(containerEl, color = 0x4dd0e1) {
    this.container = containerEl;
    this.color = color;
    this.real = null;
    this.flightTrail = [];
    this.lastFrame = null;
    this.changeTime = null;
    this.eventMarked = false;

    // View & Rendering Modes
    this.viewMode = "chase"; // 'chase' | 'orbit' | 'top' | 'side' | 'front'
    this.renderMode = "solid"; // 'solid' | 'seethrough' | 'wireframe'
    this.currentPayload = 0.5;

    // Orbit camera parameters relative to aircraft center
    this.orbit = {
      azimuth: 0.45,       // horizontal angle (radians)
      elevation: 0.32,     // vertical angle (radians)
      distance: 4.8,       // zoom radius (meters)
      panX: 0,
      panY: 0,
      minDist: 0.8,
      maxDist: 35.0,
    };
    this.isDragging = false;
    this.dragMode = "rotate"; // 'rotate' | 'pan'
    this.lastPointer = { x: 0, y: 0 };

    // Three.js Scene Setup
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(48, 1, 0.05, 1000);
    this.camera.up.set(0, 0, 1); // Z is Up

    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.container.appendChild(this.renderer.domElement);

    // Lighting
    const amb = new THREE.AmbientLight(0xffffff, 0.70);
    this.scene.add(amb);

    this.sunLight = new THREE.DirectionalLight(0xffffff, 0.85);
    this.sunLight.position.set(6, -8, 12);
    this.scene.add(this.sunLight);

    const fillLight = new THREE.DirectionalLight(0x7dd3fc, 0.35);
    fillLight.position.set(-6, 8, -4);
    this.scene.add(fillLight);

    // Ground Grid
    this.grid = new THREE.GridHelper(40, 40, 0x38bdf8, 0x172a3a);
    this.grid.rotation.x = Math.PI / 2; // XY ground plane; Z is up
    this.scene.add(this.grid);

    // Aircraft Hierarchy
    this.group = new THREE.Group();
    this.scene.add(this.group);

    // Mesh caches
    this.partMeshes = {};
    this.wireframeMeshes = {};

    // Internal X-Ray Anatomy
    this._createInternalStructures();
    this._createWindField();
    this._createFlightTrail();
    this._createEventMarker();
    this._initInputHandlers();

    this._resize();
    window.addEventListener("resize", () => this._resize());
  }

  _resize() {
    const w = this.container.clientWidth || 300;
    const h = this.container.clientHeight || 300;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // ------------------------------------------------------------------
  // Interactive Orbit / Pan / Zoom Controls
  // ------------------------------------------------------------------
  _initInputHandlers() {
    const el = this.renderer.domElement;
    el.style.touchAction = "none";
    el.style.cursor = "grab";

    el.addEventListener("contextmenu", (e) => e.preventDefault());

    el.addEventListener("pointerdown", (e) => {
      this.isDragging = true;
      this.lastPointer = { x: e.clientX, y: e.clientY };
      el.style.cursor = "grabbing";
      this.dragMode = e.button === 2 || e.shiftKey ? "pan" : "rotate";
      try { el.setPointerCapture(e.pointerId); } catch (_) {}
    });

    el.addEventListener("pointermove", (e) => {
      if (!this.isDragging) return;
      const dx = e.clientX - this.lastPointer.x;
      const dy = e.clientY - this.lastPointer.y;
      this.lastPointer = { x: e.clientX, y: e.clientY };

      if (this.dragMode === "rotate") {
        this.orbit.azimuth -= dx * 0.008;
        this.orbit.elevation = Math.max(
          -Math.PI / 2 + 0.05,
          Math.min(Math.PI / 2 - 0.05, this.orbit.elevation + dy * 0.008)
        );
        if (this.viewMode === "chase") {
          // Subtle nudge allowed in chase mode, or auto-switch to orbit
        }
      } else {
        const factor = this.orbit.distance * 0.002;
        this.orbit.panX -= dx * factor;
        this.orbit.panY += dy * factor;
      }
      this._updateCamera();
    });

    const endDrag = (e) => {
      this.isDragging = false;
      el.style.cursor = "grab";
      try { el.releasePointerCapture(e.pointerId); } catch (_) {}
    };
    el.addEventListener("pointerup", endDrag);
    el.addEventListener("pointercancel", endDrag);

    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 1.10 : 0.90;
      this.orbit.distance = Math.max(
        this.orbit.minDist,
        Math.min(this.orbit.maxDist, this.orbit.distance * zoomFactor)
      );
      this._updateCamera();
    }, { passive: false });

    el.addEventListener("dblclick", () => {
      this.centerView();
    });
  }

  // ------------------------------------------------------------------
  // Internal X-Ray Anatomy (Payload Bay, CG, AC, Wing Spar)
  // ------------------------------------------------------------------
  _createInternalStructures() {
    this.internalGroup = new THREE.Group();
    this.internalGroup.visible = false;
    this.group.add(this.internalGroup);

    // 1. Internal Payload Cargo Module at x = -0.104m
    const payloadGeom = new THREE.BoxGeometry(0.14, 0.08, 0.07);
    const payloadMat = new THREE.MeshStandardMaterial({
      color: 0xffb454,
      emissive: 0xff9900,
      emissiveIntensity: 0.6,
      roughness: 0.3,
      metalness: 0.5,
    });
    this.payloadMesh = new THREE.Mesh(payloadGeom, payloadMat);
    // In Three.js group coordinates: Body x is +Y, Body y is +X, Body z is -Z
    this.payloadMesh.position.set(0, -0.104, 0);
    this.internalGroup.add(this.payloadMesh);

    // Payload wireframe cage
    const pEdges = new THREE.EdgesGeometry(payloadGeom);
    this.payloadCage = new THREE.LineSegments(
      pEdges,
      new THREE.LineBasicMaterial({ color: 0xffd166, linewidth: 2 })
    );
    this.payloadMesh.add(this.payloadCage);

    // 2. Dynamic Center of Gravity (CG) Marker
    const cgGeom = new THREE.SphereGeometry(0.024, 16, 12);
    const cgMat = new THREE.MeshStandardMaterial({
      color: 0xffd166,
      emissive: 0xffaa00,
      emissiveIntensity: 0.8,
    });
    this.cgMesh = new THREE.Mesh(cgGeom, cgMat);
    this.cgMesh.position.set(0, 0, 0);
    this.internalGroup.add(this.cgMesh);

    // 3. Aerodynamic Center (AC) Marker at quarter-chord (~ +0.065m)
    const acGeom = new THREE.OctahedronGeometry(0.025, 0);
    const acMat = new THREE.MeshStandardMaterial({
      color: 0x06d6a0,
      emissive: 0x06d6a0,
      emissiveIntensity: 0.85,
    });
    this.acMesh = new THREE.Mesh(acGeom, acMat);
    this.acMesh.position.set(0, 0.065, 0);
    this.internalGroup.add(this.acMesh);

    // 4. Static Margin Vector Line (AC to CG)
    const lineGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0.065, 0),
      new THREE.Vector3(0, 0, 0),
    ]);
    this.staticMarginLine = new THREE.Line(
      lineGeom,
      new THREE.LineBasicMaterial({ color: 0x06d6a0, linewidth: 3 })
    );
    this.internalGroup.add(this.staticMarginLine);

    // 5. Internal Wing Spar (Quarter-chord structural spar)
    const sparGeom = new THREE.CylinderGeometry(0.007, 0.007, 2.0, 12);
    sparGeom.rotateZ(Math.PI / 2);
    const sparMat = new THREE.MeshStandardMaterial({
      color: 0x64748b,
      metalness: 0.8,
      roughness: 0.2,
    });
    this.wingSparMesh = new THREE.Mesh(sparGeom, sparMat);
    this.wingSparMesh.position.set(0, 0.065, 0);
    this.internalGroup.add(this.wingSparMesh);

    // 6. Fuselage Centerline / Tail Boom Tube
    const boomGeom = new THREE.CylinderGeometry(0.008, 0.008, 0.95, 12);
    boomGeom.rotateX(Math.PI / 2);
    this.tailBoomMesh = new THREE.Mesh(boomGeom, sparMat);
    this.tailBoomMesh.position.set(0, -0.25, 0);
    this.internalGroup.add(this.tailBoomMesh);

    // 7. Spinning Nose Propeller
    const propGroup = new THREE.Group();
    const spinnerGeom = new THREE.ConeGeometry(0.022, 0.06, 12);
    spinnerGeom.rotateX(-Math.PI / 2);
    const spinnerMat = new THREE.MeshStandardMaterial({ color: 0xd1d5db, metalness: 0.9, roughness: 0.1 });
    const spinner = new THREE.Mesh(spinnerGeom, spinnerMat);
    propGroup.add(spinner);

    const bladeGeom = new THREE.BoxGeometry(0.24, 0.015, 0.004);
    const bladeMat = new THREE.MeshStandardMaterial({ color: 0x1f2937, roughness: 0.4 });
    const blade = new THREE.Mesh(bladeGeom, bladeMat);
    propGroup.add(blade);

    propGroup.position.set(0, 0.36, 0);
    this.propeller = propGroup;
    this.group.add(this.propeller);
  }

  // ------------------------------------------------------------------
  // Atmospheric Disturbance & Visual Elements
  // ------------------------------------------------------------------
  _createWindField() {
    const COUNT = 160;
    const positions = new Float32Array(COUNT * 3);
    for (let i = 0; i < COUNT; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 8;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 8;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 4;
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
      color: 0x7dd3fc,
      size: 0.045,
      transparent: true,
      opacity: 0.70,
      depthWrite: false,
    });
    this.windField = new THREE.Points(geom, mat);
    this.scene.add(this.windField);
    this.windFieldPositions = positions;
  }

  _createFlightTrail() {
    const geometry = new THREE.BufferGeometry();
    const material = new THREE.LineBasicMaterial({
      color: 0x8affc1,
      transparent: true,
      opacity: 0.80,
    });
    this.flightTrailLine = new THREE.Line(geometry, material);
    this.scene.add(this.flightTrailLine);
  }

  _createEventMarker() {
    this.eventMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.12, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xff4d6d, wireframe: true })
    );
    this.eventMarker.visible = false;
    this.scene.add(this.eventMarker);
  }

  _updateWindField(frame) {
    if (!this.windField || !this.windFieldPositions) return;
    const t = Number(frame.t ?? 0);
    const gustBoost = 0.25 + (Math.abs(frame.da_deg || 0) + Math.abs(frame.de_deg || 0) + Math.abs(frame.dr_deg || 0)) / 60;
    const pos = this.windField.geometry.attributes.position.array;
    for (let i = 0; i < pos.length / 3; i++) {
      const idx = i * 3;
      const phase = t * 4.0 + i * 0.45;
      pos[idx] = (Math.sin(phase) * 3.0 + (i % 9) * 0.15) * (1.0 + gustBoost * 0.5);
      pos[idx + 1] = -1.0 + Math.cos(phase * 1.6) * 2.8;
      pos[idx + 2] = Math.sin(phase * 2.0) * 1.6 + (i % 5) * 0.1;
    }
    this.windField.geometry.attributes.position.needsUpdate = true;
  }

  _updateFlightTrail(state) {
    this.flightTrail.push([state.pe, state.pn, -state.pd]);
    if (this.flightTrail.length > 400) this.flightTrail.shift();
    const pts = this.flightTrail.map(([x, y, z]) => new THREE.Vector3(x, y, z));
    this.flightTrailLine.geometry.setFromPoints(pts);
  }

  // ------------------------------------------------------------------
  // Part Meshing & X-Ray Mode Management
  // ------------------------------------------------------------------
  _buildOrUpdatePart(name, partData) {
    const { points, tris } = partData;
    const positions = new Float32Array(points.length * 3);
    points.forEach((p, i) => {
      positions[i * 3] = p[0];
      positions[i * 3 + 1] = p[1];
      positions[i * 3 + 2] = p[2];
    });
    const indices = [];
    tris.forEach((t) => indices.push(t[0], t[1], t[2]));

    let mesh = this.partMeshes[name];
    let wire = this.wireframeMeshes[name];

    if (!mesh) {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geom.setIndex(indices);
      geom.computeVertexNormals();

      const mat = new THREE.MeshStandardMaterial({
        color: this.color,
        side: THREE.DoubleSide,
        metalness: 0.35,
        roughness: 0.45,
      });
      mesh = new THREE.Mesh(geom, mat);
      this.group.add(mesh);
      this.partMeshes[name] = mesh;

      // Wireframe overlay for See-Through / Wireframe modes
      const wireMat = new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        wireframe: true,
        transparent: true,
        opacity: 0.5,
      });
      wire = new THREE.Mesh(geom.clone(), wireMat);
      wire.visible = false;
      this.group.add(wire);
      this.wireframeMeshes[name] = wire;
    } else {
      mesh.geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      mesh.geometry.setIndex(indices);
      mesh.geometry.computeVertexNormals();
      mesh.geometry.attributes.position.needsUpdate = true;

      wire.geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      wire.geometry.setIndex(indices);
      wire.geometry.attributes.position.needsUpdate = true;
    }

    // Apply Material Mode (Solid vs See-Through vs Wireframe)
    if (this.renderMode === "seethrough") {
      mesh.material.transparent = true;
      mesh.material.opacity = 0.28;
      mesh.material.depthWrite = false;
      mesh.material.roughness = 0.15;
      mesh.material.metalness = 0.2;
      mesh.material.color.setHex(0x38bdf8);
      wire.visible = true;
      wire.material.color.setHex(0x7dd3fc);
      wire.material.opacity = 0.45;
    } else if (this.renderMode === "wireframe") {
      mesh.material.transparent = true;
      mesh.material.opacity = 0.05;
      mesh.material.depthWrite = false;
      wire.visible = true;
      wire.material.color.setHex(this.color);
      wire.material.opacity = 0.90;
    } else {
      // Solid mode with aerodynamic surface heating
      mesh.material.transparent = false;
      mesh.material.opacity = 1.0;
      mesh.material.depthWrite = true;
      mesh.material.roughness = 0.45;
      mesh.material.metalness = 0.35;
      mesh.material.color.setHex(this.color);
      wire.visible = false;

      const controlLoad = Math.min(1.0, ((Math.abs(this.lastFrame?.de_deg || 0) + Math.abs(this.lastFrame?.da_deg || 0) + Math.abs(this.lastFrame?.dr_deg || 0)) / 60.0));
      const pressureColor = new THREE.Color().setHSL(0.58 - controlLoad * 0.22, 0.70, 0.52 + controlLoad * 0.12);
      mesh.material.emissive.copy(pressureColor);
      mesh.material.emissiveIntensity = 0.10 + controlLoad * 0.70;
    }
  }

  // ------------------------------------------------------------------
  // Frame Update & Perfect Center-Aligned Camera Tracking
  // ------------------------------------------------------------------
  setRealDims(real) {
    this.real = real;
    // Immediate initial frame render so UAV appears centered right away
    this.updateFrame({
      t: 0, theta_deg: 0, phi_deg: 0, psi_deg: 0,
      de_deg: 0, da_deg: 0, dr_deg: 0, alt_m: 100, tail_inc_deg: 0,
      pn_m: 0, pe_m: 0, pd_m: 0,
    });
  }

  setChangeTime(t) {
    this.changeTime = Number.isFinite(Number(t)) ? Number(t) : null;
    this.eventMarked = false;
    if (this.eventMarker) this.eventMarker.visible = false;
  }

  setChaseCamera(enabled) {
    this.viewMode = enabled ? "chase" : "orbit";
    this._updateCamera();
  }

  setViewMode(mode) {
    this.viewMode = mode;
    this._updateCamera();
  }

  setRenderMode(mode) {
    this.renderMode = mode;
    this.internalGroup.visible = (mode === "seethrough");
    if (this.lastFrame) {
      this.updateFrame(this.lastFrame);
    }
  }

  setPayloadPreview(payloadLevel) {
    this.currentPayload = Math.max(0, Math.min(1.0, payloadLevel));
    this._updateInternalPhysics(this.currentPayload);
  }

  centerView() {
    this.orbit.panX = 0;
    this.orbit.panY = 0;
    this.orbit.azimuth = 0.45;
    this.orbit.elevation = 0.32;
    this.orbit.distance = 4.8;
    this._updateCamera();
  }

  captureSnapshot() {
    this.render();
    return this.renderer.domElement.toDataURL("image/png");
  }

  reset() {
    this.flightTrail = [];
    this.lastFrame = null;
    this.eventMarked = false;
    if (this.eventMarker) this.eventMarker.visible = false;
    if (this.flightTrailLine) this.flightTrailLine.geometry.setFromPoints([]);
  }

  _updateInternalPhysics(payloadLevel) {
    // Exact aerodynamic & CG equations matching aircraft_params.py:
    // payload_mass_max = 1.5kg, empty mass = 2.5kg, payload arm = -0.104m
    const plMass = payloadLevel * 1.5;
    const totalMass = 2.5 + plMass;
    const xcg = (plMass * -0.104) / totalMass; // in body X (meters aft)

    // In Three.js body frame (+Y is forward, so xcg moves along -Y)
    this.cgMesh.position.set(0, xcg, 0);

    // Scale payload box size with mass
    const scaleFactor = 0.6 + payloadLevel * 0.6;
    this.payloadMesh.scale.set(scaleFactor, scaleFactor, scaleFactor);

    // Update static margin line
    const acPos = new THREE.Vector3(0, 0.065, 0);
    const cgPos = new THREE.Vector3(0, xcg, 0);
    this.staticMarginLine.geometry.setFromPoints([acPos, cgPos]);

    // Color code static margin (green if positive/stable, red if neutral/unstable)
    const staticMargin = 0.065 - xcg;
    if (staticMargin < 0.02) {
      this.staticMarginLine.material.color.setHex(0xff4d6d);
    } else {
      this.staticMarginLine.material.color.setHex(0x06d6a0);
    }
  }

  updateFrame(frame) {
    if (!this.real) return;
    if (this.lastFrame && Number(frame.t) < Number(this.lastFrame.t)) this.reset();
    this.lastFrame = frame;

    const t = Number(frame.t ?? 0);
    const hasBackendPos = [frame.pn_m, frame.pe_m, frame.pd_m].every((v) => Number.isFinite(Number(v)));
    const flightDist = t * 0.45;

    const state = {
      pn: hasBackendPos ? Number(frame.pn_m) : flightDist,
      pe: hasBackendPos ? Number(frame.pe_m) : 0,
      pd: hasBackendPos ? Number(frame.pd_m) : 0,
      phi: (frame.phi_deg * Math.PI) / 180,
      theta: (frame.theta_deg * Math.PI) / 180,
      psi: (frame.psi_deg * Math.PI) / 180,
    };
    const controls = {
      de: ((frame.de_deg || 0) * Math.PI) / 180,
      da: ((frame.da_deg || 0) * Math.PI) / 180,
      dr: ((frame.dr_deg || 0) * Math.PI) / 180,
    };
    const tailInc = ((frame.tail_inc_deg || 0) * Math.PI) / 180;

    // Spin propeller
    if (this.propeller) {
      this.propeller.rotation.y = t * 38.0;
    }

    this._updateWindField(frame);
    this._updateFlightTrail(state);

    // World position of UAV in Three.js (X = East, Y = North, Z = Up)
    const wx = state.pe;
    const wy = state.pn;
    const wz = -state.pd;
    this.worldPos = new THREE.Vector3(wx, wy, wz);

    // The group moves through space
    this.group.position.copy(this.worldPos);

    // Keep grid and wind field anchored near the aircraft elevation
    this.grid.position.set(wx, wy, Math.floor(wz / 10) * 10 - 2.5);
    this.windField.position.set(wx, wy, wz);

    // Update internal payload and CG marker
    const plEst = frame.pl_est !== undefined ? frame.pl_est : this.currentPayload;
    this._updateInternalPhysics(plEst);

    // Assemble parts in local frame (pn=0, pe=0, pd=0)
    const localState = { ...state, pn: 0, pe: 0, pd: 0 };
    const world = UAVGeometry.assembleWorld(this.real, localState, controls, tailInc);
    for (const name of Object.keys(world)) {
      this._buildOrUpdatePart(name, world[name]);
    }

    if (this.changeTime !== null && !this.eventMarked && t >= this.changeTime) {
      this.eventMarker.position.copy(this.worldPos);
      this.eventMarker.visible = true;
      this.eventMarked = true;
    }

    this._updateCamera();
  }

  // ------------------------------------------------------------------
  // Camera Alignment (Guarantees UAV is in the Middle of Viewport)
  // ------------------------------------------------------------------
  _updateCamera() {
    const target = this.worldPos ? this.worldPos.clone() : new THREE.Vector3(0, 0, 0);

    // Offset target slightly to the geometric volumetric center of the UAV
    // (body is ~1.06m long, geometric center is ~0.10m aft of empty origin)
    const psi = this.lastFrame ? (this.lastFrame.psi_deg * Math.PI) / 180 : 0;
    const forward = new THREE.Vector3(Math.sin(psi), Math.cos(psi), 0).normalize();
    const right = new THREE.Vector3(Math.cos(psi), -Math.sin(psi), 0).normalize();
    const up = new THREE.Vector3(0, 0, 1);

    const centerTarget = target.clone().addScaledVector(forward, -0.10);

    // Add user pan offsets
    centerTarget.addScaledVector(right, this.orbit.panX);
    centerTarget.addScaledVector(up, this.orbit.panY);

    if (this.viewMode === "top") {
      this.camera.position.set(centerTarget.x, centerTarget.y, centerTarget.z + this.orbit.distance);
      this.camera.up.set(0, 1, 0);
      this.camera.lookAt(centerTarget);
      return;
    }

    if (this.viewMode === "side") {
      const sidePos = centerTarget.clone().addScaledVector(right, this.orbit.distance);
      this.camera.position.copy(sidePos);
      this.camera.up.set(0, 0, 1);
      this.camera.lookAt(centerTarget);
      return;
    }

    if (this.viewMode === "front") {
      const frontPos = centerTarget.clone().addScaledVector(forward, this.orbit.distance);
      frontPos.z += 0.4;
      this.camera.position.copy(frontPos);
      this.camera.up.set(0, 0, 1);
      this.camera.lookAt(centerTarget);
      return;
    }

    if (this.viewMode === "chase") {
      // True chase camera: positioned directly behind aircraft and looking straight at centerTarget
      const chaseDist = this.orbit.distance;
      const chaseHeight = Math.sin(this.orbit.elevation) * chaseDist;
      const chaseBack = Math.cos(this.orbit.elevation) * chaseDist;

      // Allow subtle azimuth offset
      const chaseDir = forward.clone().applyAxisAngle(up, this.orbit.azimuth);
      const camPos = centerTarget.clone().addScaledVector(chaseDir, -chaseBack);
      camPos.z += Math.max(0.6, chaseHeight);

      this.camera.position.copy(camPos);
      this.camera.up.set(0, 0, 1);
      this.camera.lookAt(centerTarget); // LOOK DIRECTLY AT THE UAV CENTER!
      return;
    }

    // Default Free Orbit Mode (Spherical Coordinates around centerTarget)
    const r = this.orbit.distance;
    const phi = this.orbit.elevation;
    const theta = this.orbit.azimuth;

    const x = centerTarget.x + r * Math.cos(phi) * Math.sin(theta);
    const y = centerTarget.y - r * Math.cos(phi) * Math.cos(theta);
    const z = centerTarget.z + r * Math.sin(phi);

    this.camera.position.set(x, y, z);
    this.camera.up.set(0, 0, 1);
    this.camera.lookAt(centerTarget);
  }

  render() {
    this._resize();
    this.renderer.render(this.scene, this.camera);
  }
}

window.AircraftViewport = AircraftViewport;
