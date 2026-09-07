import * as BABYLON from "babylonjs";

class Playground {
    public static async CreateScene(engine: BABYLON.Engine, canvas: HTMLCanvasElement): Promise<BABYLON.Scene> {
        var scene = new BABYLON.Scene(engine);

        // =========================================================
        // 1. ADD YOUR LOCAL LIGHTS AND MESHES HERE
        // =========================================================
        var camera = new BABYLON.ArcRotateCamera("camera", 0, Math.PI / 3, 5, BABYLON.Vector3.Zero(), scene);
        camera.attachControl(canvas, true);
        
        var light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 0), scene);
        light.intensity = 0.7;

        // A visual reference for the floor
        var ground = BABYLON.MeshBuilder.CreateGround("ground", {width: 4, height: 4}, scene);
        var mat = new BABYLON.StandardMaterial("groundMat", scene);
        mat.diffuseColor = new BABYLON.Color3(0.2, 0.2, 0.2);
        ground.material = mat;

        // =========================================================
        // 2. SETUP FLUID RENDERER
        // =========================================================
        const fluidRenderer = scene.enableFluidRenderer()!;
        const fluidRenderObject = fluidRenderer.addCustomParticles({}, 0, false, undefined, camera);
        
        // Visual styling for the fluid
        fluidRenderObject.targetRenderer.fluidColor = new BABYLON.Color3(0.5, 0.8, 0.95);
        fluidRenderObject.targetRenderer.density = 2.2;
        fluidRenderObject.targetRenderer.refractionStrength = 0.02;
        fluidRenderObject.targetRenderer.specularPower = 150;
        
        fluidRenderObject.object.particleSize = 0.08;
        fluidRenderObject.object.particleThicknessAlpha = 0.08;
        fluidRenderObject.object.useVelocity = true;

        // =========================================================
        // 3. SETUP SPH FLUID SIMULATOR (CPU PHYSICS)
        // =========================================================
        const numParticles = 2000;
        const fluidSim = new FluidSimulator();
        fluidSim.smoothingRadius = 0.08;
        fluidSim.densityReference = 6000;
        fluidSim.pressureConstant = 10;
        fluidSim.viscosity = 0.01;
        fluidSim.maxVelocity = 4;
        fluidSim.maxAcceleration = 2000;

        // Generate a starting block of fluid
        const positions = new Float32Array(numParticles * 3);
        const velocities = new Float32Array(numParticles * 3);
        let idx = 0;
        let side = Math.cbrt(numParticles);
        let spacing = 0.05;
        for(let x=0; x<side; x++) {
            for(let y=0; y<side; y++) {
                for(let z=0; z<side; z++) {
                    if (idx >= numParticles) break;
                    // Start 2 units in the air
                    positions[idx*3] = (x - side/2) * spacing;
                    positions[idx*3+1] = 2 + y * spacing; 
                    positions[idx*3+2] = (z - side/2) * spacing;
                    idx++;
                }
            }
        }
        
        fluidSim.setParticleData(positions, velocities);
        fluidSim.currentNumParticles = numParticles;
        (fluidRenderObject.object as BABYLON.FluidRenderingObjectCustomParticles).setNumParticles(numParticles);

        // =========================================================
        // 4. RENDER LOOP & BASIC COLLISIONS
        // =========================================================
        scene.onBeforeRenderObservable.add(() => {
            // Run fluid math
            fluidSim.update(1 / 100);

            // Basic bounding box collisions (replace with your own logic if needed)
            const pos = fluidSim.positions;
            const vel = fluidSim.velocities;
            const particleRadius = 0.04;
            const restitution = 0.5; // Bounciness

            for (let i = 0; i < fluidSim.currentNumParticles; i++) {
                // Floor collision
                if (pos[i*3 + 1] < particleRadius) {
                    pos[i*3 + 1] = particleRadius;
                    vel[i*3 + 1] *= -restitution;
                }
                // Wall collisions (to keep fluid on the 4x4 ground)
                if (pos[i*3 + 0] < -2) { pos[i*3 + 0] = -2; vel[i*3 + 0] *= -restitution; }
                if (pos[i*3 + 0] > 2) { pos[i*3 + 0] = 2; vel[i*3 + 0] *= -restitution; }
                if (pos[i*3 + 2] < -2) { pos[i*3 + 2] = -2; vel[i*3 + 2] *= -restitution; }
                if (pos[i*3 + 2] > 2) { pos[i*3 + 2] = 2; vel[i*3 + 2] *= -restitution; }
            }

            // Push updated physics to the renderer
            if (fluidRenderObject.object.vertexBuffers["position"]) {
                fluidRenderObject.object.vertexBuffers["position"].updateDirectly(fluidSim.positions, 0);
                fluidRenderObject.object.vertexBuffers["velocity"].updateDirectly(fluidSim.velocities, 0);
            } else {
                fluidRenderObject.object.vertexBuffers["position"] = new BABYLON.VertexBuffer(
                    engine, fluidSim.positions, BABYLON.VertexBuffer.PositionKind, true, false, 3, true
                );
                fluidRenderObject.object.vertexBuffers["velocity"] = new BABYLON.VertexBuffer(
                    engine, fluidSim.velocities, "velocity", true, false, 3, true
                );
            }
        });

        return scene;
    }
}

// =========================================================
// REQUIRED FLUID MATH CLASSES (Unchanged from original)
// =========================================================

interface IFluidParticle {
    mass: number;
    density: number;
    pressure: number;
    accelX: number;
    accelY: number;
    accelZ: number;
}

class FluidSimulator {
    protected _particles: IFluidParticle[];
    protected _numMaxParticles: number;
    public _positions: Float32Array;
    public _velocities: Float32Array;
    protected _hash: Hash;

    protected _smoothingRadius2: number;
    protected _poly6Constant: number;
    protected _spikyConstant: number;
    protected _viscConstant: number;

    protected _smoothingRadius = 0.2;

    public get smoothingRadius() { return this._smoothingRadius; }
    public set smoothingRadius(radius: number) {
        this._smoothingRadius = radius;
        this._computeConstants();
    }

    public densityReference = 2000;
    public pressureConstant = 20;
    public viscosity = 0.005;
    public gravity = new BABYLON.Vector3(0, -9.8, 0);
    public minTimeStep = 1 / 100;
    public maxVelocity = 75;
    public maxAcceleration = 2000;
    public currentNumParticles: number;
    private _mass: number;

    public get mass() { return this._mass; }
    public set mass(m: number) {
        for (let i = 0; i < this._particles.length; ++i) {
            this._particles[i].mass = m;
        }
    }

    private _computeConstants(): void {
        this._smoothingRadius2 = this._smoothingRadius * this._smoothingRadius;
        this._poly6Constant = 315 / (64 * Math.PI * Math.pow(this._smoothingRadius, 9));
        this._spikyConstant = -45 / (Math.PI * Math.pow(this._smoothingRadius, 6));
        this._viscConstant = 45 / (Math.PI * Math.pow(this._smoothingRadius, 6));
        this._hash = new Hash(this._smoothingRadius, this._numMaxParticles);
    }

    public get positions() { return this._positions; }
    public get velocities() { return this._velocities; }
    public get numMaxParticles() { return this._numMaxParticles; }

    public setParticleData(positions?: Float32Array, velocities?: Float32Array): void {
        this._positions = positions ?? new Float32Array();
        this._velocities = velocities ?? new Float32Array();
        this._numMaxParticles = this._positions.length / 3;
        this._hash = new Hash(this._smoothingRadius, this._numMaxParticles);

        for (let i = this._particles.length; i < this._numMaxParticles; ++i) {
            this._particles.push({ mass: this.mass, density: 0, pressure: 0, accelX: 0, accelY: 0, accelZ: 0 });
        }
    }

    constructor(positions?: Float32Array, velocities?: Float32Array, mass = 1) {
        this._positions = undefined as any;
        this._velocities = undefined as any;
        this._particles = [];
        this._numMaxParticles = 0;
        this._mass = mass;

        if (positions && velocities) { this.setParticleData(positions, velocities); }

        this._hash = new Hash(this._smoothingRadius, this._numMaxParticles);
        this.currentNumParticles = this._numMaxParticles;
        this._smoothingRadius2 = 0;
        this._poly6Constant = 0;
        this._spikyConstant = 0;
        this._viscConstant = 0;
        this._computeConstants();
    }

    public update(deltaTime: number): void {
        let timeLeft = deltaTime;
        while (timeLeft > 0) {
            this._hash.create(this._positions, this.currentNumParticles);
            this._computeDensityAndPressure();
            this._computeAcceleration();

            let timeStep = this._calculateTimeStep();
            timeLeft -= timeStep;
            if (timeLeft < 0) {
                timeStep += timeLeft;
                timeLeft = 0;
            }
            this._updatePositions(timeStep);
        }
    }

    protected _computeDensityAndPressure(): void {
        for (let a = 0; a < this.currentNumParticles; ++a) {
            const pA = this._particles[a];
            const paX = this._positions[a * 3 + 0];
            const paY = this._positions[a * 3 + 1];
            const paZ = this._positions[a * 3 + 2];

            pA.density = 0;
            this._hash.query(this._positions, a, this._smoothingRadius);

            for (let ib = 0; ib < this._hash.querySize; ++ib) {
                const b = this._hash.queryIds[ib];
                const diffX = paX - this._positions[b * 3 + 0];
                const diffY = paY - this._positions[b * 3 + 1];
                const diffZ = paZ - this._positions[b * 3 + 2];
                const r2 = diffX * diffX + diffY * diffY + diffZ * diffZ;

                if (r2 < this._smoothingRadius2) {
                    const w = this._poly6Constant * Math.pow(this._smoothingRadius2 - r2, 3);
                    pA.density += w;
                }
            }
            pA.density = Math.max(this.densityReference, pA.density);
            pA.pressure = this.pressureConstant * (pA.density - this.densityReference);
        }
    }

    protected _computeAcceleration(): void {
        for (let a = 0; a < this.currentNumParticles; ++a) {
            const pA = this._particles[a];
            const paX = this._positions[a * 3 + 0];
            const paY = this._positions[a * 3 + 1];
            const paZ = this._positions[a * 3 + 2];

            const vaX = this._velocities[a * 3 + 0];
            const vaY = this._velocities[a * 3 + 1];
            const vaZ = this._velocities[a * 3 + 2];

            let pressureAccelX = 0, pressureAccelY = 0, pressureAccelZ = 0;
            let viscosityAccelX = 0, viscosityAccelY = 0, viscosityAccelZ = 0;

            this._hash.query(this._positions, a, this._smoothingRadius);

            for (let ib = 0; ib < this._hash.querySize; ++ib) {
                const b = this._hash.queryIds[ib];
                let diffX = paX - this._positions[b * 3 + 0];
                let diffY = paY - this._positions[b * 3 + 1];
                let diffZ = paZ - this._positions[b * 3 + 2];
                const r2 = diffX * diffX + diffY * diffY + diffZ * diffZ;
                const r = Math.sqrt(r2);

                if (r > 0 && r2 < this._smoothingRadius2) {
                    const pB = this._particles[b];
                    diffX /= r; diffY /= r; diffZ /= r;

                    const w = this._spikyConstant * (this._smoothingRadius - r) * (this._smoothingRadius - r);
                    const massRatio = pB.mass / pA.mass;
                    const fp = w * ((pA.pressure + pB.pressure) / (2 * pA.density * pB.density)) * massRatio;

                    pressureAccelX -= fp * diffX;
                    pressureAccelY -= fp * diffY;
                    pressureAccelZ -= fp * diffZ;

                    const w2 = this._viscConstant * (this._smoothingRadius - r);
                    const fv = w2 * (1 / pB.density) * massRatio * this.viscosity;

                    viscosityAccelX += fv * (this._velocities[b * 3 + 0] - vaX);
                    viscosityAccelY += fv * (this._velocities[b * 3 + 1] - vaY);
                    viscosityAccelZ += fv * (this._velocities[b * 3 + 2] - vaZ);
                }
            }

            pA.accelX = pressureAccelX + viscosityAccelX + this.gravity.x;
            pA.accelY = pressureAccelY + viscosityAccelY + this.gravity.y;
            pA.accelZ = pressureAccelZ + viscosityAccelZ + this.gravity.z;

            const mag = Math.sqrt(pA.accelX * pA.accelX + pA.accelY * pA.accelY + pA.accelZ * pA.accelZ);
            if (mag > this.maxAcceleration) {
                pA.accelX = (pA.accelX / mag) * this.maxAcceleration;
                pA.accelY = (pA.accelY / mag) * this.maxAcceleration;
                pA.accelZ = (pA.accelZ / mag) * this.maxAcceleration;
            }
        }
    }

    protected _calculateTimeStep() {
        let maxVelocity = 0, maxAcceleration = 0, maxSpeedOfSound = 0;

        for (let a = 0; a < this.currentNumParticles; ++a) {
            const pA = this._particles[a];
            const velSq = this._velocities[a * 3 + 0] * this._velocities[a * 3 + 0] + this._velocities[a * 3 + 1] * this._velocities[a * 3 + 1] + this._velocities[a * 3 + 2] * this._velocities[a * 3 + 2];
            const accSq = pA.accelX * pA.accelX + pA.accelY * pA.accelY + pA.accelZ * pA.accelZ;
            const spsSq = pA.density < 0.00001 ? 0 : pA.pressure / pA.density;

            if (velSq > maxVelocity) maxVelocity = velSq;
            if (accSq > maxAcceleration) maxAcceleration = accSq;
            if (spsSq > maxSpeedOfSound) maxSpeedOfSound = spsSq;
        }

        maxVelocity = Math.sqrt(maxVelocity);
        maxAcceleration = Math.sqrt(maxAcceleration);
        maxSpeedOfSound = Math.sqrt(maxSpeedOfSound);

        const velStep = (0.4 * this.smoothingRadius) / Math.max(1, maxVelocity);
        const accStep = 0.4 * Math.sqrt(this.smoothingRadius / maxAcceleration);
        const spsStep = this.smoothingRadius / maxSpeedOfSound;

        return Math.max(this.minTimeStep, Math.min(velStep, accStep, spsStep));
    }

    protected _updatePositions(deltaTime: number): void {
        for (let a = 0; a < this.currentNumParticles; ++a) {
            const pA = this._particles[a];

            this._velocities[a * 3 + 0] += pA.accelX * deltaTime;
            this._velocities[a * 3 + 1] += pA.accelY * deltaTime;
            this._velocities[a * 3 + 2] += pA.accelZ * deltaTime;

            const mag = Math.sqrt(this._velocities[a * 3 + 0] * this._velocities[a * 3 + 0] + this._velocities[a * 3 + 1] * this._velocities[a * 3 + 1] + this._velocities[a * 3 + 2] * this._velocities[a * 3 + 2]);

            if (mag > this.maxVelocity) {
                this._velocities[a * 3 + 0] = (this._velocities[a * 3 + 0] / mag) * this.maxVelocity;
                this._velocities[a * 3 + 1] = (this._velocities[a * 3 + 1] / mag) * this.maxVelocity;
                this._velocities[a * 3 + 2] = (this._velocities[a * 3 + 2] / mag) * this.maxVelocity;
            }

            this._positions[a * 3 + 0] += deltaTime * this._velocities[a * 3 + 0];
            this._positions[a * 3 + 1] += deltaTime * this._velocities[a * 3 + 1];
            this._positions[a * 3 + 2] += deltaTime * this._velocities[a * 3 + 2];
        }
    }
}

class Hash {
    private _spacing: number;
    private _tableSize: number;
    private _cellStart: Int32Array;
    private _cellEntries: Int32Array;
    private _queryIds: Int32Array;
    private _querySize: number;

    public get querySize() { return this._querySize; }
    public get queryIds() { return this._queryIds; }

    constructor(spacing: number, maxNumObjects: number) {
        this._spacing = spacing;
        this._tableSize = 2 * maxNumObjects;
        this._cellStart = new Int32Array(this._tableSize + 1);
        this._cellEntries = new Int32Array(maxNumObjects);
        this._queryIds = new Int32Array(maxNumObjects);
        this._querySize = 0;
    }

    public hashCoords(xi: number, yi: number, zi: number) {
        const h = (xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481); 
        return Math.abs(h) % this._tableSize;
    }

    public intCoord(coord: number) { return Math.floor(coord / this._spacing); }

    public hashPos(pos: number[] | Float32Array, nr: number) {
        return this.hashCoords(this.intCoord(pos[3 * nr]), this.intCoord(pos[3 * nr + 1]), this.intCoord(pos[3 * nr + 2]));
    }

    public create(pos: number[] | Float32Array, numElements?: number) {
        numElements = numElements ?? pos.length / 3;
        const numObjects = Math.min(numElements, this._cellEntries.length);
        this._cellStart.fill(0);
        this._cellEntries.fill(0);

        for (let i = 0; i < numObjects; i++) {
            const h = this.hashPos(pos, i);
            this._cellStart[h]++;
        }

        let start = 0;
        for (let i = 0; i < this._tableSize; i++) {
            start += this._cellStart[i];
            this._cellStart[i] = start;
        }
        this._cellStart[this._tableSize] = start; 

        for (let i = 0; i < numObjects; i++) {
            const h = this.hashPos(pos, i);
            this._cellStart[h]--;
            this._cellEntries[this._cellStart[h]] = i;
        }
    }

    public query(pos: number[] | Float32Array, nr: number, maxDist: number) {
        const x0 = this.intCoord(pos[3 * nr] - maxDist);
        const y0 = this.intCoord(pos[3 * nr + 1] - maxDist);
        const z0 = this.intCoord(pos[3 * nr + 2] - maxDist);

        const x1 = this.intCoord(pos[3 * nr] + maxDist);
        const y1 = this.intCoord(pos[3 * nr + 1] + maxDist);
        const z1 = this.intCoord(pos[3 * nr + 2] + maxDist);

        this._querySize = 0;

        for (let xi = x0; xi <= x1; xi++) {
            for (let yi = y0; yi <= y1; yi++) {
                for (let zi = z0; zi <= z1; zi++) {
                    const h = this.hashCoords(xi, yi, zi);
                    const start = this._cellStart[h];
                    const end = this._cellStart[h + 1];

                    for (let i = start; i < end; i++) {
                        this._queryIds[this._querySize] = this._cellEntries[i];
                        this._querySize++;
                    }
                }
            }
        }
    }
}

// =========================================================
// ENGINE INITIALIZATION 
// =========================================================

const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
const engine = new BABYLON.Engine(canvas, true);

Playground.CreateScene(engine, canvas).then((scene) => {
    engine.runRenderLoop(() => {
        scene.render();
    });
});

window.addEventListener("resize", () => {
    engine.resize();
});